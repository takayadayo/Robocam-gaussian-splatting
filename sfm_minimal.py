import os
import glob
import json
from collections import defaultdict

import numpy as np
import cv2


def load_images(img_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    files.sort()
    return files


def load_intrinsics(path):
    if not os.path.exists(path):
        return None, None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None, None
    K_node = fs.getNode("K")
    dist_node = fs.getNode("dist")
    if K_node.empty():
        fs.release()
        return None, None
    K = K_node.mat().astype(np.float64)
    dist = None
    if not dist_node.empty():
        dist = dist_node.mat().astype(np.float64).reshape(-1, 1)
    fs.release()
    return K, dist


def undistort_images(imgs, K, dist):
    if K is None or dist is None:
        return imgs, None
    h, w = imgs[0].shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undistorted = [cv2.undistort(im, K, dist, None, newK) for im in imgs]
    return undistorted, newK


def build_initial_K(width, height, scale=1.2):
    f = scale * max(width, height)
    return np.array([[f, 0, width * 0.5],
                     [0, f, height * 0.5],
                     [0, 0, 1.0]], dtype=np.float64)


def extract_sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sift = cv2.SIFT_create()
    keypoints, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return keypoints, desc


def match_features(desc_a, desc_b, ratio=0.75):
    if len(desc_a) == 0 or len(desc_b) == 0:
        return []
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_ab = flann.knnMatch(desc_a, desc_b, k=2)
    knn_ba = flann.knnMatch(desc_b, desc_a, k=2)
    good_ab = []
    for m, n in knn_ab:
        if m.distance < ratio * n.distance:
            good_ab.append((m.queryIdx, m.trainIdx))
    good_ba = []
    for m, n in knn_ba:
        if m.distance < ratio * n.distance:
            good_ba.append((m.trainIdx, m.queryIdx))
    set_ba = set(good_ba)
    matches = [(i, j) for (i, j) in good_ab if (i, j) in set_ba]
    return matches


def essential_inliers(kp_a, kp_b, matches, K):
    if len(matches) < 8:
        return None, None
    pts_a = np.float32([kp_a[i].pt for i, _ in matches])
    pts_b = np.float32([kp_b[j].pt for _, j in matches])
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=2.0)
    if E is None or mask is None:
        return None, None
    inlier_idx = np.where(mask.ravel() > 0)[0]
    if len(inlier_idx) < 8:
        return None, None
    inlier_matches = [matches[idx] for idx in inlier_idx]
    return E, inlier_matches


def triangulate_two_view(kp_a, kp_b, matches, K):
    pts_a = np.float32([kp_a[i].pt for i, _ in matches])
    pts_b = np.float32([kp_b[j].pt for _, j in matches])
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.5)
    if E is None or mask is None:
        return None
    inlier_idx = np.where(mask.ravel() > 0)[0]
    if len(inlier_idx) < 12:
        return None
    pts_a = pts_a[inlier_idx]
    pts_b = pts_b[inlier_idx]
    matches = [matches[i] for i in inlier_idx]
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K)
    if mask_pose is None:
        return None
    pose_inliers = np.where(mask_pose.ravel() > 0)[0]
    if len(pose_inliers) < 10:
        return None
    pts_a = pts_a[pose_inliers]
    pts_b = pts_b[pose_inliers]
    matches = [matches[i] for i in pose_inliers]

    pts_a_h = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K, None).reshape(-1, 2).T
    pts_b_h = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K, None).reshape(-1, 2).T
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, t])
    X_h = cv2.triangulatePoints(P0, P1, pts_a_h, pts_b_h)
    X = X_h[:3] / X_h[3:]

    depth0 = X[2, :] > 1e-4
    depth1 = (R @ X + t)[2, :] > 1e-4
    valid = depth0 & depth1
    if np.count_nonzero(valid) < 15:
        return None

    X = X[:, valid]
    matches = [matches[i] for i, flag in enumerate(valid) if flag]
    return R, t.reshape(3, 1), X, matches


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def build_tracks(keypoints, match_dict):
    offsets = []
    gid = 0
    for kp in keypoints:
        offsets.append(gid)
        gid += len(kp)

    dsu = DSU(gid)
    for (i, j), pairs in match_dict.items():
        offset_i = offsets[i]
        offset_j = offsets[j]
        for a, b in pairs:
            dsu.union(offset_i + a, offset_j + b)

    tracks = defaultdict(list)
    for img_id, kps in enumerate(keypoints):
        offset = offsets[img_id]
        for idx in range(len(kps)):
            root = dsu.find(offset + idx)
            tracks[root].append((img_id, idx))

    filtered = [obs for obs in tracks.values() if len(obs) >= 2]
    return filtered


def triangulate_n_views(K, cams, observations):
    A = []
    for (R, t), uv in zip(cams, observations):
        P = K @ np.hstack([R, t])
        u, v = uv
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X_h /= X_h[3]
    return X_h[:3]


def project_point(K, R, t, X):
    x = R @ X + t
    if x[2] <= 1e-8:
        return None
    x = x / x[2]
    u = K @ x
    return u[:2].reshape(2)


def prune_outliers(K, cameras, points3d, observations, base_thresh=3.0, max_iter=3):
    removed_obs_total = 0
    removed_pts_total = 0
    for _ in range(max_iter):
        removed_obs = 0
        removed_pts = 0
        for pid, X in enumerate(points3d):
            if X is None:
                continue
            obs = observations[pid]
            residuals = []
            for img_id, uv in obs:
                cam = cameras[img_id]
                if cam is None:
                    continue
                R, t = cam
                proj = project_point(K, R, t, X.reshape(3, 1))
                if proj is None:
                    continue
                error = float(np.linalg.norm(proj - uv))
                residuals.append((img_id, uv, error))
            if len(residuals) < 2:
                points3d[pid] = None
                observations[pid] = []
                removed_pts += 1
                continue
            errs = np.array([r[2] for r in residuals], dtype=np.float64)
            median = float(np.median(errs))
            mad = float(np.median(np.abs(errs - median))) + 1e-6
            thresh = max(base_thresh, median + 3.0 * 1.4826 * mad)
            new_obs = []
            for img_id, uv, err in residuals:
                if err <= thresh:
                    new_obs.append((img_id, uv))
                else:
                    removed_obs += 1
            observations[pid] = new_obs
            if len(new_obs) < 2:
                points3d[pid] = None
                removed_pts += 1
        removed_obs_total += removed_obs
        removed_pts_total += removed_pts
        if removed_obs == 0 and removed_pts == 0:
            break
    return removed_obs_total, removed_pts_total


def reprojection_stats(K, cameras, points3d, observations):
    residuals = []
    per_view = defaultdict(list)
    for pid, X in enumerate(points3d):
        if X is None:
            continue
        for img_id, uv in observations[pid]:
            cam = cameras[img_id]
            if cam is None:
                continue
            R, t = cam
            proj = project_point(K, R, t, X.reshape(3, 1))
            if proj is None:
                continue
            error = float(np.linalg.norm(proj - uv))
            residuals.append(error)
            per_view[img_id].append(error)
    if not residuals:
        return dict(global_rms=None, per_view=None)
    residuals = np.array(residuals, dtype=np.float64)
    stats = dict(global_rms=float(np.sqrt(np.mean(residuals ** 2))), per_view={})
    for vid, vals in per_view.items():
        arr = np.array(vals, dtype=np.float64)
        stats["per_view"][vid] = dict(
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            std=float(arr.std()),
            count=int(arr.size),
        )
    return stats


def select_initial_pair(kps, descs, K):
    candidate_pairs = []
    match_dict = {}
    n = len(kps)
    for i in range(n - 1):
        for j in range(i + 1, n):
            matches = match_features(descs[i], descs[j])
            E, inliers = essential_inliers(kps[i], kps[j], matches, K)
            if inliers is None or len(inliers) < 25:
                continue
            result = triangulate_two_view(kps[i], kps[j], inliers, K)
            if result is None:
                continue
            R, t, X, kept = result
            baseline = np.linalg.norm(t)
            if baseline < 1e-3:
                continue
            candidate_pairs.append((len(kept), baseline, i, j, R, t, X, kept))
            match_dict[(i, j)] = kept
    if not candidate_pairs:
        raise RuntimeError("No valid initial pair could be triangulated.")
    candidate_pairs.sort(reverse=True)
    top = candidate_pairs[0]
    _, _, i0, i1, R, t, X, kept = top
    return i0, i1, R, t, X, kept, match_dict


def run_sfm(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = load_images(img_dir)
    if len(files) < 2:
        raise RuntimeError("Need at least two images.")

    imgs_raw = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
    H, W = imgs_raw[0].shape[:2]
    K = build_initial_K(W, H, scale=1.2)

    intrinsics_path = os.path.join(out_dir, "intrinsics.yaml")
    K_loaded, dist_loaded = load_intrinsics(intrinsics_path)
    if K_loaded is not None and dist_loaded is not None:
        imgs, newK = undistort_images(imgs_raw, K_loaded, dist_loaded)
        if newK is not None:
            K = newK
        else:
            K = K_loaded
        print(f"[info] using intrinsics from {intrinsics_path}; images undistorted.")
    else:
        imgs = imgs_raw
        print("[warn] intrinsics.yaml unavailable or incomplete; using approximate K and zero distortion.")

    features = [extract_sift(im) for im in imgs]
    keypoints = [f[0] for f in features]
    descriptors = [f[1] for f in features]

    i0, i1, R01, t01, X01, kept_pairs, initial_match_dict = select_initial_pair(keypoints, descriptors, K)
    match_dict = initial_match_dict.copy()

    cameras = [None] * len(files)
    cameras[i0] = (np.eye(3), np.zeros((3, 1)))
    cameras[i1] = (R01, t01)

    tracks = build_tracks(keypoints, match_dict)
    pid_map = {}
    points3d = []
    observations = []
    for obs in tracks:
        imgs_in = [img_id for img_id, _ in obs]
        if i0 in imgs_in and i1 in imgs_in:
            kp0 = [k for (img_id, k) in obs if img_id == i0][0]
            kp1 = [k for (img_id, k) in obs if img_id == i1][0]
            key = (kp0, kp1)
            if key not in {(a, b) for a, b in kept_pairs}:
                continue
            idx = len(points3d)
            X = X01[:, [kept_pairs.index((kp0, kp1))]]
            points3d.append(X.reshape(3))
            obs_list = []
            for (img_id, kp_idx) in obs:
                uv = np.array(keypoints[img_id][kp_idx].pt, dtype=np.float64)
                obs_list.append((img_id, uv))
            observations.append(obs_list)
            pid_map[tuple(sorted(obs))] = idx

    registered = set([i0, i1])
    remaining = [idx for idx in range(len(files)) if idx not in registered]

    for idx in remaining:
        X3 = []
        x2 = []
        for pid, X in enumerate(points3d):
            if X is None:
                continue
            for (img_id, uv) in observations[pid]:
                if img_id == idx:
                    X3.append(X)
                    x2.append(uv)
                    break
        if len(X3) < 6:
            continue
        X3 = np.array(X3, dtype=np.float64).reshape(-1, 1, 3)
        x2 = np.array(x2, dtype=np.float64).reshape(-1, 1, 2)
        pose = cv2.solvePnPRansac(X3, x2, K, None, flags=cv2.SOLVEPNP_AP3P,
                                  reprojectionError=2.0, confidence=0.999, iterationsCount=2000)
        if pose[0]:
            rvec, tvec = pose[1], pose[2]
            success, rvec, tvec = cv2.solvePnP(X3, x2, K, None, rvec, tvec,
                                               useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                cameras[idx] = (R, tvec)
                registered.add(idx)

        matches = match_features(descriptors[i0], descriptors[idx])
        matches += match_features(descriptors[i1], descriptors[idx])
        if matches:
            match_dict[(min(idx, i0), max(idx, i0))] = matches

    if len(registered) < 2 or not points3d:
        print("[ERROR] insufficient reconstruction results for reliable reprojection stats.")
        print(f"  -> registered cameras: {len(registered)}")
        print(f"  -> reconstructed points: {sum(1 for p in points3d if p is not None)}")
        print("  -> global_rms: None")
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({
                "global_rms_px": None,
                "per_view": None,
                "num_registered": len(registered),
                "num_points": 0,
                "note": "insufficient data to compute reliable metrics"
            }, f, indent=2)
        print(f"[INFO] wrote diagnostic outputs to {out_dir}")
        return

    prune_outliers(K, cameras, points3d, observations, base_thresh=3.0, max_iter=3)
    stats = reprojection_stats(K, cameras, points3d, observations)

    num_registered = sum(1 for cam in cameras if cam is not None)
    num_points = sum(1 for p in points3d if p is not None)
    print(f"[INFO] registered cameras: {num_registered}, reconstructed points: {num_points}")

    if stats["global_rms"] is None:
        print("[ERROR] insufficient reconstruction results for reliable reprojection stats.")
        print("  -> global_rms: None")
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({
                "global_rms_px": None,
                "per_view": None,
                "num_registered": num_registered,
                "num_points": num_points,
                "note": "insufficient data to compute reliable metrics"
            }, f, indent=2)
        print(f"[INFO] wrote diagnostic outputs to {out_dir}")
        return

    print(f"[RESULT] global RMS reprojection error (px): {stats['global_rms']:.3f}")
    for vid, s in stats["per_view"].items():
        print(f" view {vid}: mean={s['mean']:.3f} med={s['median']:.3f} std={s['std']:.3f} n={s['count']}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "global_rms_px": stats.get("global_rms"),
            "per_view": stats.get("per_view"),
            "num_registered": num_registered,
            "num_points": num_points
        }, f, indent=2)

    print(f"[INFO] finished writing outputs to {out_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python sfm_minimal.py <image_dir> [out_dir]")
    image_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sfm_out"
    run_sfm(image_dir, output_dir)
