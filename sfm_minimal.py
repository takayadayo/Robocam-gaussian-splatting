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

def refine_points(K, cameras, points3d, observations, iterations=2):
    
    for iter_count in range(iterations):
        num_refined = 0
        for pid, point in enumerate(points3d):
            if point is None:
                continue
            
            registered_views = []
            valid_obs = []
            for img_id, uv in observations[pid]:
                if cameras[img_id] is not None:
                    registered_views.append(cameras[img_id])
                    valid_obs.append(uv)
                    
            if len(registered_views) >= 2:
                new_X = triangulate_n_views(K, registered_views, valid_obs)
                
                is_valid = True
                for R_cam, t_cam in registered_views:
                    if (R_cam @ new_X + t_cam.reshape(3))[2] < 1e-4:
                        is_valid = False
                        break
                    
                if is_valid:
                    points3d[pid] = new_X
                    num_refined += 1
                    
        if num_refined == 0:
            break
    return points3d

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

def save_ply(filepath, points3d, imgs, observations, K):
    """
    3D点群をPLYファイルとして保存する。
    各点の色は、その点を観測した最初の画像のピクセル色から取得する。
    """
    if points3d is None or len(points3d) == 0:
        print("[warn] No 3D points to save.")
        return

    # 有効な3D点とその色を収集
    valid_points = []
    colors = []
    
    # どの観測がどの3D点に対応するかを逆引きするマップを作成
    point_map = {}
    for pid, obs_list in enumerate(observations):
        for img_id, uv in obs_list:
            point_map[(img_id, tuple(uv))] = pid

    # 最初に観測された画像から色を取得する
    pid_to_color_source = {}
    for pid, p in enumerate(points3d):
        if p is None: continue
        
        # この点を観測した最初の画像を探す
        first_obs = None
        for obs in observations[pid]:
            img_id, uv = obs
            if imgs[img_id] is not None:
                first_obs = obs
                break
        
        if first_obs:
            img_id, uv = first_obs
            u, v = int(round(uv[0])), int(round(uv[1]))
            
            h, w = imgs[img_id].shape[:2]
            if 0 <= v < h and 0 <= u < w:
                # OpenCVはBGR形式なので、RGBに変換
                bgr = imgs[img_id][v, u]
                rgb = (bgr[2], bgr[1], bgr[0])
                valid_points.append(p)
                colors.append(rgb)

    if not valid_points:
        print("[warn] No valid points with color information found to save.")
        return

    # PLYヘッダーを作成
    header = f"""ply
format ascii 1.0
element vertex {len(valid_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    # PLYボディを作成
    body = ""
    for p, c in zip(valid_points, colors):
        body += f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n"

    # ファイルに書き込み
    with open(filepath, 'w') as f:
        f.write(header + body)
    
    print(f"[info] Saved {len(valid_points)} points to {filepath}")

def run_sfm(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = load_images(img_dir)
    if len(files) < 2:
        raise RuntimeError("Need at least two images.")

    imgs_raw = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
    H, W = imgs_raw[0].shape[:2]
    # K = build_initial_K(W, H, scale=1.2)

    intrinsics_path = os.path.join(out_dir, "intrinsics.yaml")
    K_loaded, dist_loaded = load_intrinsics(intrinsics_path)
    if K_loaded is not None and dist_loaded is not None:
        imgs, newK = undistort_images(imgs_raw, K_loaded, dist_loaded)
        K = newK if newK is not None else K_loaded
        print(f'[info] using instrinsics from {intrinsics_path}: images undistorted.')
    
    else:
        imgs = imgs_raw
        K = build_initial_K(W, H, scale=1.2)
        print("[warn] intrinsics.yaml unavailable or incomplete; using approximate K and zero distortion.")
        
    print('[info] extracting features...')
    features = [extract_sift(im) for im in imgs]
    keypoints = [f[0] for f in features]
    descriptors = [f[1] for f in features]
    
    print('[info] selecting initial pair...')

    i0, i1, R01, t01, X01, kept_pairs, initial_match_dict = select_initial_pair(keypoints, descriptors, K)
    match_dict = initial_match_dict.copy()

    cameras = [None] * len(files)
    cameras[i0] = (np.eye(3), np.zeros((3, 1)))
    cameras[i1] = (R01, t01)

    tracks = build_tracks(keypoints, match_dict)
    pid_map = {}
    points3d = []
    observations = []
    
    for track in tracks:
        imgs_in_track = [obs[0] for obs in track]
        if i0 in imgs_in_track and i1 in imgs_in_track:
            kp0 = next(k for (img_id, k) in track if img_id == i0)
            kp1 = next(k for (img_id, k) in track if img_id == i1)
            key = (kp0, kp1)
            if key in {(a, b) for a, b in kept_pairs}:
                idx = len(points3d)
                X = X01[:, [kept_pairs.index(key)]].reshape(3)
                points3d.append(X)
                
                obs_list = []
                for (img_id, kp_idx) in track:
                    uv = np.array(keypoints[img_id][kp_idx].pt, dtype=np.float64)
                    obs_list.append((img_id, uv))
                observations.append(obs_list)
                pid_map[tuple(sorted(track))] = idx
                
    registered = set([i0, i1])
    
    while True:
        view_scores =defaultdict(int)
        for track in tracks:
            track_key = tuple(sorted(track))
            if track_key in pid_map and points3d[pid_map[track_key]] is not None:
                for img_id, _ in track:
                    if img_id not in registered:
                        view_scores[img_id] += 1
                        
        candidates = sorted([(score, vid) for vid, score in view_scores.items()], reverse=True)
        
        if not candidates or candidates[0][0] < 10:
            break
        
        best_score, best_idx = candidates[0]
        print(f'[info] registering view {best_idx} (visible 3D points: {best_score})...')
        
        X3_list = []
        x2_list = []
        
        for track in tracks:
            track_key = tuple(sorted(track))
            if track_key in pid_map and points3d[pid_map[track_key]] is not None:
                current_view_kp = None
                for img_id, kp_idx in track:
                    if img_id == best_idx:
                        current_view_kp = kp_idx
                        break
                if current_view_kp is not None:
                    X3_list.append(points3d[pid_map[track_key]])
                    uv = keypoints[best_idx][current_view_kp].pt
                    x2_list.append(uv)
                    
        X3_arr = np.array(X3_list, dtype=np.float64).reshape(-1, 1, 3)
        x2_arr = np.array(x2_list, dtype=np.float64).reshape(-1, 1, 2)
        
        pose = cv2.solvePnPRansac(X3_arr, x2_arr, K, None, flags=cv2.SOLVEPNP_AP3P, reprojectionError=4.0, confidence=0.999, iterationsCount=1000)
        
        success = False

        if pose[0]:
            rvec, tvec = pose[1], pose[2]
            success, rvec, tvec = cv2.solvePnP(X3_arr, x2_arr, K, None, rvec, tvec,
                                               useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                cameras[best_idx] = (R, tvec)
                registered.add(best_idx)
                
                print(f'  -> Success. registered views: {len(registered)} / {len(files)}')
                
                for reg_id in registered:
                    if reg_id == best_idx: continue
                    if (min(reg_id, best_idx), max(reg_id, best_idx)) not in match_dict:
                        matches = match_features(descriptors[reg_id], descriptors[best_idx])
                        if matches:
                            match_dict[(min(reg_id, best_idx), max(reg_id, best_idx))] = matches
                            
                new_points_count = 0
                
                for track in tracks:
                    track_key = tuple(sorted(track))
                    if track_key in pid_map:
                        continue
                    
                    in_current = False
                    obs_in_registered = []
                    
                    for img_id, kp_idx in track:
                        if img_id == best_idx:
                            in_current = True
                            
                        if cameras[img_id] is not None:
                            uv = np.array(keypoints[img_id][kp_idx].pt, dtype=np.float64)
                            obs_in_registered.append((cameras[img_id], uv, img_id))
                            
                    if in_current and len(obs_in_registered) >= 2:
                        cams_for_tri = [item[0] for item in obs_in_registered]
                        uvs_for_tri = [item[1] for item in obs_in_registered]
                        
                        X_new = triangulate_n_views(K, cams_for_tri, uvs_for_tri)
                        is_valid = True
                        for R_c, t_c in cams_for_tri:
                            if (R_c @ X_new + t_c.reshape(3))[2] < 1e-4:
                                is_valid = False
                                break
                        if is_valid:
                            new_pid = len(points3d)
                            points3d.append(X_new)
                            new_obs_list = [(item[2], item[1]) for item in obs_in_registered]
                            observations.append(new_obs_list)
                            pid_map[track_key] = new_pid
                            new_points_count += 1
                            
                print(f'  -> Triangulated {new_points_count} new 3D points.')
        
        if not success:
            print(f'  -> PnP failed for view {best_idx}.')
            pass


    print('[info] finished registration loop')
    num_registered = sum(1 for c in cameras if c is not None)
    if num_registered < 2 or not points3d:
        print('[ERROR] insufficient reconstruction')
        return
    
    print('[info] refining 3D points...')
    points3d = refine_points(K, cameras, points3d, observations, iterations=2)
    
    print('[info] pruning outliers...')
    prune_outliers(K, cameras, points3d, observations, base_thresh=3.0, max_iter=5)
    stats = reprojection_stats(K, cameras, points3d, observations)

    num_points = sum(1 for p in points3d if p is not None)
    print(f"[INFO] registered cameras: {num_registered}, reconstructed points: {num_points}")

    if stats["global_rms"] is not None:
        print(f"[RESULT] global RMS reprojection error (px): {stats['global_rms']:.3f}")

    with open(os.path.join(out_dir, "sfm_minimal", "metrics.json"), "w") as f:
        json.dump({
            "global_rms_px": stats.get("global_rms"),
            "per_view": stats.get("per_view"),
            "num_registered": num_registered,
            "num_points": num_points
        }, f, indent=2)
        
    ply_path = os.path.join(out_dir, "sfm_minimal", "point_cloud.ply")
    save_ply(ply_path, points3d, imgs, observations, K)

    print(f"[INFO] finished writing outputs to {out_dir}/sfm_minimal")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python sfm_minimal.py <image_dir> [out_dir]")
    image_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sfm_out"
    run_sfm(image_dir, output_dir)
