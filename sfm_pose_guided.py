import json
import os
from collections import defaultdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


EULER_ORDER = "xyz"
MIN_MATCHES_PAIR = 20
SAMPSON_THRESHOLD = 4.0  # pixels
MIN_OBSERVATIONS_POINT = 2
DEPTH_THRESHOLD = 1e-4


def load_handeye(path: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open hand-eye file: {path}")
    Rg_c = fs.getNode("Rg_c").mat().astype(np.float64)
    tg_c = fs.getNode("tg_c").mat().reshape(3, 1).astype(np.float64)
    fs.release()
    return Rg_c, tg_c


def load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics file: {path}")
    K = fs.getNode("K").mat().astype(np.float64)
    dist = fs.getNode("dist").mat().astype(np.float64)
    fs.release()
    if dist.ndim == 1:
        dist = dist.reshape(-1, 1)
    return K, dist


def load_robot_captures(json_path: str, image_dir: str, handeye_path: str) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    Rg_c, tg_c = load_handeye(handeye_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_paths = []
    cameras = []
    for capture in data["captures"]:
        basename = os.path.basename(capture["image_file"])
        img_path = os.path.join(image_dir, basename)
        if not os.path.isfile(img_path):
            continue

        t_mm = np.array([
            capture["pose"]["x_mm"],
            capture["pose"]["y_mm"],
            capture["pose"]["z_mm"]
        ], dtype=np.float64)
        t_m = t_mm / 1000.0

        angles = np.array([
            capture["pose"]["rx_deg"],
            capture["pose"]["ry_deg"],
            capture["pose"]["rz_deg"]
        ], dtype=np.float64)
        Rb_g = R.from_euler(EULER_ORDER, angles, degrees=True).as_matrix()

        Rb_c = Rb_g @ Rg_c
        tb_c = Rb_g @ tg_c + t_m.reshape(3, 1)

        image_paths.append(img_path)
        cameras.append((Rb_c, tb_c))
    return image_paths, cameras


def load_images(paths: List[str]) -> List[np.ndarray]:
    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        images.append(img)
    return images


def build_initial_K(width: int, height: int, scale: float = 1.2) -> np.ndarray:
    f = scale * max(width, height)
    return np.array([[f, 0, width * 0.5],
                     [0, f, height * 0.5],
                     [0, 0, 1.0]], dtype=np.float64)


def undistort_images(images: List[np.ndarray], K: np.ndarray, dist: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    if K is None or dist is None:
        return images, None
    h, w = images[0].shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undistorted = [cv2.undistort(im, K, dist, None, newK) for im in images]
    return undistorted, newK


def extract_sift_features(images: List[np.ndarray]):
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            desc = np.zeros((0, 128), dtype=np.float32)
        keypoints.append(kp)
        descriptors.append(desc)
    return keypoints, descriptors


def match_features(desc_a, desc_b, ratio=0.75):
    if len(desc_a) == 0 or len(desc_b) == 0:
        return []
    index_params = dict(algorithm=1, trees=5)
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
    matches = [(a, b) for (a, b) in good_ab if (a, b) in set_ba]
    return matches


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec.flatten()
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)


def compute_fundamental(K: np.ndarray, cam_i, cam_j) -> np.ndarray:
    R_i, t_i = cam_i
    R_j, t_j = cam_j
    C_i = -R_i.T @ t_i
    C_j = -R_j.T @ t_j
    t_ij_world = (C_j - C_i).reshape(3, 1)
    t_ij_cam_i = R_i @ t_ij_world
    R_ij = R_j @ R_i.T
    E = skew(t_ij_cam_i) @ R_ij
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F


def filter_matches_epipolar(matches, kp_i, kp_j, F, threshold=SAMPSON_THRESHOLD):
    filtered = []
    for a, b in matches:
        pt1 = np.array([*kp_i[a].pt, 1.0], dtype=np.float64)
        pt2 = np.array([*kp_j[b].pt, 1.0], dtype=np.float64)
        Fx1 = F @ pt1
        Ftx2 = F.T @ pt2
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
        if denom < 1e-12:
            continue
        num = (pt2 @ F @ pt1) ** 2
        sampson = num / denom
        if sampson <= threshold:
            filtered.append((a, b))
    return filtered


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
    for kps in keypoints:
        offsets.append(gid)
        gid += len(kps)

    dsu = DSU(gid)
    for (i, j), matches in match_dict.items():
        off_i = offsets[i]
        off_j = offsets[j]
        for a, b in matches:
            dsu.union(off_i + a, off_j + b)

    tracks = defaultdict(list)
    for img_id, kps in enumerate(keypoints):
        offset = offsets[img_id]
        for idx in range(len(kps)):
            root = dsu.find(offset + idx)
            tracks[root].append((img_id, idx))

    filtered = [obs for obs in tracks.values() if len(obs) >= MIN_OBSERVATIONS_POINT]
    return filtered


def triangulate_track(K, cameras, obs):
    A = []
    for img_id, uv in obs:
        R_i, t_i = cameras[img_id]
        P = K @ np.hstack([R_i, t_i])
        x, y = uv
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.asarray(A)
    if A.shape[0] < 4:
        return None
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        return None
    X = X_h[:3] / X_h[3]

    positive = 0
    for img_id, _ in obs:
        R_i, t_i = cameras[img_id]
        depth = (R_i @ X.reshape(3, 1) + t_i)[2, 0]
        if depth > DEPTH_THRESHOLD:
            positive += 1
    if positive < max(1, len(obs) // 2):
        return None
    return X


def project_point(K, R_i, t_i, X):
    x_cam = R_i @ X.reshape(3, 1) + t_i
    if x_cam[2, 0] <= 1e-6:
        return None
    x_norm = x_cam / x_cam[2, 0]
    pixel = K @ x_norm
    return pixel[:2].reshape(2)


def refine_point(K, cameras, obs, X_init):
    def residual(x):
        X = x.reshape(3)
        res = []
        for img_id, uv in obs:
            R_i, t_i = cameras[img_id]
            proj = project_point(K, R_i, t_i, X)
            if proj is None:
                res.extend([1e3, 1e3])
            else:
                res.extend(proj - uv)
        return res

    try:
        opt = least_squares(residual, X_init, method="lm")
        return opt.x
    except Exception:
        return X_init


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
                R_i, t_i = cameras[img_id]
                proj = project_point(K, R_i, t_i, X)
                if proj is None:
                    continue
                error = float(np.linalg.norm(proj - uv))
                residuals.append((img_id, uv, error))
            if len(residuals) < 2:
                points3d[pid] = None
                observations[pid] = []
                removed_pts += 1
                continue
            errs = np.array([e for _, _, e in residuals], dtype=np.float64)
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
            R_i, t_i = cameras[img_id]
            proj = project_point(K, R_i, t_i, X)
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


def write_metrics(path, stats, num_registered, num_points):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "global_rms_px": stats.get("global_rms"),
            "per_view": stats.get("per_view"),
            "num_registered": num_registered,
            "num_points": num_points
        }, f, indent=2)


def write_ply(path, points):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z} 200 200 200\n")


def bundle_adjust_soft(K: np.ndarray,
                       cameras: List[Tuple[np.ndarray, np.ndarray]],
                       points3d: List[Optional[np.ndarray]],
                       observations: List[List[Tuple[int, np.ndarray]]],
                       prior_rvecs: List[np.ndarray],
                       prior_tvecs: List[np.ndarray],
                       lam_rot: float = 0.5,
                       lam_trans: float = 5.0,
                       max_nfev: int = 50) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Optional[np.ndarray]]]:
    active_points = [pid for pid, X in enumerate(points3d) if X is not None and len(observations[pid]) >= 2]
    if not active_points:
        return cameras, points3d

    cam_count = len(cameras)
    point_map = {pid: idx for idx, pid in enumerate(active_points)}

    obs_cam_idx = []
    obs_point_idx = []
    obs_uv = []
    for pid in active_points:
        for cam_id, uv in observations[pid]:
            obs_cam_idx.append(cam_id)
            obs_point_idx.append(point_map[pid])
            obs_uv.append(uv)

    if not obs_cam_idx:
        return cameras, points3d

    cam_params = np.zeros((cam_count, 6), dtype=np.float64)
    for i, (R_i, t_i) in enumerate(cameras):
        rvec_i, _ = cv2.Rodrigues(R_i)
        cam_params[i, :3] = rvec_i.ravel()
        cam_params[i, 3:] = t_i.ravel()

    point_params = np.array([points3d[pid] for pid in active_points], dtype=np.float64).reshape(-1, 3)
    x0 = np.hstack([cam_params.ravel(), point_params.ravel()])

    obs_cam_idx = np.asarray(obs_cam_idx, dtype=np.int32)
    obs_point_idx = np.asarray(obs_point_idx, dtype=np.int32)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)
    prior_rvecs_arr = np.asarray(prior_rvecs, dtype=np.float64)
    prior_tvecs_arr = np.asarray(prior_tvecs, dtype=np.float64)

    sqrt_lam_rot = np.sqrt(lam_rot)
    sqrt_lam_trans = np.sqrt(lam_trans)

    def residual(vec):
        cam_vec = vec[:cam_count * 6].reshape(cam_count, 6)
        pt_vec = vec[cam_count * 6:].reshape(-1, 3)

        reproj = []
        for cam_id, pt_id, uv in zip(obs_cam_idx, obs_point_idx, obs_uv):
            rvec = cam_vec[cam_id, :3].reshape(3, 1)
            tvec = cam_vec[cam_id, 3:].reshape(3, 1)
            X = pt_vec[pt_id].reshape(3, 1)
            R_i, _ = cv2.Rodrigues(rvec)
            proj = project_point(K, R_i, tvec, X)
            if proj is None:
                reproj.extend([1e3, 1e3])
            else:
                reproj.extend(proj - uv)

        prior_res = []
        for idx in range(cam_count):
            r_curr = cam_vec[idx, :3]
            t_curr = cam_vec[idx, 3:]
            prior_res.extend((r_curr - prior_rvecs_arr[idx]) * sqrt_lam_rot)
            prior_res.extend((t_curr - prior_tvecs_arr[idx]) * sqrt_lam_trans)

        return np.hstack([reproj, prior_res])

    try:
        opt = least_squares(residual, x0, method="lm", max_nfev=max_nfev)
        params_opt = opt.x
    except Exception:
        return cameras, points3d

    cam_opt = params_opt[:cam_count * 6].reshape(cam_count, 6)
    pt_opt = params_opt[cam_count * 6:].reshape(-1, 3)

    new_cameras: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(cam_count):
        rvec = cam_opt[i, :3].reshape(3, 1)
        tvec = cam_opt[i, 3:].reshape(3, 1)
        R_i, _ = cv2.Rodrigues(rvec)
        new_cameras.append((R_i, tvec))

    new_points = list(points3d)
    for pid, idx in point_map.items():
        new_points[pid] = pt_opt[idx].reshape(3)

    return new_cameras, new_points


def run_pose_guided_sfm(image_dir, pose_json, handeye_yaml, intrinsics_yaml, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_paths, cameras = load_robot_captures(pose_json, image_dir, handeye_yaml)
    if len(image_paths) < 2:
        raise RuntimeError("Not enough captures in JSON to perform reconstruction.")

    images_raw = load_images(image_paths)
    H, W = images_raw[0].shape[:2]
    K = build_initial_K(W, H)
    K_loaded, dist_loaded = load_intrinsics(intrinsics_yaml)
    if K_loaded is not None:
        images, newK = undistort_images(images_raw, K_loaded, dist_loaded)
        K = newK if newK is not None else K_loaded
        print(f"[info] using intrinsics from {intrinsics_yaml}; images undistorted.")
    else:
        images = images_raw
        print("[warn] intrinsics.yaml unavailable; using approximate intrinsics.")

    keypoints, descriptors = extract_sift_features(images)

    match_dict = {}
    total_pairs = 0
    kept_pairs = 0
    for i in range(len(images) - 1):
        for j in range(i + 1, len(images)):
            total_pairs += 1
            matches = match_features(descriptors[i], descriptors[j])
            if not matches:
                continue
            F_ij = compute_fundamental(K, cameras[i], cameras[j])
            filtered = filter_matches_epipolar(matches, keypoints[i], keypoints[j], F_ij, threshold=SAMPSON_THRESHOLD)
            if len(filtered) < MIN_MATCHES_PAIR:
                continue
            match_dict[(i, j)] = filtered
            kept_pairs += 1
    print(f"[INFO] evaluated {total_pairs} image pairs, retained {kept_pairs} with sufficient matches.")

    tracks = build_tracks(keypoints, match_dict)
    print(f"[INFO] constructed {len(tracks)} multi-view tracks.")

    points3d = []
    observations = []
    for obs in tracks:
        obs_pixels = []
        for img_id, kp_idx in obs:
            uv = np.array(keypoints[img_id][kp_idx].pt, dtype=np.float64)
            obs_pixels.append((img_id, uv))
        candidate = triangulate_track(K, cameras, obs_pixels)
        if candidate is None:
            continue
        points3d.append(candidate)
        observations.append(obs_pixels)

    print(f"[INFO] triangulated {len([p for p in points3d if p is not None])} initial points.")

    prune_outliers(K, cameras, points3d, observations, base_thresh=2.5, max_iter=3)

    refined_points = []
    for pid, X in enumerate(points3d):
        if X is None:
            refined_points.append(None)
            continue
        refined = refine_point(K, cameras, observations[pid], X)
        refined_points.append(refined)
    points3d = refined_points

    prior_rvecs = []
    prior_tvecs = []
    for R_i, t_i in cameras:
        rvec_i, _ = cv2.Rodrigues(R_i)
        prior_rvecs.append(rvec_i.ravel())
        prior_tvecs.append(t_i.ravel())

    cameras, points3d = bundle_adjust_soft(K, cameras, points3d, observations, prior_rvecs, prior_tvecs)

    stats = reprojection_stats(K, cameras, points3d, observations)

    num_registered = len([c for c in cameras if c is not None])
    num_points = len([p for p in points3d if p is not None])

    metrics_path = os.path.join(out_dir, "pose_metrics.json")
    write_metrics(metrics_path, stats, num_registered, num_points)

    valid_points = [p for p in points3d if p is not None]
    if valid_points:
        ply_path = os.path.join(out_dir, "pose_points.ply")
        write_ply(ply_path, valid_points)

    if stats["global_rms"] is not None:
        print(f"[RESULT] pose-guided RMS reprojection error (px): {stats['global_rms']:.3f}")
        for vid, s in stats["per_view"].items():
            print(f" view {vid}: mean={s['mean']:.3f} med={s['median']:.3f} std={s['std']:.3f} n={s['count']}")
    else:
        print("[WARN] unable to compute reliable reprojection statistics.")
    print(f"[INFO] pose-guided metrics written to {metrics_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        raise SystemExit("Usage: python sfm_pose_guided.py <image_dir> <poses_json> <handeye_yaml> <intrinsics_yaml> [out_dir]")

    image_dir = sys.argv[1]
    poses_json = sys.argv[2]
    handeye_yaml = sys.argv[3]
    intrinsics_yaml = sys.argv[4]
    output_dir = sys.argv[5] if len(sys.argv) > 5 else "sfm_out"

    run_pose_guided_sfm(image_dir, poses_json, handeye_yaml, intrinsics_yaml, output_dir)
