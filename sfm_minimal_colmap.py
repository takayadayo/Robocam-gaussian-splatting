import os
import glob
import json
from collections import defaultdict
import argparse
import sys

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


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

def parse_colmap_poses(images_txt_path):
    poses = {}
    with open(images_txt_path, 'r') as f:
        # ヘッダーをスキップ
        for line in f:
            if line.strip().startswith('#'): continue
            
            parts = line.strip().split()
            if len(parts) < 10: continue
            
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]
            
            # scipyのquatは [x, y, z, w]
            q_scipy = np.array([qx, qy, qz, qw])
            t_vec = np.array([tx, ty, tz]).reshape(3, 1)
            
            # COLMAPのq,tは T_cam_world (R_cw, t_cw) を直接定義している
            R_cw = R.from_quat(q_scipy).as_matrix()
            
            # 逆変換は行わず、そのまま T_cam_world として保存
            poses[name] = (R_cw, t_vec)
            
            # 2行目はPOINTS2Dなので読み飛ばす
            next(f, None)
    return poses

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

def triangulate_n_views(K, cams, observations, min_parallax_deg=1.0, max_reproj_error_px=1.0):
    """
    N視点から3D点を三角測量し、品質チェックを行う。
    品質が基準を満たさない場合は None を返す。
    """
    if len(cams) < 2:
        return None

    # 1. 三角測量 (SVDによるDLT)
    A = []
    for (R, t), uv in zip(cams, observations):
        P = K @ np.hstack([R, t])
        A.append(uv[0] * P[2, :] - P[0, :])
        A.append(uv[1] * P[2, :] - P[1, :])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    if abs(X_h[3]) < 1e-6:
        print("[DEBUG|triangulate_n_views] Rejected: SVD failed (w is near zero)")
        return None
    X_h /= X_h[3]
    X = X_h[:3]

    # 2. Cheirality Check
    for i, (R, t) in enumerate(cams):
        if (R @ X + t.reshape(3))[2] <= 1e-4:
            print(f"[DEBUG|triangulate_n_views] Rejected: Cheirality failed for cam {i}")
            return None

    # 3. 個別再投影誤差チェック
    for i, ((R, t), uv_obs) in enumerate(zip(cams, observations)):
        proj = project_point(K, R, t, X.reshape(3, 1))
        if proj is None:
            print(f"[DEBUG|triangulate_n_views] Rejected: Projection failed for cam {i}")
            return None
        error = np.linalg.norm(proj - uv_obs)
        if error > max_reproj_error_px:
            print(f"[DEBUG|triangulate_n_views] Rejected: Reprojection error ({error:.2f}px > {max_reproj_error_px}px) for cam {i}")
            return None

    # 4. 視差角チェック
    cam_centers = [-R.T @ t for R, t in cams]
    
    max_angle_rad = 0.0
    for i in range(len(cam_centers)):
        for j in range(i + 1, len(cam_centers)):
            vec1 = X - cam_centers[i].flatten()
            vec2 = X - cam_centers[j].flatten()
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            max_angle_rad = max(max_angle_rad, angle_rad)
    
    if np.rad2deg(max_angle_rad) < min_parallax_deg:
        print(f"[DEBUG|triangulate_n_views] Rejected: Parallax angle ({np.rad2deg(max_angle_rad):.2f}deg < {min_parallax_deg}deg)")
        return None

    return X


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
                
                if new_X is not None:
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

def save_ply(filepath, points3d, imgs_raw, observations):
    """
    3D点群をPLYファイルとして保存する。
    各点の色は、その点を観測した最初の画像のピクセル色から取得する。
    """
    if not points3d:
        print("[warn] No 3D points to save.")
        return

    valid_points = []
    colors = []
    
    for pid, point in enumerate(points3d):
        if point is None:
            continue
        
        # この点を観測した最初の画像を探す
        first_obs = None
        for img_id, uv in observations[pid]:
            if imgs_raw[img_id] is not None:
                first_obs = (img_id, uv)
                break
        
        if first_obs:
            img_id, uv = first_obs
            u, v = int(round(uv[0])), int(round(uv[1]))
            h, w = imgs_raw[img_id].shape[:2]
            
            if 0 <= v < h and 0 <= u < w:
                bgr = imgs_raw[img_id][v, u]
                rgb = (bgr[2], bgr[1], bgr[0]) # BGR -> RGB
                valid_points.append(point)
                colors.append(rgb)

    if not valid_points:
        print("[warn] No valid points with color information found to save.")
        return

    ply_header = f"""ply
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
    with open(filepath, 'w') as f:
        f.write(ply_header)
        for p, c in zip(valid_points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    
    print(f"[info] Saved {len(valid_points)} points to {filepath}")

def run_incremental_sfm(files, K, keypoints, descriptors, imgs):
    """
    画像と特徴量から、増分型SfMを実行してカメラポーズと3D点群を推定する。
    """
    print("[info] Running in Incremental SfM mode.")

    cameras = [None] * len(files)
    points3d = []
    observations = []

    print("[info] Selecting initial pair...")
    result = select_initial_pair(keypoints, descriptors, K)
    if result is None:
        print("[ERROR] Failed to find a valid initial pair for SfM.")
        return None, None, None
    i0, i1, R01, t01, X01, kept_pairs, match_dict = result
    print(f"[info] Initial pair: {os.path.basename(files[i0])}, {os.path.basename(files[i1])}")

    cameras[i0] = (np.eye(3), np.zeros((3, 1)))
    cameras[i1] = (R01, t01)
    registered = {i0, i1}

    # --- トラック構築と初期化 ---
    # まず全画像間のマッチングを実行し、完全なトラックを事前に構築
    print("[info] Matching all pairs to build full tracks...")
    all_match_dict = {}
    for i in range(len(files) - 1):
        for j in range(i + 1, len(files)):
            # 既に計算済みの初期ペアのマッチングは再利用
            if (i, j) == (i0, i1) or (j, i) == (i0, i1):
                all_match_dict[(i0, i1)] = kept_pairs
                continue
            
            matches = match_features(descriptors[i], descriptors[j])
            if len(matches) > 15:
                all_match_dict[(i,j)] = matches
    
    print("[info] Building full tracks...")
    full_tracks = build_tracks(keypoints, all_match_dict)
    print(f"[info] Found {len(full_tracks)} tracks in total.")
    
    # 3D化されたトラックを管理するためのセット
    triangulated_tracks = set()
    
    # 初期ペアから最初の3D点群を生成し、観測を登録
    for track_idx, track in enumerate(full_tracks):
        kp_indices = {img_id: kp_idx for img_id, kp_idx in track}
        if i0 in kp_indices and i1 in kp_indices:
            key = (kp_indices[i0], kp_indices[i1])
            if key in {(a, b) for a, b in kept_pairs}:
                p_idx_in_X01 = kept_pairs.index(key)
                points3d.append(X01[:, p_idx_in_X01].reshape(3))
                
                obs = [(img_id, np.array(keypoints[img_id][kp_idx].pt)) for img_id, kp_idx in track]
                observations.append(obs)
                triangulated_tracks.add(track_idx)

    # --- 増分的な登録ループ ---
    while len(registered) < len(files):
        # 次に登録する最適なビューを選択
        view_scores = defaultdict(int)
        point_indices_for_view = defaultdict(list)
        
        for pid, obs_list in enumerate(observations):
            if points3d[pid] is None: continue
            for img_id, _ in obs_list:
                if img_id not in registered:
                    view_scores[img_id] += 1
                    point_indices_for_view[img_id].append(pid)
        
        if not view_scores:
            print("[info] No more views can be registered with current 3D points.")
            break
        
        best_idx = max(view_scores, key=view_scores.get)
        if view_scores[best_idx] < 8:
            print(f"[info] Best candidate view {best_idx} has too few matches ({view_scores[best_idx]}). Stopping.")
            break
        
        print(f"[info] Registering view {best_idx} (visible 3D points: {view_scores[best_idx]})...")

        # PnPのためのデータ準備
        pids_for_pnp = point_indices_for_view[best_idx]
        X3_list = [points3d[pid] for pid in pids_for_pnp]
        x2_list = [next(uv for img_id, uv in observations[pid] if img_id == best_idx) for pid in pids_for_pnp]

        X3_arr = np.array(X3_list, dtype=np.float64).reshape(-1, 1, 3)
        x2_arr = np.array(x2_list, dtype=np.float64).reshape(-1, 1, 2)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(X3_arr, x2_arr, K, None, flags=cv2.SOLVEPNP_AP3P, reprojectionError=2.0)
        
        if success and inliers is not None and len(inliers) >= 6:
            R_new, _ = cv2.Rodrigues(rvec)
            cameras[best_idx] = (R_new, tvec)
            registered.add(best_idx)
            print(f"  -> Success. Registered views: {len(registered)}/{len(files)}")
            
            # =============================================================
            #  新規点の三角測量ロジック (ここからが核心部分)
            # =============================================================
            new_points_count = 0
            for track_idx, track in enumerate(full_tracks):
                if track_idx in triangulated_tracks:
                    continue  # 既に3D化済み

                # このトラックが、今回登録したビューを含んでいるか？
                track_img_ids = {obs[0] for obs in track}
                if best_idx not in track_img_ids:
                    continue

                # 登録済みのビューからの観測を収集
                cams_for_tri, uvs_for_tri = [], []
                for img_id, kp_idx in track:
                    if cameras[img_id] is not None:
                        cams_for_tri.append(cameras[img_id])
                        uvs_for_tri.append(np.array(keypoints[img_id][kp_idx].pt))

                # 2視点以上から観測されていれば三角測量可能
                if len(cams_for_tri) >= 2:
                    # 品質チェック付きの三角測量を実行
                    X_new = triangulate_n_views(K, cams_for_tri, uvs_for_tri, 
                                                min_parallax_deg=1.5, 
                                                max_reproj_error_px=2.0)
                    
                    if X_new is not None:
                        # 成功した場合、新しい3D点と観測を登録
                        points3d.append(X_new)
                        full_obs_list = [(img_id, np.array(keypoints[img_id][kp_idx].pt)) for img_id, kp_idx in track]
                        observations.append(full_obs_list)
                        triangulated_tracks.add(track_idx)
                        new_points_count += 1
            
            print(f"  -> Triangulated {new_points_count} new 3D points.")
            # =============================================================

        else:
            print(f"  -> PnP failed for view {best_idx}. Stopping.")
            break
            
    return cameras, points3d, observations

def reconstruct_with_fixed_poses(files, K, cameras, keypoints, descriptors):
    """
    固定されたカメラポーズを用いて、ロバストな2段階三角測量により3D点群を再構成する。
    """
    print("[info] Running Robust Reconstruction with Fixed Poses...")

    # --- Step 0: 全ペアマッチングとトラック構築 ---
    print("[info] Matching features for all pairs with known poses...")
    match_dict = {}
    valid_indices = [i for i, c in enumerate(cameras) if c is not None]
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            idx1, idx2 = valid_indices[i], valid_indices[j]
            matches = match_features(descriptors[idx1], descriptors[idx2])
            if len(matches) > 15:
                match_dict[(idx1, idx2)] = matches
    
    print("[info] Building full tracks...")
    full_tracks = build_tracks(keypoints, match_dict)
    print(f"[info] Found {len(full_tracks)} potential tracks.")

    # --- Step 1: 頑健な初期化 (Two-View Triangulation) ---
    print("[info] Step 1: Initializing 3D points from robust two-view pairs...")
    initial_points = {}  # track_idx -> (point3D, initial_pair_indices)
    
    # 視差角を効率的に計算するためのカメラ中心
    cam_centers = {i: -cam[0].T @ cam[1] for i, cam in enumerate(cameras) if cam is not None}
    # === デバッグ用: 視差角の分布を記録 ===
    all_best_angles_deg = []
    for track_idx, track in enumerate(full_tracks):
        if len(track) < 2:
            continue
        
        best_pair_angle = -1.0
        best_pair_3d_point = None
        best_pair_indices = None

        # トラック内の全てのペアを試し、最も視差角の大きいペアを探す
        for i in range(len(track)):
            for j in range(i + 1, len(track)):
                img_id1, kp_idx1 = track[i]
                img_id2, kp_idx2 = track[j]

                cam1, cam2 = cameras[img_id1], cameras[img_id2]
                if cam1 is None or cam2 is None:
                    continue

                # 2視点三角測量
                cams_tri = [cam1, cam2]
                uvs_tri = [np.array(keypoints[img_id1][kp_idx1].pt), 
                           np.array(keypoints[img_id2][kp_idx2].pt)]
                
                # Cheiralityチェックのみ行う簡易三角測量
                A = [uv[0] * (K @ np.hstack(c))[2,:] - (K @ np.hstack(c))[0,:] for uv, c in zip(uvs_tri, cams_tri)]
                A.extend([uv[1] * (K @ np.hstack(c))[2,:] - (K @ np.hstack(c))[1,:] for uv, c in zip(uvs_tri, cams_tri)])
                A = np.array(A)
                _, _, Vt = np.linalg.svd(A)
                X_h = Vt[-1]
                if abs(X_h[3]) < 1e-6: continue
                X_h /= X_h[3]
                X = X_h[:3]

                if (cam1[0] @ X + cam1[1].reshape(3))[2] <= 1e-4 or \
                   (cam2[0] @ X + cam2[1].reshape(3))[2] <= 1e-4:
                    continue

                # 視差角の計算
                vec1 = X - cam_centers[img_id1].flatten()
                vec2 = X - cam_centers[img_id2].flatten()
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if norm1 < 1e-6 or norm2 < 1e-6: continue
                
                cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                if angle_rad > best_pair_angle:
                    best_pair_angle = angle_rad
                    best_pair_3d_point = X
                    best_pair_indices = (img_id1, img_id2)
        
        if best_pair_angle > 0:
            all_best_angles_deg.append(np.rad2deg(best_pair_angle))

        if np.rad2deg(best_pair_angle) > 2.0:
            initial_points[track_idx] = (best_pair_3d_point, best_pair_indices)

    # === デバッグ用: 視差角の統計情報を表示 ===
    if all_best_angles_deg:
        angles_arr = np.array(all_best_angles_deg)
        print(f"\n[DEBUG] Parallax angle stats (deg) for {len(angles_arr)} tracks:")
        print(f"  Min: {angles_arr.min():.2f}, Median: {np.median(angles_arr):.2f}, Max: {angles_arr.max():.2f}")
        print(f"  Tracks exceeding 2.0 deg threshold: {np.sum(angles_arr > 2.0)} / {len(angles_arr)}\n")
    else:
        print("\n[DEBUG] No valid pairs found to calculate parallax angles.\n")

    print(f"[info] Initialized {len(initial_points)} points with sufficient parallax.")

    # ★★★ ここで処理を中断して確認 ★★★
    if len(initial_points) == 0:
        print("[ERROR] No 3D points could be initialized. The parallax angle threshold (2.0 deg) might be too strict for this dataset, or feature matches are poor.")
        return [], []

    # --- Step 2: トラックの成長とリファインメント ---
    print("[info] Step 2: Growing and refining tracks...")
    points3d = []
    observations = []
    
    for track_idx, (initial_point, _) in initial_points.items():
        track = full_tracks[track_idx]
        
        current_point = initial_point
        valid_observations_in_track = []
        
        # トラック内の全観測を検証し、インライアを収集
        for img_id, kp_idx in track:
            cam = cameras[img_id]
            if cam is None: continue
            
            uv_obs = np.array(keypoints[img_id][kp_idx].pt)
            proj = project_point(K, cam[0], cam[1], current_point.reshape(3, 1))
            
            if proj is not None:
                reproj_error = np.linalg.norm(proj - uv_obs)
                if reproj_error < 2.5: # 再投影誤差の閾値
                    valid_observations_in_track.append((img_id, kp_idx))
        
        # 2つ以上の有効な観測があれば、リファインメントを行う
        if len(valid_observations_in_track) >= 2:
            cams_refine = [cameras[img_id] for img_id, _ in valid_observations_in_track]
            uvs_refine = [np.array(keypoints[img_id][kp_idx].pt) for img_id, kp_idx in valid_observations_in_track]
            
            # 最終的な3D点を、インライア観測のみを使って再計算
            final_point = triangulate_n_views(K, cams_refine, uvs_refine, 
                                              min_parallax_deg=0, # 視差は初期化で保証済み
                                              max_reproj_error_px=2.5) # 再チェック
            
            if final_point is not None:
                points3d.append(final_point)
                # 観測リストには、トラック全体の情報ではなく、インライアのみを保存することもできるが、
                # prune_outliersなどで全情報が必要になる可能性を考慮し、ここではトラック全体を保存する。
                # 後の処理でインライアのみを使いたい場合は、この情報からフィルタリングする。
                full_obs_list = [(img_id, np.array(keypoints[img_id][kp_idx].pt)) for img_id, kp_idx in track]
                observations.append(full_obs_list)

    print(f"[info] Successfully reconstructed {len(points3d)} points after growing.")
    return points3d, observations

def save_poses_to_json(filepath, cameras, files):
    """ SfMで計算されたカメラポーズをJSONファイルに保存する """
    pose_data = {}
    valid_poses = 0
    for i, cam in enumerate(cameras):
        if cam is not None:
            R_wc, t_wc = cam  # SfMは T_world_cam を計算する
            # JSONに保存しやすいようにリストに変換
            pose_data[os.path.basename(files[i])] = {
                'R_wc': R_wc.tolist(),
                't_wc': t_wc.flatten().tolist()
            }
            valid_poses += 1
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(pose_data, f, indent=4)
    print(f"[INFO] Saved {valid_poses} SfM-computed poses to {filepath}")
    
# --- ADD: COLMAP export helpers ------------------------------------
def _export_colmap_files(out_dir, files, K_used, cameras):
    """
    Export COLMAP text files for visualization:
      - cameras.txt (PINHOLE, undistorted intrinsics)
      - images.txt  (world->camera pose: QW QX QY QZ TX TY TZ)
      - points3D.txt (empty)
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = None, None
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if im is not None:
            H, W = im.shape[:2]
            break
    if H is None:
        raise RuntimeError("Could not read any image to infer size.")

    fx, fy, cx, cy = float(K_used[0,0]), float(K_used[1,1]), float(K_used[0,2]), float(K_used[1,2])

    cam_path   = os.path.join(out_dir, "cameras.txt")
    img_path   = os.path.join(out_dir, "images.txt")
    pts3d_path = os.path.join(out_dir, "points3D.txt")

    # --- cameras.txt (PINHOLE, no distortion) ---
    with open(cam_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # --- images.txt ---
    with open(img_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        img_id = 1
        for path, cam in zip(files, cameras):
            name = os.path.basename(path)
            if cam is None:
                f.write(f"# {img_id} {name} : pose missing\n\n")
                img_id += 1
                continue
            R_cw, t_cw = cam  # world->camera
            q_xyzw = R.from_matrix(R_cw).as_quat()  # [x,y,z,w]
            qw, qx, qy, qz = float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])
            tx, ty, tz = float(t_cw[0]), float(t_cw[1]), float(t_cw[2])
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}\n")
            f.write("\n")
            img_id += 1

    # --- empty points3D.txt ---
    with open(pts3d_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list is empty for visualization\n")

    print(f"[INFO] Wrote COLMAP txt to: {out_dir}")
# --------------------------------------------------------------------


def main(image_dir, out_dir, poses_path=None):
    os.makedirs(out_dir, exist_ok=True)
    files = load_images(image_dir)
    if len(files) < 2: 
        raise RuntimeError("Need at least two images.")

    print("[info] Reading images...")
    imgs_raw = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
    H, W = imgs_raw[0].shape[:2]

    # --- Intrinsics Setup ---
    intrinsics_path = os.path.join(out_dir, "intrinsics.yaml")
    if not os.path.exists(intrinsics_path):
        raise RuntimeError(f"Intrinsics file not found at {intrinsics_path}. Please run calibration first or provide the file.")
    
    K_loaded, dist_loaded = load_intrinsics(intrinsics_path)
    if K_loaded is None: 
        raise RuntimeError(f"Failed to load intrinsics from {intrinsics_path}")
    
    imgs, K_undistort = undistort_images(imgs_raw, K_loaded, dist_loaded)
    K = K_undistort if K_undistort is not None else K_loaded
    print(f"[info] Using intrinsics from {intrinsics_path}")

    print("[info] Extracting features...")
    features = [extract_sift(im) for im in imgs]
    keypoints = [f[0] for f in features]
    descriptors = [f[1] for f in features]
    
    # ここで変数を初期化
    cameras = [None] * len(files)
    points3d = []
    observations = []

    # =================================================================
    #  モード分岐: ポーズ指定モードか、SfMモードか
    # =================================================================
    # =================================================================
    #  MODE B: Reconstruction with Fixed Poses
    # =================================================================
    if poses_path:
        print(f"[info] Running in Fixed Pose Reconstruction mode using {poses_path}")
        ext_poses = parse_colmap_poses(poses_path)
        print(f"[info] Loaded {len(ext_poses)} poses.")
        
        fname_to_idx = {os.path.basename(f): i for i, f in enumerate(files)}
        for name, pose in ext_poses.items():
            if name in fname_to_idx:
                cameras[fname_to_idx[name]] = pose
        
        num_loaded = sum(1 for c in cameras if c is not None)
        if num_loaded < 2: raise RuntimeError(f"Could not match enough poses. Matched: {num_loaded}")
        print(f"[info] Matched {num_loaded}/{len(files)} poses.")
        
        # 新しいロバストな再構成関数を呼び出す
        points3d, observations = reconstruct_with_fixed_poses(files, K, cameras, keypoints, descriptors)
        mode_name = 'FixPoseMode'

    else:
        # =================================================================
        #  MODE A: Incremental SfM
        # =================================================================
        cameras, points3d, observations = run_incremental_sfm(files, K, keypoints, descriptors, imgs)
        mode_name = 'sfmMode'
        if cameras is None:
            print("[ERROR] Incremental SfM failed. Exiting.")
            sys.exit(1)
        
        # sfm_poses_path = os.path.join(out_dir, "sfm_minimal", "sfm_computed_poses.json")
        # save_poses_to_json(sfm_poses_path, cameras, files)
        
        colmap_out = os.path.join(out_dir, "sfm_minimal", "colmap_export")
        _export_colmap_files(colmap_out, files, K, cameras)
    # --- Finalization (両モード共通) ---
    if not points3d:
        print("[ERROR] No 3D points were reconstructed.")
        # sys.exit(1) # 状況によっては終了させなくても良い
        return # main関数から抜ける

    print("[info] Refining 3D points...")
    points3d = refine_points(K, cameras, points3d, observations, iterations=2)
    
    print("[info] Pruning outliers...")
    prune_outliers(K, cameras, points3d, observations)
    
    stats = reprojection_stats(K, cameras, points3d, observations)
    num_registered = sum(1 for c in cameras if c is not None)
    num_points = sum(1 for p in points3d if p is not None)
    print(f"\n[INFO] FINAL: registered cameras: {num_registered}, reconstructed points: {num_points}")

    if stats["global_rms"] is not None:
        print(f"[RESULT] global RMS reprojection error (px): {stats['global_rms']:.4f}")
        metrics_path = os.path.join(out_dir, "sfm_minimal", f"metrics_{mode_name}.json")
        with open(metrics_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[info] Saved detailed metrics to {metrics_path}")

    ply_path = os.path.join(out_dir, 'sfm_minimal', f"point_cloud_{mode_name}.ply")
    save_ply(ply_path, points3d, imgs_raw, observations)
    print(f"[INFO] Finished writing outputs to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A minimal Structure-from-Motion pipeline with two modes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image_dir", help="Directory containing input images.")
    parser.add_argument("out_dir", help="Directory to save outputs (must contain intrinsics.yaml).")
    parser.add_argument(
        "--poses", 
        help="Path to a COLMAP images.txt file to use fixed camera poses.\n"
             "If not provided, the script runs in incremental SfM mode.",
        default=None
    )
    args = parser.parse_args()

    main(args.image_dir, args.out_dir, args.poses)
