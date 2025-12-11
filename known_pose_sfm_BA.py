"""
Known-pose SfM baseline:
- Load camera poses from COLMAP images.txt
- Load intrinsics from YAML (K, dist)
- Extract SIFT + RootSIFT features
- Select image pairs using known poses
- Match features with Lowe ratio + cross-check
- Geometric verification with Essential matrix (RANSAC)
- Build multi-view tracks
- Multi-view triangulation with fixed known poses
- Point-only bundle adjustment (SciPy least_squares, Huber loss)
- Filter 3D points and visualize result

Usage example:
    python known_pose_sfm_baseline.py \
        --images_txt path/to/images.txt \
        --intrinsics_yaml path/to/intrinsics.yaml \
        --image_dir path/to/image_folder \
        --output_ply output_points.ply
"""

import argparse
import os
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

@dataclass
class BaselineConfig:
    # Feature extraction
    max_image_size: int = 2400
    max_num_features: int = 8192
    use_affine_shape: bool = True
    use_domain_size_pooling: bool = False

    # Pair selection (pose-based)
    theta_min_pair_deg: float = 6.0
    theta_max_pair_deg: float = 65.0
    baseline_over_depth_min: float = 0.05
    baseline_over_depth_max: float = 0.6
    max_pairs_per_image: int = 10
    seq_neighbor_window: int = 3

    # Matching
    sift_ratio: float = 0.7
    cross_check: bool = True
    geom_max_error_px: float = 0.01   # エピポーラ誤差
    geom_confidence: float = 0.999
    geom_min_inlier_ratio: float = 0.25
    geom_min_num_inliers: int = 10
    guided_matching: bool = False

    # Triangulation & BA
    tri_min_angle_deg: float = 2.0   # ★ 1.5→2.0deg（COLMAP推奨より少しだけ強め）
    use_two_view_tracks_for_eval: bool = False
    use_two_view_tracks_for_gs: bool = True
    
    # Track quality classification
    strong_track_min_angle_deg: float = 6.0  # ≧この視差角 or 長さ≥4なら強トラック
    weak_track_min_angle_deg: float = 3.0    # ≧この視差角なら弱トラック候補

    # BA loss / filtering (px → 後で正規化)
    ba_loss_scale_px: float = 1.0    # ★ ロバスト閾値もやや厳しく
    point_mean_reproj_max_px: float = 1.0  # ★ 平均再投影誤差を 1px 以内
    point_max_reproj_max_px: float = 1.0   # ★ 最大誤差 2px 以内
    obs_drop_thresh_px: float = 1.0        # ★ 観測落としは 3px

    # BA iterations
    ba_max_nfev: int = 500

    # Joint BA with Priors settings
    # カメラの回転・並進が初期値から動くことに対するペナルティ重み
    # 値が大きいほど「既知ポーズ」を強く信じる。小さいほど画像合わせを優先する。
    # 目安: 1.0e-2 〜 1.0e2 程度。
    # 3DGS用なら「ポーズはかなり正確」前提なので、少し強めにかけてドリフトを防ぐ。
    ba_pose_weight_rot: float = 0.5
    ba_pose_weight_trans: float = 0.5
    
    # ポーズ最適化に参加させるトラックの条件
    # すべての点を使うと計算が重く、かつ低品質な点がポーズを汚染するため
    ba_opt_min_track_len: int = 3        # 3視点以上で見えている点のみポーズ最適化に使う
    ba_opt_min_angle_deg: float = 2.0    # または、視差角がこれ以上ある場合

# ============================================================
# Data structures
# ============================================================

@dataclass
class ImageInfo:
    image_id: int
    name: str
    R_cw: np.ndarray  # (3,3) world -> camera
    t_cw: np.ndarray  # (3,)  world -> camera
    C: np.ndarray     # (3,)  camera center in world
    K: np.ndarray     # (3,3)
    dist: np.ndarray  # (N,) distortion
    keypoints_px: np.ndarray = field(default=None)    # (N,2)
    keypoints_norm: np.ndarray = field(default=None)  # (N,2) undistorted normalized
    descriptors: np.ndarray = field(default=None)     # (N,128) RootSIFT


# ============================================================
# Util: quaternion -> rotation (COLMAP)
# ============================================================

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    COLMAP convention: qvec = [qw, qx, qy, qz], world->camera rotation.
    R = qvec2rotmat(qvec) such that x_cam = R * X_world + t.
    """
    assert qvec.shape == (4,)
    w, x, y, z = qvec
    # 標準的な unit quaternion -> rotation matrix
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),         1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix (3x3) to Quaternion (w, x, y, z) for COLMAP.
    """
    # scipy is (x, y, z, w)
    quat = Rotation.from_matrix(R).as_quat()
    # Convert to COLMAP convention (w, x, y, z)
    return np.array([quat[3], quat[0], quat[1], quat[2]])

def compute_rotation_diff_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    R1とR2の間の回転角（degree）を計算
    """
    # R_diff = R2 * R1.T
    R_diff = R2 @ R1.T
    # trace = 1 + 2 cos(theta)
    tr = np.trace(R_diff)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def report_pose_changes(
    images_init: Dict[int, ImageInfo],
    images_opt: Dict[int, ImageInfo]
) -> None:
    """
    最適化前後のカメラポーズの変化量を計算して表示する
    """
    trans_diffs = []
    rot_diffs = []

    for img_id, info_opt in images_opt.items():
        if img_id not in images_init:
            continue
        info_init = images_init[img_id]

        # Translation diff (L2 norm)
        dt = np.linalg.norm(info_opt.t_cw - info_init.t_cw)
        trans_diffs.append(dt)

        # Rotation diff (Geodesic angle)
        dr = compute_rotation_diff_deg(info_init.R_cw, info_opt.R_cw)
        rot_diffs.append(dr)

    if not trans_diffs:
        print("No cameras to compare.")
        return

    trans_diffs = np.array(trans_diffs)
    rot_diffs = np.array(rot_diffs)

    print("\n" + "="*60)
    print("  Camera Pose Changes (Optimization Result)")
    print("="*60)
    print(f"Translation Change (World Units):")
    print(f"  Mean:   {trans_diffs.mean():.6f}")
    print(f"  Median: {np.median(trans_diffs):.6f}")
    print(f"  Max:    {trans_diffs.max():.6f}")
    print(f"  Min:    {trans_diffs.min():.6f}")
    print("-" * 40)
    print(f"Rotation Change (Degrees):")
    print(f"  Mean:   {rot_diffs.mean():.6f} deg")
    print(f"  Median: {np.median(rot_diffs):.6f} deg")
    print(f"  Max:    {rot_diffs.max():.6f} deg")
    print(f"  Min:    {rot_diffs.min():.6f} deg")
    print("="*60 + "\n")

# ============================================================
# Load intrinsics from YAML (K, dist)
# ============================================================

def load_intrinsics_from_yaml(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    YAML structure (example):
        K: !!opencv-matrix
          rows: 3
          cols: 3
          dt: d
          data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        dist: !!opencv-matrix
          rows: 1
          cols: 5
          dt: d
          data: [k1, k2, p1, p2, k3]
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Intrinsics YAML not found: {yaml_path}")

    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open YAML: {yaml_path}")

    K_node = fs.getNode("K")
    dist_node = fs.getNode("dist")

    if K_node.empty():
        fs.release()
        raise ValueError("Node 'K' not found in YAML")
    if dist_node.empty():
        fs.release()
        raise ValueError("Node 'dist' not found in YAML")

    K = K_node.mat().astype(np.float64)
    dist = dist_node.mat().astype(np.float64).ravel()
    fs.release()

    if K.shape != (3, 3):
        raise ValueError(f"Invalid K shape: {K.shape}")
    return K, dist


# ============================================================
# Load COLMAP images.txt (poses)
# ============================================================

def load_colmap_images_txt(images_txt_path: str, K: np.ndarray, dist: np.ndarray) -> Dict[int, ImageInfo]:
    """
    Read COLMAP images.txt (text format) and return dict[image_id] -> ImageInfo.

    images.txt format (per COLMAP):
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    """
    if not os.path.exists(images_txt_path):
        raise FileNotFoundError(f"images.txt not found: {images_txt_path}")

    image_infos: Dict[int, ImageInfo] = {}
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            # 10 要素以上あれば pose 行とみなす（2D points 行はもっと短い）
            if len(elems) > 10:
                continue

            image_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])), dtype=np.float64)
            tvec = np.array(list(map(float, elems[5:8])), dtype=np.float64)
            camera_id = int(elems[8])  # 未使用だが読み込んでおく
            name = elems[9]

            R_cw = qvec2rotmat(qvec)
            t_cw = tvec
            # camera center in world coordinates: C = -R^T * t
            C = -R_cw.T @ t_cw  # 

            image_infos[image_id] = ImageInfo(
                image_id=image_id,
                name=name,
                R_cw=R_cw,
                t_cw=t_cw,
                C=C,
                K=K.copy(),
                dist=dist.copy(),
            )

    if not image_infos:
        raise RuntimeError(f"No image poses found in {images_txt_path}")
    return image_infos


# ============================================================
# Feature extraction: SIFT + RootSIFT
# ============================================================

def rootsift(descriptors: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    RootSIFT: L1-normalize, then element-wise sqrt, then (optionally) L2 normalize.
    (Arandjelovic & Zisserman, 2012; implementation参考) 
    """
    if descriptors is None or len(descriptors) == 0:
        return descriptors

    desc = descriptors.astype(np.float32)
    # L1-normalize
    l1_norm = np.sum(np.abs(desc), axis=1, keepdims=True) + eps
    desc /= l1_norm
    # sqrt
    desc = np.sqrt(desc)
    # (必要なら L2 正規化を追加してもよい)
    return desc


def resize_for_max_size(img: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_resized, scale
    else:
        return img, scale

def extract_features(
    image_dir: str,
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
) -> None:
    """
    For each image:
      - load image
      - resize (max_image_size)
      - detect SIFT keypoints
      - compute descriptors
      - convert to RootSIFT
      - select features with grid-based spatial distribution
      - compute undistorted normalized coordinates using K, dist
    """
    # COLMAPに近いSIFT設定（peak_threshold ≈ 0.0067）
    sift = cv2.SIFT_create(
        nfeatures=0,              # 上限無し、後段で max_num_features 制限
        contrastThreshold=0.0067,
        edgeThreshold=10,
        sigma=1.6,
    )

    grid_cols = 8
    grid_rows = 6

    for image_id, info in images.items():
        img_path = os.path.join(image_dir, info.name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        h_orig, w_orig = img.shape[:2]
        img_resized, scale = resize_for_max_size(img, config.max_image_size)

        # Detect and compute
        keypoints, descriptors = sift.detectAndCompute(img_resized, None)

        if descriptors is None or len(keypoints) == 0:
            info.keypoints_px = np.zeros((0, 2), dtype=np.float32)
            info.keypoints_norm = np.zeros((0, 2), dtype=np.float32)
            info.descriptors = np.zeros((0, 128), dtype=np.float32)
            continue

        # RootSIFT
        descriptors = rootsift(descriptors)

        # 元スケール座標に戻す
        pts_px = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        pts_px /= scale

        num_kp = pts_px.shape[0]

        # グリッドごとに特徴点を均等に分布させる
        if num_kp > config.max_num_features:
            # 各セルに [ (response, idx), ... ]
            cell_lists: List[List[List[Tuple[float, int]]]] = [
                [[] for _ in range(grid_cols)] for _ in range(grid_rows)
            ]

            for idx, kp in enumerate(keypoints):
                x, y = pts_px[idx]
                c = int(x / max(w_orig, 1e-6) * grid_cols)
                r = int(y / max(h_orig, 1e-6) * grid_rows)
                c = min(max(c, 0), grid_cols - 1)
                r = min(max(r, 0), grid_rows - 1)
                cell_lists[r][c].append((kp.response, idx))

            # 各セルから quota 個ずつピックアップ
            total_cells = grid_rows * grid_cols
            base_quota = max(config.max_num_features // total_cells, 1)

            selected_indices = set()
            leftovers: List[Tuple[float, int]] = []

            for r in range(grid_rows):
                for c in range(grid_cols):
                    cell = cell_lists[r][c]
                    if not cell:
                        continue
                    cell_sorted = sorted(cell, key=lambda x: -x[0])
                    # まずセルごとの quota を取る
                    for resp, idx in cell_sorted[:base_quota]:
                        selected_indices.add(idx)
                    # 余りは後でグローバル選抜用に保存
                    for resp, idx in cell_sorted[base_quota:]:
                        leftovers.append((resp, idx))

            # quota 合計が max_num_features に足りなければ、余りから補充
            if len(selected_indices) < config.max_num_features and leftovers:
                remaining = config.max_num_features - len(selected_indices)
                leftovers_sorted = sorted(leftovers, key=lambda x: -x[0])
                for resp, idx in leftovers_sorted[:remaining]:
                    selected_indices.add(idx)

            # それでも多すぎる場合は全体レスポンス上位に制限
            if len(selected_indices) > config.max_num_features:
                # 選ばれた idx だけで再度レスポンス上位を取る
                selected_list = list(selected_indices)
                responses = np.array([keypoints[i].response for i in selected_list])
                order = np.argsort(-responses)[: config.max_num_features]
                final_indices = [selected_list[i] for i in order]
            else:
                final_indices = sorted(selected_indices)

            pts_px = pts_px[final_indices]
            descriptors = descriptors[final_indices]
        # num_kp <= max_num_features の場合はそのまま

        # undistort + normalized
        K = info.K
        dist = info.dist
        pts_undist = cv2.undistortPoints(
            pts_px.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=dist,
            P=None,
        )
        pts_norm = pts_undist.reshape(-1, 2).astype(np.float32)

        info.keypoints_px = pts_px
        info.keypoints_norm = pts_norm
        info.descriptors = descriptors


# ============================================================
# Pair selection using known poses
# ============================================================

def compute_scene_center(images: Dict[int, ImageInfo]) -> np.ndarray:
    centers = np.array([info.C for info in images.values()], dtype=np.float64)
    return centers.mean(axis=0)


def build_image_pairs(
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
) -> List[Tuple[int, int]]:
    """
    Use camera centers and viewing directions to select good-baseline pairs.

    以前は「閾値を満たすペアだけ採用」だったが、
    - データセットや撮像パスによっては条件を満たすペアが激減し、track が伸びない
    という問題があり得る。

    ここでは、全ペアについて「幾何スコア」を計算し、
    - 連番近傍には強いボーナス
    - 視差角 / baseline-depth 比が良いペアほど高スコア
    とし、各画像あたり上位 K 件を採用する方式に変更する。
    """

    image_ids = sorted(images.keys())
    scene_center = compute_scene_center(images)

    centers = {iid: images[iid].C for iid in image_ids}
    directions = {}
    depths = {}
    for iid in image_ids:
        C = centers[iid]
        v = scene_center - C
        d = np.linalg.norm(v)
        if d < 1e-8:
            v_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            d = 1.0
        else:
            v_dir = v / d
        directions[iid] = v_dir
        depths[iid] = d

    # スコアテーブル: per_image_scores[i] = [(score, j), ...]
    per_image_scores: Dict[int, List[Tuple[float, int]]] = {iid: [] for iid in image_ids}

    theta_min = config.theta_min_pair_deg
    theta_max = config.theta_max_pair_deg
    theta_mid = 0.5 * (theta_min + theta_max)
    theta_span = max(theta_max - theta_min, 1e-6)

    r_min = config.baseline_over_depth_min
    r_max = config.baseline_over_depth_max
    r_mid = 0.5 * (r_min + r_max)
    r_span = max(r_max - r_min, 1e-6)

    for i_idx, i in enumerate(image_ids):
        for j_idx in range(i_idx + 1, len(image_ids)):
            j = image_ids[j_idx]

            C_i = centers[i]
            C_j = centers[j]
            b = np.linalg.norm(C_i - C_j)

            d_i = depths[i]
            d_j = depths[j]
            d_ij = 0.5 * (d_i + d_j)
            if d_ij < 1e-8:
                continue
            r_ij = b / d_ij

            v_i = directions[i]
            v_j = directions[j]
            cos_theta = float(np.clip(np.dot(v_i, v_j), -1.0, 1.0))
            theta_ij = np.degrees(np.arccos(cos_theta))

            seq_dist = abs(i_idx - j_idx)

            # 幾何スコア（0〜1 付近）：theta_mid, r_mid からの距離で減衰
            w_theta = max(0.0, 1.0 - abs(theta_ij - theta_mid) / theta_span)
            w_r = max(0.0, 1.0 - abs(r_ij - r_mid) / r_span)
            geom_score = w_theta * w_r

            # 連番近傍には強いボーナスを付与（接続性確保）
            score = geom_score
            if seq_dist <= config.seq_neighbor_window:
                score += 1.0  # 連続撮像を優先

            # 負のスコアにならないようクリップ
            score = max(score, 0.0)

            per_image_scores[i].append((score, j))
            per_image_scores[j].append((score, i))

    selected_pairs_set = set()
    for i in image_ids:
        pairs = per_image_scores[i]
        if not pairs:
            continue
        # スコア順に並べて上位 K 件だけ採用
        pairs_sorted = sorted(pairs, key=lambda x: -x[0])
        top_js = [j for _, j in pairs_sorted[: config.max_pairs_per_image]]
        for j in top_js:
            if i < j:
                selected_pairs_set.add((i, j))
            elif j < i:
                selected_pairs_set.add((j, i))

    selected_pairs = sorted(list(selected_pairs_set))
    return selected_pairs



# ============================================================
# Feature matching + geometric verification
# ============================================================

def collect_best_matches(
    desc1: np.ndarray,
    desc2: np.ndarray,
    config: BaselineConfig,
) -> Dict[int, cv2.DMatch]:
    """
    For each query descriptor i in desc1, keep best match m to desc2
    that passes Lowe ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    best: Dict[int, cv2.DMatch] = {}
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < config.sift_ratio * n.distance:
            if m.queryIdx not in best or m.distance < best[m.queryIdx].distance:
                best[m.queryIdx] = m
    return best


def compute_known_essential_matrix(
    img_i: ImageInfo,
    img_j: ImageInfo,
) -> np.ndarray:
    """
    既知ポーズ (world->camera) から、カメラ i -> j の相対姿勢 (R_ji, t_ji) を求め、
    Essential 行列 E_ji = [t_ji]_x R_ji を計算する。

    x_i, x_j は正規化画像座標 (undistortPoints の出力) を想定。

    導出:
      x_i = R_i X + t_i
      x_j = R_j X + t_j

      x_j = R_j R_i^T (x_i - t_i) + t_j
          = R_ji x_i + (t_j - R_ji t_i)
      よって R_ji = R_j R_i^T, t_ji = t_j - R_ji t_i
      E = [t_ji]_x R_ji
    """
    R_i = img_i.R_cw
    t_i = img_i.t_cw.reshape(3, 1)
    R_j = img_j.R_cw
    t_j = img_j.t_cw.reshape(3, 1)

    R_ji = R_j @ R_i.T
    t_ji = t_j - R_ji @ t_i  # camera i -> j の平行移動（cam_j座標系）

    tx, ty, tz = float(t_ji[0]), float(t_ji[1]), float(t_ji[2])
    t_skew = np.array(
        [
            [0.0, -tz, ty],
            [tz, 0.0, -tx],
            [-ty, tx, 0.0],
        ],
        dtype=np.float64,
    )
    E = t_skew @ R_ji
    return E


def symmetric_epipolar_distance_sq(
    E: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    """
    対応点 x1 (N,2), x2 (N,2) [正規化座標] に対する symmetric epipolar distance^2 を計算。

    d_sym^2 = d1^2 + d2^2
      d1 = 距離( x2, l2 = E x1 )
      d2 = 距離( x1, l1 = E^T x2 )

    ここでは homogeneous 座標 x=(x,y,1), l=(a,b,c) に対し
      dist(x,l) = (a x + b y + c)^2 / (a^2 + b^2)
    を用いる。
    """
    assert x1.shape == x2.shape
    N = x1.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    x1_h = np.hstack([x1.astype(np.float64), ones])  # (N,3)
    x2_h = np.hstack([x2.astype(np.float64), ones])

    Ex1 = (E @ x1_h.T).T      # (N,3)
    Etx2 = (E.T @ x2_h.T).T   # (N,3)

    # x2^T E x1
    s = np.sum(x2_h * (E @ x1_h.T).T, axis=1)

    # 距離^2
    num = s * s
    denom1 = Ex1[:, 0]**2 + Ex1[:, 1]**2 + 1e-12
    denom2 = Etx2[:, 0]**2 + Etx2[:, 1]**2 + 1e-12

    d1_sq = num / denom1
    d2_sq = num / denom2
    d_sym_sq = d1_sq + d2_sq
    return d_sym_sq

def geom_verify_with_ransac_essential(
    pts1_norm: np.ndarray,
    pts2_norm: np.ndarray,
    th_norm: float,
    prob: float = 0.999,
) -> np.ndarray:
    """
    正規化座標 (x,y) で Essential を RANSAC 推定し、inlier mask を返す。
    OpenCV findEssentialMat は focal=1, pp=(0,0) で正規化座標扱いにできる。
    """
    if len(pts1_norm) < 5:
        return np.zeros((len(pts1_norm),), dtype=bool)

    E, mask = cv2.findEssentialMat(
        pts1_norm, pts2_norm,
        focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=prob,
        threshold=float(th_norm),
    )
    if mask is None:
        return np.zeros((len(pts1_norm),), dtype=bool)
    return (mask.ravel() > 0)

def match_features_for_pair(
    img_i: ImageInfo,
    img_j: ImageInfo,
    config: BaselineConfig,
    f_mean: float,
    scene_center: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Returns list of (kp_idx_i, kp_idx_j) after:
      - Lowe ratio
      - cross-check
      - known-pose-based geometric verification using Essential matrix E_known
        with angle-dependent threshold.

    ポイント:
      - Two-view 幾何モデルは known pose から一意に決まる。
      - ペアの視差角が小さいほどエピポーラ誤差に厳しく、大きいほどやや緩める。
    """
    desc1 = img_i.descriptors
    desc2 = img_j.descriptors
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    # 1) Lowe ratio + cross-check
    best_ij = collect_best_matches(desc1, desc2, config)
    best_ji = collect_best_matches(desc2, desc1, config)

    mutual_pairs: List[Tuple[int, int]] = []
    for qi, m in best_ij.items():
        tj = m.trainIdx
        m_back = best_ji.get(tj, None)
        if m_back is not None and m_back.trainIdx == qi:
            mutual_pairs.append((qi, tj))

    if len(mutual_pairs) < config.geom_min_num_inliers:
        return []

    # 2) ペア視差角の近似評価（カメラ中心→scene_center の方向）
    Ci = img_i.C
    Cj = img_j.C
    vi = scene_center - Ci
    vj = scene_center - Cj
    ni = np.linalg.norm(vi)
    nj = np.linalg.norm(vj)
    if ni < 1e-8 or nj < 1e-8:
        angle_pair_deg = 5.0
    else:
        vi /= ni
        vj /= nj
        cos_th = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
        angle_pair_deg = float(np.degrees(np.arccos(cos_th)))

    # 3) Known pose から Essential 行列
    pts1 = np.float64([img_i.keypoints_norm[i] for (i, _) in mutual_pairs])
    pts2 = np.float64([img_j.keypoints_norm[j] for (_, j) in mutual_pairs])

    # 視差角スケールまでは既存ロジックを踏襲
    base_px = config.geom_max_error_px
    if angle_pair_deg <= 2.0:
        scale = 0.5
    elif angle_pair_deg >= 5.0:
        scale = 1.0
    else:
        t = (angle_pair_deg - 2.0) / (5.0 - 2.0)
        scale = 0.5 + 0.5 * t

    th_norm = (base_px * scale) / max(f_mean, 1e-6)

    # まず RANSAC(E) でインライア抽出（COLMAP の思想に寄せる）
    inlier_mask = geom_verify_with_ransac_essential(pts1, pts2, th_norm, prob=0.999)

    inlier_pairs = [pair for pair, ok in zip(mutual_pairs, inlier_mask) if ok]

    # fallback: それでも足りない場合のみ known E（現状の実装）を使う
    if len(inlier_pairs) < config.geom_min_num_inliers:
        E_known = compute_known_essential_matrix(img_i, img_j)
        d_sym_sq = symmetric_epipolar_distance_sq(E_known, pts1, pts2)
        inlier_mask2 = (d_sym_sq <= (th_norm * th_norm))
        inlier_pairs = [pair for pair, ok in zip(mutual_pairs, inlier_mask2) if ok]

    if len(inlier_pairs) < config.geom_min_num_inliers:
        return []

    return inlier_pairs



def match_all_pairs(
    images: Dict[int, ImageInfo],
    pairs: List[Tuple[int, int]],
    config: BaselineConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    returns list of quadruples (image_id_i, kp_i, image_id_j, kp_j)
    after geometric verification.
    """
    # 平均焦点距離
    some_info = next(iter(images.values()))
    fx = some_info.K[0, 0]
    fy = some_info.K[1, 1]
    f_mean = 0.5 * (fx + fy)

    # 視差角評価用のシーン中心
    scene_center = compute_scene_center(images)

    all_matches: List[Tuple[int, int, int, int]] = []

    for (i, j) in pairs:
        img_i = images[i]
        img_j = images[j]
        inlier_pairs = match_features_for_pair(img_i, img_j, config, f_mean, scene_center)
        for kp_i, kp_j in inlier_pairs:
            all_matches.append((i, kp_i, j, kp_j))

    return all_matches



# ============================================================
# Track construction (Union-Find)
# ============================================================

class UnionFind:
    def __init__(self):
        self.parent: List[int] = []
        self.rank: List[int] = []

    def make_set(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def build_tracks(
    images: Dict[int, ImageInfo],
    all_matches: List[Tuple[int, int, int, int]],
) -> List[List[Tuple[int, int]]]:
    """
    Build multi-view tracks from pairwise matches.
    all_matches: list of (image_id_i, kp_i, image_id_j, kp_j)

    returns:
      tracks: list of [(image_id, kp_idx), ...]
    """
    uf = UnionFind()
    node_index: Dict[Tuple[int, int], int] = {}

    def get_node_id(img_id: int, kp_idx: int) -> int:
        key = (img_id, kp_idx)
        if key not in node_index:
            node_index[key] = uf.make_set()
        return node_index[key]

    # Union for all matches
    for img_i, kp_i, img_j, kp_j in all_matches:
        ni = get_node_id(img_i, kp_i)
        nj = get_node_id(img_j, kp_j)
        uf.union(ni, nj)

    # Group by root
    clusters: Dict[int, List[Tuple[int, int]]] = {}
    for (img_id, kp_idx), nid in node_index.items():
        root = uf.find(nid)
        clusters.setdefault(root, []).append((img_id, kp_idx))

    # 1画像1観測制約: 同一 image_id の重複がある場合は最初の1つだけ残す
    tracks: List[List[Tuple[int, int]]] = []
    for obs_list in clusters.values():
        per_image: Dict[int, int] = {}
        for img_id, kp_idx in obs_list:
            if img_id not in per_image:
                per_image[img_id] = kp_idx
        if len(per_image) >= 2:
            track = [(img_id, kp_idx) for img_id, kp_idx in per_image.items()]
            tracks.append(track)

    return tracks


# ============================================================
# Triangulation & Point-only BA
# ============================================================

def triangulate_track_linear(
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
) -> np.ndarray:
    """
    Multi-view linear triangulation (normalized coordinates).
    x_cam = R * X + t, with normalized image coordinate u = (x/z, y/z).

    Build A X = 0 using standard DLT.
    """
    A_rows = []
    for img_id, kp_idx in track:
        info = images[img_id]
        R = info.R_cw
        t = info.t_cw.reshape(3, 1)
        P = np.hstack([R, t])  # 3x4, normalized intrinsics (K=I)

        xn, yn = info.keypoints_norm[kp_idx]
        # x * P[2,:] - P[0,:], y * P[2,:] - P[1,:]
        row1 = xn * P[2, :] - P[0, :]
        row2 = yn * P[2, :] - P[1, :]
        A_rows.append(row1)
        A_rows.append(row2)

    A = np.asarray(A_rows, dtype=np.float64)
    if A.shape[0] < 4:
        return None

    # SVD
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-8:
        return None
    X = X_h[0:3] / X_h[3]
    return X

def triangulate_optimal_pair(
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
    min_angle_deg: float
) -> Optional[np.ndarray]:
    """
    COLMAP方式: トラック内の全ペアを探索し、
    「視差角が最大かつ十分大きいペア」を用いて三角測量を行う。
    全視点を混ぜるDLTよりも、デプス方向の初期値精度が高くなる。
    """
    best_angle = -1.0
    best_X = None
    
    # トラック内の全ペアを総当り (Nが小さいので高速)
    # Nが大きい場合は、時間的に離れたペアだけ見るなどの枝刈りも可能だが、
    # ここでは厳密さを優先して全探索する。
    n = len(track)
    for i in range(n):
        img_i, kp_i = track[i]
        Ci = images[img_i].C
        
        for j in range(i + 1, n):
            img_j, kp_j = track[j]
            Cj = images[img_j].C
            
            # 視差角の計算 (Center-to-Center ベースの簡易版ではなく、レイ同士の角度を見るべきだが
            # ここでは計算コストとロバスト性の兼ね合いでベースライン方向を見る)
            baseline = np.linalg.norm(Ci - Cj)
            if baseline < 1e-6:
                continue

            # 2視点での三角測量 (DLT for 2 views)
            # ペア単位で解く
            subset = [track[i], track[j]]
            X_pair = triangulate_track_linear(subset, images) # 既存のDLT関数を再利用
            
            if X_pair is None:
                continue
                
            # 視差角の厳密な評価: 生成された点 X_pair に対する角度
            v_i = Ci - X_pair
            v_j = Cj - X_pair
            n_i = np.linalg.norm(v_i)
            n_j = np.linalg.norm(v_j)
            
            if n_i < 1e-6 or n_j < 1e-6:
                continue
                
            cos_th = np.dot(v_i / n_i, v_j / n_j)
            angle_deg = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))
            
            # チラリティチェック (カメラの前にあるか)
            # 両方のカメラ座標系で Z > 0 であること
            info_i = images[img_i]
            X_cam_i = info_i.R_cw @ X_pair.reshape(3,1) + info_i.t_cw.reshape(3,1)
            
            info_j = images[img_j]
            X_cam_j = info_j.R_cw @ X_pair.reshape(3,1) + info_j.t_cw.reshape(3,1)
            
            if X_cam_i[2,0] > 0 and X_cam_j[2,0] > 0:
                if angle_deg > best_angle:
                    best_angle = angle_deg
                    best_X = X_pair

    # 最大視差角が閾値未満なら、このトラックは信頼できないので破棄
    if best_X is not None and best_angle >= min_angle_deg:
        return best_X
    
    return None

def compute_reproj_errors_norm(
    X: np.ndarray,
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
) -> np.ndarray:
    """
    Reprojection error in normalized coordinates (2D).
    Returns array of shape (len(track), 2) of residual vectors.
    """
    X = X.reshape(3, 1)
    residuals = []
    for img_id, kp_idx in track:
        info = images[img_id]
        R = info.R_cw
        t = info.t_cw.reshape(3, 1)
        X_cam = R @ X + t
        z = float(X_cam[2, 0])
        if z <= 1e-8:
            # Behind camera: large penalty vector
            residuals.append(np.array([1e3, 1e3], dtype=np.float64))
            continue
        xn = X_cam[0, 0] / z
        yn = X_cam[1, 0] / z
        obs = info.keypoints_norm[kp_idx]
        res = np.array([xn - obs[0], yn - obs[1]], dtype=np.float64)
        residuals.append(res)
    return np.vstack(residuals)


def compute_min_triangulation_angle_deg(
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
    Xw: np.ndarray,
) -> float:
    """
    現在の3D点 Xw を仮定したときの、track内の全カメラペアにおける
    三角測量角の最小値（deg）を返す。
    ここでは Xw 周りのカメラ中心ベクトルのなす角を計算する。
    """
    if Xw.ndim == 1:
        X = Xw.reshape(3)
    else:
        X = Xw.reshape(3)

    min_angle_deg = 180.0
    n = len(track)
    if n < 2:
        return 0.0

    for i in range(n):
        img_id_i, _ = track[i]
        Ci = images[img_id_i].C
        vi = X - Ci
        nvi = np.linalg.norm(vi)
        if nvi < 1e-8:
            continue
        vi /= nvi
        for j in range(i + 1, n):
            img_id_j, _ = track[j]
            Cj = images[img_id_j].C
            vj = X - Cj
            nvj = np.linalg.norm(vj)
            if nvj < 1e-8:
                continue
            vj /= nvj
            cos_th = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
            ang = np.degrees(np.arccos(cos_th))
            if ang < min_angle_deg:
                min_angle_deg = ang

    if min_angle_deg == 180.0:
        return 0.0
    return float(min_angle_deg)



def refine_point_ba(
    X0: np.ndarray,
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
    f_mean: float,
) -> np.ndarray:
    """
    Point-only BA with Huber loss on normalized residuals.
    三角測量角が小さいトラックほど Huber のスケールを小さくしてロバスト化する。
    """

    # 初期三角測量角（deg）を評価
    alpha = compute_min_triangulation_angle_deg(track, images, X0)
    # tri_min_angle_deg〜strong_track_min_angle_degの間で [0.5, 1.0] に線形写像
    tri_min = config.tri_min_angle_deg
    tri_strong = max(config.strong_track_min_angle_deg, tri_min + 1e-3)

    if alpha <= tri_min:
        angle_scale = 0.5
    elif alpha >= tri_strong:
        angle_scale = 1.0
    else:
        t = (alpha - tri_min) / (tri_strong - tri_min)
        angle_scale = 0.5 + 0.5 * t

    # px → 正規化座標での Huber f_scale
    base_scale_px = config.ba_loss_scale_px
    f_scale_norm = (base_scale_px / max(f_mean, 1e-6)) * angle_scale

    def residuals(x: np.ndarray) -> np.ndarray:
        X = x.reshape(3, 1)
        res_list = []
        for img_id, kp_idx in track:
            info = images[img_id]
            R = info.R_cw
            t = info.t_cw.reshape(3, 1)
            X_cam = R @ X + t
            z = float(X_cam[2, 0])
            if z <= 1e-8:
                res_list.extend([1e3, 1e3])
                continue
            xn = X_cam[0, 0] / z
            yn = X_cam[1, 0] / z
            obs = info.keypoints_norm[kp_idx]
            res_list.append(xn - obs[0])
            res_list.append(yn - obs[1])
        return np.array(res_list, dtype=np.float64)

    result = least_squares(
        residuals,
        X0,
        method="trf",
        loss="huber",
        f_scale=f_scale_norm,
        max_nfev=config.ba_max_nfev,
        verbose=0,
    )

    X_refined = result.x
    return X_refined



def collect_points_for_joint_ba(
    tracks: List[List[Tuple[int, int]]],
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], np.ndarray]:
    
    points_init = []
    tracks_kept = []
    mask_strong = []

    # COLMAP推奨値に近い厳し目の角度閾値
    # これ未満の点はデプス推定が数倍〜数十倍の誤差を持つため捨てる
    strict_min_angle = max(config.tri_min_angle_deg, 2.0) 

    print(f"Triangulating with Best-Pair selection (Threshold: {strict_min_angle} deg)...")

    for track in tracks:
        if len(track) < 2:
            continue
        
        # ★ 変更点: 全点DLTではなく、ベストペア探索を行う
        X = triangulate_optimal_pair(track, images, min_angle_deg=strict_min_angle)
        
        if X is None:
            continue
            
        # 以降の処理（Strong/Weak判定）は同じ
        # ただし、ベストペアで閾値を超えている時点で、ある程度Strongであることは保証されている
        
        # 念のため全トラック視点での最大角を再計算して Strong フラグを立てる
        angles = []
        track_dirs = []
        for img_id, _ in track:
            Ci = images[img_id].C
            v = X - Ci
            vn = np.linalg.norm(v)
            if vn > 1e-8:
                track_dirs.append(v / vn)
        
        max_angle = 0.0
        if len(track_dirs) >= 2:
            for i in range(len(track_dirs)):
                for j in range(i+1, len(track_dirs)):
                    cos_th = np.dot(track_dirs[i], track_dirs[j])
                    ang = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))
                    max_angle = max(max_angle, ang)
        
        # ポーズ最適化に使うのは、さらに角度がある信頼できる点のみ (例: 5度以上)
        # 2度〜5度の点は、点として残すがポーズ計算には使わない、といった運用も可能
        is_strong = (max_angle >= config.ba_opt_min_angle_deg)

        points_init.append(X)
        tracks_kept.append(track)
        mask_strong.append(is_strong)

    if not points_init:
        return np.zeros((0,3)), [], np.zeros((0,), dtype=bool)

    return np.vstack(points_init), tracks_kept, np.array(mask_strong, dtype=bool)



def compute_point_colors_from_observations(
    reps: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
    image_dir: str,
) -> np.ndarray:
    """
    各 3D 点に対して代表観測 (image_id, kp_idx) が与えられているとき、
    その観測画素から色をサンプリングして RGB 色を返す。

    Returns:
        colors: (N,3) uint8, RGB
    """
    # 画像のキャッシュ（image_id -> BGR画像）
    img_cache: Dict[int, np.ndarray] = {}

    colors = []
    for (img_id, kp_idx) in reps:
        info = images[img_id]
        if img_id not in img_cache:
            img_path = os.path.join(image_dir, info.name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                # 読み込めない場合はグレーを入れておく
                img = np.zeros((10, 10, 3), dtype=np.uint8)
            img_cache[img_id] = img

        img = img_cache[img_id]
        h, w = img.shape[:2]

        u, v = info.keypoints_px[kp_idx]  # (x,y) in pixels
        x = int(round(u))
        y = int(round(v))
        if x < 0 or x >= w or y < 0 or y >= h:
            # 範囲外ならとりあえず黒
            bgr = np.array([0, 0, 0], dtype=np.uint8)
        else:
            bgr = img[y, x, :]

        # BGR -> RGB
        rgb = bgr[::-1]
        colors.append(rgb)

    if len(colors) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    return np.stack(colors, axis=0).astype(np.uint8)



def statistical_outlier_removal(
    points: np.ndarray,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCL / Open3D の StatisticalOutlierRemoval と同じ発想の 3D 点群フィルタ。
    各点について k 近傍までの距離の平均を計算し、その分布の平均±std_ratio*std
    から外れた点をアウトライアとして除去する。

    Returns:
        filtered_points: (M,3)
        mask: bool array of shape (N,) indicating which original points are kept
    """
    N = points.shape[0]
    if N == 0:
        return points, np.zeros((0,), dtype=bool)
    if N <= k_neighbors:
        # 近傍数を満たさない場合はそのまま返す
        mask = np.ones((N,), dtype=bool)
        return points.copy(), mask

    tree = cKDTree(points)
    # k+1 (自分自身を含むので) 最近傍検索
    dists, _ = tree.query(points, k=k_neighbors + 1)
    # 自分自身との距離0を除外
    mean_dists = dists[:, 1:].mean(axis=1)

    mu = mean_dists.mean()
    sigma = mean_dists.std(ddof=1) if N > 1 else 0.0
    if sigma < 1e-12:
        mask = np.ones((N,), dtype=bool)
        return points.copy(), mask

    # PCL/CloudCompare と同様: μ + std_ratio * σ を上限にする 
    thresh = mu + std_ratio * sigma
    mask = mean_dists <= thresh
    filtered_points = points[mask]
    return filtered_points, mask


# ============================================================
# Joint bundle adjustment (optimize poses + points)
# ============================================================

class IterativeRefiner:
    def __init__(self, images: Dict[int, ImageInfo], config: BaselineConfig):
        self.images = images
        self.config = config
        
        # 焦点距離の目安（Lossのスケール調整用）
        some_info = next(iter(images.values()))
        self.f_scale = 0.5 * (some_info.K[0,0] + some_info.K[1,1])
        
        # 初期ポーズの保存（Priors用）
        self.priors = {}
        for iid, info in images.items():
            rvec, _ = cv2.Rodrigues(info.R_cw)
            self.priors[iid] = {
                'rvec': rvec.ravel(),
                'tvec': info.t_cw.copy()
            }

    def _optimize_single_camera(
        self, 
        img_id: int, 
        anchor_tracks: List[Tuple[int, np.ndarray]], # [(pt_idx, point3d), ...]
        w_rot: float, 
        w_trans: float
    ):
        """
        単一カメラのポーズを、アンカー点との再投影誤差 + 初期値Priors で最適化する。
        """
        info = self.images[img_id]
        prior = self.priors[img_id]
        
        # 初期値
        rvec_curr, _ = cv2.Rodrigues(info.R_cw)
        tvec_curr = info.t_cw
        x0 = np.hstack([rvec_curr.ravel(), tvec_curr.ravel()])
        
        # 観測データ
        obs_uvs = []
        pts_3d = []
        for kp_idx, Xw in anchor_tracks:
            obs_uvs.append(info.keypoints_norm[kp_idx])
            pts_3d.append(Xw)
            
        obs_uvs = np.array(obs_uvs)
        pts_3d = np.array(pts_3d)
        n_obs = len(obs_uvs)
        
        if n_obs < 4:
            return # 拘束不足なら更新しない

        def residuals(x):
            r = x[:3]
            t = x[3:]
            
            # 1. Reprojection
            R, _ = cv2.Rodrigues(r)
            # Vectorized projection
            X_cam = (R @ pts_3d.T).T + t # (N, 3)
            z = X_cam[:, 2]
            
            valid = z > 1e-6
            res_reproj = np.zeros(n_obs * 2)
            
            if np.any(valid):
                u = X_cam[valid, 0] / z[valid]
                v = X_cam[valid, 1] / z[valid]
                res_reproj[0::2][valid] = u - obs_uvs[valid, 0]
                res_reproj[1::2][valid] = v - obs_uvs[valid, 1]
            
            # Invalid points penalty
            if np.any(~valid):
                res_reproj[0::2][~valid] = 10.0
                res_reproj[1::2][~valid] = 10.0

            # 2. Priors
            res_prior_r = (r - prior['rvec']) * w_rot
            res_prior_t = (t - prior['tvec']) * w_trans
            
            return np.hstack([res_reproj, res_prior_r, res_prior_t])

        # Optimize
        res = least_squares(
            residuals, x0, 
            method='trf', 
            loss='huber', f_scale=1.0/self.f_scale,
            max_nfev=50, # 小刻みに回すので少なめでOK
            verbose=0
        )
        
        # Update
        r_opt = res.x[:3]
        t_opt = res.x[3:]
        R_opt, _ = cv2.Rodrigues(r_opt)
        
        info.R_cw = R_opt
        info.t_cw = t_opt
        info.C = -R_opt.T @ t_opt

    def _optimize_points_subset(self, points: np.ndarray, tracks: List[List[Tuple[int, int]]], subset_indices: List[int]):
        """
        指定されたインデックスの3D点のみを最適化する（カメラ固定）。
        """
        for i in subset_indices:
            track = tracks[i]
            Xw = points[i]
            
            # 高速化のため、scipyを使わず簡易Gauss-Newton等を自作するか、
            # あるいは既存の refine_point_ba を呼び出す。
            # ここでは既存関数を流用（ただし呼び出しコストに注意）
            X_new = refine_point_ba(Xw, track, self.images, self.config, self.f_scale)
            points[i] = X_new

    def run(self, points: np.ndarray, tracks: List[List[Tuple[int, int]]], iterations: int = 3):
        """
        交互最適化のメインループ
        """
        num_points = points.shape[0]
        if num_points == 0:
            return points

        # 設定値（少し強めのPriorから始める）
        w_rot = self.config.ba_pose_weight_rot
        w_trans = self.config.ba_pose_weight_trans
        
        print(f"Starting Iterative Refinement ({iterations} iters)...")
        
        for it in range(iterations):
            print(f"  Iteration {it+1}/{iterations}")
            
            # --- Step 1: Select Anchors ---
            # 各反復で、「現在」のポーズと点群から見て信頼できる点を再選定する
            # (幾何が改善されれば、アンカーとして使える点が増える可能性がある)
            anchor_indices = []
            
            # Image ID -> List of (pt_idx, point3d)
            cam_observations = {iid: [] for iid in self.images.keys()}
            
            # シーン中心（視差角計算用）
            # 簡易的に計算（厳密でなくても良い）
            # scene_center = points.mean(axis=0) 
            
            for i in range(num_points):
                track = tracks[i]
                Xw = points[i]
                
                # 条件: トラック長 >= 3 かつ 視差角が十分大きい
                # ここでは厳し目にフィルタリングする
                min_ang = compute_min_triangulation_angle_deg(track, self.images, Xw)
                
                # アンカー条件: 視差角が 2.0度以上 (設定依存)
                if len(track) >= 3 and min_ang >= 2.0:
                    anchor_indices.append(i)
                    for img_id, kp_idx in track:
                        # ここで point のコピーを渡さないと、後で point が更新されたときに整合性が崩れるが、
                        # Step 2 では point は固定なので参照でもOK。念のためコピー。
                        cam_observations[img_id].append((kp_idx, Xw.copy()))
            
            print(f"    Anchors selected: {len(anchor_indices)} / {num_points} points")
            if len(anchor_indices) < 10:
                print("    Too few anchors! Skipping pose update.")
            else:
                # --- Step 2: Optimize Cameras (Independent) ---
                # 各カメラを独立に最適化（並列化可能だがここではforループ）
                # アンカー点のみを使用する
                for img_id in self.images:
                    obs = cam_observations[img_id]
                    if len(obs) < 4: continue # 観測が少なすぎるカメラはスキップ
                    
                    self._optimize_single_camera(img_id, obs, w_rot, w_trans)
                
            # --- Step 3: Optimize All Points (Independent) ---
            # 更新されたカメラに対して、全点を最適化
            # ここは並列化すると速いが、Pythonスレッドオーバーヘッドもあるので愚直に
            # 全点を一気に回す
            all_indices = list(range(num_points))
            self._optimize_points_subset(points, tracks, all_indices)

        return points

def filter_bad_points_after_ba(
    points: np.ndarray,
    tracks: List[List[Tuple[int, int]]],
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], List[Tuple[int, int]]]:
    """
    Joint BA後の点群から、再投影誤差が大きい点や、
    三角測量角が極端に小さい点（デプスが不安定な点）を除去する。
    また、色取得用の代表観測（最も中心に近い、または誤差が小さい観測）を選出する。
    """
    if points.shape[0] == 0:
        return points, tracks, []

    # 平均焦点距離（ピクセル換算用）
    some_info = next(iter(images.values()))
    fx = some_info.K[0, 0]
    fy = some_info.K[1, 1]
    f_mean = 0.5 * (fx + fy)
    
    # 閾値設定（BA後なので、やや厳し目に見る）
    # 平均誤差が 2.0px を超えるものは除去（設定次第）
    max_mean_err_px = 2.0 
    
    kept_points = []
    kept_tracks = []
    kept_reps = [] # 色サンプリング用の代表観測 (image_id, kp_idx)

    for i in range(points.shape[0]):
        Xw = points[i]
        track = tracks[i]
        
        # 1. 再投影誤差の計算 (px)
        rms_err, errs = compute_reproj_errors_px_point(Xw, track, images)
        
        # 平均誤差チェック
        if rms_err > max_mean_err_px:
            continue
            
        # 2. 三角測量角のチェック
        min_angle = compute_min_triangulation_angle_deg(track, images, Xw)
        if min_angle < config.tri_min_angle_deg:
            continue
            
        # 3. 前方性チェック（念のため）
        is_front = True
        for (img_id, _) in track:
            info = images[img_id]
            X_cam = info.R_cw @ Xw.reshape(3,1) + info.t_cw.reshape(3,1)
            if X_cam[2] < 1e-4:
                is_front = False
                break
        if not is_front:
            continue

        # 4. 代表観測の選出
        # ここでは再投影誤差が最も小さい観測を代表とする
        best_idx = np.argmin(errs)
        rep_obs = track[best_idx]

        kept_points.append(Xw)
        kept_tracks.append(track)
        kept_reps.append(rep_obs)

    if not kept_points:
        return np.zeros((0, 3)), [], []

    return np.array(kept_points), kept_tracks, kept_reps

def compute_reproj_errors_px_point(
    Xw: np.ndarray,
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
) -> Tuple[float, np.ndarray]:
    """
    単一3D点 Xw とそのトラックに対する再投影誤差 [px] を計算する。

    Returns:
        rms_error_px: 各観測誤差の RMS (root-mean-square) [px]
        errs_px: 各観測ごとの誤差ノルム [px] の配列 shape (M,)
    """
    if Xw.ndim == 1:
        X = Xw.reshape(1, 1, 3).astype(np.float64)
    else:
        X = Xw.reshape(1, 1, 3).astype(np.float64)

    errs = []
    for (img_id, kp_idx) in track:
        info = images[img_id]
        # R_cw (world→cam) を Rodrigues へ
        rvec, _ = cv2.Rodrigues(info.R_cw.astype(np.float64))
        tvec = info.t_cw.reshape(3, 1).astype(np.float64)

        proj, _ = cv2.projectPoints(
            X, rvec, tvec, info.K.astype(np.float64), info.dist.astype(np.float64)
        )
        u, v = proj[0, 0]
        obs = info.keypoints_px[kp_idx]
        du = float(u - obs[0])
        dv = float(v - obs[1])
        errs.append(math.hypot(du, dv))

    if len(errs) == 0:
        return 0.0, np.zeros((0,), dtype=np.float64)

    errs_px = np.array(errs, dtype=np.float64)
    rms = float(np.sqrt(np.mean(errs_px ** 2)))
    return rms, errs_px


def compute_reproj_errors_px_for_all(
    points: np.ndarray,
    tracks: List[List[Tuple[int, int]]],
    images: Dict[int, ImageInfo],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    全3D点について RMS 誤差と全観測誤差を計算。

    Returns:
        per_point_rms: shape (N,) 各点の RMS reprojection error [px]
        all_obs_errs:  shape (M_total,) 全観測の誤差ノルム [px] を連結したもの
    """
    N = points.shape[0]
    per_point_rms = []
    all_errs_list = []

    for Xw, track in zip(points, tracks):
        rms, errs = compute_reproj_errors_px_point(Xw, track, images)
        per_point_rms.append(rms)
        if errs.size > 0:
            all_errs_list.append(errs)

    if N == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    per_point_rms = np.array(per_point_rms, dtype=np.float64)

    if len(all_errs_list) > 0:
        all_obs_errs = np.concatenate(all_errs_list, axis=0)
    else:
        all_obs_errs = np.zeros((0,), dtype=np.float64)

    return per_point_rms, all_obs_errs


def report_reprojection_error_stats(all_obs_errs: np.ndarray) -> None:
    """
    全観測の再投影誤差 [px] から、代表値をログ出力。
    """
    if all_obs_errs.size == 0:
        print("No reprojection errors to report.")
        return

    mean_err = float(all_obs_errs.mean())
    median_err = float(np.median(all_obs_errs))
    max_err = float(all_obs_errs.max())

    print(
        f"Reprojection error over all observations (px): "
        f"mean={mean_err:.3f}, median={median_err:.3f}, max={max_err:.3f}"
    )


def write_points3D_txt(
    points: np.ndarray,
    tracks: List[List[Tuple[int, int]]],
    colors: Optional[np.ndarray],
    per_point_rms: Optional[np.ndarray],
    path: str,
) -> None:
    """
    COLMAP の points3D.txt 形式で3D点群を書き出す。

    - POINT3D_ID は 1..N の連番とする
    - R,G,B は 0–255 の uchar
    - ERROR は各点の RMS 再投影誤差 [px]
    - TRACK は (IMAGE_ID, POINT2D_IDX) のペア列

    注意:
      POINT2D_IDX は、このパイプライン内の keypoints 配列の index として扱う。
      既存の COLMAP images.txt / points2D.txt と完全互換にしたい場合は、
      2D点のインデックスとの対応付けが必要になる。
    """
    N = points.shape[0]
    assert N == len(tracks)

    if colors is None or colors.shape[0] != N:
        colors = np.full((N, 3), 255, dtype=np.uint8)

    if per_point_rms is None or per_point_rms.shape[0] != N:
        per_point_rms = np.zeros((N,), dtype=np.float64)

    if N == 0:
        mean_track_len = 0.0
    else:
        mean_track_len = float(sum(len(tr) for tr in tracks) / N)

    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {N}, mean track length: {mean_track_len}\n")

        for idx in range(N):
            point3D_id = idx + 1
            X = points[idx]
            c = colors[idx]
            err = float(per_point_rms[idx])

            header_vals = [
                point3D_id,
                X[0],
                X[1],
                X[2],
                int(c[0]),
                int(c[1]),
                int(c[2]),
                err,
            ]
            f.write(" ".join(str(v) for v in header_vals))

            tr = tracks[idx]
            if tr:
                f.write(" ")
                track_tokens = []
                for (img_id, kp_idx) in tr:
                    track_tokens.append(str(img_id))
                    track_tokens.append(str(kp_idx))
                f.write(" ".join(track_tokens))

            f.write("\n")

    print(f"Wrote COLMAP points3D.txt: {path} (N={N})")



# ============================================================
# Optional: write PLY
# ============================================================

def write_points_to_ply(points: np.ndarray, ply_path: str, colors: np.ndarray = None) -> None:
    """
    Write 3D points as ASCII PLY.
    colors: (N,3) uint8 (RGB) があれば色情報も出力する。
    """
    N = points.shape[0]
    if N == 0:
        print("No points to write. Skipping PLY export.")
        return

    use_color = colors is not None and colors.shape[0] == N

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if use_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if use_color:
            for X, c in zip(points, colors):
                f.write(f"{X[0]} {X[1]} {X[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for X in points:
                f.write(f"{X[0]} {X[1]} {X[2]}\n")

    print(f"Wrote {N} points to {ply_path} (colors={'yes' if use_color else 'no'})")



def save_colmap_reconstruction(
    output_dir: str,
    images: Dict[int, ImageInfo],
    points: np.ndarray,
    tracks: List[List[Tuple[int, int]]],
    colors: Optional[np.ndarray],
    per_point_rms: Optional[np.ndarray]
) -> None:
    """
    最適化後のカメラと点群をCOLMAP形式 (images.txt, points3D.txt) で出力する。
    画像上の特徴点と3D点のID対応付けも行う。
    """
    os.makedirs(output_dir, exist_ok=True)
    images_txt_path = os.path.join(output_dir, "images.txt")
    points3d_txt_path = os.path.join(output_dir, "points3D.txt")
    
    num_points = points.shape[0]
    
    # 色とエラー情報のデフォルト値
    if colors is None or colors.shape[0] != num_points:
        colors = np.full((num_points, 3), 255, dtype=np.uint8)
    if per_point_rms is None or per_point_rms.shape[0] != num_points:
        per_point_rms = np.zeros((num_points,), dtype=np.float64)

    # -------------------------------------------------------
    # 1. 3D点IDのマッピング構築
    # -------------------------------------------------------
    # trackに含まれる (img_id, kp_idx) -> point3d_id の逆引き辞書を作成
    # COLMAPのPoint3D IDは1から始めるのが通例
    
    # key: img_id, val: { kp_idx: point3d_id }
    feature_to_point_map: Dict[int, Dict[int, int]] = {}
    
    # 全画像を初期化
    for img_id in images:
        feature_to_point_map[img_id] = {}

    for i in range(num_points):
        point3d_id = i + 1
        for (img_id, kp_idx) in tracks[i]:
            if img_id in feature_to_point_map:
                feature_to_point_map[img_id][kp_idx] = point3d_id

    # -------------------------------------------------------
    # 2. Write images.txt
    # -------------------------------------------------------
    print(f"Writing images.txt to {images_txt_path} ...")
    with open(images_txt_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: ...\n")

        # image_id順にソートして出力
        sorted_img_ids = sorted(images.keys())
        
        for img_id in sorted_img_ids:
            info = images[img_id]
            qvec = rotmat2qvec(info.R_cw)
            tvec = info.t_cw
            
            # Line 1: Pose
            # CAMERA_ID はここでは簡易的に 1 固定とする（単一カメラ想定）
            cam_id = 1 
            f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                    f"{tvec[0]} {tvec[1]} {tvec[2]} {cam_id} {info.name}\n")
            
            # Line 2: 2D Points
            # 特徴点配列がある場合のみ出力
            if info.keypoints_px is not None:
                kps = info.keypoints_px
                num_kps = kps.shape[0]
                
                point_strings = []
                # マッピング辞書
                kp_map = feature_to_point_map.get(img_id, {})
                
                for k_idx in range(num_kps):
                    x, y = kps[k_idx]
                    # この特徴点が3D点に関連付けられていればそのID、なければ -1
                    p3d_id = kp_map.get(k_idx, -1)
                    point_strings.append(f"{x} {y} {p3d_id}")
                
                f.write(" ".join(point_strings) + "\n")
            else:
                f.write("\n")

    # -------------------------------------------------------
    # 3. Write points3D.txt
    # -------------------------------------------------------
    print(f"Writing points3D.txt to {points3d_txt_path} ...")
    with open(points3d_txt_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {num_points}\n")

        for i in range(num_points):
            point3d_id = i + 1
            X = points[i]
            c = colors[i]
            err = per_point_rms[i]
            tr = tracks[i]

            # Header info
            f.write(f"{point3d_id} {X[0]} {X[1]} {X[2]} {c[0]} {c[1]} {c[2]} {err}")
            
            # Track info
            for (img_id, kp_idx) in tr:
                f.write(f" {img_id} {kp_idx}")
            f.write("\n")

    print("COLMAP export finished.")


# ============================================================
# Visualization (matplotlib 3D)
# ============================================================

def visualize_reconstruction(
    images: Dict[int, ImageInfo],
    points: np.ndarray,
    title: str = "Reconstructed sparse point cloud",
    save_path: str = "reconstruction.png",
) -> None:
    """
    3D点群とカメラ中心を matplotlib で可視化。
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if points.size > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1,
            alpha=0.8,
        )
    else:
        print("Warning: no 3D points to visualize.")

    # Camera centers
    cams = np.array([info.C for info in images.values()], dtype=np.float64)
    ax.scatter(
        cams[:, 0],
        cams[:, 1],
        cams[:, 2],
        marker="^",
        s=20,
    )

    ax.set_xlabel("X [world]")
    ax.set_ylabel("Y [world]")
    ax.set_zlabel("Z [world]")
    ax.set_title(title)
    ax.view_init(elev=20.0, azim=-60.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved visualization to {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-pose sparse reconstruction baseline (for 3DGS initialization)"
    )
    parser.add_argument(
        "--images_txt",
        required=True,
        help="Path to COLMAP images.txt (poses in text format)",
    )
    parser.add_argument(
        "--intrinsics_yaml",
        required=True,
        help="Path to intrinsics YAML file (with K, dist)",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing input images (filenames must match images.txt)",
    )
    parser.add_argument(
        "--output_ply",
        default=None,
        help="Optional: path to output PLY file for 3DGS point cloud",
    )

    parser.add_argument(
        "--output_points3d",
        type=str,
        default="points3D.txt",
        help="Path to write COLMAP-style points3D.txt (3D points, colors, tracks, and reprojection error).",
    )

    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open interactive matplotlib window (still saves PNG)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = BaselineConfig()

    # 1. Load Data
    K, dist = load_intrinsics_from_yaml(args.intrinsics_yaml)
    images = load_colmap_images_txt(args.images_txt, K, dist)

    # 画像ポーズの初期状態をディープコピーして保存（比較用）
    print("Snapshotting initial poses for comparison...")
    images_init = copy.deepcopy(images)

    # 2. Features & Matching & Tracks
    # (既存のコードと同じ)
    extract_features(args.image_dir, images, config)
    pairs = build_image_pairs(images, config)
    all_matches = match_all_pairs(images, pairs, config)
    tracks = build_tracks(images, all_matches)

    # 3. Initialization (DLT Only)
    # ここで "filter_and_collect_points" は使わず、DLTで初期値を決めるだけの関数を使う
    # (前回の回答にある collect_points_for_joint_ba のロジックを使用)
    print("\n--- Phase 1: Initialization (DLT) ---")
    points_init, tracks_init, _ = collect_points_for_joint_ba(tracks, images, config)
    # mask_strong は IterativeRefiner 内部で動的に再計算するため、ここでは不要
    
    print(f"Initial raw points: {points_init.shape[0]}")
    if points_init.shape[0] == 0:
        return

    # 4. Iterative Refinement (Round-Robin)
    print("\n--- Phase 2: Iterative Refinement (Alternating Optimization) ---")
    # 交互最適化の実行
    refiner = IterativeRefiner(images, config)
    points_refined = refiner.run(points_init, tracks_init, iterations=15)

    # --- ここで変化量を出力 ---
    # images は refiner.run の内部で直接更新されている
    report_pose_changes(images_init, images)

    # 5. Final Filtering & Output
    # (既存のコードと同じ)
    print("\n--- Phase 3: Filtering Bad Points ---")
    points_filtered, tracks_filtered, reps_filtered = filter_bad_points_after_ba(
        points_refined, tracks_init, images, config
    )
    
    if points_filtered.shape[0] > 0:
        print("Applying Statistical Outlier Removal (SOR)...")
        points_final, mask_sor = statistical_outlier_removal(points_filtered)
        tracks_final = [t for t, m in zip(tracks_filtered, mask_sor) if m]
        reps_final = [r for r, m in zip(reps_filtered, mask_sor) if m]
    else:
        points_final = points_filtered
        tracks_final = tracks_filtered
        reps_final = reps_filtered

    # Output PLY / points3D.txt
    if points_final.shape[0] > 0:
        colors = compute_point_colors_from_observations(reps_final, images, args.image_dir)
    # 再投影誤差の計算
    print("Computing final reprojection errors...")
    per_point_rms, all_obs_errs = compute_reproj_errors_px_for_all(
        points_final, tracks_final, images
    )
    report_reprojection_error_stats(all_obs_errs)

    # --- 最終出力: COLMAP形式 ---
    output_dir = "known_pose_BA"
    print(f"Exporting optimized model to directory: {output_dir}")
    save_colmap_reconstruction(
        output_dir,
        images,          # 最適化後の画像ポーズ
        points_final,    # フィルタリング後の点群
        tracks_final,    # 対応するトラック
        colors,          # 色
        per_point_rms    # エラー値
    )

    # 必要であれば PLY も出力
    if args.output_ply:
        write_points_to_ply(points_final, args.output_ply, colors=colors)

if __name__ == "__main__":
    main()
