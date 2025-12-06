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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
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
    sift_ratio: float = 0.8
    cross_check: bool = True
    geom_max_error_px: float = 0.4   # ★ 4→3 に引き締め（エピポーラ誤差）
    geom_confidence: float = 0.999
    geom_min_inlier_ratio: float = 0.25
    geom_min_num_inliers: int = 10
    guided_matching: bool = False

    # Triangulation & BA
    tri_min_angle_deg: float = 4.0   # ★ 1.5→2.0deg（COLMAP推奨より少しだけ強め）
    use_two_view_tracks_for_eval: bool = False
    use_two_view_tracks_for_gs: bool = True
    
    # Track quality classification
    strong_track_min_angle_deg: float = 10.0  # ≧この視差角 or 長さ≥4なら強トラック
    weak_track_min_angle_deg: float = 4.0    # ≧この視差角なら弱トラック候補

    # BA loss / filtering (px → 後で正規化)
    ba_loss_scale_px: float = 1.0    # ★ ロバスト閾値もやや厳しく
    point_mean_reproj_max_px: float = 1.0  # ★ 平均再投影誤差を 1px 以内
    point_max_reproj_max_px: float = 1.0   # ★ 最大誤差 2px 以内
    obs_drop_thresh_px: float = 1.0        # ★ 観測落としは 3px

    # BA iterations
    ba_max_nfev: int = 500



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

    E_known = compute_known_essential_matrix(img_i, img_j)
    d_sym_sq = symmetric_epipolar_distance_sq(E_known, pts1, pts2)

    # 4) 視差角に応じてエピポーラ閾値をスケーリング
    #    - angle <= 2deg なら 0.5 * geom_max_error_px
    #    - angle >= 5deg なら 1.0 * geom_max_error_px
    #    - その間は線形補間
    base_px = config.geom_max_error_px
    if angle_pair_deg <= 2.0:
        scale = 0.5
    elif angle_pair_deg >= 5.0:
        scale = 1.0
    else:
        t = (angle_pair_deg - 2.0) / (5.0 - 2.0)
        scale = 0.5 + 0.5 * t

    th_norm = (base_px * scale) / max(f_mean, 1e-6)
    th_norm_sq = th_norm * th_norm

    inlier_mask = d_sym_sq <= th_norm_sq
    inlier_pairs = [pair for pair, ok in zip(mutual_pairs, inlier_mask) if ok]

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



def filter_and_collect_points(
    tracks: List[List[Tuple[int, int]]],
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
    """
    Triangulate + iterative point-only BA + filtering.

    Returns:
      points_gs:  3D points for 3DGS (track_len >= 2, after filtering)
      points_eval: 3D points for evaluation (track_len >= 3, after filtering)
      reps_gs: representative observation (image_id, kp_idx) for each 3DGS point
      tracks_gs: final multi-view track (list of (image_id, kp_idx)) per 3DGS point
    """

    if not tracks:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            [],
        )

    # 平均焦点距離
    some_info = next(iter(images.values()))
    fx = some_info.K[0, 0]
    fy = some_info.K[1, 1]
    f_mean = 0.5 * (fx + fy)

    # pxベースの閾値を正規化座標系へ
    base_mean_thresh_norm = config.point_mean_reproj_max_px / max(f_mean, 1e-6)
    base_max_thresh_norm = config.point_max_reproj_max_px / max(f_mean, 1e-6)
    obs_drop_thresh_norm = config.obs_drop_thresh_px / max(f_mean, 1e-6)

    # シーン中心で擬似的な視差角を計算し、トラックを強／弱に分類
    scene_center = compute_scene_center(images)
    strong_tracks: List[List[Tuple[int, int]]] = []
    weak_tracks: List[List[Tuple[int, int]]] = []

    for track in tracks:
        if len(track) < 2:
            continue

        # track に含まれる画像の視線方向（カメラ中心→scene_center）から近似三角測量角を計算
        min_angle_deg = 180.0
        n = len(track)
        dirs = []
        for img_id, _ in track:
            Ci = images[img_id].C
            v = scene_center - Ci
            nv = np.linalg.norm(v)
            if nv < 1e-8:
                continue
            dirs.append(v / nv)
        if len(dirs) < 2:
            # 視差角評価不可 → 弱トラック扱い
            weak_tracks.append(track)
            continue

        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                cos_th = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
                ang = np.degrees(np.arccos(cos_th))
                if ang < min_angle_deg:
                    min_angle_deg = ang
        if min_angle_deg == 180.0:
            min_angle_deg = 0.0

        # 分類ロジック：
        # - 視差角が十分大きい (>= strong_track_min_angle_deg) または track長>=4 → 強
        # - 視差角が弱閾値以上 → 弱
        # - それ未満 → 処理対象外（幾何的にあまり意味がない）
        if min_angle_deg >= config.strong_track_min_angle_deg or len(track) >= 4:
            strong_tracks.append(track)
        elif min_angle_deg >= config.weak_track_min_angle_deg:
            weak_tracks.append(track)
        else:
            # 非常に小さな視差角しかないトラックはスキップ
            continue

    points_gs: List[np.ndarray] = []
    points_eval: List[np.ndarray] = []
    reps_gs: List[Tuple[int, int]] = []
    tracks_gs: List[List[Tuple[int, int]]] = []

    def process_track_list(track_list: List[List[Tuple[int, int]]]) -> None:
        nonlocal points_gs, points_eval, reps_gs

        for track in track_list:
            if len(track) < 2:
                continue

            # ---- 最大2回の (線形三角測量 → 点のみBA → 観測除外) 反復 ----
            current_track = list(track)
            X_ref = None
            success = False

            for _iter in range(2):
                # (a) 線形三角測量
                X0 = triangulate_track_linear(current_track, images)
                if X0 is None:
                    success = False
                    break

                # (b) 点のみBA（角度依存の Huber スケール）
                X_ref = refine_point_ba(X0, current_track, images, config, f_mean)

                # (c) 再投影誤差で高誤差観測を除外
                res = compute_reproj_errors_norm(X_ref, current_track, images)  # (M,2)
                errs = np.linalg.norm(res, axis=1)

                good_idx = np.where(errs <= obs_drop_thresh_norm)[0]
                if len(good_idx) < 2:
                    success = False
                    break

                new_track = [current_track[k] for k in good_idx]

                # 観測が変化しなくなれば収束
                if len(new_track) == len(current_track):
                    success = True
                    current_track = new_track
                    break
                else:
                    current_track = new_track

            if not success or X_ref is None:
                continue

            # ---- 最終観測集合での誤差・角度評価 ----
            res_final = compute_reproj_errors_norm(X_ref, current_track, images)
            errs_final = np.linalg.norm(res_final, axis=1)

            mean_err = float(errs_final.mean())
            max_err = float(errs_final.max())

            # 実三角測量角（deg）を計算
            Xw = X_ref.reshape(3, 1)
            min_angle_deg = compute_min_triangulation_angle_deg(current_track, images, Xw)

            # tri_min_angle_deg 未満なら不採用
            if min_angle_deg < config.tri_min_angle_deg:
                continue

            # 角度に応じて reprojection 閾値をスケーリング
            tri_min = config.tri_min_angle_deg
            tri_strong = max(config.strong_track_min_angle_deg, tri_min + 1e-3)
            if min_angle_deg <= tri_min:
                angle_scale = 0.5
            elif min_angle_deg >= tri_strong:
                angle_scale = 1.0
            else:
                t = (min_angle_deg - tri_min) / (tri_strong - tri_min)
                angle_scale = 0.5 + 0.5 * t

            mean_thresh_norm = base_mean_thresh_norm * angle_scale
            max_thresh_norm = base_max_thresh_norm * angle_scale

            if mean_err > mean_thresh_norm or max_err > max_thresh_norm:
                continue

            # 全視点で前方性チェック
            front_ok = True
            for img_id, _ in current_track:
                info = images[img_id]
                X_cam = info.R_cw @ Xw + info.t_cw.reshape(3, 1)
                if X_cam[2, 0] <= 1e-6:
                    front_ok = False
                    break
            if not front_ok:
                continue

            # ---- 出力集合への登録 ----
            X_final = Xw.ravel()

            # 代表観測: 最も reprojection error が小さい観測
            best_idx = int(np.argmin(errs_final))
            rep_obs = current_track[best_idx]  # (image_id, kp_idx)

            # 3DGS用 (>=2視点)
            if config.use_two_view_tracks_for_gs and len(current_track) >= 2:
                points_gs.append(X_final)
                reps_gs.append(rep_obs)
                tracks_gs.append(list(current_track))

            # 評価用 (>=3視点)
            if (not config.use_two_view_tracks_for_eval) and len(current_track) >= 3:
                points_eval.append(X_final)

    # まず強トラックを処理して骨格点群を得る
    process_track_list(strong_tracks)
    # その後、弱トラックを処理して不足分を補う
    process_track_list(weak_tracks)

    if len(points_gs) == 0:
        points_gs_arr = np.zeros((0, 3), dtype=np.float64)
    else:
        points_gs_arr = np.vstack(points_gs)

    if len(points_eval) == 0:
        points_eval_arr = np.zeros((0, 3), dtype=np.float64)
    else:
        points_eval_arr = np.vstack(points_eval)

    return points_gs_arr, points_eval_arr, reps_gs, tracks_gs



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

def rotmat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.reshape(3)


def rodrigues_to_rotmat(rvec: np.ndarray) -> np.ndarray:
    vec = rvec.astype(np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(vec)
    return R


def build_joint_ba_sparsity(
    cam_indices: np.ndarray,
    point_indices: np.ndarray,
    num_cams_opt: int,
    num_points: int,
) -> lil_matrix:
    """
    Each residual depends on one camera (6 params) and one 3D point (3 params).
    """
    num_obs = cam_indices.shape[0]
    rows = num_obs * 2
    cols = num_cams_opt * 6 + num_points * 3
    sparsity = lil_matrix((rows, cols), dtype=int)

    for obs_idx in range(num_obs):
        pt_idx = int(point_indices[obs_idx])
        row0 = 2 * obs_idx
        row1 = row0 + 1
        col_p = num_cams_opt * 6 + pt_idx * 3
        sparsity[row0, col_p : col_p + 3] = 1
        sparsity[row1, col_p : col_p + 3] = 1

        cam_idx = int(cam_indices[obs_idx])
        if cam_idx >= 0:
            col_c = cam_idx * 6
            sparsity[row0, col_c : col_c + 6] = 1
            sparsity[row1, col_c : col_c + 6] = 1

    return sparsity


def bundle_adjustment_points_and_poses(
    points: np.ndarray,
    tracks: List[List[Tuple[int, int]]],
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
) -> np.ndarray:
    """
    Optimize camera poses (extrinsics) and 3D points jointly.
    All poses start from the known COLMAP poses to keep the global gauge.
    The first participating camera is kept fixed to avoid gauge freedom.
    """
    num_points = points.shape[0]
    if num_points == 0 or len(tracks) == 0:
        return points

    participating_ids = sorted({img_id for track in tracks for (img_id, _) in track})
    if not participating_ids:
        return points

    ref_cam_id = participating_ids[0]
    opt_cam_ids = [cid for cid in participating_ids if cid != ref_cam_id]
    cam_id_to_opt = {cid: idx for idx, cid in enumerate(opt_cam_ids)}

    obs_img_ids = []
    obs_cam_indices = []
    obs_point_indices = []
    obs_measurements = []

    for pt_idx, track in enumerate(tracks):
        for (img_id, kp_idx) in track:
            info = images.get(img_id)
            if info is None or info.keypoints_norm is None:
                continue
            if kp_idx >= info.keypoints_norm.shape[0]:
                continue
            obs_img_ids.append(img_id)
            obs_cam_indices.append(cam_id_to_opt.get(img_id, -1))
            obs_point_indices.append(pt_idx)
            obs_measurements.append(info.keypoints_norm[kp_idx])

    num_obs = len(obs_img_ids)
    if num_obs < 4:
        print("Not enough observations for joint BA. Skipping pose refinement.")
        return points

    obs_img_ids = np.array(obs_img_ids, dtype=np.int32)
    obs_cam_indices = np.array(obs_cam_indices, dtype=np.int32)
    obs_point_indices = np.array(obs_point_indices, dtype=np.int32)
    obs_measurements = np.array(obs_measurements, dtype=np.float64)

    num_cams_opt = len(opt_cam_ids)

    camera_params0 = np.zeros((num_cams_opt, 6), dtype=np.float64)
    for idx, cam_id in enumerate(opt_cam_ids):
        info = images[cam_id]
        camera_params0[idx, :3] = rotmat_to_rodrigues(info.R_cw)
        camera_params0[idx, 3:] = info.t_cw.astype(np.float64)

    points0 = points.astype(np.float64).copy()

    x0 = np.hstack([camera_params0.ravel(), points0.ravel()])

    sparsity = build_joint_ba_sparsity(
        obs_cam_indices,
        obs_point_indices,
        num_cams_opt,
        num_points,
    )

    ref_R = images[ref_cam_id].R_cw.copy()
    ref_t = images[ref_cam_id].t_cw.astype(np.float64).copy()

    f_mean = 0.0
    first_info = next(iter(images.values()))
    fx = float(first_info.K[0, 0])
    fy = float(first_info.K[1, 1])
    f_mean = 0.5 * (fx + fy)
    f_scale_norm = config.ba_loss_scale_px / max(f_mean, 1e-6)

    def residuals(param_vec: np.ndarray) -> np.ndarray:
        camera_params = (
            param_vec[: num_cams_opt * 6].reshape(num_cams_opt, 6)
            if num_cams_opt > 0
            else np.zeros((0, 6), dtype=np.float64)
        )
        points_3d = param_vec[num_cams_opt * 6 :].reshape(num_points, 3)
        res = np.zeros(num_obs * 2, dtype=np.float64)

        for obs_idx in range(num_obs):
            pt_idx = obs_point_indices[obs_idx]
            cam_idx = obs_cam_indices[obs_idx]
            img_id = obs_img_ids[obs_idx]
            obs = obs_measurements[obs_idx]

            if cam_idx < 0:
                R = ref_R
                t = ref_t
            else:
                rvec = camera_params[cam_idx, :3]
                tvec = camera_params[cam_idx, 3:]
                R = rodrigues_to_rotmat(rvec)
                t = tvec

            X = points_3d[pt_idx].reshape(3, 1)
            X_cam = R @ X + t.reshape(3, 1)
            z = float(X_cam[2, 0])
            if z <= 1e-8:
                res[2 * obs_idx : 2 * obs_idx + 2] = 1e3
                continue
            xn = X_cam[0, 0] / z
            yn = X_cam[1, 0] / z
            res[2 * obs_idx] = xn - obs[0]
            res[2 * obs_idx + 1] = yn - obs[1]

        return res

    print(
        f"Running joint bundle adjustment on {num_points} points "
        f"and {len(participating_ids)} cameras (ref={ref_cam_id})."
    )
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        loss="huber",
        f_scale=f_scale_norm,
        max_nfev=config.ba_max_nfev,
        verbose=1,
    )
    print(f"Joint BA finished. success={result.success}, cost={result.cost:.3e}")

    params_opt = result.x
    if num_cams_opt > 0:
        camera_params_opt = params_opt[: num_cams_opt * 6].reshape(num_cams_opt, 6)
    else:
        camera_params_opt = np.zeros((0, 6), dtype=np.float64)
    points_opt = params_opt[num_cams_opt * 6 :].reshape(num_points, 3)

    # Update poses (non-reference cameras)
    for idx, cam_id in enumerate(opt_cam_ids):
        info = images[cam_id]
        rvec = camera_params_opt[idx, :3]
        tvec = camera_params_opt[idx, 3:]
        R_new = rodrigues_to_rotmat(rvec)
        t_new = tvec.astype(np.float64)
        info.R_cw = R_new
        info.t_cw = t_new
        info.C = -R_new.T @ t_new.reshape(3, 1)
        info.C = info.C.ravel()

    # Reference camera stays unchanged but ensure C is up to date
    ref_info = images[ref_cam_id]
    ref_info.C = -ref_info.R_cw.T @ ref_info.t_cw.reshape(3, 1)
    ref_info.C = ref_info.C.ravel()

    points[:, :] = points_opt
    return points


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

    # 1) Load intrinsics
    K, dist = load_intrinsics_from_yaml(args.intrinsics_yaml)
    print("Loaded intrinsics:")
    print(K)
    print("dist:", dist)

    # 2) Load poses (COLMAP images.txt)
    images = load_colmap_images_txt(args.images_txt, K, dist)
    print(f"Loaded {len(images)} camera poses from images.txt")

    # 3) Feature extraction
    print("Extracting SIFT + RootSIFT features...")
    extract_features(args.image_dir, images, config)
    print("Feature extraction done.")

    # 4) Pair selection
    print("Selecting image pairs using known poses...")
    pairs = build_image_pairs(images, config)
    print(f"Selected {len(pairs)} pairs.")

    # 5) Matching + geometric verification
    print("Matching features for all pairs...")
    all_matches = match_all_pairs(images, pairs, config)
    print(f"Total verified matches: {len(all_matches)}")

    if len(all_matches) == 0:
        print("No verified matches. Aborting.")
        return

    # 6) Track construction
    print("Building multi-view tracks...")
    tracks = build_tracks(images, all_matches)
    print(f"Constructed {len(tracks)} tracks (>=2 views).")

    if len(tracks) == 0:
        print("No valid tracks. Aborting.")
        return

    # 7) Triangulation + point-only BA + filtering
    print("Triangulating and refining points with fixed known poses...")
    points_gs, points_eval, reps_gs, tracks_gs = filter_and_collect_points(tracks, images, config)
    print(f"Raw points for 3DGS (before 3D outlier removal): {points_gs.shape[0]}")
    print(f"Points for evaluation (>=3 views): {points_eval.shape[0]}")

    # 7.5) 3Dレベルでの Statistical Outlier Removal（従来実装そのまま）
    if points_gs.shape[0] > 0:
        print("Applying 3D Statistical Outlier Removal to 3DGS points...")
        points_gs_filtered, mask = statistical_outlier_removal(
            points_gs,
            k_neighbors=20,
            std_ratio=2.0,
        )
        print(
            f"3DGS points after SOR: {points_gs_filtered.shape[0]} "
            f"(removed {points_gs.shape[0] - points_gs_filtered.shape[0]})"
        )

        reps_gs_filtered = [r for r, m in zip(reps_gs, mask) if m]
        tracks_gs_filtered = [tr for tr, m in zip(tracks_gs, mask) if m]  # ★ track も同様にフィルタ
    else:
        points_gs_filtered = points_gs
        reps_gs_filtered = reps_gs
        tracks_gs_filtered = tracks_gs

    # 7.75) Joint bundle adjustment (poses + points)
    if points_gs_filtered.shape[0] > 0 and len(tracks_gs_filtered) > 0:
        points_gs_filtered = bundle_adjustment_points_and_poses(
            points_gs_filtered,
            tracks_gs_filtered,
            images,
            config,
        )

    # 8) Output PLY (for 3DGS) with colors
    colors_gs = None
    if points_gs_filtered.shape[0] > 0:
        print("Sampling colors for 3DGS points from representative observations...")
        colors_gs = compute_point_colors_from_observations(
            reps_gs_filtered,
            images,
            args.image_dir,
        )

    if args.output_ply is not None:
        write_points_to_ply(points_gs_filtered, args.output_ply, colors=colors_gs)

    # 8.5) 再投影誤差 (px) の算出と統計
    if points_gs_filtered.shape[0] > 0:
        print("Computing reprojection errors (px) for reconstructed points...")
        per_point_rms_px, all_obs_errs_px = compute_reproj_errors_px_for_all(
            points_gs_filtered,
            tracks_gs_filtered,
            images,
        )
        report_reprojection_error_stats(all_obs_errs_px)
    else:
        per_point_rms_px = np.zeros((0,), dtype=np.float64)
        print("No points for reprojection error computation.")


    # 9) Write COLMAP-style points3D.txt
    if args.output_points3d is not None and points_gs_filtered.shape[0] > 0:
        write_points3D_txt(
            points_gs_filtered,
            tracks_gs_filtered,
            colors_gs,
            per_point_rms_px,
            args.output_points3d,
        )

    # 10) Visualization
    if not args.no_show:
        visualize_reconstruction(
            images,
            points_gs_filtered if points_gs_filtered.size > 0 else points_eval,
            title="Known-pose sparse reconstruction (3DGS initialization, filtered)",
            save_path="reconstruction.png",
        )



if __name__ == "__main__":
    main()
