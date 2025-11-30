#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
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
    theta_min_pair_deg: float = 3.0
    theta_max_pair_deg: float = 45.0
    baseline_over_depth_min: float = 0.02
    baseline_over_depth_max: float = 0.6
    max_pairs_per_image: int = 10
    seq_neighbor_window: int = 3

    # Matching
    sift_ratio: float = 0.75
    cross_check: bool = True
    geom_max_error_px: float = 1.0   # ★ 4→3 に引き締め（エピポーラ誤差）
    geom_confidence: float = 0.999
    geom_min_inlier_ratio: float = 0.25
    geom_min_num_inliers: int = 15
    guided_matching: bool = False

    # Triangulation & BA
    tri_min_angle_deg: float = 3.0   # ★ 1.5→2.0deg（COLMAP推奨より少しだけ強め）
    use_two_view_tracks_for_eval: bool = False
    use_two_view_tracks_for_gs: bool = True

    # BA loss / filtering (px → 後で正規化)
    ba_loss_scale_px: float = 1.0    # ★ ロバスト閾値もやや厳しく
    point_mean_reproj_max_px: float = 1.0  # ★ 平均再投影誤差を 1px 以内
    point_max_reproj_max_px: float = 1.0   # ★ 最大誤差 2px 以内
    obs_drop_thresh_px: float = 1.0        # ★ 観測落としは 3px

    # BA iterations
    ba_max_nfev: int = 200



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
            if len(elems) < 10:
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
      - keep top max_num_features by response
      - compute undistorted normalized coordinates using K, dist
    """
    # COLMAP の SiftExtraction に近づける:
    # - nfeatures = 0 で内部制限なし（後段で max_num_features にクリップ）
    # - contrastThreshold ≈ peak_threshold = 0.0067 に合わせて低めに設定
    # - edgeThreshold = 10, sigma = 1.6 は論文・COLMAPと同等
    sift = cv2.SIFT_create(
        nfeatures=0,              # 後段で手動で max_num_features に制限
        contrastThreshold=0.0067, # COLMAP の peak_threshold に合わせる 
        edgeThreshold=10,
        sigma=1.6,
    )

    for image_id, info in images.items():
        img_path = os.path.join(image_dir, info.name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        img_resized, scale = resize_for_max_size(img, config.max_image_size)

        # Detect and compute
        keypoints, descriptors = sift.detectAndCompute(img_resized, None)

        if descriptors is None or len(keypoints) == 0:
            # 空でも後で処理できるようにしておく
            info.keypoints_px = np.zeros((0, 2), dtype=np.float32)
            info.keypoints_norm = np.zeros((0, 2), dtype=np.float32)
            info.descriptors = np.zeros((0, 128), dtype=np.float32)
            continue

        # RootSIFT 正規化
        descriptors = rootsift(descriptors)

        # 元のスケールに戻す (keypoint 座標)
        pts_px = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        pts_px /= scale

        # response 上位 max_num_features のみ残す
        if len(keypoints) > config.max_num_features:
            responses = np.array([kp.response for kp in keypoints])
            idx = np.argsort(-responses)[: config.max_num_features]
            pts_px = pts_px[idx]
            descriptors = descriptors[idx]

        # undistort + normalized coordinates
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
) -> List[Tuple[int, int]]:
    """
    Returns list of (kp_idx_i, kp_idx_j) after:
      - Lowe ratio
      - cross-check
      - known-pose-based geometric verification using Essential matrix E_known

    ポイント:
      - Two-view の幾何モデルは既に既知ポーズから一意に決まるので、
        RANSAC で E を推定し直さず、E_known に対する symmetric epipolar distance で
        インライア判定を行う。
      - 阈値は「px 単位のエピポーラ誤差」を f_mean で割って正規化した値に基づく。
    """
    desc1 = img_i.descriptors
    desc2 = img_j.descriptors
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    # 1) Lowe ratio + cross-check による候補マッチ
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

    # 2) Known pose から Essential 行列を計算し、symmetric epipolar distance でフィルタ
    pts1 = np.float64([img_i.keypoints_norm[i] for (i, _) in mutual_pairs])
    pts2 = np.float64([img_j.keypoints_norm[j] for (_, j) in mutual_pairs])

    E_known = compute_known_essential_matrix(img_i, img_j)
    d_sym_sq = symmetric_epipolar_distance_sq(E_known, pts1, pts2)

    # px ベースの許容エピポーラ誤差 → 正規化座標系へ換算
    #   th_norm ≈ (geom_max_error_px / f_mean)
    th_norm = config.geom_max_error_px / max(f_mean, 1e-6)
    th_norm_sq = th_norm * th_norm

    inlier_mask = d_sym_sq <= th_norm_sq
    inlier_pairs = [pair for pair, ok in zip(mutual_pairs, inlier_mask) if ok]

    if len(inlier_pairs) < config.geom_min_num_inliers:
        return []

    # Two-view RANSAC は行わず、known E による幾何検証のみでインライア集合を確定
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

    all_matches: List[Tuple[int, int, int, int]] = []

    for (i, j) in pairs:
        img_i = images[i]
        img_j = images[j]
        inlier_pairs = match_features_for_pair(img_i, img_j, config, f_mean)
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


def refine_point_ba(
    X0: np.ndarray,
    track: List[Tuple[int, int]],
    images: Dict[int, ImageInfo],
    config: BaselineConfig,
    f_mean: float,
) -> np.ndarray:
    """
    Point-only BA with Huber loss on normalized residuals.
    """

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
                # Behind camera -> large residual to push away
                res_list.extend([1e3, 1e3])
                continue
            xn = X_cam[0, 0] / z
            yn = X_cam[1, 0] / z
            obs = info.keypoints_norm[kp_idx]
            res_list.append(xn - obs[0])
            res_list.append(yn - obs[1])
        return np.array(res_list, dtype=np.float64)

    # px -> normalized のスケール
    f_scale_norm = config.ba_loss_scale_px / max(f_mean, 1e-6)

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
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Triangulate + iterative point-only BA + filtering.

    Returns:
      points_gs:  3D points for 3DGS (track_len >= 2, after filtering)
      points_eval: 3D points for evaluation (track_len >= 3, after filtering)
      reps_gs: list of (image_id, kp_idx) representing each 3DGS point
    """
    # 平均焦点距離
    some_info = next(iter(images.values()))
    fx = some_info.K[0, 0]
    fy = some_info.K[1, 1]
    f_mean = 0.5 * (fx + fy)

    # px ベースの閾値を正規化座標系に換算
    mean_thresh_norm = config.point_mean_reproj_max_px / max(f_mean, 1e-6)
    max_thresh_norm = config.point_max_reproj_max_px / max(f_mean, 1e-6)
    obs_drop_thresh_norm = config.obs_drop_thresh_px / max(f_mean, 1e-6)

    points_gs = []
    points_eval = []
    reps_gs: List[Tuple[int, int]] = []

    for track in tracks:
        if len(track) < 2:
            continue

        # ---- 反復: (線形三角測量 → 点のみBA → 観測除外 → 再三角測量) を最大2回 ----
        current_track = list(track)
        X_ref = None
        success = False
        iteration_num = 2

        for _iter in range(iteration_num):
            # (a) 線形三角測量
            X0 = triangulate_track_linear(current_track, images)
            if X0 is None:
                success = False
                break

            # (b) 点のみBA
            X_ref = refine_point_ba(X0, current_track, images, config, f_mean)

            # (c) 再投影誤差を計算し、高誤差観測を除外
            res = compute_reproj_errors_norm(X_ref, current_track, images)  # (M,2)
            errs = np.linalg.norm(res, axis=1)  # (M,)

            good_idx = np.where(errs <= obs_drop_thresh_norm)[0]
            if len(good_idx) < 2:
                success = False
                break

            # 観測更新
            new_track = [current_track[k] for k in good_idx]

            # 観測が一切落ちていなければ収束とみなす
            if len(new_track) == len(current_track):
                success = True
                current_track = new_track
                break
            else:
                current_track = new_track
                # もう1回ループして「更新された観測集合」で再三角測量・BA を行う

        if not success or X_ref is None:
            # 2回の反復で安定しなければ破棄
            continue

        # ---- 最終観測集合での誤差評価 ----
        res_final = compute_reproj_errors_norm(X_ref, current_track, images)
        errs_final = np.linalg.norm(res_final, axis=1)

        mean_err = float(errs_final.mean())
        max_err = float(errs_final.max())
        if mean_err > mean_thresh_norm or max_err > max_thresh_norm:
            continue

        # 全視点で前方性チェック
        Xw = X_ref.reshape(3, 1)
        front_ok = True
        for img_id, _ in current_track:
            info = images[img_id]
            X_cam = info.R_cw @ Xw + info.t_cw.reshape(3, 1)
            if X_cam[2, 0] <= 1e-6:
                front_ok = False
                break
        if not front_ok:
            continue

        # 最小視差角チェック
        min_angle_deg = 180.0
        for i in range(len(current_track)):
            img_id_i, _ = current_track[i]
            Ci = images[img_id_i].C
            vi = (Xw.ravel() - Ci)
            norm_vi = np.linalg.norm(vi)
            if norm_vi < 1e-8:
                continue
            vi /= norm_vi
            for j in range(i + 1, len(current_track)):
                img_id_j, _ = current_track[j]
                Cj = images[img_id_j].C
                vj = (Xw.ravel() - Cj)
                norm_vj = np.linalg.norm(vj)
                if norm_vj < 1e-8:
                    continue
                vj /= norm_vj
                cos_ang = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
                ang_deg = np.degrees(np.arccos(cos_ang))
                if ang_deg < min_angle_deg:
                    min_angle_deg = ang_deg

        if min_angle_deg < config.tri_min_angle_deg:
            continue

        # ---- 出力集合への登録 ----
        X_final = Xw.ravel()

        # この点に対する「最も誤差の小さい観測」を代表観測として記録
        res_final = compute_reproj_errors_norm(X_ref, current_track, images)
        errs_final = np.linalg.norm(res_final, axis=1)
        best_idx = int(np.argmin(errs_final))
        rep_obs = current_track[best_idx]  # (image_id, kp_idx)

        # 3DGS用 (>=2視点)
        if config.use_two_view_tracks_for_gs and len(current_track) >= 2:
            points_gs.append(X_final)
            reps_gs.append(rep_obs)

        # 評価用 (>=3視点)
        if (not config.use_two_view_tracks_for_eval) and len(current_track) >= 3:
            points_eval.append(X_final)

    if len(points_gs) == 0:
        points_gs_arr = np.zeros((0, 3), dtype=np.float64)
    else:
        points_gs_arr = np.vstack(points_gs)

    if len(points_eval) == 0:
        points_eval_arr = np.zeros((0, 3), dtype=np.float64)
    else:
        points_eval_arr = np.vstack(points_eval)

    return points_gs_arr, points_eval_arr, reps_gs


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
    points_gs, points_eval, reps_gs = filter_and_collect_points(tracks, images, config)
    print(f"Raw points for 3DGS (before 3D outlier removal): {points_gs.shape[0]}")
    print(f"Points for evaluation (>=3 views): {points_eval.shape[0]}")

    # 7.5) 3Dレベルでの Statistical Outlier Removal
    if points_gs.shape[0] > 0:
        print("Applying 3D Statistical Outlier Removal to 3DGS points...")
        points_gs_filtered, mask = statistical_outlier_removal(
            points_gs,
            k_neighbors=20,
            std_ratio=2.0,
        )
        print(f"3DGS points after SOR: {points_gs_filtered.shape[0]} (removed {points_gs.shape[0] - points_gs_filtered.shape[0]})")

        # reps_gs も同じマスクでフィルタ
        reps_gs_filtered = [r for r, m in zip(reps_gs, mask) if m]
    else:
        points_gs_filtered = points_gs
        reps_gs_filtered = reps_gs

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

    # 9) Visualization
    if not args.no_show:
        visualize_reconstruction(
            images,
            points_gs_filtered if points_gs_filtered.size > 0 else points_eval,
            title="Known-pose sparse reconstruction (3DGS initialization, filtered)",
            save_path="reconstruction.png",
        )


if __name__ == "__main__":
    main()
