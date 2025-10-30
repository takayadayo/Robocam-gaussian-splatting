#!/usr/bin/env python3
"""
Triangulate a sparse point cloud using KNOWN intrinsics + extrinsics derived from
hand–eye calibration and robot poses (no bundle adjustment).

Inputs
  - poses.json           : robot base->gripper poses per image (mm, deg)
  - image_dir            : directory of corresponding RGB images
  - intrinsics.yaml      : K, dist from calibrate_robot_charuco.py
  - handeye.yaml         : Rg_c, tg_c (gripper->camera) from calibrate_robot_charuco.py
  - out_dir              : where to write outputs

Outputs
  - out_dir/
      cameras.txt, images.txt, points3D.txt   # COLMAP-compatible text headers (poses only)
      triangulated.ply                        # sparse point cloud
      matches/                                # optional debug of matched pairs
      stats.json                              # reprojection & filtering stats

Notes
  - We assume the WORLD frame == robot BASE frame.
  - We use ORB by default (widely available). If SIFT exists, we prefer it.
  - We undistort points (to remove lens distortion) and keep P = K[R|t] for triangulation.
  - We filter by Sampson error (from known relative pose), cheirality, angle-of-triangulation, and reprojection error.
  - This script is intentionally simple—no BA—to validate correctness of known poses.
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------- Config (feel free to tweak) -----------------------
EULER_ORDER = "xyz"   # must match your robot JSON convention
MIN_MATCHES = 20       # required matches to triangulate a pair (LightGlue is strong)
SAMPSON_THRESH = 10   # pixels; epipolar consistency threshold
REPROJ_THRESH = 20    # pixels; per-view reprojection error threshold
MIN_TRI_ANGLE_DEG = 1  # filter poorly conditioned triangulations
PAIR_GAP = 1           # match i with i+PAIR_GAP (1=consecutive)
MAX_PAIRS = None       # limit number of pairs for speed (None = all)

# ----------------------- Utilities -----------------------

def read_yaml_matrix(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise RuntimeError(f"Key '{key}' not found in YAML.")
    return node.mat()


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open intrinsics: {path}")
    K = read_yaml_matrix(fs, "K").astype(np.float64)
    dist = read_yaml_matrix(fs, "dist").astype(np.float64)
    fs.release()
    return K, dist


def load_handeye(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open handeye: {path}")
    Rg_c = read_yaml_matrix(fs, "Rg_c").astype(np.float64)
    tg_c = read_yaml_matrix(fs, "tg_c").astype(np.float64)
    fs.release()
    return Rg_c, tg_c.reshape(3, 1)


def load_robot_json(json_path: Path, image_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    caps = []
    for c in js["captures"]:
        img = image_dir / Path(c["image_file"]).name
        t = np.array([c["pose"]["x_mm"], c["pose"]["y_mm"], c["pose"]["z_mm"]], dtype=np.float64) / 1000.0
        angles = np.array([c["pose"]["rx_deg"], c["pose"]["ry_deg"], c["pose"]["rz_deg"]], dtype=np.float64)
        Rb_g = R.from_euler(EULER_ORDER, angles, degrees=True).as_matrix()
        caps.append({
            "image": str(img),
            "Rb_g": Rb_g,
            "tb_g": t.reshape(3, 1),
        })
    return caps


def compose_base_to_cam(caps, Rg_c, tg_c):
    Tbc = []
    for cap in caps:
        Rb_g = cap["Rb_g"]; tb_g = cap["tb_g"]
        Rb_c = Rb_g @ Rg_c
        tb_c = Rb_g @ tg_c + tb_g
        Tbc.append((Rb_c, tb_c))
    return Tbc


def world_to_cam(Rb_c: np.ndarray, tb_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Rcw, tcw) that maps X_w (base) -> X_c: X_c = Rcw X_w + tcw."""
    Rcw = Rb_c.T
    tcw = -Rcw @ tb_c
    return Rcw, tcw


def make_projection(K: np.ndarray, Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    return K @ np.hstack([Rcw, tcw])


def undistort_px(pts_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort pixel points and keep pixel coordinates by setting P=K."""
    pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return undist.reshape(-1, 2)

# --- Helpers for deep matchers (LightGlue) ---
def cv2_to_torch(img: np.ndarray, device: str):
    """Convert OpenCV BGR uint8 HxWx3 to torch float tensor (1,3,H,W) in [0,1]."""
    import torch
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = torch.from_numpy(img[:, :, ::-1].copy()).float() / 255.0  # BGR->RGB
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    return t


def relative_pose(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R21 = R2 @ R1.T
    t21 = t2 - R21 @ t1
    return R21, t21


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.reshape(-1)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def fundamental_from_poses(K: np.ndarray, Rcw1: np.ndarray, tcw1: np.ndarray, Rcw2: np.ndarray, tcw2: np.ndarray) -> np.ndarray:
    R21, t21 = relative_pose(Rcw1, tcw1, Rcw2, tcw2)
    E = skew(t21.reshape(3)) @ R21
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv
    return F


def sampson_errors(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1h = np.hstack([x1, np.ones((len(x1), 1))])
    x2h = np.hstack([x2, np.ones((len(x2), 1))])
    Fx1 = (F @ x1h.T).T
    Ftx2 = (F.T @ x2h.T).T
    x2Fx1 = np.sum(x2h * (F @ x1h.T).T, axis=1)
    errs = (x2Fx1 ** 2) / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)
    return errs


def triangulate_pair(K: np.ndarray, dist: np.ndarray, P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate with OpenCV, return (Xw Nx3, reproj_errs Nx2, valid_mask).
    pts1/pts2: pixel coordinates (undistorted already)
    """
    assert pts1.shape == pts2.shape and pts1.ndim == 2 and pts1.shape[1] == 2
    # OpenCV expects 2xN arrays
    p1 = pts1.T.astype(np.float64)
    p2 = pts2.T.astype(np.float64)
    Xh = cv2.triangulatePoints(P1, P2, p1, p2)  # 4xN
    X = (Xh[:3] / Xh[3]).T  # Nx3 in WORLD (base) because P contains world->cam

    # Cheirality: positive depth in both cameras
    def depth(Rcw, tcw, Xw):
        Xc = (Rcw @ Xw.T + tcw).T
        return Xc[:, 2]
    # recover (Rcw,tcw)
    Rcw1, tcw1 = P1[:, :3], P1[:, 3:4]
    Rcw1 = np.linalg.inv(np.linalg.inv(K) @ Rcw1)  # not robust; compute directly below
    # Instead compute from P by factorization:
    Rcw1 = np.linalg.inv(K) @ P1[:, :3]
    tcw1 = np.linalg.inv(K) @ P1[:, 3:4]
    Rcw2 = np.linalg.inv(K) @ P2[:, :3]
    tcw2 = np.linalg.inv(K) @ P2[:, 3:4]

    z1 = depth(Rcw1, tcw1, X)
    z2 = depth(Rcw2, tcw2, X)
    cheir = (z1 > 0) & (z2 > 0)

    # Reprojection errors
    def reproj(P, Xw, pts):
        xh = (P @ np.hstack([Xw, np.ones((len(Xw), 1))]).T).T
        x = (xh[:, :2].T / xh[:, 2]).T
        return np.linalg.norm(x - pts, axis=1)

    e1 = reproj(P1, X, pts1)
    e2 = reproj(P2, X, pts2)
    errs = np.vstack([e1, e2]).T

    valid = cheir & (errs[:, 0] < REPROJ_THRESH) & (errs[:, 1] < REPROJ_THRESH)
    return X, errs, valid


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def color_from_image(img: np.ndarray, pt: Tuple[float, float]) -> Tuple[int, int, int]:
    u, v = int(round(pt[0])), int(round(pt[1]))
    h, w = img.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        bgr = img[v, u]
        return int(bgr[2]), int(bgr[1]), int(bgr[0])  # RGB
    return (200, 200, 200)


def write_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None):
    points = points.astype(np.float32)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        if colors is None:
            f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def export_colmap_text(out_dir: Path, K: np.ndarray, image_paths: List[str], Rcw_list: List[np.ndarray], tcw_list: List[np.ndarray], width: int, height: int):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    k1 = k2 = p1 = p2 = k3 = 0.0  # distortion is assumed handled in K; COLMAP OPENCV expects 5 coeffs

    cam_txt = out_dir / "cameras.txt"
    with open(cam_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 OPENCV {width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}\n")

    img_txt = out_dir / "images.txt"
    with open(img_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (p, Rcw, tcw) in enumerate(zip(image_paths, Rcw_list, tcw_list), start=1):
            qx, qy, qz, qw = R.from_matrix(Rcw).as_quat()  # OpenCV-order [qx,qy,qz,qw]
            f.write(f"{i} {qw} {qx} {qy} {qz} {tcw[0,0]} {tcw[1,0]} {tcw[2,0]} 1 {Path(p).name}\n\n")

    pts_txt = out_dir / "points3D.txt"
    with open(pts_txt, "w") as f:
        f.write("# 3D point list is empty for known-poses bootstrap\n")


# ----------------------- Main pipeline -----------------------

def main(args):
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    K, dist = load_intrinsics(Path(args.intrinsics))
    Rg_c, tg_c = load_handeye(Path(args.handeye))
    caps = load_robot_json(Path(args.poses), image_dir)

    # sort by image name for deterministic order
    caps.sort(key=lambda c: Path(c["image"]).name)
    image_paths = [c["image"] for c in caps]

    # Camera extrinsics: base->cam and world->cam
    Tbc = compose_base_to_cam(caps, Rg_c, tg_c)
    Rcw_list, tcw_list = [], []
    for Rb_c, tb_c in Tbc:
        Rcw, tcw = world_to_cam(Rb_c, tb_c)
        Rcw_list.append(Rcw)
        tcw_list.append(tcw)

    # load one image to get size
    img0 = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Cannot read first image: {image_paths[0]}")
    H, W = img0.shape[:2]

    # Export poses in COLMAP text (optional but handy for 3DGS)
    export_colmap_text(out_dir, K, image_paths, Rcw_list, tcw_list, W, H)

    # ---- Feature extractor & matcher: LightGlue + SuperPoint ----
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using LightGlue+SuperPoint on {device}")
        # Accuracy-first preset per official README: disable adaptivity for max accuracy
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)
    except Exception as e:
        raise RuntimeError("LightGlue not available. Please `pip install lightglue torch torchvision` (and a CUDA-enabled Torch if you have a GPU).") from e

    # Triangulation across pairs
    all_X, all_rgb = [], []
    pair_count = 0

    for i in range(0, len(image_paths) - PAIR_GAP):
        if MAX_PAIRS is not None and pair_count >= MAX_PAIRS:
            break
        j = i + PAIR_GAP
        img1 = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        img2 = cv2.imread(image_paths[j], cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            continue

        # extract & match with LightGlue
        t1 = cv2_to_torch(img1, device)
        t2 = cv2_to_torch(img2, device)
        with torch.no_grad():
            feats1 = extractor.extract(t1)
            feats2 = extractor.extract(t2)
            matches_dict = matcher({'image0': feats1, 'image1': feats2})
        feats1, feats2, matches_dict = [rbd(x) for x in [feats1, feats2, matches_dict]]
        matches = matches_dict['matches'].cpu().numpy()
        if matches.size == 0:
            continue
        kpts1 = feats1['keypoints'].cpu().numpy()
        kpts2 = feats2['keypoints'].cpu().numpy()
        pts1_px = kpts1[matches[:, 0]]
        pts2_px = kpts2[matches[:, 1]]
        if len(pts1_px) < MIN_MATCHES:
            continue

        # undistort to pixel coords (P=K keeps pixel domain but removes distortion)
        pts1_ud = undistort_px(pts1_px, K, dist)
        pts2_ud = undistort_px(pts2_px, K, dist)

        # epipolar/Sampson filtering using KNOWN relative pose
        Rcw1, tcw1 = Rcw_list[i], tcw_list[i]
        Rcw2, tcw2 = Rcw_list[j], tcw_list[j]
        F = fundamental_from_poses(K, Rcw1, tcw1, Rcw2, tcw2)
        se = sampson_errors(F, pts1_ud, pts2_ud)
        keep = se < (SAMPSON_THRESH)
        if keep.sum() < MIN_MATCHES:
            continue

        pts1 = pts1_ud[keep]
        pts2 = pts2_ud[keep]

        # projection matrices
        P1 = make_projection(K, Rcw1, tcw1)
        P2 = make_projection(K, Rcw2, tcw2)

        # triangulate
        Xw, errs, valid = triangulate_pair(K, dist, P1, P2, pts1, pts2)

        # angle-of-triangulation filtering
        C1 = (-Rcw1.T @ tcw1).reshape(3)
        C2 = (-Rcw2.T @ tcw2).reshape(3)
        dirs1 = Xw - C1
        dirs2 = Xw - C2
        ang = np.array([angle_between(a, b) for a, b in zip(dirs1, dirs2)])
        angle_ok = ang >= MIN_TRI_ANGLE_DEG

        good_mask = valid & angle_ok
        X_good = Xw[good_mask]
        if len(X_good) == 0:
            continue

        # colorize from first image
        colors = np.array([color_from_image(img1, tuple(p)) for p in pts1[good_mask]], dtype=np.uint8)

        all_X.append(X_good)
        all_rgb.append(colors)
        pair_count += 1
        print(f"[INFO] pair {i}-{j}: matches={len(matches)} kept={len(X_good)}")

        # optional debug dump
        if args.dump_matches:
            md = Path(out_dir) / "matches"; md.mkdir(exist_ok=True)
            canvas = np.hstack([img1, img2])
            off = img1.shape[1]
            for a, b in zip(pts1[good_mask].astype(int), pts2[good_mask].astype(int)):
                cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0])+off, int(b[1])), (0,255,0), 1)
            cv2.imwrite(str(md / f"pair_{i:04d}_{j:04d}.jpg"), canvas)

    if not all_X:
        raise RuntimeError("No triangulated points survived filtering. Check thresholds or inputs.")

    # concatenate and (optionally) downsample
    X_cat = np.vstack(all_X)
    RGB_cat = np.vstack(all_rgb)

    # Write PLY
    ply_path = out_dir / "triangulated.ply"
    write_ply(ply_path, X_cat, RGB_cat)
    print(f"[INFO] wrote {ply_path} with {len(X_cat)} points")

    # Stats JSON
    stats = {
        "num_images": len(image_paths),
        "num_pairs": pair_count,
        "num_points": int(len(X_cat)),
        "params": {
            "MIN_MATCHES": MIN_MATCHES,
            "SAMPSON_THRESH": SAMPSON_THRESH,
            "REPROJ_THRESH": REPROJ_THRESH,
            "MIN_TRI_ANGLE_DEG": MIN_TRI_ANGLE_DEG,
            "PAIR_GAP": PAIR_GAP,
        },
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

        raise RuntimeError("No triangulated points survived filtering. Check thresholds or inputs.")

    # concatenate and (optionally) downsample
    X_cat = np.vstack(all_X)
    RGB_cat = np.vstack(all_rgb)

    # Write PLY
    ply_path = out_dir / "triangulated.ply"
    write_ply(ply_path, X_cat, RGB_cat)
    print(f"[INFO] wrote {ply_path} with {len(X_cat)} points")

    # Stats JSON
    stats = {
        "num_images": len(image_paths),
        "num_pairs": pair_count,
        "num_points": int(len(X_cat)),
        "params": {
            "MIN_MATCHES": MIN_MATCHES,
                        "SAMPSON_THRESH": SAMPSON_THRESH,
            "REPROJ_THRESH": REPROJ_THRESH,
            "MIN_TRI_ANGLE_DEG": MIN_TRI_ANGLE_DEG,
            "PAIR_GAP": PAIR_GAP,
        },
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Triangulate sparse points from known poses")
    ap.add_argument("poses", type=str, help="hand_eye_calibration/json/poses.json")
    ap.add_argument("image_dir", type=str, help="hand_eye_calibration/image")
    ap.add_argument("intrinsics", type=str, help="sfm_out/intrinsics.yaml")
    ap.add_argument("handeye", type=str, help="sfm_out/handeye.yaml")
    ap.add_argument("out_dir", type=str, help="output directory")
    ap.add_argument("--dump_matches", action="store_true", help="write match visualizations")
    args = ap.parse_args()
    main(args)
