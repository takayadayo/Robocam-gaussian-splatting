import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


'''
python pose_align.py --ref_images_txt /path/to/colmap_ref/images.txt --est_images_txt /path/to/colmap_est/images.txt --with_scale --ransac --ransac_thresh 0.03 --ransac_iters 2000 --min_inliers 20
'''

# -------------------------
# Quaternion / SO(3) utils
# -------------------------
def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """
    q: (4,) as [qw, qx, qy, qz] (Hamilton convention as COLMAP) :contentReference[oaicite:2]{index=2}
    """
    qw, qx, qy, qz = q.astype(np.float64)
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n <= 0:
        raise ValueError("Invalid quaternion norm.")
    q = q / math.sqrt(n)
    qw, qx, qy, qz = q

    # Standard Hamilton quaternion to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of R in degrees."""
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


# -------------------------
# COLMAP images.txt parser
# -------------------------
@dataclass
class PoseWC:
    # camera->world rotation
    R_wc: np.ndarray  # (3,3)
    # camera center in world
    C_w: np.ndarray   # (3,)


def load_colmap_images_txt(images_txt: str) -> Dict[str, PoseWC]:
    """
    Parse COLMAP images.txt.
    Each image has 2 lines; first line:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    Pose is world->camera. :contentReference[oaicite:3]{index=3}
    """
    poses: Dict[str, PoseWC] = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Invalid images.txt line: {line}")

        # IMAGE_ID = parts[0] (unused)
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        # CAMERA_ID = parts[8] (unused)
        name = parts[9]

        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        R_cw = quat_wxyz_to_R(q)                 # world->cam
        t_cw = np.array([tx, ty, tz], dtype=np.float64)

        # camera center in world: C = -R^T t
        C_w = -R_cw.T @ t_cw

        # camera->world rotation
        R_wc = R_cw.T

        poses[name] = PoseWC(R_wc=R_wc, C_w=C_w)

        # skip the second line (2D points)
        if i < len(lines):
            i += 1

    return poses


# -------------------------
# Generic pose CSV loader
# -------------------------
def load_pose_csv(path: str) -> Dict[str, PoseWC]:
    """
    Minimal CSV format (one row per image):
      name, qw, qx, qy, qz, tx, ty, tz, convention
    where:
      - convention = "w2c" (world->camera) or "c2w" (camera->world)
      - quaternion is [qw,qx,qy,qz] Hamilton
      - translation is t of that convention
    """
    poses: Dict[str, PoseWC] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                raise ValueError(f"Invalid CSV at line {ln}: {line}")

            name = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            conv = parts[8].lower()

            q = np.array([qw, qx, qy, qz], dtype=np.float64)
            R = quat_wxyz_to_R(q)
            t = np.array([tx, ty, tz], dtype=np.float64)

            if conv == "w2c":
                R_cw = R
                t_cw = t
                C_w = -R_cw.T @ t_cw
                R_wc = R_cw.T
            elif conv == "c2w":
                R_wc = R
                C_w = t  # if provided as camera center in world
            else:
                raise ValueError(f"Unknown convention '{conv}' at line {ln}")

            poses[name] = PoseWC(R_wc=R_wc, C_w=C_w)

    return poses


# -------------------------
# Umeyama Sim(3) alignment
# -------------------------
@dataclass
class Sim3:
    s: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def apply_points(self, P: np.ndarray) -> np.ndarray:
        """P: (N,3) in source frame -> (N,3) in target frame"""
        return (self.s * (P @ self.R.T)) + self.t  # (N,3)

    def apply_rotations_wc(self, R_wc_list: np.ndarray) -> np.ndarray:
        """
        Apply the world-frame rotation of the alignment to camera->world rotations:
        X_ref = s * R_align * X_est + t
        => world basis is rotated by R_align, so camera->world becomes:
        R_wc_ref = R_align * R_wc_est
        """
        # R_wc_list: (N,3,3)
        return np.einsum("ij,njk->nik", self.R, R_wc_list)



def umeyama_alignment(
    X_tgt: np.ndarray,  # (N,3) target
    Y_src: np.ndarray,  # (N,3) source to be aligned
    with_scale: bool = True
) -> Sim3:
    """
    Find s,R,t minimizing || X - (s R Y + t) ||^2 (Umeyama 1991). :contentReference[oaicite:4]{index=4}
    """
    assert X_tgt.shape == Y_src.shape
    N = X_tgt.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points for Sim(3) alignment.")

    mu_x = X_tgt.mean(axis=0)
    mu_y = Y_src.mean(axis=0)

    Xc = X_tgt - mu_x
    Yc = Y_src - mu_y

    # covariance
    Sigma = (Xc.T @ Yc) / N  # (3,3)

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_y = (Yc * Yc).sum() / N
        if var_y <= 0:
            raise ValueError("Degenerate point configuration (zero variance).")
        s = float(np.trace(np.diag(D) @ S) / var_y)
    else:
        s = 1.0

    t = mu_x - s * (R @ mu_y)
    return Sim3(s=s, R=R, t=t)


def sim3_ransac(
    X_tgt: np.ndarray,
    Y_src: np.ndarray,
    with_scale: bool,
    thresh: float,
    max_iters: int,
    min_inliers: int,
    seed: int = 0
) -> Tuple[Sim3, np.ndarray]:
    """
    Robust Sim(3) via RANSAC (minimal set=3). Returns (best_model, inlier_mask).
    """
    rng = random.Random(seed)
    N = X_tgt.shape[0]
    if N < 3:
        raise ValueError("Need >=3 correspondences for RANSAC.")

    best_inliers = None
    best_model = None
    best_cnt = -1

    idx_all = list(range(N))

    for _ in range(max_iters):
        samp = rng.sample(idx_all, 3)
        Xs = X_tgt[samp]
        Ys = Y_src[samp]

        # Reject nearly-collinear samples (stability)
        if np.linalg.matrix_rank((Ys - Ys.mean(axis=0))) < 2:
            continue

        try:
            model = umeyama_alignment(Xs, Ys, with_scale=with_scale)
        except Exception:
            continue

        Y_al = model.apply_points(Y_src)
        err = np.linalg.norm(X_tgt - Y_al, axis=1)
        inliers = err < thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
            best_model = model

    if best_model is None or best_inliers is None or best_cnt < min_inliers:
        # fallback to least-squares on all points
        model = umeyama_alignment(X_tgt, Y_src, with_scale=with_scale)
        inliers = np.ones((N,), dtype=bool)
        return model, inliers

    # Refit on inliers
    model = umeyama_alignment(X_tgt[best_inliers], Y_src[best_inliers], with_scale=with_scale)
    return model, best_inliers


# -------------------------
# Metrics
# -------------------------
def compute_errors(
    ref: Dict[str, PoseWC],
    est: Dict[str, PoseWC],
    keys: List[str],
    model: Sim3
) -> Dict[str, float]:
    # Camera centers
    C_ref = np.stack([ref[k].C_w for k in keys], axis=0)  # (N,3)
    C_est = np.stack([est[k].C_w for k in keys], axis=0)  # (N,3)
    C_est_al = model.apply_points(C_est)                  # (N,3)

    t_err = np.linalg.norm(C_ref - C_est_al, axis=1)

    # Rotations (camera->world)
    R_ref = np.stack([ref[k].R_wc for k in keys], axis=0)  # (N,3,3)
    R_est = np.stack([est[k].R_wc for k in keys], axis=0)  # (N,3,3)
    R_est_al = model.apply_rotations_wc(R_est)             # (N,3,3)

    r_err = np.empty((len(keys),), dtype=np.float64)
    for i in range(len(keys)):
        R_delta = R_ref[i].T @ R_est_al[i]
        r_err[i] = rotation_angle_deg(R_delta)

    return {
        "N": float(len(keys)),
        "scale": float(model.s),
        "trans_rmse": float(np.sqrt(np.mean(t_err**2))),
        "trans_mean": float(np.mean(t_err)),
        "trans_median": float(np.median(t_err)),
        "trans_max": float(np.max(t_err)),
        "rot_mean_deg": float(np.mean(r_err)),
        "rot_median_deg": float(np.median(r_err)),
        "rot_max_deg": float(np.max(r_err)),
    }


def compute_per_frame_errors(
    ref: Dict[str, PoseWC],
    est: Dict[str, PoseWC],
    keys: List[str],
    model: Sim3
) -> List[Tuple[str, float, float]]:
    """
    Per-frame errors after applying the estimated alignment model.

    Returns:
      list of (name, translation_error, rotation_error_deg)
      - translation_error: Euclidean error between camera centers (aligned est vs ref)
      - rotation_error_deg: geodesic angle of (R_ref^T * R_est_aligned)
    """
    C_ref = np.stack([ref[k].C_w for k in keys], axis=0)  # (N,3)
    C_est = np.stack([est[k].C_w for k in keys], axis=0)  # (N,3)
    C_est_al = model.apply_points(C_est)                  # (N,3)
    t_err = np.linalg.norm(C_ref - C_est_al, axis=1)

    R_ref = np.stack([ref[k].R_wc for k in keys], axis=0)  # (N,3,3)
    R_est = np.stack([est[k].R_wc for k in keys], axis=0)  # (N,3,3)
    R_est_al = model.apply_rotations_wc(R_est)             # (N,3,3)

    r_err = np.empty((len(keys),), dtype=np.float64)
    for i in range(len(keys)):
        R_delta = R_ref[i].T @ R_est_al[i]
        r_err[i] = rotation_angle_deg(R_delta)

    return [(k, float(t_err[i]), float(r_err[i])) for i, k in enumerate(keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_images_txt", required=True, help="COLMAP images.txt as reference (optimized poses)")
    ap.add_argument("--est_images_txt", required=False, help="COLMAP images.txt as estimate")
    ap.add_argument("--est_csv", required=False, help="Alternative estimate pose CSV (see loader doc)")
    ap.add_argument("--with_scale", action="store_true", help="Use Sim(3) (scale+SE3). If off -> SE(3).")
    ap.add_argument("--ransac", action="store_true", help="Enable Sim3 RANSAC for outlier rejection")
    ap.add_argument("--ransac_thresh", type=float, default=0.03, help="Inlier threshold in meters (or your unit)")
    ap.add_argument("--ransac_iters", type=int, default=2000)
    ap.add_argument("--min_inliers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ref = load_colmap_images_txt(args.ref_images_txt)

    if args.est_images_txt:
        est = load_colmap_images_txt(args.est_images_txt)
    elif args.est_csv:
        est = load_pose_csv(args.est_csv)
    else:
        raise ValueError("Provide --est_images_txt or --est_csv")

    keys = sorted(set(ref.keys()) & set(est.keys()))
    if len(keys) < 3:
        raise ValueError(f"Not enough correspondences. common={len(keys)}")

    C_ref = np.stack([ref[k].C_w for k in keys], axis=0)
    C_est = np.stack([est[k].C_w for k in keys], axis=0)

    if args.ransac:
        model, inliers = sim3_ransac(
            X_tgt=C_ref,
            Y_src=C_est,
            with_scale=args.with_scale,
            thresh=args.ransac_thresh,
            max_iters=args.ransac_iters,
            min_inliers=args.min_inliers,
            seed=args.seed
        )
        keys_in = [k for k, m in zip(keys, inliers) if m]
        print(f"[RANSAC] inliers {len(keys_in)}/{len(keys)}")
        keys_eval = keys_in if len(keys_in) >= 3 else keys
    else:
        model = umeyama_alignment(C_ref, C_est, with_scale=args.with_scale)
        keys_eval = keys


    # Per-frame errors (computed after applying the estimated alignment)
    keys_report = keys  # report all common frames (including RANSAC outliers, if any)
    per_frame = compute_per_frame_errors(ref, est, keys_report, model)

    if args.ransac:
        inlier_map = {k: bool(m) for k, m in zip(keys, inliers)}
    else:
        inlier_map = {k: True for k in keys}

    print("===== Per-frame Errors AFTER optimal alignment =====")
    print("name\tinlier\ttrans_err\trot_err_deg")
    for name, te, re_deg in per_frame:
        flag = "1" if inlier_map.get(name, False) else "0"
        print(f"{name}\t{flag}\t{te:.6f}\t{re_deg:.6f}")

    stats = compute_errors(ref, est, keys_eval, model)

    print("===== Alignment (target=ref, source=est) =====")
    print(f"common frames: {len(keys)}  eval frames: {int(stats['N'])}")
    print(f"scale: {stats['scale']:.9f}  (Sim3 if --with_scale else 1.0)")
    print("R (3x3):")
    print(model.R)
    print(f"t: {model.t}")

    print("===== Errors AFTER optimal alignment =====")
    print(f"translation RMSE:   {stats['trans_rmse']:.6f}")
    print(f"translation mean:   {stats['trans_mean']:.6f}")
    print(f"translation median: {stats['trans_median']:.6f}")
    print(f"translation max:    {stats['trans_max']:.6f}")
    print(f"rotation mean [deg]:   {stats['rot_mean_deg']:.6f}")
    print(f"rotation median [deg]: {stats['rot_median_deg']:.6f}")
    print(f"rotation max [deg]:    {stats['rot_max_deg']:.6f}")


if __name__ == "__main__":
    main()
