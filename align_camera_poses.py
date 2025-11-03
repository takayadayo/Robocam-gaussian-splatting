# align_camera_poses.py  (JSONレポートのみ出力)
import os, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_images_txt(path):
    poses = {}
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln.startswith("#"):
            i += 1; continue
        parts = ln.split()
        if len(parts) < 10:
            i += 1; continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz     = map(float, parts[5:8])
        name = parts[9]
        R_cw = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_cw = np.array([[tx],[ty],[tz]], np.float64)
        poses[name] = (R_cw, t_cw)   # world->camera
        i += 2
    return poses

def load_sfm_json(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    poses = {}
    for name, d in data.items():
        R_wc = np.array(d["R_wc"], np.float64)
        t_wc = np.array(d["t_wc"], np.float64).reshape(3,1)
        poses[name] = (R_wc, t_wc)  # world->camera
    return poses

def cam_centers(poses):
    C = {}
    for k,(R_cw,t_cw) in poses.items():
        Cw = -R_cw.T @ t_cw
        C[k] = Cw.reshape(3)
    return C

def umeyama_sim3(A, B, allow_reflection=True):
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    S = (A0.T @ B0) / A.shape[0]
    U, D, Vt = np.linalg.svd(S)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1,:] *= -1.0
        Rm = Vt.T @ U.T
    varA = (A0**2).sum()/A.shape[0]
    s = np.trace(np.diag(D)) / (varA + 1e-12)
    t = muB - s*(Rm @ muA)
    return float(s), Rm, t.reshape(3)

def robust_sim3(A, B, trim=0.7, iters=5):
    s, Rm, t = umeyama_sim3(A, B, allow_reflection=True)
    for _ in range(iters):
        X = (s*(A @ Rm.T)) + t
        res = np.linalg.norm(X - B, axis=1)
        k = max(3, int(trim*len(res)))
        idx = np.argpartition(res, k)[:k]
        # Huber
        med = np.median(res[idx]); sigma = 1.4826*np.median(np.abs(res[idx]-med))+1e-9
        tau = 1.5*sigma
        w = np.clip(tau/np.maximum(res,1e-9), 0.0, 1.0)
        W = np.diag(w[idx])
        A0 = A[idx] - (W @ A[idx]).sum(0)/(w[idx].sum()+1e-9)
        B0 = B[idx] - (W @ B[idx]).sum(0)/(w[idx].sum()+1e-9)
        S = A0.T @ (W @ B0) / (w[idx].sum()+1e-9)
        U,D,Vt = np.linalg.svd(S)
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[-1,:] *= -1.0
            Rm = Vt.T @ U.T
        varA = ((W @ A0)*A0).sum()/(w[idx].sum()+1e-9)
        s = np.trace(np.diag(D))/(varA + 1e-12)
        muA = (W @ A[idx]).sum(0)/(w[idx].sum()+1e-9)
        muB = (W @ B[idx]).sum(0)/(w[idx].sum()+1e-9)
        t = muB - s*(Rm @ muA)
    X = (s*(A @ Rm.T)) + t
    res = np.linalg.norm(X - B, axis=1)
    inlier_ratio = float((res <= np.percentile(res, 70)).mean())
    return s, Rm, t, res, inlier_ratio

def rot_angle_deg(Rm):
    tr = np.clip((np.trace(Rm)-1)/2, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def compute_ate(A_aligned, B):
    d = np.linalg.norm(A_aligned - B, axis=1)
    stats = lambda x: dict(
        rmse=float(np.sqrt(np.mean(x**2))),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        std=float(np.std(x)),
        min=float(np.min(x)),
        p90=float(np.percentile(x,90)),
        p95=float(np.percentile(x,95)),
        max=float(np.max(x)),
    )
    return d, stats(d)

def compute_rpe(Ts_est, Ts_ref, delta=1):
    # Ts: list of 4x4 world->camera after alignment (est) and reference
    rpe_t, rpe_r = [], []
    for i in range(len(Ts_est)-delta):
        Ei = np.linalg.inv(Ts_est[i]) @ Ts_est[i+delta]
        Gi = np.linalg.inv(Ts_ref[i]) @ Ts_ref[i+delta]
        D  = np.linalg.inv(Gi) @ Ei
        Re, te = D[:3,:3], D[:3,3]
        rpe_t.append(np.linalg.norm(te))
        rpe_r.append(rot_angle_deg(Re))
    rpe_t, rpe_r = np.array(rpe_t), np.array(rpe_r)
    s = lambda x: dict(
        rmse=float(np.sqrt(np.mean(x**2))),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p90=float(np.percentile(x,90))
    )
    return s(rpe_t), s(rpe_r)

def build_Tcw(R_cw, t_cw):  # 4x4
    T = np.eye(4); T[:3,:3]=R_cw; T[:3,3]=t_cw.reshape(3); return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_json",  required=True)   # sfm_computed_poses.json (world->camera)
    ap.add_argument("--dst_images", required=True)  # images.txt (world->camera)
    ap.add_argument("--image_dir",  required=True)  # ordering basis
    ap.add_argument("--out", default="pose_eval_report.json")
    ap.add_argument("--delta", type=int, default=1) # RPE delta
    args = ap.parse_args()

    src = load_sfm_json(args.src_json)
    dst = load_images_txt(args.dst_images)

    files = sorted([f for f in os.listdir(args.image_dir)
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))])
    names = [f for f in files if f in src and f in dst]
    if len(names) < 3: raise RuntimeError("Not enough common views (>=3).")

    A = np.stack([(-src[n][0].T @ src[n][1]).reshape(3) for n in names], axis=0)
    B = np.stack([(-dst[n][0].T @ dst[n][1]).reshape(3) for n in names], axis=0)

    s, Rm, t, res, inlier_ratio = robust_sim3(A, B, trim=0.7, iters=5)
    A_aligned = (s*(A @ Rm.T)) + t

    ate_per_frame, ate_stats = compute_ate(A_aligned, B)

    # world->camera を再構成（位置だけ Sim(3) を反映）
    T_est = []; T_ref = []
    for n in names:
        Rcw_est, tcw_est = src[n]
        Cw = -Rcw_est.T @ tcw_est
        Cw_al = (s*(Rm @ Cw.reshape(3))) + t
        tcw_new = -Rcw_est @ Cw_al.reshape(3,1)
        T_est.append(build_Tcw(Rcw_est, tcw_new))
        Rcw_ref, tcw_ref = dst[n]
        T_ref.append(build_Tcw(Rcw_ref, tcw_ref))

    rpe_t_stats, rpe_r_stats = compute_rpe(T_est, T_ref, delta=args.delta)

    report = {
        "meta": {
            "source": os.path.basename(args.src_json),
            "target": os.path.basename(args.dst_images),
            "num_common": len(names),
            "delta_for_rpe": args.delta,
            "units": {"translation": "meters", "angle": "degrees"}
        },
        "alignment": {
            "scale": float(s),
            "rotation_matrix": Rm.tolist(),
            "rotation_angle_deg": rot_angle_deg(Rm),
            "translation": t.tolist(),
            "detR": float(np.linalg.det(Rm)),
            "inlier_ratio": inlier_ratio,
            "method": "Umeyama init + Trimmed L2 + Huber IRLS"
        },
        "ATE": ate_stats,
        "RPE": { "translation": rpe_t_stats, "rotation_deg": rpe_r_stats },
        "per_frame": [
            {"name": n, "ate": float(e)}
            for n, e in zip(names, ate_per_frame)
        ]
    }
    json.dump(report, open(args.out,"w",encoding="utf-8"), indent=2)
    print(f"[INFO] wrote {args.out}")

if __name__ == "__main__":
    main()
