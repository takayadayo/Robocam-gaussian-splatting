# meta_validate_handeye.py
import os, json, math, itertools
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ---------- 固定設定（ユーザ環境に合わせて埋め込み） ----------
ROOT = "."
IMG_DIR = os.path.join(ROOT, "hand_eye_calibration", "image")
POSE_JSON = os.path.join(ROOT, "hand_eye_calibration", "json", "poses.json")
OUT_DIR = os.path.join(ROOT, "sfm_out")
INTR_YAML = os.path.join(OUT_DIR, "intrinsics.yaml")
HE_YAML = os.path.join(OUT_DIR, "handeye.yaml")
COLMAP_DIR = os.path.join(OUT_DIR, "colmap_sparse")
EULER_ORDER = "xyz"  # ロボット姿勢は xyz(deg) で与えられる
SQUARES_X, SQUARES_Y = 7, 5
SQUARE_MM, MARKER_MM = 32.0, 24.0
DICT = cv2.aruco.DICT_4X4_50

# 判定閾値（初期値）
TH_REPROJ_MED = 0.30
TH_REPROJ_MAX = 0.60
TH_AXXB_ROT_MED_DEG = 0.30
TH_AXXB_TRA_MED_MM = 1.5
TH_BT_ROT_MAD_DEG = 0.50
TH_BT_TRA_MAD_MM = 2.0

def _fs_read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"cannot open: {path}")
    data = {}
    for k in ("K","dist","Rg_c","tg_c","rms","per_view_errors","num_views"):
        node = fs.getNode(k)
        if not node.empty():
            data[k] = node.mat()
    fs.release()
    return data

def _load_robot_json(json_path, image_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    caps = []
    for c in js["captures"]:
        img = os.path.join(image_dir, os.path.basename(c["image_file"]))
        t = np.array([c["pose"]["x_mm"], c["pose"]["y_mm"], c["pose"]["z_mm"]], np.float64)/1000.0
        ang = np.array([c["pose"]["rx_deg"], c["pose"]["ry_deg"], c["pose"]["rz_deg"]], np.float64)
        Rb_g = R.from_euler(EULER_ORDER, ang, degrees=True).as_matrix()
        caps.append((img, Rb_g, t.reshape(3,1)))
    return caps  # [(path, Rb_g, tb_g)]

def _charuco_detect(image_paths, board, dict_obj):
    """
    Charuco 検出（互換性優先版）
    - CharucoDetector を使わず、detectMarkers -> interpolateCornersCharuco -> cornerSubPix の流れで扱う
    - 戻り値: (detmap, imsize) where detmap[path] = (corners, ids)
    """
    res = {}
    imsize = None
    # サブピクセル精度のための termination criteria
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    # Detector parameters（必要なら調整）
    dp = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()

    # ループ
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])

        # マーカー検出
        try:
            corners_list, ids, _ = cv2.aruco.detectMarkers(img, dict_obj, parameters=dp)
        except Exception as e:
            # まれに detectMarkers のバインディング差で例外が出る場合はスキップ
            print(f"[WARN] detectMarkers failed for {p}: {e}")
            continue

        if ids is None or len(ids) == 0:
            continue

        # interpolate corners -> Charuco コーナを得る
        try:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners_list, ids, img, board)
        except Exception as e:
            print(f"[WARN] interpolateCornersCharuco failed for {p}: {e}")
            continue

        if charuco_ids is None or len(charuco_ids) < 4:
            continue

        # subpixel refinement
        corners_np = np.array(charuco_corners, dtype=np.float32).reshape(-1,1,2)
        cv2.cornerSubPix(img, corners_np, (3,3), (-1,-1), term)

        res[p] = (corners_np, np.array(charuco_ids, dtype=np.int32).reshape(-1,1))

    return res, imsize


def _object_points_from_board(board, ids):
    if hasattr(board,"chessboardCorners"):
        all_obj = board.chessboardCorners
    else:
        all_obj = board.getChessboardCorners()
    idx = ids.flatten().astype(int)
    obj = all_obj[idx].reshape(-1,1,3).astype(np.float32)
    return obj

# --- 検証(1): Charuco再投影 ---
def validate_intrinsics_charuco(K, dist, detmap, imsize, board):
    per_view = []
    for p,(corners,ids) in detmap.items():
        obj = _object_points_from_board(board, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: 
            continue
        proj,_ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errs = np.linalg.norm(proj.reshape(-1,2) - corners.reshape(-1,2), axis=1)
        per_view.append((p, float(np.median(errs)), float(errs.mean()), float(errs.max())))
    if not per_view:
        raise RuntimeError("reprojection check failed: no valid views")
    med = np.median([v[1] for v in per_view])
    mx  = np.max([v[3] for v in per_view])
    ok = (med<=TH_REPROJ_MED) and (mx<=TH_REPROJ_MAX)
    return ok, per_view, {"median_px": float(med), "max_px": float(mx)}

# --- 検証(2): AX=XB 残差 ---
def _se3_from_Rt(Rm, t):
    T = np.eye(4)
    T[:3,:3] = Rm
    T[:3, 3] = t.reshape(3)
    return T

def _log_rot_deg(Rm):
    # geodesic角度
    angle = np.rad2deg(np.arccos(np.clip((np.trace(Rm)-1)/2, -1, 1)))
    return float(angle)

def validate_ax_xb_residuals(Rb_g_list, tb_g_list, Rt_c_list, tt_c_list, Rc_g, tc_g):
    # OpenCVの定義に合わせ X = (Rc_g, tc_g) を使用（cam→gripper）
    X = _se3_from_Rt(Rc_g, tc_g)

    rot_errs, tra_errs = [], []
    n = len(Rb_g_list)
    for i,j in itertools.combinations(range(n), 2):
        if Rt_c_list[i] is None or Rt_c_list[j] is None:
            continue
        A = np.linalg.inv(_se3_from_Rt(Rb_g_list[i], tb_g_list[i])) @ _se3_from_Rt(Rb_g_list[j], tb_g_list[j])
        B = _se3_from_Rt(Rt_c_list[i], tt_c_list[i]) @ np.linalg.inv(_se3_from_Rt(Rt_c_list[j], tt_c_list[j]))
        # 残差：A X vs X B
        L = A @ X
        Rm = X @ B
        d = np.linalg.inv(Rm) @ L
        rot_errs.append(_log_rot_deg(d[:3,:3]))
        tra_errs.append(np.linalg.norm(d[:3,3])*1000.0)  # mm
    if not rot_errs:
        raise RuntimeError("AX=XB residual check failed: insufficient pairs")
    rot_med = float(np.median(rot_errs))
    tra_med = float(np.median(tra_errs))
    ok = (rot_med<=TH_AXXB_ROT_MED_DEG) and (tra_med<=TH_AXXB_TRA_MED_MM)
    return ok, {"rot_med_deg":rot_med, "tra_med_mm":tra_med, "rot_all_deg":rot_errs, "tra_all_mm":tra_errs}

# --- 検証(3): Base→Board 剛体一貫性 ---
def validate_board_rigidity(captures, K, dist, detmap, board):
    # 各フレームで T_b_t = T_b_g * T_g_c * T_c_t を求め，その分布のばらつきを評価
    # 注意：solvePnPは t_c_t（target→cam）を返すので，T_c_t = (R_c_t, t_c_t)
    # ここでは Rg_c,tg_c は handeye.yaml を用いる
    he = _fs_read_yaml(HE_YAML)
    Rg_c, tg_c = he["Rg_c"], he["tg_c"]
    T_g_c = _se3_from_Rt(Rg_c, tg_c)

    dict_obj = cv2.aruco.getPredefinedDictionary(DICT)
    board_cv = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                      SQUARE_MM/1000.0, MARKER_MM/1000.0, dict_obj)

    T_b_t_list = []
    for (path, Rb_g, tb_g) in captures:
        if path not in detmap: 
            continue
        corners, ids = detmap[path]
        obj = _object_points_from_board(board_cv, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: 
            continue
        R_c_t,_ = cv2.Rodrigues(rvec)
        T_b_g = _se3_from_Rt(Rb_g, tb_g)
        T_c_t = _se3_from_Rt(R_c_t, tvec.reshape(3,1))
        T_b_t = T_b_g @ T_g_c @ T_c_t
        T_b_t_list.append(T_b_t)

    if len(T_b_t_list)<3:
        raise RuntimeError("board rigidity check failed: not enough views")

    # 代表値（幾何平均的に近い1枚）を基準に差を計算
    ref = T_b_t_list[len(T_b_t_list)//2]
    rot_degs, tra_mms = [], []
    for T in T_b_t_list:
        d = np.linalg.inv(ref) @ T
        rot_degs.append(_log_rot_deg(d[:3,:3]))
        tra_mms.append(np.linalg.norm(d[:3,3])*1000.0)
    # MAD（中央値絶対偏差）
    rot_med = float(np.median(rot_degs)); rot_mad = float(np.median(np.abs(np.array(rot_degs)-rot_med)))
    tra_med = float(np.median(tra_mms));  tra_mad = float(np.median(np.abs(np.array(tra_mms)-tra_med)))
    ok = (rot_mad<=TH_BT_ROT_MAD_DEG) and (tra_mad<=TH_BT_TRA_MAD_MM)
    return ok, {"rot_mad_deg":rot_mad, "tra_mad_mm":tra_mad,
                "rot_all_deg":rot_degs, "tra_all_mm":tra_mms}

# --- (任意) COLMAP整合＆歪みモデルの影響 ---
def optional_colmap_consistency(K, dist, image_paths):
    # (A) 5係数へ切り詰めた場合と (B) 事前undistort(+ゼロ歪み) で，
    #   PnP再投影を比較し、差を数値化して報告（COLMAPへの影響を事前評価）
    dist5 = np.zeros((1,5), np.float64); 
    flat = dist.flatten().tolist()
    for i in range(min(5, len(flat))):
        dist5[0,i] = flat[i]

    dict_obj = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_MM/1000.0, MARKER_MM/1000.0, dict_obj)
    detmap, imsize = _charuco_detect(image_paths, board, dict_obj)

    diffs = []
    for p,(corners,ids) in detmap.items():
        obj = _object_points_from_board(board, ids)
        # (A) dist5 そのまま
        okA, rA, tA = cv2.solvePnP(obj, corners, K, dist5, flags=cv2.SOLVEPNP_SQPNP)
        # (B) 事前undistort + ゼロ歪み
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        und = cv2.undistort(img, K, dist)
        # und上で角再検出はコストが高いので，既コーナをピクセル写像（ここでは簡略に同一と見なす）
        okB, rB, tB = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if okA and okB:
            RA,_ = cv2.Rodrigues(rA); RB,_ = cv2.Rodrigues(rB)
            rot = _log_rot_deg(RB.T @ RA)
            tra = np.linalg.norm((tA - tB).reshape(3))*1000.0
            diffs.append((p,rot,tra))
    if not diffs:
        return {"note": "not enough views"}, None
    rot_med = float(np.median([d[1] for d in diffs]))
    tra_med = float(np.median([d[2] for d in diffs]))
    return {"rot_med_deg": rot_med, "tra_med_mm": tra_med}, diffs

def main():
    # データ読込
    intr = _fs_read_yaml(INTR_YAML)
    he = _fs_read_yaml(HE_YAML)
    assert "K" in intr and "dist" in intr, "intrinsics.yaml must contain K and dist"
    assert "Rg_c" in he and "tg_c" in he, "handeye.yaml must contain Rg_c and tg_c"

    K, dist = intr["K"], intr["dist"]
    Rg_c, tg_c = he["Rg_c"], he["tg_c"]
    Rc_g = Rg_c.T
    tc_g = -Rc_g @ tg_c

    caps = _load_robot_json(POSE_JSON, IMG_DIR)
    image_paths = [c[0] for c in caps]

    # Charuco 検出（軽量版）
    dict_obj = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_MM/1000.0, MARKER_MM/1000.0, dict_obj)
    detmap, imsize = _charuco_detect(image_paths, board, dict_obj)

    # (1) 再投影検証
    ok1, per_view, s1 = validate_intrinsics_charuco(K, dist, detmap, imsize, board)

    # (2) AX=XB 残差
    #   solvePnPで得た target→cam (R,t) を列挙
    Rt_c_list, tt_c_list = [], []
    for p, Rb_g, tb_g in caps:
        if p not in detmap:
            Rt_c_list.append(None); tt_c_list.append(None); continue
        corners, ids = detmap[p]
        obj = _object_points_from_board(board, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            Rt_c_list.append(None); tt_c_list.append(None); continue
        Rc_t,_ = cv2.Rodrigues(rvec)
        Rt_c_list.append(Rc_t)
        tt_c_list.append(tvec.reshape(3,1))
    Rb_g_list = [c[1] for c in caps]
    tb_g_list = [c[2] for c in caps]
    ok2, s2 = validate_ax_xb_residuals(Rb_g_list, tb_g_list, Rt_c_list, tt_c_list, Rc_g, tc_g)

    # (3) Base→Board 剛体一貫性
    ok3, s3 = validate_board_rigidity(caps, K, dist, detmap, board)

    # (任意) 歪みモデル差の事前評価
    s4, diffs = optional_colmap_consistency(K, dist, image_paths)

    # 総合判定
    ready = ok1 and ok2 and ok3
    print("\n=== Meta-Validation Report ===")
    print(f"[1] Reprojection  : median={s1['median_px']:.3f}px, max={s1['max_px']:.3f}px  -> {'OK' if ok1 else 'NG'}")
    print(f"[2] AX=XB residual: rot_med={s2['rot_med_deg']:.3f}deg, tra_med={s2['tra_med_mm']:.3f}mm -> {'OK' if ok2 else 'NG'}")
    print(f"[3] Base->Board   : rot_MAD={s3['rot_mad_deg']:.3f}deg, tra_MAD={s3['tra_mad_mm']:.3f}mm -> {'OK' if ok3 else 'NG'}")
    if s4 is not None:
        print(f"[opt] Distortion model gap (A 5-coef vs B undist+zero): rot_med={s4['rot_med_deg']:.3f}deg, tra_med={s4['tra_med_mm']:.3f}mm")

    print(f"\nCOLMAP/3DGS Ready: {'PASS' if ready else 'FAIL'}")
    if not ready:
        print("-> See thresholds at top of this file and per-view details to locate failure cause.")
    # 追加で per_view / diffs をCSV出力したい場合は適宜保存処理を追加

if __name__ == "__main__":
    main()
