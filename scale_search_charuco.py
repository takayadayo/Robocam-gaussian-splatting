# scale_search_charuco.py
import os, json, numpy as np, cv2
from math import degrees
from scipy.spatial.transform import Rotation as R
import importlib.util

# === 設定（必要なら調整） ===
ROOT = "."
POSE_JSON = os.path.join("hand_eye_calibration","json","poses.json")
IMG_DIR   = os.path.join("hand_eye_calibration","image")
OUT_JSON  = "scale_search_report.json"

# スケール探索レンジ（例：±8%を0.2%刻み）。最初は広めに→後で局所に絞るとよい
SCALE_START, SCALE_STOP, SCALE_STEP = 0.3, 0.9, 0.001

# === ユーティリティ ===
def T_from(Rm, t):
    T = np.eye(4); T[:3,:3]=Rm; T[:3,3]=t.reshape(3); return T

def rot_deg_of_Tdiff(A,B):
    Rdiff = A[:3,:3].T @ B[:3,:3]
    ang = np.arccos(np.clip((np.trace(Rdiff)-1)/2, -1, 1))
    return float(np.degrees(ang))

def trans_mm_of_Tdiff(A,B):
    return float(np.linalg.norm(A[:3,3]-B[:3,3])*1000.0)

def MAD(vals):
    med = np.median(vals)
    return float(np.median(np.abs(np.array(vals)-med)))

# === あなたの実装をロード（コードパス差の排除） ===
spec = importlib.util.spec_from_file_location("ch", "charuco_handeye_calib.py")
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)

# 1) ロボット姿勢・画像パスの取得（あなたの関数を利用）
captures = ch.load_robot_json(POSE_JSON, IMG_DIR)  # List[Capture] with .path,.Rb_g,.tb_g
image_paths = [c.path for c in captures]

# 2) Charuco検出→内パラ→PnP（ここを一次固定資産として以後使い回す）
dict_obj = cv2.aruco.getPredefinedDictionary(ch.DICT)
# 元のスケール（あなたの定義値）
SQUARES_X, SQUARES_Y = ch.SQUARES_X, ch.SQUARES_Y
SQ_MM, MK_MM = ch.SQUARE_LEN_MM, ch.MARKER_LEN_MM

# Board（元スケール）生成
board_ref = cv2.aruco.CharucoBoard((SQUARES_X,SQUARES_Y), SQ_MM/1000.0, MK_MM/1000.0, dict_obj)

# 検出・校正（あなたの関数をそのまま呼ぶ）
det_map, imsize = ch.detect_charuco_points(image_paths, board_ref, dict_obj)
det_map = ch.filter_detections(det_map)
calib = ch.calibrate_intrinsics(det_map, imsize, board_ref)
K, dist = calib["K"], calib["dist"]
# ここで使う検出は採択後の dict（path->Detection）を固定
det_map_fixed = {d.path: d for d in calib["detections"]}

# 3) Hand-Eye の X（cam->gripper）を「as_is-as_is」側で一度だけ解く
#    （相対運動の整合は既に良好と判明しているため、Xは固定で良い）
Rt_list, _ = ch.estimate_board_poses(captures, det_map_fixed, board_ref, K, dist)
Rg2b_list = [T_from(c.Rb_g, c.tb_g).copy() for c in captures]  # absolute (gripper->base)
# 修正（正しい呼び出しと変換）
Rc_g, tc_g = ch.run_handeye(captures, Rt_list)   # OpenCV returns camera->gripper (Rc_g, tc_g)
# convert to gripper->camera (一致する後続実装に合わせる)
Rg_c = Rc_g.T
tg_c = -Rg_c @ tc_g
T_g2c = T_from(Rg_c, tg_c)


# 4) スケール探索：2Dは固定し、Boardの3D角座標のみスケール
def object_points_from_board(board, ids):
    all_obj = board.getChessboardCorners() if hasattr(board,"getChessboardCorners") else board.chessboardCorners
    return all_obj[ids.flatten().astype(int)].reshape(-1,1,3).astype(np.float32)

def pnp_target_pose(det, board_scaled, K, dist):
    # det.corners: (N,1,2) float32 ; det.ids: (N,1) int32
    obj = object_points_from_board(board_scaled, det.ids)
    ok, rvec, tvec = cv2.solvePnP(obj, det.corners.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok: return None
    R_c_t, _ = cv2.Rodrigues(rvec)
    return T_from(R_c_t, tvec.reshape(3))

# “as_is-as_is” と “inv-inv” の両系統で比較
def compose_rigidity_MAD(T_g2c, inv_switch=False):
    # inv_switch=True の場合、仮想的に gripper/base と target/camera を両方反転した系統を模す
    rot_mads = []; tra_mads = []
    # 各スケールごとの結果を格納
    sweep = []
    s = SCALE_START
    while s <= SCALE_STOP + 1e-12:
        # スケール適用のBoard
        board_s = cv2.aruco.CharucoBoard((SQUARES_X,SQUARES_Y), (SQ_MM*s)/1000.0, (MK_MM*s)/1000.0, dict_obj)
        # ビューごとに T_b_t を構成
        T_list = []
        for cap in captures:
            det = det_map_fixed.get(cap.path, None)
            if det is None: 
                continue
            T_c_t = pnp_target_pose(det, board_s, K, dist)
            if T_c_t is None:
                continue
            # 反転系の切替（相対運動としては等価になるが、合成の「位相」を確認する）
            T_b_g = T_from(cap.Rb_g, cap.tb_g)
            if not inv_switch:
                T_b_t = T_b_g @ T_g2c @ T_c_t
            else:
                # gripper/base と target/camera を両方反転した世界を模擬（位相チェック用）
                # T_b_g_inv = inv(T_b_g), T_g2c_inv = inv(T_g2c), T_c_t_inv = inv(T_c_t)
                # ただし Base->Board の一貫性は「相対」で評価するため、参照系は何れでも良い
                T_b_t = np.linalg.inv(T_b_g) @ np.linalg.inv(T_g2c) @ np.linalg.inv(T_c_t)

            T_list.append(T_b_t)

        if len(T_list) < 3:
            sweep.append({"s": s, "rot_MAD_deg": None, "tra_MAD_mm": None, "n": len(T_list)})
        else:
            # 剛体一貫性：代表姿勢との差分でMADを取る
            ref = T_list[len(T_list)//2]
            rot_errs = [rot_deg_of_Tdiff(ref, T) for T in T_list]
            tra_errs = [trans_mm_of_Tdiff(ref, T) for T in T_list]
            sweep.append({"s": s, "rot_MAD_deg": MAD(rot_errs), "tra_MAD_mm": MAD(tra_errs), "n": len(T_list)})
        s += SCALE_STEP
    # 最良スケールの選抜（回転→並進の優先順位でレキシコ最小）
    cand = [x for x in sweep if x["rot_MAD_deg"] is not None]
    best = sorted(cand, key=lambda x: (x["rot_MAD_deg"], x["tra_MAD_mm"]))[0] if cand else None
    return sweep, best

sweep_as_is, best_as_is = compose_rigidity_MAD(T_g2c, inv_switch=False)
sweep_invinv, best_invinv = compose_rigidity_MAD(T_g2c, inv_switch=True)

report = {
    "grid": {"start": SCALE_START, "stop": SCALE_STOP, "step": SCALE_STEP},
    "best_as_is": best_as_is,
    "best_invinv": best_invinv,
    "note": "2D corners, K, dist, accepted views are fixed from charuco_handeye_calib.py; only board 3D is scaled."
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("wrote", OUT_JSON)
