# test_handeye_permutations.py
import os, json, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from math import degrees

'''
calibrateHandEye に渡す入力（gripper2base, target2cam）を「そのまま／逆（inverse）」の組み合わせで4パターン試し、各パターンで出力 X を得て AX−XB 残差（回転：deg、並進：mm） を定量化します。最良パターンが「正しい表現」

なぜ「総当たり検証」が必要か：calibrateHandEye は内部で A,B を相対運動に変換して AX=XB を解く。A と B の「向き」あるいは「座標に表すベクトルの定義」が 1 箇所でも逆になると、解 X が意味を成さなくなり、回転差で数度、並進で数cm〜数十cmのずれが生じる。これは理論的に当たり前（行列の逆を掛けるだけで誤差が累積）で、Hの値（今回の ~5°/50–180mm）はこの種の「向きミス」で説明可能

出力で4ケース (as_is-as_is, as_is-inv, inv-as_is, inv-inv) が出る。最小の rot_med/tra_med を示すケースが正しい定義に極めて近い。

想定ケース：
・最小ケースが as_is-as_is なら現在の流れは正しい（別原因を探す）。
・最小ケースが inv-as_is や as_is-inv なら、どちらか（robot pose / target pose）を反転して与えるべきであることが確定する。
・最小ケースが inv-inv なら両方とも反転が必要（珍しいがあり得る）。

（根拠：AX=XB の行列代数的性質により、A,B の「逆」を渡すと得られる X が逆になるため、残差が比較で極めて明確に変化する。）
'''


# --- adjust paths ---
ROOT = "."
POSE_JSON = os.path.join("hand_eye_calibration","json","poses.json")
IMG_DIR = os.path.join("hand_eye_calibration","image")

euler_order_validation = "xyz"

# --- load poses (same as your code) ---
def load_robot_json(json_path, image_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    caps = []
    for c in js["captures"]:
        img = os.path.join(image_dir, os.path.basename(c["image_file"]))
        t = np.array([c["pose"]["x_mm"], c["pose"]["y_mm"], c["pose"]["z_mm"]], np.float64)/1000.0
        ang = np.array([c["pose"]["rx_deg"], c["pose"]["ry_deg"], c["pose"]["rz_deg"]], np.float64)
        Rb_g = R.from_euler(euler_order_validation, ang, degrees=True).as_matrix()
        caps.append((img, Rb_g, t.reshape(3,1)))
    return caps

# --- helper: SE3 from R,t ---
def T_from(Rm, t):
    T = np.eye(4)
    T[:3,:3] = Rm
    T[:3,3] = t.reshape(3)
    return T

def inv_T(T):
    Rm = T[:3,:3]; t = T[:3,3].reshape(3,1)
    Rinv = Rm.T
    tinv = -Rinv @ t
    T2 = np.eye(4); T2[:3,:3]=Rinv; T2[:3,3]=tinv.reshape(3)
    return T2

def rot_err_deg(A,B):
    # geodesic angle between rotations A and B
    Rdiff = A.T @ B
    ang = degrees(np.arccos(np.clip((np.trace(Rdiff)-1)/2, -1,1)))
    return ang

def analyze_residuals(A_list, B_list, R_X, t_X):
    # A_i = relative gripper motion (4x4), B_i = relative target motion (4x4)
    X = T_from(R_X, t_X)
    rots=[]; tras=[]
    n=0
    for i in range(len(A_list)):
        for j in range(i+1, len(A_list)):
            A = np.linalg.inv(A_list[i]) @ A_list[j]
            B = B_list[i] @ np.linalg.inv(B_list[j])
            L = A @ X
            Rm = X @ B
            D = np.linalg.inv(Rm) @ L
            rots.append(rot_err_deg(np.eye(3), D[:3,:3])) # angle of D
            tras.append(np.linalg.norm(D[:3,3])*1000.0) # mm
            n+=1
    if n==0:
        return None
    return {"rot_med_deg": float(np.median(rots)), "tra_med_mm": float(np.median(tras)),
            "rot_all": rots, "tra_all": tras}

# --- prepare A_list,B_list from captures & estimated board poses (use your estimate_board_poses output) ---
# Here we assume you previously ran charuco_handeye_calib.py and have:
#   - sfm_out/intrinsics.yaml (K,dist)
#   - estimate_board_poses outputs: a list Rt_list of (R_c_t, t_c_t) for each capture path (or you can recreate calling your functions)
# To simplify, we'll re-run the pose estimation using your existing code functions if available.
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ch", "charuco_handeye_calib.py")
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)

captures = ch.load_robot_json(POSE_JSON, IMG_DIR)
image_paths = [c.path for c in captures]

# reuse your detection->estimate_board_poses pipeline
dict_obj = cv2.aruco.getPredefinedDictionary(ch.DICT)
board = cv2.aruco.CharucoBoard((ch.SQUARES_X,ch.SQUARES_Y), ch.SQUARE_LEN_MM/1000.0, ch.MARKER_LEN_MM/1000.0, dict_obj)
detections, imsize = ch.detect_charuco_points(image_paths, board, dict_obj)
detections = ch.filter_detections(detections)
calib = ch.calibrate_intrinsics(detections, imsize, board)
K = calib["K"]; dist = calib["dist"]
Rt_list, pose_stats = ch.estimate_board_poses(captures, {d.path:d for d in calib["detections"]}, board, K, dist)

# build absolute transforms:
A_abs = []  # gripper->base transforms as 4x4
B_abs = []  # target->cam transforms as 4x4
for cap, Rt in zip(captures, Rt_list):
    if Rt is None:
        continue
    Rc_t, tc_t = Rt
    Rb_g = cap.Rb_g; tb_g = cap.tb_g
    A_abs.append(T_from(Rb_g, tb_g))
    B_abs.append(T_from(Rc_t, tc_t))

# create various candidate lists (original / inverted)
def make_candidates(abs_list):
    as_is = abs_list
    inv = [inv_T(T) for T in abs_list]
    return {"as_is": as_is, "inv": inv}

A_cands = make_candidates(A_abs)
B_cands = make_candidates(B_abs)

# try 4 combos: (A:as_is/inv) x (B:as_is/inv)
results = {}
for akey, Al in A_cands.items():
    for bkey, Bl in B_cands.items():
        # extract Rs and ts to pass to cv2.calibrateHandEye (list of (3,3) and (3,1))
        Rg2b = [T[:3,:3] for T in Al]
        tg2b = [T[:3,3].reshape(3,1) for T in Al]
        Rt2c = [T[:3,:3] for T in Bl]
        tt2c = [T[:3,3].reshape(3,1) for T in Bl]
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
        except Exception as e:
            results[f"{akey}-{bkey}"] = {"error": str(e)}
            continue
        # convert to X (R,t) as returned (no fix)
        res = analyze_residuals(Al, Bl, R_cam2gripper, t_cam2gripper)
        results[f"{akey}-{bkey}"] = {"R_cam2gripper": R_cam2gripper.tolist(), "t_cam2gripper": t_cam2gripper.reshape(-1).tolist(), "residuals": res}

# print summary
for k,v in results.items():
    print("===", k, "===")
    if "error" in v:
        print("ERROR:", v["error"])
    else:
        r = v["residuals"]
        if r is None:
            print("No residuals computed (insufficient pairs).")
        else:
            print(f"rot_med_deg={r['rot_med_deg']:.4f}, tra_med_mm={r['tra_med_mm']:.3f}")

# Save results to JSON
import json
with open("handeye_permutation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("wrote handeye_permutation_results.json")
