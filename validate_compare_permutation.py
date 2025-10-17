# compose_compare.py
import os, json, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from math import degrees

'''
背景：handeye_permutation の出力は「as_is-as_is と inv-inv の両方が非常に低残差（rot_med≈0.26°、tra_med≈2.9mm）」で、as_is-inv / inv-as_is は大きく悪化する（→ 方向ミスマッチは“片側だけ反転”が最も悪い）。

これは「入力（A,B）の向きが両方とも現在の定義（as_is）で一致しているか、両方とも逆（inv）で一致している」ことを示す。どちらが“物理的に正しい”かは 最終的に合成した world→camera（base→camera）を PnP／Charuco によるカメラ姿勢と照合して決めれば確定できる


処理：
as_is-as_is と inv-inv の両方について（a）hand-eye の X を保存、（b）各フレームの T_base->camera（合成）を計算、（c）それを PnP（Charuco から得た camera pose）と比較する。より小さく整合する方を採用する。
'''

ROOT="."
# paths (適宜変更)
PERM="handeye_permutation_results.json"
POSE_JSON=os.path.join("hand_eye_calibration","json","poses.json")
IMG_DIR=os.path.join("hand_eye_calibration","image")
INTR=os.path.join("sfm_out","intrinsics.yaml")
HE=os.path.join("sfm_out","handeye.yaml")

# --- helper ---
def load_json(p): 
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def T_from(Rm, t):
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=t.reshape(3)
    return T
def inv_T(T):
    Rm=T[:3,:3]; t=T[:3,3].reshape(3,1)
    Rinv=Rm.T; tinv=-Rinv@t
    T2=np.eye(4); T2[:3,:3]=Rinv; T2[:3,3]=tinv.reshape(3)
    return T2
def rot_deg_of_Tdiff(A,B):
    Rdiff=A[:3,:3].T@B[:3,:3]
    ang = degrees(np.arccos(max(-1,min(1,(np.trace(Rdiff)-1)/2))))
    return ang
def trans_mm_of_Tdiff(A,B):
    return float(np.linalg.norm((A[:3,3]-B[:3,3]))*1000.0)

# --- load permutation results ---
pr = load_json(PERM)

# --- load captures (use your charuco_handeye_calib.py loader) ---
spec_path = "charuco_handeye_calib.py"
import importlib.util
spec = importlib.util.spec_from_file_location("ch", spec_path)
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)
captures = ch.load_robot_json(POSE_JSON, IMG_DIR)  # returns list of Capture objects with .path,.Rb_g,.tb_g

# --- get PnP camera poses per view from existing pipeline (recompute using functions in ch if necessary) ---
# We'll reuse the detection & estimate pipeline from ch:
dict_obj = cv2.aruco.getPredefinedDictionary(ch.DICT)
board = cv2.aruco.CharucoBoard((ch.SQUARES_X,ch.SQUARES_Y), ch.SQUARE_LEN_MM/1000.0, ch.MARKER_LEN_MM/1000.0, dict_obj)
dets, ims = ch.detect_charuco_points([c.path for c in captures], board, dict_obj)
dets = ch.filter_detections(dets)
calib = ch.calibrate_intrinsics(dets, ims, board)
K = calib["K"]; dist = calib["dist"]
detection_map = {d.path: d for d in dets}
Rt_list, _ = ch.estimate_board_poses(captures, detection_map, board, K, dist)  # list of (Rc_t, tc_t) or None
# map path->(Rc_t, tc_t)
pnp_map={}
for cap, Rt in zip(captures, Rt_list):
    if Rt is None: continue
    Rc_t, tc_t = Rt
    # camera pose in base (via composition later), but store as T_c_t (cam<-target)
    pnp_map[cap.path] = T_from(Rc_t, tc_t)

# --- for each candidate case (as_is-as_is and inv-inv), compute handeye X and then composed base->camera per view ---
results={}
for key in ["as_is-as_is","inv-inv"]:
    entry = pr.get(key)
    if entry is None:
        results[key] = {"error":"no permutation entry"}
        continue
    R_cam2gripper = np.array(entry["R_cam2gripper"])
    t_cam2gripper = np.array(entry["t_cam2gripper"]).reshape(3,1)
    # convert to Rg_c,tg_c (gripper->camera) as your code expects if needed (OpenCV returns cam->gripper)
    Rg_c = R_cam2gripper.T
    tg_c = -Rg_c @ t_cam2gripper
    # compose base->camera for each cap: Rb_c = Rb_g @ Rg_c ; tb_c = Rb_g @ tg_c + tb_g
    rot_errs=[]; tra_errs=[]
    matches=0
    for cap, Rt in zip(captures, Rt_list):
        if Rt is None: continue
        Rb_g = cap.Rb_g; tb_g = cap.tb_g.reshape(3,1)
        T_b_g = T_from(Rb_g, tb_g)
        T_g_c = T_from(Rg_c, tg_c)
        T_b_c = T_b_g @ T_g_c
        # compare to PnP camera transform: PnP returned T_c_t (camera <- target)
        # we need a comparable camera pose in base coords; PnP gives T_c_t; to get T_b_c we need T_b_t via T_b_t = T_b_g @ T_g_c @ T_c_t ? 
        # but ch.estimate_board_poses gives Rc_t, tc_t where solvePnP returns target->camera, we used T_from(Rc_t, tc_t) as T_c_t
        # Using earlier composition from your script: T_b_t = T_b_g @ T_g_c @ T_c_t. Then T_b_c = T_b_t @ inv(T_c_t) = T_b_g @ T_g_c
        # So T_b_c we computed is the correct comparable pose to camera in base frame
        if cap.path not in pnp_map:
            continue
        # pnp_map stores T_c_t; but we don't have direct T_b_c from PnP alone, so compare using alternative:
        # reconstruct T_b_t from T_b_g @ T_g_c @ T_c_t and compare across views consistency
        T_b_t = T_b_g @ T_g_c @ pnp_map[cap.path]
        # choose a reference (first valid) to compute diffs (rigidity check)
        if matches==0:
            ref_T_b_t = T_b_t
        d_rot = rot_deg_of_Tdiff(ref_T_b_t, T_b_t)
        d_tra = trans_mm_of_Tdiff(ref_T_b_t, T_b_t)
        rot_errs.append(d_rot); tra_errs.append(d_tra)
        matches+=1
    if matches==0:
        results[key]={"error":"no matches"}
        continue
    results[key]={"rot_MAD_deg": float(np.median(np.abs(np.array(rot_errs)-np.median(rot_errs)))),
                  "tra_MAD_mm": float(np.median(np.abs(np.array(tra_errs)-np.median(tra_errs)))),
                  "n_views": matches}
# save
with open("compose_compare_report.json","w") as f:
    json.dump(results,f,indent=2)
print("wrote compose_compare_report.json")
