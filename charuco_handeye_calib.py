# calibrate_robot_charuco.py
import json
import glob
import os
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

'''
intrinsics.yaml
    - K：カメラ行列
    - dist：ゆがみ係数（Distortion coefficients）OpenCV の plumb-bob（k1,k2,p1,p2,k3, [k4,k5,k6]）や別モデル（fisheye）で表現。出力は配列。
    - rms：再投影誤差.pixel単位（高精度なら0.5px以下、一般用では1.0px以下が良好）
    
handeye.yaml
    - rvec:回転ベクトル。物体-カメラの回転
    - tvec:カメラの並進。
    cv.projectPoints(objectPoints, rvec, tvec, K, dist) に入れると、再投影された pixel 位置が得られる（再投影誤差の計算に使う）
'''

# ---------- Config ----------
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LEN_MM = 32.0
MARKER_LEN_MM = 24.0
DICT = cv2.aruco.DICT_4X4_50   # 必ず具体的に 4X4_50/100/250/1000 を選ぶ
EULER_ORDER = "xyz"            # 実機に合わせて変更可
MIN_CORNERS = 20               # 品質フィルタ
PNP_FLAG = cv2.SOLVEPNP_SQPNP  # or SOLVEPNP_ITERATIVE

# ---------- Data ----------
@dataclass
class Capture:
    path: str
    Rb_g: np.ndarray  # (3,3) gripper->base
    tb_g: np.ndarray  # (3,1) meters

def load_robot_json(json_path: str, image_dir: str) -> list[Capture]:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    caps = []
    for c in js["captures"]:
        img = os.path.join(image_dir, os.path.basename(c["image_file"]))
        t = np.array([c["pose"]["x_mm"], c["pose"]["y_mm"], c["pose"]["z_mm"]], dtype=np.float64) / 1000.0
        angles = np.array([c["pose"]["rx_deg"], c["pose"]["ry_deg"], c["pose"]["rz_deg"]], dtype=np.float64)
        Rb_g = R.from_euler(EULER_ORDER, angles, degrees=True).as_matrix()  # gripper->base
        caps.append(Capture(img, Rb_g, t.reshape(3,1)))
    return caps

# 変更点：MIN_CORNERSを例えば12に変更（スクリプト上部で定義）
MIN_CORNERS = 21  # 必要であれば 16 などに調整

def detect_charuco_points(image_paths: list[str], board, dict_obj):
    det_params = cv2.aruco.DetectorParameters()
    charuco = cv2.aruco.CharucoDetector(board)

    obj_points, img_points = [], []
    imsize = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] can't read image: {p}")
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])

        ch_corners, ch_ids, marker_corners, marker_ids = charuco.detectBoard(img)

        if ch_ids is None or len(ch_ids) < MIN_CORNERS:
            print(f"[INFO] insufficient charuco corners in {p}: {0 if ch_ids is None else len(ch_ids)}")
            continue

        # board.matchImagePoints は objPoints (Nx3) と imgPoints (Nx2) を返す
        obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or imgp is None:
            print(f"[WARN] matchImagePoints failed for {p}")
            continue

        # --- ここが重要：shape/dtype を OpenCV が期待する形式に揃える ---
        # obj: (N,3) -> (N,1,3) float32
        # imgp: (N,2) -> (N,1,2) float32
        obj_arr = np.asarray(obj, dtype=np.float32).reshape(-1, 1, 3)
        img_arr = np.asarray(imgp, dtype=np.float32).reshape(-1, 1, 2)

        # 再投影チェック（任意だが推奨）
        if obj_arr.shape[0] < MIN_CORNERS:
            print(f"[INFO] after reshape, too few points in {p}: {obj_arr.shape[0]}")
            continue

        obj_points.append(obj_arr)
        img_points.append(img_arr)

    return obj_points, img_points, imsize



def calibrate_intrinsics(obj_points, img_points, imsize):
    # 事前チェック
    if len(obj_points) == 0 or len(img_points) == 0:
        raise RuntimeError("No valid charuco detections collected. Check MIN_CORNERS or images.")
    if len(obj_points) != len(img_points):
        raise RuntimeError("objectPoints and imagePoints count mismatch.")

    # 確認用ログ（デバッグ）
    print(f"[INFO] collected {len(obj_points)} valid views")
    for i,(o,i2) in enumerate(zip(obj_points,img_points)):
        print(f" view {i}: obj {o.shape} dtype {o.dtype}, img {i2.shape} dtype {i2.dtype}")

    flags = cv2.CALIB_RATIONAL_MODEL  # 必要なら調整
    # 注意: 今 obj_points, img_points の各要素は (N,1,3) / (N,1,2) の np.float32 になっているはず
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, imsize, None, None, flags=flags)
    return rms, K, dist


def estimate_board_poses(image_paths: list[str], board, dict_obj, K, dist):
    charuco = cv2.aruco.CharucoDetector(board)
    Rt_list = []  # (Rc_t, tc_t) per image
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            Rt_list.append(None); continue

        ch_corners, ch_ids, marker_corners, marker_ids = charuco.detectBoard(img)
        if ch_ids is None or len(ch_ids) < MIN_CORNERS:
            Rt_list.append(None); continue

        obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or imgp is None:
            Rt_list.append(None); continue

        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=PNP_FLAG)
        if not ok:
            Rt_list.append(None); continue
        Rc_t, _ = cv2.Rodrigues(rvec)
        Rt_list.append((Rc_t, tvec.reshape(3,1)))
    return Rt_list


def run_handeye(captures: list[Capture], target2cam_list):
    Rg2b, tg2b, Rt2c, tt2c = [], [], [], []
    for cap, Rt in zip(captures, target2cam_list):
        if Rt is None: 
            continue
        Rc_t, tc_t = Rt
        Rg2b.append(cap.Rb_g)          # gripper->base
        tg2b.append(cap.tb_g)
        Rt2c.append(Rc_t)              # target->camera
        tt2c.append(tc_t)
    # OpenCVは行列orロドリゲス両対応
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        Rg2b, tg2b, Rt2c, tt2c, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
    return R_cam2gripper, t_cam2gripper  # {}^gR_c, {}^g t_c

def compose_base_to_cam(caps: list[Capture], Rg_c, tg_c):
    Tbc_list = []
    for cap in caps:
        Rb_g, tb_g = cap.Rb_g, cap.tb_g
        # {}^bT_c = {}^bT_g * {}^gT_c
        Rb_c = Rb_g @ Rg_c
        tb_c = Rb_g @ tg_c + tb_g
        Tbc_list.append((Rb_c, tb_c))
    return Tbc_list

def to_colmap_quat_t(Rbc, tbc):
    # images.txtは World→Camera。いまWorld=Baseなので {}^cT_b = inv({}^bT_c)
    Rc_b = Rbc.T
    tc_b = - Rc_b @ tbc
    q = R.from_matrix(Rc_b).as_quat()  # returns [qx,qy,qz,qw]
    QW, QX, QY, QZ = q[3], q[0], q[1], q[2]
    return (QW, QX, QY, QZ), tc_b.reshape(-1)

def save_yaml(path, data: dict):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for k,v in data.items():
        fs.write(k, np.array(v))
    fs.release()

def export_colmap(cameras_txt, images_txt, points_txt, K, dist, image_paths, Tbc_list, width, height):
    # cameras.txt
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # OPENCV: fx fy cx cy k1 k2 p1 p2 k3
    k1,k2,p1,p2,k3 = (dist.flatten().tolist()+[0,0,0,0,0])[:5]
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 OPENCV {width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}\n")

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i,(p,bc) in enumerate(zip(image_paths, Tbc_list), start=1):
            if bc is None: continue
            (Rbc, tbc) = bc
            (qw,qx,qy,qz), t = to_colmap_quat_t(Rbc, tbc)
            f.write(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {os.path.basename(p)}\n")
            f.write("\n")  # no 2D points

    with open(points_txt, "w") as f:
        f.write("# 3D point list is empty for known-poses bootstrap\n")

def main(json_path, image_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    caps = load_robot_json(json_path, image_dir)
    image_paths = [c.path for c in caps]

    # Board（メートル単位）
    dict_obj = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_LEN_MM/1000.0, MARKER_LEN_MM/1000.0, dict_obj)

    # 1) Intrinsics
    obj_pts, img_pts, imsize = detect_charuco_points(image_paths, board, dict_obj)
    assert imsize is not None and len(obj_pts) >= 5, "Charuco検出不足"
    rms, K, dist = calibrate_intrinsics(obj_pts, img_pts, imsize)
    save_yaml(os.path.join(out_dir, "intrinsics.yaml"), {"K":K, "dist":dist, "rms":[rms]})

    # 2) Board poses (target->cam)
    Rt_list = estimate_board_poses(image_paths, board, dict_obj, K, dist)

    # 3) Hand-Eye
    Rg_c, tg_c = run_handeye(caps, Rt_list)
    save_yaml(os.path.join(out_dir, "handeye.yaml"), {"Rg_c":Rg_c, "tg_c":tg_c})

    # 4) Base->Camera per image & export to COLMAP
    Tbc = compose_base_to_cam(caps, Rg_c, tg_c)
    colmap_dir = os.path.join(out_dir, "colmap_sparse")
    os.makedirs(colmap_dir, exist_ok=True)
    export_colmap(os.path.join(colmap_dir, "cameras.txt"),
                  os.path.join(colmap_dir, "images.txt"),
                  os.path.join(colmap_dir, "points3D.txt"),
                  K, dist, image_paths, Tbc, imsize[0], imsize[1])

if __name__ == "__main__":
    # 例:
    # python calibrate_robot_charuco.py /path/to/json_info.json /path/to/images out/
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
