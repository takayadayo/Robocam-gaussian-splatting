#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Charuco + Hand-Eye calibration script

1. Charuco 画像からカメラ内部パラメータ (K, dist) を推定
2. calibrateCamera の出力 rvecs, tvecs を target->cam (board->camera) として利用
3. ロボット JSON (base->flange pose) から R_gripper2base, t_gripper2base を生成
4. cv2.calibrateHandEye で camera->gripper 変換を推定
5. Matplotlib で base 座標系における gripper / camera 軌跡を 3D 可視化
6. OpenCV で検出結果・補正結果をウィンドウ表示（任意）

依存:
- Python 3.x
- OpenCV 4.7.0 (pip の opencv-contrib-python==4.7.* 推奨)
- numpy, matplotlib
"""

import argparse
import json
import os
import glob

import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D)

# ---------------------------------------------------------------
# 基本ユーティリティ
# ---------------------------------------------------------------
def euler_extrinsic_to_R(rx_deg, ry_deg, rz_deg, order="XYZ"):
    """
    世界座標系の X,Y,Z 軸まわりの外部回転 (deg) を、指定された順序で合成して回転行列を返す。

    order: 文字列 "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX" のいずれか。
           例: order="ZYX" の場合、R = Rz(rz) * Ry(ry) * Rx(rx)
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    def Rx(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0],
                         [0, ca, -sa],
                         [0, sa,  ca]], dtype=np.float64)

    def Ry(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ ca, 0, sa],
                         [  0, 1,  0],
                         [-sa, 0, ca]], dtype=np.float64)

    def Rz(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca, -sa, 0],
                         [sa,  ca, 0],
                         [ 0,   0, 1]], dtype=np.float64)

    R_map = {
        "X": Rx(rx),
        "Y": Ry(ry),
        "Z": Rz(rz),
    }

    R = np.eye(3, dtype=np.float64)
    # 右から掛けていく（外部回転: fixed axes）
    for ax in order:
        R = R_map[ax] @ R

    return R

def pose_to_T_base_gripper(pose_dict, euler_order="XYZ"):
    """
    JSON の pose から base->gripper (フランジ中心) 同次変換 T_b_g を生成。
    euler_order: "XYZ", "ZYX" など。実機仕様が確定したら固定。

    x_mm, y_mm, z_mm: 世界座標系での制御点（フランジ中心）位置 [mm]
    rx_deg, ry_deg, rz_deg: 世界座標系 XYZ 軸まわりの外部回転 [deg]
    """
    x = pose_dict["x_mm"] * 1e-3  # m
    y = pose_dict["y_mm"] * 1e-3
    z = pose_dict["z_mm"] * 1e-3

    rx = pose_dict["rx_deg"]
    ry = pose_dict["ry_deg"]
    rz = pose_dict["rz_deg"]

    R_b_g = euler_extrinsic_to_R(rx, ry, rz, order=euler_order)
    t_b_g = np.array([x, y, z], dtype=np.float64).reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_b_g
    T[:3, 3] = t_b_g[:, 0]
    return T

def evaluate_euler_order(
    euler_order,
    robot_json_path,
    used_image_paths,
    K,
    dist,
    rvecs, tvecs,
):
    """
    特定の Euler 順序 (e.g. "XYZ") を仮定して
    1) base->gripper を計算
    2) Hand-Eye を解いて camera->gripper を求める
    3) 各画像で base->board を計算し、その位置・姿勢のばらつきを評価

    戻り値:
        dict(cost=..., pos_std=..., ori_std_deg=...)
    """
    robot_poses = load_robot_captures(robot_json_path)

    # 1) base->gripper リストを構成
    R_gripper2base = []
    t_gripper2base = []
    for img_path in used_image_paths:
        img_name = os.path.basename(img_path)
        T_b_g = pose_to_T_base_gripper(robot_poses[img_name], euler_order=euler_order)
        R_bg, t_bg = T_to_R_t(T_b_g)
        R_gripper2base.append(R_bg)
        t_gripper2base.append(t_bg)

    # 2) Hand-Eye
    R_cam2gripper, t_cam2gripper = run_handeye(
        R_gripper2base,
        t_gripper2base,
        rvecs,
        tvecs,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # camera->gripper を 4x4 に
    T_g_c = np.eye(4)
    T_g_c[:3, :3] = R_cam2gripper
    T_g_c[:3, 3] = t_cam2gripper[:, 0]
    T_c_g = np.linalg.inv(T_g_c)

    # 3) 各画像で base->board を計算
    board_positions = []
    board_normals = []

    for i, img_path in enumerate(used_image_paths):
        img_name = os.path.basename(img_path)

        # base->gripper
        T_b_g = pose_to_T_base_gripper(robot_poses[img_name], euler_order=euler_order)

        # base->camera
        T_b_c = T_b_g @ T_c_g

        # camera->board (calibrateCamera の rvec, tvec は object->camera)
        R_cb, _ = cv2.Rodrigues(rvecs[i])
        t_cb = tvecs[i].reshape(3, 1)

        T_c_B = np.eye(4)
        T_c_B[:3, :3] = R_cb
        T_c_B[:3, 3] = t_cb[:, 0]

        # base->board
        T_b_B = T_b_c @ T_c_B

        # board 原点位置 (base 座標系)
        p_b = T_b_B[:3, 3]
        board_positions.append(p_b)

        # board の法線ベクトル (board Z軸を base に写像)
        n_b = T_b_B[:3, :3] @ np.array([0, 0, 1], dtype=np.float64)
        board_normals.append(n_b / np.linalg.norm(n_b))

    board_positions = np.stack(board_positions, axis=0)  # (N,3)
    board_normals = np.stack(board_normals, axis=0)      # (N,3)

    # 位置ばらつき [m]
    pos_mean = board_positions.mean(axis=0)
    pos_dev = np.linalg.norm(board_positions - pos_mean, axis=1)
    pos_std = pos_dev.std()

    # 姿勢ばらつき [deg]：法線ベクトルの角度散らばり
    n_mean = board_normals.mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)
    # 各法線との角度
    dots = np.clip(np.einsum("ij,j->i", board_normals, n_mean), -1.0, 1.0)
    angles = np.arccos(dots)  # [rad]
    ori_std_deg = np.rad2deg(angles).std()

    # cost: 単純に加重和
    cost = pos_std + 0.1 * np.deg2rad(ori_std_deg)  # 例

    print(f"[EVAL] order={euler_order}: pos_std={pos_std*1000:.2f} mm, "
          f"ori_std={ori_std_deg:.3f} deg, cost={cost:.6f}")

    return dict(
        cost=cost,
        pos_std=pos_std,
        ori_std_deg=ori_std_deg,
        R_cam2gripper=R_cam2gripper,
        t_cam2gripper=t_cam2gripper,
    )



def T_to_R_t(T):
    """4x4 同次変換から (R, t) を抽出"""
    R = T[:3, :3].copy()
    t = T[:3, 3].reshape(3, 1).copy()
    return R, t


# ---------------------------------------------------------------
# Charuco ボード & 検出
# ---------------------------------------------------------------

def create_charuco_board():
    """
    今回の実験条件に合わせた Charuco ボードオブジェクトを生成。

    - squaresX = 7, squaresY = 5
    - squareLength = 32 mm
    - markerLength = 24 mm
    - dictionary = 4x4_50
    """
    squares_x = 7
    squares_y = 5
    square_length_m = 32.0e-3  # 32 mm -> m
    marker_length_m = 24.0e-3  # 24 mm -> m

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_m,
        marker_length_m,
        dictionary,
    )
    return board, dictionary


def load_robot_captures(json_path):
    """
    JSON ファイルから captures を読み込み、
    画像ファイル名 (basename) -> pose_dict の dict を返す。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for cap in data["captures"]:
        img_name = os.path.basename(cap["image_file"])
        mapping[img_name] = cap["pose"]
    return mapping


def collect_charuco_and_robot_pairs(
    image_dir,
    json_path,
    min_corners=15,
    show_detections=False,
):
    """
    Charuco 検出と robot pose をペアにして収集。

    Returns:
        obj_points_all: list of (Ni x 3) float32, board座標系の3D点
        img_points_all: list of (Ni x 2) float32, 画像上のコーナ座標
        R_gripper2base: list of (3x3) float64
        t_gripper2base: list of (3x1) float64
        image_size: (width, height)
        used_image_paths: list of 使われた画像のフルパス
    """
    board, dictionary = create_charuco_board()
    robot_poses = load_robot_captures(json_path)

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"[INFO] Found {len(image_paths)} images")

    obj_points_all = []
    img_points_all = []
    R_gripper2base = []
    t_gripper2base = []
    used_image_paths = []

    image_size = None

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        if img_name not in robot_poses:
            print(f"[WARN] No robot pose for image {img_name}, skip")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read {img_path}, skip")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # ArUco 検出
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) == 0:
            print(f"[WARN] No markers detected in {img_name}, skip")
            continue

        # Charuco コーナ補間
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
        )

        if charuco_ids is None or len(charuco_ids) < min_corners:
            print(
                f"[WARN] Not enough Charuco corners in {img_name}: "
                f"{0 if charuco_ids is None else len(charuco_ids)} < {min_corners}, skip"
            )
            continue

        # Charuco コーナを描画して確認（任意）
        if show_detections:
            vis = img.copy()
            aruco.drawDetectedMarkers(vis, corners, ids)
            aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            cv2.imshow("charuco_detection", vis)
            key = cv2.waitKey(10)
            if key == 27:  # ESC で停止
                show_detections = False
                cv2.destroyWindow("charuco_detection")

        # board.chessboardCorners は (N x 3) の 3D 座標（ボード座標系）
        corners3d_all = board.getChessboardCorners()  # list of Point3f
        # numpy 配列化するなら：
        corners3d_arr = np.array(corners3d_all, dtype=np.float32)  # shape (N,3)
        obj_points = corners3d_arr[charuco_ids.flatten(), :]
        img_points = charuco_corners.reshape(-1, 2)

        obj_points_all.append(obj_points.astype(np.float32))
        img_points_all.append(img_points.astype(np.float32))

        # ロボット base->gripper 同次変換
        T_b_g = pose_to_T_base_gripper(robot_poses[img_name])
        R_bg, t_bg = T_to_R_t(T_b_g)

        # calibrateHandEye の仕様上、R_gripper2base = R_bg, t_gripper2base = t_bg
        # （グリッパ座標で表現された点を base 座標に写像する変換）
        R_gripper2base.append(R_bg)
        t_gripper2base.append(t_bg)

        used_image_paths.append(img_path)

    if len(obj_points_all) < 3:
        raise RuntimeError("Not enough valid Charuco+robot pairs for calibration (need >= 3)")

    print(f"[INFO] Using {len(obj_points_all)} image/pose pairs for calibration")
    return (
        obj_points_all,
        img_points_all,
        R_gripper2base,
        t_gripper2base,
        image_size,
        used_image_paths,
    )


# ---------------------------------------------------------------
# カメラ内部パラメータ推定 & Hand-Eye
# ---------------------------------------------------------------

def calibrate_intrinsics(obj_points_all, img_points_all, image_size):
    """
    Charuco コーナ群からカメラ行列 K と歪み dist を推定。
    OpenCV の標準関数 calibrateCamera を利用する。
    """
    print("[INFO] Running cv2.calibrateCamera ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points_all,
        imagePoints=img_points_all,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    print(f"[INFO] Calibration RMS reprojection error: {rms:.4f} px")
    print("[INFO] Camera matrix K:")
    print(K)
    print("[INFO] Distortion coefficients:")
    print(dist.ravel())
    return rms, K, dist, rvecs, tvecs


def run_handeye(
    R_gripper2base,
    t_gripper2base,
    rvecs_board2cam,
    tvecs_board2cam,
    method=cv2.CALIB_HAND_EYE_TSAI,
):
    """
    OpenCV の calibrateHandEye を用いて Hand-Eye を解く。

    入力:
        - R_gripper2base, t_gripper2base:
            base 座標系に対する gripper の姿勢 ( ^bT_g )
        - rvecs_board2cam, tvecs_board2cam:
            board 座標系から camera 座標系への変換 ( ^cT_t )
            calibrateCamera の rvecs, tvecs をそのまま利用

    出力:
        - R_cam2gripper, t_cam2gripper:
            camera -> gripper ( ^gT_c )
    """
    R_target2cam = []
    t_target2cam = []
    for rvec, tvec in zip(rvecs_board2cam, tvecs_board2cam):
        R_tc, _ = cv2.Rodrigues(rvec)  # target(board) -> cam
        R_target2cam.append(R_tc)
        t_target2cam.append(tvec.reshape(3, 1))

    print("[INFO] Running cv2.calibrateHandEye ...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )

    print("[INFO] Hand-Eye result (camera -> gripper):")
    print("R_cam2gripper =")
    print(R_cam2gripper)
    print("t_cam2gripper (m) =")
    print(t_cam2gripper.ravel())
    return R_cam2gripper, t_cam2gripper


def triangulate_n_views(normalized_points, camera_poses):
    """
    多視点 (>=2) からの観測から、1つの3D点 X を線形三角測量で推定する。

    Args:
        normalized_points: list of (u, v) in normalized image coordinates
                           (すでに undistort され、K^-1 で正規化された座標)
        camera_poses:      list of (R, t) for each view, base->camera
                           R: (3x3), t: (3x1), 同じ長さで normalized_points と1対1対応

    Returns:
        X: (3,) np.ndarray in base coordinate
    """

    assert len(normalized_points) == len(camera_poses)
    n = len(normalized_points)
    if n < 2:
        raise ValueError("Need at least two views for triangulation")

    # A X = 0 を解く (X は同次座標 (X,Y,Z,1))
    A = []

    for (u, v), (R, t) in zip(normalized_points, camera_poses):
        # P = [R | t], 正規化済みなので K は不要
        P = np.hstack([R, t])
        # 行ごとに
        p0 = P[0, :]
        p1 = P[1, :]
        p2 = P[2, :]

        A.append(u * p2 - p0)
        A.append(v * p2 - p1)

    A = np.stack(A, axis=0)  # (2n, 4)

    # SVD で最小固有ベクトルを取る
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]  # (4,)
    X_h /= X_h[3]

    return X_h[:3]

def reconstruct_charuco_points_in_base(
    image_dir,
    json_path,
    K,
    dist,
    R_cam2gripper,
    t_cam2gripper,
    min_corners=15,
    min_views_per_id=3,
    max_reproj_error_px=1.5,
    show_reprojection=False,
):
    """
    推定されたカメラポーズを固定利用して、Charuco コーナを
    ロボットベース座標系で再構成する。

    Returns:
        id_to_Xb: dict {charuco_id: 3D point (3,) in base frame}
        reproj_errors: list of reprojection errors [px]
        T_base_cam_list: list of 4x4 base->camera for each used image
    """
    board, dictionary = create_charuco_board()
    robot_poses = load_robot_captures(json_path)

    # camera->gripper ( ^gT_c ) を 4x4 に
    T_g_c = np.eye(4)
    T_g_c[:3, :3] = R_cam2gripper
    T_g_c[:3, 3] = t_cam2gripper[:, 0]

    # gripper->camera ( ^cT_g )
    T_c_g = np.linalg.inv(T_g_c)

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir} for reconstruction")

    # Charuco ID ごとに、(u,v) とカメラ姿勢を集める
    id_to_measurements = {}  # id -> list of (u, v, R_bc, t_bc)
    T_base_cam_list = []     # 可視化用
    used_image_paths = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        if img_name not in robot_poses:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) == 0:
            continue

        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
        )
        if charuco_ids is None or len(charuco_ids) < min_corners:
            continue

        # base->gripper
        T_b_g = pose_to_T_base_gripper(robot_poses[img_name])
        # base->camera = base->gripper * gripper->camera
        T_b_c = T_b_g @ T_c_g
        R_bc, t_bc = T_to_R_t(T_b_c)

        # 画像座標 -> 正規化座標 (undistort)
        # charuco_corners: (N,1,2)
        pts = charuco_corners.reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(pts, K, dist)  # (N,1,2), normalized

        for idx, cid in enumerate(charuco_ids.flatten()):
            u, v = undistorted[idx, 0, :]  # 正規化座標
            if cid not in id_to_measurements:
                id_to_measurements[cid] = []
            id_to_measurements[cid].append((u, v, R_bc, t_bc))

        T_base_cam_list.append(T_b_c)
        used_image_paths.append(img_path)

    # id ごとに多視点三角測量
    id_to_Xb = {}
    all_reproj_errors = []

    for cid, meas_list in id_to_measurements.items():
        if len(meas_list) < min_views_per_id:
            continue

        normalized_points = []
        camera_poses = []
        for (u, v, R_bc, t_bc) in meas_list:
            normalized_points.append((u, v))
            camera_poses.append((R_bc, t_bc))

        try:
            Xb = triangulate_n_views(normalized_points, camera_poses)
        except np.linalg.LinAlgError:
            continue

        # 再投影誤差評価
        reproj_errors = []
        for (u, v, R_bc, t_bc) in meas_list:
            # 正規化座標系で投影
            X_cam = R_bc @ Xb.reshape(3, 1) + t_bc  # 3x1
            x_proj = X_cam[0, 0] / X_cam[2, 0]
            y_proj = X_cam[1, 0] / X_cam[2, 0]
            err = np.sqrt((x_proj - u) ** 2 + (y_proj - v) ** 2)
            reproj_errors.append(err)

        rms_err = float(np.sqrt(np.mean(np.square(reproj_errors))))
        # 正規化座標での誤差 [rad] 程度なので、px に近似換算する場合は fx を掛けるなどもありだが、
        # ここでは閾値を「正規化空間の誤差」で扱う:
        # だいたい u,v ~ x/z なので、1px ~ 1/fx と見なすと、この閾値は ~1/fx オーダ。
        # fx ~ 900 のとき、1px ≈ 1/900 ≈ 0.0011 なので、ざっくり 0.002〜0.003 程度を「1〜3px 程度」と見なせる。
        if rms_err > max_reproj_error_px / K[0, 0]:
            # 外れ値ぽいのでスキップ
            continue

        id_to_Xb[cid] = Xb
        all_reproj_errors.extend(reproj_errors)

    if not id_to_Xb:
        print("[WARN] No Charuco 3D points reconstructed after filtering")
    else:
        # 正規化空間の誤差をピクセル換算して統計表示
        errs_px = np.array(all_reproj_errors) * K[0, 0]
        print(f"[INFO] Reconstructed {len(id_to_Xb)} Charuco IDs in base frame")
        print(f"[INFO] Reprojection error (px): mean={errs_px.mean():.3f}, "
              f"median={np.median(errs_px):.3f}, max={errs_px.max():.3f}")

    # 任意で、1枚目に再投影結果を描画して確認
    if show_reprojection and used_image_paths and id_to_Xb:
        img_path = used_image_paths[0]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is not None and img_name in robot_poses:
            T_b_g = pose_to_T_base_gripper(robot_poses[img_name])
            T_b_c = T_b_g @ T_c_g
            R_bc, t_bc = T_to_R_t(T_b_c)

            # base->camera 外部と K, dist から reproject
            for cid, Xb in id_to_Xb.items():
                X_cam = R_bc @ Xb.reshape(3, 1) + t_bc
                x, y, z = X_cam[:, 0]
                if z <= 0:
                    continue
                xn = x / z
                yn = y / z
                uv = np.array([[[xn, yn]]], dtype=np.float32)
                # ここで再度 distort をかけて画像座標へ
                uv_dist = cv2.projectPoints(
                    Xb.reshape(1, 3).astype(np.float32),
                    cv2.Rodrigues(R_bc)[0],
                    t_bc.astype(np.float32),
                    K,
                    dist
                )[0]
                u_px, v_px = uv_dist[0, 0, :]

                cv2.circle(img, (int(u_px), int(v_px)), 4, (0, 0, 255), -1)

            cv2.imshow("charuco_reprojection", img)
            print("[INFO] Showing reprojected Charuco points. Press any key to close.")
            cv2.waitKey(0)
            try:
                cv2.destroyWindow("charuco_reprojection")
            except cv2.error:
                cv2.destroyAllWindows()

    return id_to_Xb, all_reproj_errors, T_base_cam_list


# ---------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------

def plot_frames_3d(ax, T, length=0.05, label=None, alpha=1.0):
    """
    3D 軸フレームを描画 (T: base->frame)。
    length: 軸の長さ [m]
    """
    R = T[:3, :3]
    t = T[:3, 3]

    origin = t
    x_axis = origin + R[:, 0] * length
    y_axis = origin + R[:, 1] * length
    z_axis = origin + R[:, 2] * length

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]])
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]])
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]])

    if label is not None:
        ax.text(origin[0], origin[1], origin[2], label, fontsize=8, alpha=alpha)


def visualize_poses_3d(
    T_base_gripper_list,
    R_cam2gripper,
    t_cam2gripper,
    title="Robot & Camera Poses",
):
    """
    base 座標系における gripper および camera の軌跡を 3D で可視化。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # base frame
    T_base = np.eye(4)
    plot_frames_3d(ax, T_base, length=0.05, label="base")

    # camera->gripper ( ^gT_c )
    T_g_c = np.eye(4)
    T_g_c[:3, :3] = R_cam2gripper
    T_g_c[:3, 3] = t_cam2gripper[:, 0]

    gripper_centers = []
    camera_centers = []

    for i, T_b_g in enumerate(T_base_gripper_list):
        # gripper 軸
        plot_frames_3d(ax, T_b_g, length=0.03, label=None, alpha=0.5)

        # base->camera = base->gripper * gripper->camera
        # gripper->camera は camera->gripper の逆
        T_c_g = np.linalg.inv(T_g_c)
        T_b_c = T_b_g @ T_c_g

        gripper_centers.append(T_b_g[:3, 3])
        camera_centers.append(T_b_c[:3, 3])

    gripper_centers = np.array(gripper_centers)
    camera_centers = np.array(camera_centers)

    ax.scatter(
        gripper_centers[:, 0],
        gripper_centers[:, 1],
        gripper_centers[:, 2],
        marker="o",
        label="gripper centers",
    )
    ax.scatter(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        marker="^",
        label="camera centers",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    # 軸範囲を少し余裕持たせて設定
    all_pts = np.vstack([gripper_centers, camera_centers])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = max(maxs - mins)
    center = (mins + maxs) / 2.0
    for i, axis in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        axis(center[i] - span * 0.6, center[i] + span * 0.6)

    plt.show()


def undistort_and_show_sample(image_path, K, dist, window_name="undistorted_sample"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read {image_path} for undistort preview")
        return

    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, dist, None, newK)

    vis = np.hstack([img, undistorted])
    cv2.imshow(window_name, vis)
    print("[INFO] Showing distorted (left) vs undistorted (right). Press any key to close.")
    cv2.waitKey(0)

    # ロバストに閉じる
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        cv2.destroyAllWindows()



# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Charuco-based camera calibration + Hand-Eye calibration"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing Charuco images",
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to captures JSON with robot poses",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open OpenCV visualization windows",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    (
        obj_points_all,
        img_points_all,
        R_gripper2base,
        t_gripper2base,
        image_size,
        used_image_paths,
    ) = collect_charuco_and_robot_pairs(
        image_dir=args.image_dir,
        json_path=args.json,
        min_corners=15,
        show_detections=not args.no_show,
    )

    # 1) intrinsics
    rms, K, dist, rvecs, tvecs = calibrate_intrinsics(
        obj_points_all,
        img_points_all,
        image_size,
    )

    # --- 新規: Euler 順序の候補評価 ---
    candidate_orders = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]
    results = {}
    for order in candidate_orders:
        print(f"[INFO] Evaluating Euler order: {order}")
        res = evaluate_euler_order(
            euler_order=order,
            robot_json_path=args.json,
            used_image_paths=used_image_paths,
            K=K,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
        )
        results[order] = res

    # cost が最小の順序を採用
    best_order = min(results.keys(), key=lambda o: results[o]["cost"])
    print(f"[INFO] Best Euler order estimated from data: {best_order}")


    # 2) Hand-Eye
    R_cam2gripper, t_cam2gripper = run_handeye(
        R_gripper2base,
        t_gripper2base,
        rvecs,
        tvecs,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # 3) 3D 可視化（gripper & camera 軌跡）
    robot_poses = load_robot_captures(args.json)
    T_base_gripper_list = []
    for img_path in used_image_paths:
        img_name = os.path.basename(img_path)
        T_b_g = pose_to_T_base_gripper(robot_poses[img_name])
        T_base_gripper_list.append(T_b_g)

    visualize_poses_3d(
        T_base_gripper_list,
        R_cam2gripper,
        t_cam2gripper,
        title="Robot base, gripper, and camera trajectories",
    )

    # 4) Undistort サンプル表示（任意）
    if not args.no_show and len(used_image_paths) > 0:
        undistort_and_show_sample(used_image_paths[0], K, dist)

    # 5) 推定カメラポーズを固定利用した Charuco 3D 再構成
    print("[INFO] Reconstructing Charuco points in base frame using fixed camera poses ...")
    id_to_Xb, reproj_errors_norm, T_base_cam_list = reconstruct_charuco_points_in_base(
        image_dir=args.image_dir,
        json_path=args.json,
        K=K,
        dist=dist,
        R_cam2gripper=R_cam2gripper,
        t_cam2gripper=t_cam2gripper,
        min_corners=15,
        min_views_per_id=3,
        max_reproj_error_px=1.5,
        show_reprojection=not args.no_show,
    )

    if id_to_Xb:
        # 3D 可視化: base座標系における camera 軌跡 + Charuco 3D 点
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Charuco 3D points in robot base frame")

        # base frame
        plot_frames_3d(ax, np.eye(4), length=0.05, label="base")

        # camera centers
        cam_centers = []
        for T_b_c in T_base_cam_list:
            cam_centers.append(T_b_c[:3, 3])
            plot_frames_3d(ax, T_b_c, length=0.03, alpha=0.4)

        cam_centers = np.array(cam_centers)
        ax.scatter(
            cam_centers[:, 0],
            cam_centers[:, 1],
            cam_centers[:, 2],
            marker="^",
            label="camera centers",
        )

        # Charuco 3D points
        Xbs = np.stack(list(id_to_Xb.values()), axis=0)
        ax.scatter(
            Xbs[:, 0],
            Xbs[:, 1],
            Xbs[:, 2],
            marker="o",
            label="Charuco 3D",
        )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        # 軸範囲の調整
        all_pts = np.vstack([cam_centers, Xbs])
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = max(maxs - mins)
        center = (mins + maxs) / 2.0
        ax.set_xlim(center[0] - span * 0.6, center[0] + span * 0.6)
        ax.set_ylim(center[1] - span * 0.6, center[1] + span * 0.6)
        ax.set_zlim(center[2] - span * 0.6, center[2] + span * 0.6)

        plt.show()
    else:
        print("[WARN] No 3D Charuco points reconstructed; skipping 3D plot.")


if __name__ == "__main__":
    main()
