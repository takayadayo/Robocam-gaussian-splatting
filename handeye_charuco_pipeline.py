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
    世界座標系の X,Y,Z 軸まわりの外因性回転 (deg) を、
    指定された順序で合成して回転行列を返す。

    order: "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"
           例: "XYZ" の場合 R = Rz(rz) * Ry(ry) * Rx(rx)
           （Yaskawa の Pose Format と整合させるなら "XYZ"）
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

    Rx_m = Rx(rx)
    Ry_m = Ry(ry)
    Rz_m = Rz(rz)

    R_map = {"X": Rx_m, "Y": Ry_m, "Z": Rz_m}

    # 外因性: fixed axes なので、右から掛けていく
    R = np.eye(3, dtype=np.float64)
    for ax in order:
        R = R_map[ax] @ R

    return R


def pose_to_T_base_gripper(pose_dict, euler_order="XYZ"):
    """
    JSONの pose から base->gripper (フランジ中心) の同次変換 T_b_g を生成。

    - x_mm, y_mm, z_mm : ベース座標系でのフランジ中心(TCP)位置 [mm]
    - rx_deg, ry_deg, rz_deg : ベース座標系 XYZ 軸まわりの外因性回転 [deg]
      （YaskawaのRECTAN/BASEポーズと整合）
    """
    x = pose_dict["x_mm"] * 1e-3  # [m]
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
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
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
        T_b_c = T_b_g @ T_g_c

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


def refine_handeye_and_board_pose_bundle_adjustment(
    K,
    dist,
    R_cam2gripper_init,
    t_cam2gripper_init,
    obj_points_all,
    img_points_all,
    robot_json_path,
    used_image_paths,
    rvecs,
    tvecs,
    euler_order="XYZ",
    max_iters=20,
    lambda_init=1e-3,
    verbose=True,
):
    """
    Hand-Eye (^gT_c) と Charuco ボードの base 座標系における姿勢 (^bT_B) を
    同時に最適化するバンドル調整処理。

    - カメラ内部パラメータ K, dist は固定
    - base->gripper はロボット JSON から取得して固定
    - 未知変数は:
        * Hand-Eye (camera->gripper) の 6 自由度
        * base->board の 6 自由度
    - 目的関数は、全画像・全 Charuco コーナに対する 2D 再投影誤差 (px) の二乗和

    引数:
        K, dist:
            カメラ内部・歪みパラメータ (calibrateCamera の結果)
        R_cam2gripper_init, t_cam2gripper_init:
            calibrateHandEye の初期解 (^gR_c, ^gt_c)
        obj_points_all, img_points_all:
            collect_charuco_and_robot_pairs() の返り値の一部。
            obj_points_all[i] : (Ni x 3) board 座標系の 3D 点
            img_points_all[i] : (Ni x 2) 画像上の 2D ピクセル座標
        robot_json_path:
            ロボット captures JSON へのパス
        used_image_paths:
            対応する画像パス列 (obj_points_all/img_points_all, rvecs/tvecs と同じ順序)
        rvecs, tvecs:
            calibrateCamera の外部パラメータ (board->camera)。初期 board 姿勢推定にのみ利用。
    戻り値:
        R_cam2gripper_opt, t_cam2gripper_opt, R_base2board_opt, t_base2board_opt
    """
    # --- 補助関数: SO(3) の平均 ---
    def average_rotations(R_list):
        """
        複数の回転行列 R_i のフレシェ平均を、
        A = sum R_i に対する最近傍の回転行列として求める。
        """
        if len(R_list) == 0:
            return np.eye(3, dtype=np.float64)
        A = np.zeros((3, 3), dtype=np.float64)
        for R in R_list:
            A += R
        U, _, Vt = np.linalg.svd(A)
        R_avg = U @ Vt
        if np.linalg.det(R_avg) < 0.0:
            U[:, -1] *= -1.0
            R_avg = U @ Vt
        return R_avg

    robot_poses = load_robot_captures(robot_json_path)

    # --- 初期 Hand-Eye から T_g_c_init を構成 ---
    rvec_cg_init, _ = cv2.Rodrigues(R_cam2gripper_init)
    T_g_c_init = np.eye(4, dtype=np.float64)
    T_g_c_init[:3, :3] = R_cam2gripper_init
    T_g_c_init[:3, 3] = t_cam2gripper_init.reshape(3)

    # --- calibrateCamera の外部パラメータとロボットポーズから、
    #     各画像の base->board を一旦計算し、その平均を board 初期値にする ---
    T_b_B_list = []
    for i, img_path in enumerate(used_image_paths):
        img_name = os.path.basename(img_path)
        if img_name not in robot_poses:
            continue

        # base->gripper
        T_b_g = pose_to_T_base_gripper(
            robot_poses[img_name],
            euler_order=euler_order,
        )

        # base->camera
        T_b_c = T_b_g @ T_g_c_init

        # camera->board (calibrateCamera の rvec, tvec は object->camera)
        R_cb, _ = cv2.Rodrigues(rvecs[i])
        t_cb = tvecs[i].reshape(3, 1)

        T_c_B = np.eye(4, dtype=np.float64)
        T_c_B[:3, :3] = R_cb
        T_c_B[:3, 3] = t_cb[:, 0]

        # base->board
        T_b_B = T_b_c @ T_c_B
        T_b_B_list.append(T_b_B)

    # T_b_B_list から base->board の初期値を平均として構成
    if len(T_b_B_list) == 0:
        # 安全側: 原点 & 単位回転
        R_bB_init = np.eye(3, dtype=np.float64)
        t_bB_init = np.zeros((3, 1), dtype=np.float64)
    else:
        R_list = [T[:3, :3] for T in T_b_B_list]
        t_list = [T[:3, 3] for T in T_b_B_list]
        R_bB_init = average_rotations(R_list)
        t_bB_init = np.mean(np.stack(t_list, axis=0), axis=0).reshape(3, 1)

    rvec_bB_init, _ = cv2.Rodrigues(R_bB_init)

    # --- パラメータベクトル [rvec_cg(3), t_cg(3), rvec_bB(3), t_bB(3)] ---
    param = np.concatenate(
        [
            rvec_cg_init.reshape(3),
            t_cam2gripper_init.reshape(3),
            rvec_bB_init.reshape(3),
            t_bB_init.reshape(3),
        ]
    ).astype(np.float64)  # (12,)

    # --- 画像ごとのデータを整理 ---
    view_data = []
    for i, img_path in enumerate(used_image_paths):
        img_name = os.path.basename(img_path)
        if img_name not in robot_poses:
            continue
        if i >= len(obj_points_all) or i >= len(img_points_all):
            continue
        obj_pts = obj_points_all[i]
        img_pts = img_points_all[i]
        if obj_pts.shape[0] == 0:
            continue
        view_data.append(
            dict(
                img_name=img_name,
                obj_pts=obj_pts.copy(),  # (Ni,3), board frame
                img_pts=img_pts.copy(),  # (Ni,2), pixels
            )
        )

    if len(view_data) == 0:
        raise RuntimeError("No valid views for bundle adjustment")

    # --- 残差関数: 全画像・全点の 2D 再投影誤差 (px) を連結 ---
    def residuals(param_vec):
        """
        現在のパラメータでの 2D 再投影誤差ベクトルを返す。
        形状: (2 * Σ Ni, )
        """
        # パラメータのデコード
        rvec_cg = param_vec[0:3].reshape(3, 1)
        t_cg = param_vec[3:6].reshape(3, 1)
        rvec_bB = param_vec[6:9].reshape(3, 1)
        t_bB = param_vec[9:12].reshape(3, 1)

        R_cg, _ = cv2.Rodrigues(rvec_cg)  # ^gR_c
        R_bB, _ = cv2.Rodrigues(rvec_bB)  # ^bR_B

        # camera->gripper を 4x4
        T_g_c = np.eye(4, dtype=np.float64)
        T_g_c[:3, :3] = R_cg
        T_g_c[:3, 3] = t_cg[:, 0]

        # base->board を 4x4
        T_b_B = np.eye(4, dtype=np.float64)
        T_b_B[:3, :3] = R_bB
        T_b_B[:3, 3] = t_bB[:, 0]

        residual_list = []

        for vd in view_data:
            img_name = vd["img_name"]
            obj_pts = vd["obj_pts"]  # (Ni,3) in board frame
            img_pts = vd["img_pts"]  # (Ni,2) in pixels

            pose_dict = robot_poses[img_name]

            # base->gripper
            T_b_g = pose_to_T_base_gripper(
                pose_dict,
                euler_order=euler_order,
            )

            # base->camera
            T_b_c = T_b_g @ T_g_c

            # world(=base)->camera
            R_c_b, t_c_b = base_to_cam_to_world_to_cam(T_b_c)

            # board->base を通じて 3D 点を base 座標系に変換
            X_B = obj_pts.astype(np.float64)               # board frame
            X_b = (R_bB @ X_B.T + t_bB).T                  # (Ni,3), base frame

            # projectPoints は「world座標系 -> camera」の変換 (R_c_b, t_c_b) と
            # world座標系上の 3D 点 (ここでは base 座標系の X_b) を受け取る
            rvec_cb, _ = cv2.Rodrigues(R_c_b)
            X_b32 = X_b.reshape(-1, 1, 3).astype(np.float32)
            uv_pred, _ = cv2.projectPoints(
                X_b32,
                rvec_cb.astype(np.float32),
                t_c_b.astype(np.float32),
                K,
                dist,
            )
            uv_pred = uv_pred.reshape(-1, 2).astype(np.float64)

            # 残差: ピクセル誤差
            res_2d = (uv_pred - img_pts.astype(np.float64)).reshape(-1)  # (2*Ni,)
            residual_list.append(res_2d)

        if len(residual_list) == 0:
            return np.zeros((0,), dtype=np.float64)

        return np.concatenate(residual_list, axis=0)

    # --- 初期コストを計算 ---
    r0 = residuals(param)
    cost0 = 0.5 * np.dot(r0, r0)
    if verbose:
        print(
            f"[BA] Initial reprojection cost = {cost0:.6e}, "
            f"RMS = {np.sqrt(2 * cost0 / r0.size):.3f} px"
        )

    # --- Levenberg-Marquardt ループ ---
    lam = lambda_init
    eps = 1e-6
    prev_cost = cost0

    for it in range(max_iters):
        r = residuals(param)
        cost = 0.5 * np.dot(r, r)
        if verbose:
            print(
                f"[BA] iter={it:02d}, cost={cost:.6e}, "
                f"RMS={np.sqrt(2 * cost / r.size):.3f} px, lambda={lam:.3e}"
            )

        m = r.size
        n = param.size
        J = np.zeros((m, n), dtype=np.float64)

        # 数値ヤコビアン（中心差分）
        for j in range(n):
            dp = np.zeros_like(param)
            dp[j] = eps
            r_p = residuals(param + dp)
            r_m = residuals(param - dp)
            J[:, j] = (r_p - r_m) / (2.0 * eps)

        JTJ = J.T @ J
        JTr = J.T @ r

        try:
            step = np.linalg.solve(JTJ + lam * np.eye(n), JTr)
        except np.linalg.LinAlgError:
            print("[BA] Singular JTJ, abort refinement.")
            break

        param_new = param - step
        r_new = residuals(param_new)
        cost_new = 0.5 * np.dot(r_new, r_new)

        if cost_new < cost:
            # 改善したので採用 & lambda を小さく
            param = param_new
            lam *= 0.5
            if abs(cost - cost_new) < 1e-9:
                if verbose:
                    print("[BA] Converged (cost change small).")
                break
            prev_cost = cost_new
        else:
            # 悪化したのでロールバック & lambda を増加
            lam *= 2.0
            if lam > 1e6:
                print("[BA] lambda too large, abort refinement.")
                break

        if np.linalg.norm(step) < 1e-8:
            if verbose:
                print("[BA] Converged (step norm small).")
            break

    # --- 最終パラメータから行列を復元 ---
    rvec_cg_opt = param[0:3].reshape(3, 1)
    t_cg_opt = param[3:6].reshape(3, 1)
    rvec_bB_opt = param[6:9].reshape(3, 1)
    t_bB_opt = param[9:12].reshape(3, 1)

    R_cg_opt, _ = cv2.Rodrigues(rvec_cg_opt)
    R_bB_opt, _ = cv2.Rodrigues(rvec_bB_opt)

    # 結果の評価として、base->board の位置・姿勢ばらつきを計算しておく
    # （board は本来 base で静止しているので、ここはほぼ 0 に近いはず）
    board_positions = []
    board_normals = []

    T_g_c_opt = np.eye(4, dtype=np.float64)
    T_g_c_opt[:3, :3] = R_cg_opt
    T_g_c_opt[:3, 3] = t_cg_opt[:, 0]

    T_b_B_opt = np.eye(4, dtype=np.float64)
    T_b_B_opt[:3, :3] = R_bB_opt
    T_b_B_opt[:3, 3] = t_bB_opt[:, 0]

    for vd in view_data:
        img_name = vd["img_name"]
        _pose_dict = robot_poses[img_name]

        # board は T_b_B_opt で固定なので、全 view で同じになる想定だが、
        # 数値安定性確認のため一応 scatter を算出しておく。
        p_b = T_b_B_opt[:3, 3]
        n_b = T_b_B_opt[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        n_b = n_b / np.linalg.norm(n_b)

        board_positions.append(p_b)
        board_normals.append(n_b)

    board_positions = np.stack(board_positions, axis=0)
    board_normals = np.stack(board_normals, axis=0)

    pos_mean = board_positions.mean(axis=0)
    pos_dev = np.linalg.norm(board_positions - pos_mean, axis=1)
    pos_std = pos_dev.std()

    n_mean = board_normals.mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)
    dots = np.clip(np.einsum("ij,j->i", board_normals, n_mean), -1.0, 1.0)
    angles = np.arccos(dots)
    ori_std_deg = np.rad2deg(angles).std()

    if verbose:
        print(
            "[BA] Final board scatter: "
            f"pos_std={pos_std*1000:.3f} mm, ori_std={ori_std_deg:.4f} deg"
        )
        print("[BA] Hand-Eye + board pose bundle adjustment done.")

    return R_cg_opt, t_cg_opt, R_bB_opt, t_bB_opt



def T_to_R_t(T):
    """4x4 同次変換から (R, t) を抽出"""
    R = T[:3, :3].copy()
    t = T[:3, 3].reshape(3, 1).copy()
    return R, t

def base_to_cam_to_world_to_cam(T_b_c):
    """
    base->camera (^bT_c) から world(=base)->camera (^cT_b) を得る。

        X_b = ^bR_c X_c + ^b t_c
        <=> X_c = ^cR_b X_b + ^c t_b

    なので ^cR_b = (^bR_c)^T,  ^c t_b = -(^bR_c)^T ^b t_c
    """
    R_b_c, t_b_c = T_to_R_t(T_b_c)  # ^bR_c, ^bt_c
    R_c_b = R_b_c.T
    t_c_b = -R_c_b @ t_b_c
    return R_c_b, t_c_b

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
    Charuco から得た 3D-2D 対応を使って、カメラ内部パラメータを推定。
    """
    obj_pts = [np.asarray(o, dtype=np.float32) for o in obj_points_all]
    img_pts = [np.asarray(i, dtype=np.float32) for i in img_points_all]

    K = np.zeros((3, 3), dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, K, dist,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    print(f"[INFO] Calibration RMS reprojection error: {rms:.4f} px")
    print("[INFO] Camera matrix K:")
    print(K)
    print("[INFO] Distortion coefficients:")
    print(dist.ravel())

    return rms, K, dist, rvecs, tvecs


def run_handeye(R_gripper2base, t_gripper2base, rvecs, tvecs, method=cv2.CALIB_HAND_EYE_DANIILIDIS):
    """
    OpenCV の Hand-Eye を呼び出す。

    - R_gripper2base[i] : ^bR_g(i)
    - t_gripper2base[i] : ^bt_g(i)
    - rvecs[i], tvecs[i]: ^cR_B(i), ^ct_B(i) （PnP: board->camera）

    戻り:
    - R_cam2gripper : ^gR_c
    - t_cam2gripper : ^gt_c
    """
    R_target2cam = []
    t_target2cam = []
    for rvec, tvec in zip(rvecs, tvecs):
        R_cb, _ = cv2.Rodrigues(rvec)   # ^cR_B
        t_cb = tvec.reshape(3, 1)       # ^ct_B
        R_target2cam.append(R_cb)
        t_target2cam.append(t_cb)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=method
    )
    t_cam2gripper = t_cam2gripper.reshape(3, 1)

    print("[INFO] Running cv2.calibrateHandEye ...")
    print("[INFO] Hand-Eye result (camera -> gripper):")
    print("R_cam2gripper =")
    print(R_cam2gripper)
    print("t_cam2gripper (m) =")
    print(t_cam2gripper.ravel())

    return R_cam2gripper, t_cam2gripper

def triangulate_n_views(norm_points, cam_poses):
    """
    多視点 (>=2) からの観測から、1つの3D点 X_b (base frame) を線形三角測量。

    Args:
        norm_points: list of (u_n, v_n) 正規化画像座標 (undistortPoints の結果)
        cam_poses:   list of (R_c_b, t_c_b) world(=base)->camera

    Returns:
        X_b: (3,) np.ndarray, base frame 座標
    """
    assert len(norm_points) == len(cam_poses)
    n = len(norm_points)
    if n < 2:
        raise ValueError("Need at least two views for triangulation")

    A = []
    for (u, v), (R_c_b, t_c_b) in zip(norm_points, cam_poses):
        # P = [R | t] with world->cam, normalized coords
        P = np.hstack([R_c_b, t_c_b])  # 3x4
        p0, p1, p2 = P[0, :], P[1, :], P[2, :]

        A.append(u * p2 - p0)
        A.append(v * p2 - p1)

    A = np.stack(A, axis=0)  # (2n, 4)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]
    X_h /= X_h[3]
    return X_h[:3]

def reconstruct_charuco_points_in_base(
    image_dir,
    json_path,
    K,
    dist,
    R_cam2gripper,
    t_cam2gripper,
    euler_order="XYZ",
    min_corners=15,
    min_views_per_id=3,
    max_reproj_error_px=4.0,   # None ならフィルタしない
    show_reprojection=True,
):
    """
    Hand-Eye で求めたカメラポーズ (^bT_c) を固定利用し、
    Charuco コーナを base 座標系で多視点再構成する。

    - JSON: base->gripper (^bT_g)
    - Hand-Eye: gripper->camera (^gT_c)
    - => base->camera (^bT_c) = ^bT_g · ^gT_c
    - 投影・三角測量は world->camera (^cT_b) を使う
    """

    # --- Charuco ボード定義（あなたの条件にあわせる） ---
    squares_x = 7
    squares_y = 5
    marker_length_m = 0.024  # 24mm
    square_length_m = 0.032  # 32mm

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_m,
        marker_length_m,
        dictionary
    )

    # --- ロボット JSON ロード ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    robot_poses = {cap["image_file"]: cap["pose"] for cap in data["captures"]}

    # --- Hand-Eye: ^gT_c ---
    T_g_c = np.eye(4, dtype=np.float64)
    T_g_c[:3, :3] = R_cam2gripper
    T_g_c[:3, 3] = t_cam2gripper[:, 0]

    # --- 画像列挙 ---
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir} for reconstruction")

    # 各 Charuco ID ごとに観測集約
    # id -> list of dict{u_px,v_px,u_n,v_n,R_c_b,t_c_b}
    id_to_meas = {}

    T_b_c_list = []     # base->camera
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
        T_b_g = pose_to_T_base_gripper(robot_poses[img_name], euler_order=euler_order)
        # base->camera
        T_b_c = T_b_g @ T_g_c
        T_b_c_list.append(T_b_c)
        used_image_paths.append(img_path)

        # world->camera (base->cam の逆)
        R_c_b, t_c_b = base_to_cam_to_world_to_cam(T_b_c)

        # Charuco 2D 座標（画素 & 正規化）
        pts_px = charuco_corners.reshape(-1, 1, 2).astype(np.float32)  # (N,1,2)
        pts_norm = cv2.undistortPoints(pts_px, K, dist)                # (N,1,2)

        for idx, cid in enumerate(charuco_ids.flatten()):
            u_px, v_px = pts_px[idx, 0, :]
            u_n, v_n = pts_norm[idx, 0, :]

            cid_int = int(cid)
            if cid_int not in id_to_meas:
                id_to_meas[cid_int] = []
            id_to_meas[cid_int].append(
                {
                    "u_px": float(u_px),
                    "v_px": float(v_px),
                    "u_n": float(u_n),
                    "v_n": float(v_n),
                    "R_c_b": R_c_b,
                    "t_c_b": t_c_b,
                }
            )

    # --- ID ごとに多視点三角測量 ---
    id_to_Xb = {}
    all_err_px = []

    for cid, meas_list in id_to_meas.items():
        if len(meas_list) < min_views_per_id:
            continue

        # --- まず全ビューを使って初期三角測量 ---
        norm_points_all = []
        cam_poses_all = []
        for m in meas_list:
            norm_points_all.append((m["u_n"], m["v_n"]))
            cam_poses_all.append((m["R_c_b"], m["t_c_b"]))

        try:
            X_b = triangulate_n_views(norm_points_all, cam_poses_all)
        except np.linalg.LinAlgError:
            continue

        def compute_per_view_errors(X_b, meas_list_local):
            """1つの 3D 点 X_b に対する、各ビューの再投影誤差 [px] を返す。"""
            errs = []
            for m in meas_list_local:
                R_c_b = m["R_c_b"]
                t_c_b = m["t_c_b"]
                u_px_obs = m["u_px"]
                v_px_obs = m["v_px"]

                # world(=base)->cam を rvec,tvec に変換
                rvec, _ = cv2.Rodrigues(R_c_b)
                X_obj = X_b.reshape(1, 3).astype(np.float32)

                uv_proj, _ = cv2.projectPoints(
                    X_obj,
                    rvec.astype(np.float32),
                    t_c_b.astype(np.float32),
                    K,
                    dist,
                )
                u_px_pred, v_px_pred = uv_proj[0, 0, :]

                err = float(np.hypot(u_px_pred - u_px_obs, v_px_pred - v_px_obs))
                errs.append(err)
            return np.array(errs, dtype=np.float32)

        # --- 初期解に対する per-view 誤差 ---
        per_view_err = compute_per_view_errors(X_b, meas_list)
        rms_px = float(np.sqrt(np.mean(per_view_err ** 2)))

        # --- ビュー単位の外れ値除去＋再三角測量 ---
        if (max_reproj_error_px is not None):
            inlier_mask = per_view_err <= max_reproj_error_px

            # inlier が少なすぎる場合はこの ID を破棄
            if np.count_nonzero(inlier_mask) < min_views_per_id:
                continue

            # inlier のみで再三角測量
            norm_points_in = [
                (m["u_n"], m["v_n"])
                for m, keep in zip(meas_list, inlier_mask) if keep
            ]
            cam_poses_in = [
                (m["R_c_b"], m["t_c_b"])
                for m, keep in zip(meas_list, inlier_mask) if keep
            ]

            try:
                X_b = triangulate_n_views(norm_points_in, cam_poses_in)
            except np.linalg.LinAlgError:
                continue

            # 最終的な誤差を inlier のみで再計算
            meas_list_in = [m for m, keep in zip(meas_list, inlier_mask) if keep]
            per_view_err = compute_per_view_errors(X_b, meas_list_in)

        # --- ID 採用 ---
        id_to_Xb[cid] = X_b
        all_err_px.extend(per_view_err.tolist())


    if not id_to_Xb:
        print("[WARN] No Charuco 3D points reconstructed after filtering")
    else:
        errs = np.array(all_err_px)
        print(f"[INFO] Reconstructed {len(id_to_Xb)} Charuco IDs in base frame")
        print(
            "[INFO] Reprojection error (px): "
            f"mean={errs.mean():.3f}, median={np.median(errs):.3f}, max={errs.max():.3f}"
        )

    # --- 任意の1枚で再投影を可視化（観測点と対応を描画） ---
    if show_reprojection and used_image_paths and id_to_Xb:
        # 1番目に使われた画像を可視化対象とする
        img_path = used_image_paths[0]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is not None and img_name in robot_poses:
            # base->gripper, base->camera, world->camera を再計算
            T_b_g = pose_to_T_base_gripper(
                robot_poses[img_name],
                euler_order=euler_order,
            )
            T_b_c = T_b_g @ T_g_c              # ^bT_c (cam -> base)
            R_c_b, t_c_b = base_to_cam_to_world_to_cam(T_b_c)  # ^cR_b, ^ct_b

            # この画像で観測された Charuco corner を (id -> (u_px,v_px)) にまとめる
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)
            if ids is not None and len(ids) > 0:
                _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=board,
                )
            else:
                charuco_corners, charuco_ids = None, None

            obs2d = {}
            if charuco_ids is not None:
                pts_px = charuco_corners.reshape(-1, 2)
                for idx, cid in enumerate(charuco_ids.flatten()):
                    obs2d[int(cid)] = (
                        float(pts_px[idx, 0]),
                        float(pts_px[idx, 1]),
                    )

            # 観測点（緑）と再投影点（赤）を描く
            rvec, _ = cv2.Rodrigues(R_c_b)
            for cid, X_b in id_to_Xb.items():
                X_obj = X_b.reshape(1, 3).astype(np.float32)
                uv_proj, _ = cv2.projectPoints(
                    X_obj,
                    rvec.astype(np.float32),
                    t_c_b.astype(np.float32),
                    K,
                    dist,
                )
                u_px_pred, v_px_pred = uv_proj[0, 0, :]

                # 再投影点（赤）
                cv2.circle(
                    img,
                    (int(round(u_px_pred)), int(round(v_px_pred))),
                    4,
                    (0, 0, 255),
                    -1,
                )

                # 観測点がこの画像にも存在するなら（緑）
                if cid in obs2d:
                    u_px_obs, v_px_obs = obs2d[cid]
                    cv2.circle(
                        img,
                        (int(round(u_px_obs)), int(round(v_px_obs))),
                        4,
                        (0, 255, 0),
                        -1,
                    )
                    # 緑→赤の線を引いてズレを視覚化
                    cv2.line(
                        img,
                        (int(round(u_px_obs)), int(round(v_px_obs))),
                        (int(round(u_px_pred)), int(round(v_px_pred))),
                        (0, 255, 255),
                        1,
                    )

            cv2.imshow("charuco_reprojection", img)
            print("[INFO] Showing observed (green) vs reprojected (red) Charuco points. "
                  "Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return id_to_Xb, T_b_c_list, used_image_paths

def rotation_matrix_to_quaternion(R):
    """
    3x3 回転行列 -> (qw,qx,qy,qz)  [Hamilton convention, COLMAP互換]
    """
    # 数値安定性を考慮した標準実装
    m00, m01, m02 = R[0, :]
    m10, m11, m12 = R[1, :]
    m20, m21, m22 = R[2, :]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz], dtype=np.float64)

def export_colmap_cameras(K, image_size, out_path, camera_id=1, model="PINHOLE"):
    """
    cameras.txt を COLMAP 形式で出力する。
    3DGS で推奨される PINHOLE モデル (fx, fy, cx, cy) をデフォルトとする。
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    width, height = image_size

    # ディレクトリが無ければ作成
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

        if model == "PINHOLE":
            # params: fx, fy, cx, cy
            f.write(f"{camera_id} {model} {width} {height} "
                    f"{fx:.16f} {fy:.16f} {cx:.16f} {cy:.16f}\n")
        elif model == "SIMPLE_PINHOLE":
            # params: f, cx, cy
            f_avg = (fx + fy) / 2.0
            f.write(f"{camera_id} {model} {width} {height} "
                    f"{f_avg:.16f} {cx:.16f} {cy:.16f}\n")
        else:
            print(f"[WARN] Model {model} not fully supported in this simple exporter, using PINHOLE fallback.")
            f.write(f"{camera_id} PINHOLE {width} {height} "
                    f"{fx:.16f} {fy:.16f} {cx:.16f} {cy:.16f}\n")

    print(f"[INFO] Wrote COLMAP cameras.txt to: {out_path}")

def export_colmap_images(
    T_b_c_list,
    image_paths,
    out_path,
    camera_id=1,
):
    """
    Hand-Eye + ロボットから得た base->camera (^bT_c) の列を
    COLMAP の images.txt 形式で書き出す。

    world は base 座標系とみなす。

    各行:
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
        <空行>
    """
    assert len(T_b_c_list) == len(image_paths)
    
    # ディレクトリが無ければ作成
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (T_b_c, img_path) in enumerate(zip(T_b_c_list, image_paths), start=1):
            R_c_b, t_c_b = base_to_cam_to_world_to_cam(T_b_c)  # world->cam

            q = rotation_matrix_to_quaternion(R_c_b)
            qw, qx, qy, qz = q
            tx, ty, tz = t_c_b.flatten().tolist()

            img_name = os.path.basename(img_path)

            f.write(
                f"{i} {qw:.16f} {qx:.16f} {qy:.16f} {qz:.16f} "
                f"{tx:.16f} {ty:.16f} {tz:.16f} {camera_id} {img_name}\n"
            )
            f.write("\n")

    print(f"[INFO] Wrote COLMAP images.txt to: {out_path}")


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
        T_b_c = T_b_g @ T_g_c

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
    # args.image_dir, args.json, args.colmap_images_out など

    # 1) Charuco + ロボットで対応抽出
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

    # 2) intrinsics
    rms, K, dist, rvecs, tvecs = calibrate_intrinsics(
        obj_points_all,
        img_points_all,
        image_size,
    )

    # 3) Hand-Eye (XYZ オーダで十分良いことは既に確認済み) を初期解として取得
    R_cam2gripper, t_cam2gripper = run_handeye(
        R_gripper2base,
        t_gripper2base,
        rvecs,
        tvecs,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )

    # 3') Hand-Eye + board pose に対するバンドル調整
    (
        R_cam2gripper_refined,
        t_cam2gripper_refined,
        R_base2board,
        t_base2board,
    ) = refine_handeye_and_board_pose_bundle_adjustment(
        K=K,
        dist=dist,
        R_cam2gripper_init=R_cam2gripper,
        t_cam2gripper_init=t_cam2gripper,
        obj_points_all=obj_points_all,
        img_points_all=img_points_all,
        robot_json_path=args.json,
        used_image_paths=used_image_paths,
        rvecs=rvecs,
        tvecs=tvecs,
        euler_order="XYZ",
        max_iters=20,
        lambda_init=1e-3,
        verbose=True,
    )

    # 以降の処理では BA 後の Hand-Eye を使用
    R_cam2gripper = R_cam2gripper_refined
    t_cam2gripper = t_cam2gripper_refined

    # 4) カメラポーズ固定利用の Charuco 再構成
    id_to_Xb, T_b_c_list, used_image_paths = reconstruct_charuco_points_in_base(
        image_dir=args.image_dir,
        json_path=args.json,
        K=K,
        dist=dist,
        R_cam2gripper=R_cam2gripper,
        t_cam2gripper=t_cam2gripper,
        euler_order="XYZ",
        min_corners=15,
        min_views_per_id=3,
        max_reproj_error_px=4.0,  # 例: 3.0
        show_reprojection=not args.no_show,
    )


    # 5) COLMAP 出力 (cameras.txt, images.txt)
    output_dir = 'output'  # 出力先ディレクトリ
    colmap_images_out = os.path.join(output_dir, 'images.txt')
    colmap_cameras_out = os.path.join(output_dir, 'cameras.txt')

    if colmap_images_out is not None:
        # images.txt 出力
        export_colmap_images(
            T_b_c_list=T_b_c_list,
            image_paths=used_image_paths,
            out_path=colmap_images_out,
            camera_id=1,
        )

    if colmap_cameras_out is not None:
        # cameras.txt 出力 (PINHOLE モデル)
        export_colmap_cameras(
            K=K,
            image_size=image_size,
            out_path=colmap_cameras_out,
            camera_id=1,
            model="PINHOLE"
        )

# -----可視化ブロック------

    # 1) Undistort サンプル表示（任意）
    if not args.no_show and len(used_image_paths) > 0:
        undistort_and_show_sample(used_image_paths[0], K, dist)

    # 2) 3D 可視化（gripper & camera 軌跡）
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

    # 3) 3D 可視化: base座標系における camera 軌跡 + Charuco 3D 点
    if id_to_Xb:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Charuco 3D points in robot base frame")

        # base frame
        plot_frames_3d(ax, np.eye(4), length=0.05, label="base")

        # camera centers
        cam_centers = []
        for T_b_c in T_b_c_list:
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
        print("[warn] No 3D charuco points reconstructed; skipping 3D plot")

if __name__ == "__main__":
    main()
