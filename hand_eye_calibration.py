import cv2
import numpy as np
import json
import glob
import argparse
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# --- 定数定義 ---
# ChArUcoボードの仕様
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LEN_MM = 32.0
MARKER_LEN_MM = 24.0
# OpenCV 4.7.0以降、辞書の指定方法が変更された
# DICT_4X4_50 = cv2.aruco.DICT_4X4_50
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 3D可視化用のヘルパー関数
def draw_axis(ax, T, scale=50.0):
    """3Dプロットに同次変換行列Tの座標軸を描画する"""
    origin = T[0:3, 3]
    x_axis = T[0:3, 0] * scale
    y_axis = T[0:3, 1] * scale
    z_axis = T[0:3, 2] * scale
    
    # X軸 (赤)
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r')
    # Y軸 (緑)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g')
    # Z軸 (青)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b')

import numpy as np
from scipy.spatial.transform import Rotation

import numpy as np
from scipy.spatial.transform import Rotation

def pose_json_to_matrix(pose):
    """
    ロボットのポーズ情報 (JSON) を、Hand-Eyeキャリブレーションで
    使用可能な4x4同次変換行列に正しく変換する。
    - オイラー角の順序: ZYX
    - 並進ベクトルの座標系: Y軸とZ軸の符号を反転
    """
    # 並進ベクトル (mm)
    t_vec = np.array([
        pose['x_mm'], 
        -pose['y_mm'], # Y軸の符号を反転
        -pose['z_mm']  # Z軸の符号を反転
    ])

    # 回転 (deg)
    rx = pose['rx_deg']
    ry = pose['ry_deg']
    rz = pose['rz_deg']

    # Scipyを使用してオイラー角から回転行列を生成
    r_obj = Rotation.from_euler('zyx', [rz, ry, rx], degrees=True)
    r_mat = r_obj.as_matrix()

    # 4x4 同次変換行列を作成
    T = np.eye(4)
    T[0:3, 0:3] = r_mat
    T[0:3, 3] = t_vec
    return T

def base_pose_json_to_matrix(pose):
    """
    【ベースライン】JSONポーズを最もシンプルに変換する
    """
    t_vec = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']])
    rx = pose['rx_deg']
    ry = pose['ry_deg']
    rz = pose['rz_deg']
    r_obj = Rotation.from_euler('zyx', [rz, ry, rx], degrees=True)
    r_mat = r_obj.as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = r_mat
    T[0:3, 3] = t_vec
    return T

def visualize_robot_trajectory(json_data):
    """
    JSONデータからロボットアーム先端の軌道を可視化する。
    キャリブレーション前のデータ確認用。
    """
    print("\n--- Visualizing Robot End-Effector Trajectory (from JSON) ---")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_points = []
    for capture in json_data['captures']:
        T_base_ee = pose_json_to_matrix(capture['pose'])
        draw_axis(ax, T_base_ee, scale=50)
        all_points.append(T_base_ee[0:3, 3])
    
    all_points = np.array(all_points)

    # プロット範囲の設定
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max()
    if max_range == 0: max_range = 100 # 点が1つの場合など
    
    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Robot End-Effector Trajectory (from JSON)")
    plt.show()

def estimate_all_poses(image_files, json_data, board, camera_matrix, dist_coeffs):
    """
    全画像に対してポーズ推定を行い、Hand-Eyeキャリブレーション用のリストを作成する
    """
    json_map = {item['image_file']: item for item in json_data['captures']}
    
    R_base_to_endeffector, t_base_to_endeffector = [], []
    R_camera_to_board, t_camera_to_board = [], []
    
    print(f"\nProcessing {len(image_files)} images for pose estimation...")
    valid_count = 0
    for img_file in image_files:
        filename = img_file.split('/')[-1].split('\\')[-1]
        
        # マーカーが検出された画像に対応するJSONポーズ情報があるか確認
        if filename not in json_map:
            continue
            
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICTIONARY)

        if ids is not None and len(ids) > 4: # 十分なマーカーが検出されたか
            # ChArUcoコーナーをサブピクセル精度で検出
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            
            if retval and charuco_corners is not None and len(charuco_corners) > 4:
                # ボードのポーズを推定 (カメラ座標系)
                pose_found, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None, useExtrinsicGuess=False)

                if pose_found:
                    # ロボットのポーズを取得し、行列に変換
                    T_base_ee = pose_json_to_matrix(json_map[filename]['pose'])
                    R_base_to_endeffector.append(T_base_ee[0:3, 0:3])
                    t_base_to_endeffector.append(T_base_ee[0:3, 3])

                    # カメラから見たボードのポーズをリストに追加
                    R_cam_board, _ = cv2.Rodrigues(rvec)
                    R_camera_to_board.append(R_cam_board)
                    t_camera_to_board.append(tvec.flatten())
                    valid_count += 1
    
    print(f"Successfully estimated poses for {valid_count} images.")
    return R_base_to_endeffector, t_base_to_endeffector, R_camera_to_board, t_camera_to_board

def validate_calibration(R_base_ee_list, t_base_ee_list, R_cam_board_list, t_cam_board_list, R_ee_cam, t_ee_cam):
    """
    キャリブレーション結果を数値的に検証する（最重要）
    ロボットベースから見たボードのポーズが一定になることを確認する
    """
    print("\n--- Quantitative Validation ---")
    
    T_ee_cam = np.eye(4)
    T_ee_cam[0:3, 0:3] = R_ee_cam
    T_ee_cam[0:3, 3] = t_ee_cam.flatten()

    T_base_board_list = []

    for i in range(len(R_base_ee_list)):
        T_base_ee = np.eye(4)
        T_base_ee[0:3, 0:3] = R_base_ee_list[i]
        T_base_ee[0:3, 3] = t_base_ee_list[i]
        
        T_cam_board = np.eye(4)
        T_cam_board[0:3, 0:3] = R_cam_board_list[i]
        T_cam_board[0:3, 3] = t_cam_board_list[i]

        # T_base->board = T_base->ee * T_ee->cam * T_cam->board
        T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
        T_base_board_list.append(T_base_board)

    # 並進成分のばらつきを評価
    translations = np.array([T[0:3, 3] for T in T_base_board_list])
    mean_translation = np.mean(translations, axis=0)
    std_translation = np.std(translations, axis=0)
    
    print(f"Board position consistency (in base frame):")
    print(f"  - Mean translation (x, y, z) [mm]: {mean_translation}")
    print(f"  - Std deviation  (x, y, z) [mm]: {std_translation}")

    # 回転成分のばらつきを評価
    rotations_obj = Rotation.from_matrix([T[0:3, 0:3] for T in T_base_board_list])
    mean_rotation_obj = rotations_obj.mean()
    
    # 平均回転からの角度差を計算
    dev_angles_deg = []
    for r_obj in rotations_obj:
        # 差分回転を計算
        delta_rot = r_obj * mean_rotation_obj.inv()
        # 差分回転を軸-角度表現に変換し、角度(rad)を取得
        angle_rad = np.linalg.norm(delta_rot.as_rotvec())
        dev_angles_deg.append(np.rad2deg(angle_rad))
        
    mean_dev_angle = np.mean(dev_angles_deg)
    std_dev_angle = np.std(dev_angles_deg)

    print(f"Board orientation consistency (in base frame):")
    print(f"  - Mean angular deviation [deg]: {mean_dev_angle:.4f}")
    print(f"  - Std dev angular deviation [deg]: {std_dev_angle:.4f}")
    
    # 評価
    # 目安: 並進の標準偏差が数mm以下、回転の標準偏差が1度未満であれば良好な結果
    if np.all(std_translation < 5.0) and std_dev_angle < 1.0:
        print("\n[Validation Result]: Consistent and likely successful calibration.")
    else:
        print("\n[Validation Result]: High variance detected. Check input data, coordinate system assumptions, or calibration process.")

    return T_base_board_list

def systematic_validation(json_data, image_files, board, camera_matrix, dist_coeffs):
    """
    JSONポーズと計算カメラポーズの相対関係を比較し、座標系解釈を体系的に検証する
    """
    print("\n--- Systematic Validation: Analyzing Pose Consistency ---")

    # --- Step 1: 必要なポーズリストを準備 ---
    json_map = {item['image_file']: item for item in json_data['captures']}
    
    # 有効だった画像ファイル名と、それに対応するポーズ情報を格納
    valid_filenames = []
    T_base_ee_list = [] # JSONから読み取ったアーム先端ポーズ
    T_cam_board_list = [] # 画像から推定したボードのポーズ

    for img_file in image_files:
        filename = img_file.split('/')[-1].split('\\')[-1]
        if filename not in json_map:
            continue
        
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICTIONARY)
        if ids is None or len(ids) <= 4:
            continue
            
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)
        if not (retval and charuco_corners is not None and len(charuco_corners) > 4):
            continue
            
        pose_found, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None, useExtrinsicGuess=False)

        if pose_found:
            valid_filenames.append(filename)
            # 現在の最善の解釈でアームポーズを変換
            T_base_ee_list.append(base_pose_json_to_matrix(json_map[filename]['pose']))
            
            # カメラから見たボードのポーズ
            T_cam_board = np.eye(4)
            T_cam_board[0:3, 0:3], _ = cv2.Rodrigues(rvec)
            T_cam_board[0:3, 3] = tvec.flatten()
            T_cam_board_list.append(T_cam_board)
    
    # --- Step 2: 上記データでHand-Eyeキャリブレーションを実行 ---
    R_base_ee_list = [T[0:3, 0:3] for T in T_base_ee_list]
    t_base_ee_list = [T[0:3, 3] for T in T_base_ee_list]
    R_cam_board_list = [T[0:3, 0:3] for T in T_cam_board_list]
    t_cam_board_list = [T[0:3, 3] for T in T_cam_board_list]

    R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
        R_base_ee_list, t_base_ee_list, R_cam_board_list, t_cam_board_list, 
        method=cv2.CALIB_HAND_EYE_PARK)

    T_ee_cam = np.eye(4)
    T_ee_cam[0:3, 0:3] = R_ee_cam
    T_ee_cam[0:3, 3] = t_ee_cam.flatten()

    # --- Step 3: 各画像での相対ポーズ T_ee_cam_i を計算し、ばらつきを評価 ---
    T_ee_cam_individual_list = []
    T_base_cam_list = []

    # まず、ロボットベースから見たボードのポーズを平均化して基準とする
    T_base_board_list = []
    for i in range(len(T_base_ee_list)):
        T_base_board = T_base_ee_list[i] @ T_ee_cam @ T_cam_board_list[i]
        T_base_board_list.append(T_base_board)
    
    translations = np.array([T[0:3, 3] for T in T_base_board_list])
    rotations_obj = Rotation.from_matrix([T[0:3, 0:3] for T in T_base_board_list])
    T_base_board_avg = np.eye(4)
    T_base_board_avg[0:3, 3] = np.mean(translations, axis=0)
    T_base_board_avg[0:3, 0:3] = rotations_obj.mean().as_matrix()

    # 各視点でのカメラポーズを計算
    for i in range(len(T_base_ee_list)):
        # T_base_cam = T_base_board * (T_cam_board)^-1
        T_base_cam = T_base_board_avg @ np.linalg.inv(T_cam_board_list[i])
        T_base_cam_list.append(T_base_cam)

        # 本題：各視点での相対ポーズを計算
        # T_ee_cam_i = (T_base_ee_i)^-1 * T_base_cam
        T_ee_cam_i = np.linalg.inv(T_base_ee_list[i]) @ T_base_cam
        T_ee_cam_individual_list.append(T_ee_cam_i)
        
    # --- Step 4: 結果の定量化と分析 ---
    translations = np.array([T[0:3, 3] for T in T_ee_cam_individual_list])
    rotations_obj = Rotation.from_matrix([T[0:3, 0:3] for T in T_ee_cam_individual_list])

    mean_translation = np.mean(translations, axis=0)
    std_translation = np.std(translations, axis=0)
    
    mean_rotation_obj = rotations_obj.mean()
    dev_angles_deg = []
    for r_obj in rotations_obj:
        delta_rot = r_obj * mean_rotation_obj.inv()
        angle_rad = np.linalg.norm(delta_rot.as_rotvec())
        dev_angles_deg.append(np.rad2deg(angle_rad))
        
    mean_dev_angle = np.mean(dev_angles_deg)
    std_dev_angle = np.std(dev_angles_deg)

    print("\n[Analysis of T_endeffector_to_camera Consistency]")
    print(f"Mean Translation (x,y,z): {mean_translation}")
    print(f"Std Deviation [mm]: {std_translation} <-- THIS IS THE KEY METRIC FOR TRANSLATION")
    print(f"Mean Angular Deviation [deg]: {mean_dev_angle:.4f}")
    print(f"Std Dev Angular Deviation [deg]: {std_dev_angle:.4f} <-- THIS IS THE KEY METRIC FOR ROTATION")

    # 課題の診断
    print("\n[Diagnostic Insights]")
    if std_dev_angle > 2.0:
        print("- ROTATION interpretation is likely INCORRECT. (Euler order, sign, etc.)")
    else:
        print("- ROTATION interpretation is likely CORRECT. (StdDev < 2.0 deg)")

    if np.any(std_translation > 10.0):
        print("- TRANSLATION interpretation is likely INCORRECT. (Axis mapping, sign, etc.)")
    else:
        print("- TRANSLATION interpretation is likely CORRECT. (StdDev < 10.0 mm)")
        
    # --- Step 5: 可視化 ---
    print("\nVisualizing calculated camera poses vs. JSON end-effector poses...")
    fig = plt.figure(figsize=(12, 6))
    
    # 左: JSONのアーム先端ポーズ
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Robot End-Effector Poses (from JSON)")
    for T in T_base_ee_list:
        draw_axis(ax1, T, scale=50)
    
    # 右: 計算されたカメラポーズ
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Calculated Camera Poses")
    for T in T_base_cam_list:
        draw_axis(ax2, T, scale=50)
    
    # 範囲を揃える
    all_points = np.vstack([T[0:3, 3] for T in T_base_ee_list] + [T[0:3, 3] for T in T_base_cam_list])
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max); ax1.set_zlim(z_min, z_max)
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max); ax2.set_zlim(z_min, z_max)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    
    plt.tight_layout()
    plt.show()

def main(image_dir, json_path):
    # --- 1. データ準備 ---
    image_files = glob.glob(f"{image_dir}/*.png")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN_MM, MARKER_LEN_MM, DICTIONARY)

    # --- 2. カメラ内部パラメータのキャリブレーション ---
    print("--- Calibrating camera intrinsics ---")
    # (この部分は変更なし)
    all_charuco_corners, all_charuco_ids = [], []
    image_size = None
    # ... (カメラキャリブレーションのコード) ...
    for img_file in image_files:
        img = cv2.imread(img_file)
        if image_size is None: image_size = img.shape[:2][::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICTIONARY)
        if ids is not None:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
    ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, image_size, None, None)
    print(f"Intrinsics calibrated. Reprojection Error: {ret:.4f}")

    # --- 3. 新しい体系的検証を実行 ---
    systematic_validation(json_data, image_files, board, camera_matrix, dist_coeffs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Hand-Eye calibration.')
    parser.add_argument('image_dir', type=str, help='Directory containing ChArUco images.')
    parser.add_argument('json_path', type=str, help='Path to the JSON file with pose data.')
    
    args = parser.parse_args()
    
    main(args.image_dir, args.json_path)