import cv2
import numpy as np
import json
import argparse
import glob
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 定数定義 ---
# ChArUcoボードの仕様
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LEN_MM = 32.0
MARKER_LEN_MM = 24.0
# OpenCV 4.7.0以降では cv2.aruco.DICT_4X4_50 のようにアクセス
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def read_data(image_dir, json_path):
    """画像パスとポーズ情報のリストを読み込む"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"エラー: JSONファイルが見つかりません: {json_path}")
        return None, None

    captures = data.get('captures', [])
    if not captures:
        print("エラー: JSONファイルに 'captures' データが含まれていません。")
        return None, None

    image_files = []
    poses = []
    for cap in captures:
        img_path = os.path.join(image_dir, cap['image_file'])
        if os.path.exists(img_path):
            image_files.append(img_path)
            poses.append(cap['pose'])
        else:
            print(f"警告: 画像ファイルが見つかりません: {img_path}")
            
    return image_files, poses

def pose_to_matrix(pose):
    """
    JSONから読み込んだポーズ情報を4x4同次変換行列に変換する。
    ロボットの特異な座標系を補正する。
    """
    # オイラー角 (deg) から回転行列を計算
    r = Rotation.from_euler(
        'xyz',
        [pose['rx_deg'], pose['ry_deg'], pose['rz_deg']],
        degrees=True
    )
    R_raw = r.as_matrix()
    
    # 並進ベクトル
    t_raw = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]).reshape(3, 1)

    # --- 座標系の補正 ---
    # ピッチ角0で真下を向く仕様を、Z軸が前方を向く標準的なツール座標系に補正する。
    # これは、ツールフランジに対してX軸周りに+90度回転させることに相当する。
    R_fix = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    R_std = R_raw @ R_fix
    
    # 4x4同次変換行列を作成
    T = np.identity(4)
    T[:3, :3] = R_std
    T[:3, 3] = t_raw.flatten()
    
    return T

def calibrate_camera_charuco(image_files, board):
    """
    ChArUcoボードを用いてカメラキャリブレーションを行う。
    検出に成功した画像のインデックスも返す。
    """
    all_corners = []
    all_ids = []
    all_obj_points = []
    img_size = None
    valid_indices = [] # 検出に成功した画像のインデックス

    obj_points = board.getChessboardCorners()

    print("カメラキャリブレーションを開始します...")
    for i, fname in enumerate(image_files):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告: 画像を読み込めませんでした: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if ret and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                all_obj_points.append(obj_points)
                valid_indices.append(i)
        
        print(f"\r- 画像 {i+1}/{len(image_files)} を処理中...", end="")
    print("\nマーカー検出完了。")

    if not valid_indices:
        print("エラー: どの画像からもChArUcoボードを検出できませんでした。")
        return None

    print(f"{len(valid_indices)}/{len(image_files)} 枚の画像でボードを検出しました。")
    print("キャリブレーション計算を実行中...")

    # cv2.aruco.calibrateCameraCharuco には all_obj_points ではなく、
    # ボードオブジェクトそのものを渡すことができないため、対応するオブジェクトポイントリストを作成する
    obj_points_for_calib = [obj_points for _ in range(len(all_corners))]


    # calibrateCameraCharucoExtended を使用して、より多くの情報を得る
    ret, camera_matrix, dist_coeffs, rvecs, tvecs, _, _, per_view_errors = cv2.aruco.calibrateCameraCharucoExtended(
        all_corners, all_ids, board, img_size, None, None
    )

    if not ret:
        print("エラー: カメラキャリブレーションに失敗しました。")
        return None

    mean_error = np.mean(per_view_errors)
    print(f"キャリブレーション完了。再投影誤差: {mean_error:.4f} ピクセル")
    
    return {
        "ret": ret,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "valid_indices": valid_indices
    }


def plot_axes(ax, T, scale=50.0, labels=True):
    """3Dプロットに座標軸を描画するヘルパー関数"""
    origin = T[:3, 3]
    R = T[:3, :3]
    
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=scale, color='r')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=scale, color='g')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=scale, color='b')
    
    if labels:
        ax.text(origin[0] + scale * x_axis[0], origin[1] + scale * x_axis[1], origin[2] + scale * x_axis[2], 'X')
        ax.text(origin[0] + scale * y_axis[0], origin[1] + scale * y_axis[1], origin[2] + scale * y_axis[2], 'Y')
        ax.text(origin[0] + scale * z_axis[0], origin[1] + scale * z_axis[1], origin[2] + scale * z_axis[2], 'Z')


def main(image_dir, json_path):
    """メインの処理フロー"""
    
    # --- 1. データ読み込みとボード定義 ---
    print("--- ステップ1: データ読み込み ---")
    image_files, all_poses = read_data(image_dir, json_path)
    if not image_files or not all_poses:
        return
        
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LEN_MM,
        MARKER_LEN_MM,
        ARUCO_DICT
    )
    
    # --- 2. カメラキャリブレーション ---
    print("\n--- ステップ2: カメラキャリブレーション ---")
    calib_result = calibrate_camera_charuco(image_files, board)
    if calib_result is None:
        return
        
    camera_matrix = calib_result["camera_matrix"]
    dist_coeffs = calib_result["dist_coeffs"]
    rvecs_cam_target = calib_result["rvecs"] # カメラから見たターゲットの回転
    tvecs_cam_target = calib_result["tvecs"] # カメラから見たターゲットの並進
    valid_indices = calib_result["valid_indices"]

    print("\n[結果] カメラ内部パラメータ:")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # --- 3. Hand-Eyeキャリブレーションの準備 ---
    print("\n--- ステップ3: Hand-Eyeキャリブレーション ---")
    # マーカーが検出できた画像のポーズのみを使用
    valid_poses = [all_poses[i] for i in valid_indices]

    # ロボットベースからグリッパーへの変換行列リストを作成
    T_base_grippers = [pose_to_matrix(p) for p in valid_poses]
    R_base_grippers = [T[:3, :3] for T in T_base_grippers]
    t_base_grippers = [T[:3, 3] for T in T_base_grippers]

    # カメラからターゲットへの変換行列リストを作成
    R_cam_targets = [cv2.Rodrigues(rvec)[0] for rvec in rvecs_cam_target]
    t_cam_targets = [tvec.flatten() for tvec in tvecs_cam_target]
    
    # --- 4. Hand-Eyeキャリブレーション実行 ---
    # OpenCVは target2cam ではなく cam2target を期待する場合があるため、逆変換も用意しておく
    # 今回のケースでは、`R_gripper2base` と `R_target2cam` の組み合わせで正しい結果が得られることが多い
    R_target_cams = [R.T for R in R_cam_targets]
    t_target_cams = [-R.T @ t for R, t in zip(R_cam_targets, t_cam_targets)]

    # `cv2.calibrateHandEye` を使用
    # メソッドは複数あるが、DANIILIDISがロバストとされる
    print("Hand-Eyeキャリブレーション計算を実行中...")
    R_gripper_cam, t_gripper_cam = cv2.calibrateHandEye(
        R_base_grippers, t_base_grippers,
        R_target_cams, t_target_cams, # ここは R_target2cam, t_target2cam を使用
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )

    # Hand-Eye変換行列 (T_gripper_cam) を作成
    T_gripper_cam = np.identity(4)
    T_gripper_cam[:3, :3] = R_gripper_cam
    T_gripper_cam[:3, 3] = t_gripper_cam.flatten()

    print("\n[結果] Hand-Eye変換行列 (T_gripper_cam):")
    print(T_gripper_cam)
    
    # ユーザー指摘のロール角反転が結果に現れているか確認
    hand_eye_euler_deg = Rotation.from_matrix(R_gripper_cam).as_euler('xyz', degrees=True)
    print(f"Hand-Eyeのオイラー角 (xyz, deg): {hand_eye_euler_deg}")
    if abs(hand_eye_euler_deg[0]) > 150:
        print("-> ロール角が約180度となっており、カメラの反転取り付けが正しく推定された可能性が高いです。")

    # --- 5. カメラポーズとボードポーズの計算 ---
    print("\n--- ステップ4: ワールド座標系でのポーズ計算 ---")
    
    # ワールド座標系でのカメラポーズを計算 (T_base_cam = T_base_gripper * T_gripper_cam)
    T_base_cams = [T_bg @ T_gripper_cam for T_bg in T_base_grippers]

    # ワールド座標系でのボードのポーズを計算 (最初の有効フレームを基準)
    # T_base_target = T_base_cam * T_cam_target
    T_cam_target_0 = np.identity(4)
    T_cam_target_0[:3,:3] = R_cam_targets[0]
    T_cam_target_0[:3,3] = t_cam_targets[0]

    T_base_target = T_base_cams[0] @ T_cam_target_0
    print("\n[結果] ワールド座標系におけるボードのポーズ (T_base_target):")
    print(T_base_target)

    # --- 6. 3D可視化 ---
    print("\n--- ステップ5: 結果の3D可視化 ---")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ワールド座標系の描画
    plot_axes(ax, np.identity(4), scale=100.0, labels=True)

    # ChArUcoボードの描画
    board_corners_3d = board.getChessboardCorners()
    board_corners_world = (T_base_target @ np.vstack((board_corners_3d.T, np.ones((1, board_corners_3d.shape[0])))))[:3, :].T
    ax.scatter(board_corners_world[:, 0], board_corners_world[:, 1], board_corners_world[:, 2], c='black', marker='.')
    plot_axes(ax, T_base_target, scale=50.0)

    # カメラポーズの描画
    cam_positions = []
    for T_base_cam in T_base_cams:
        plot_axes(ax, T_base_cam, scale=30.0, labels=False)
        cam_positions.append(T_base_cam[:3, 3])
    
    cam_positions = np.array(cam_positions)
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], c='cyan', marker='o', label='Camera Positions')

    # プロット範囲の設定
    all_points = np.vstack([cam_positions, board_corners_world, np.zeros((1,3))])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    center = (max_coords + min_coords) / 2
    max_range = (max_coords - min_coords).max()
    
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Hand-Eye Calibration Result Visualization')
    ax.legend()
    # アスペクト比を維持
    ax.set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Hand-Eye calibration using ChArUco board.')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing images.')
    parser.add_argument('json_path', type=str, help='Path to the JSON file with pose information.')
    
    args = parser.parse_args()
    
    main(args.image_dir, args.json_path)