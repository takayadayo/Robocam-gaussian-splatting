import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# -----------------------------------------------------------------------------
# ChArUcoボードの物理パラメータ (※ご自身のボードに合わせて要変更)
# -----------------------------------------------------------------------------
# ボードのチェッカー柄の、内側の頂点数を指定します
# 例: 7x5のマス目があるボードの場合、内側の頂点数は (7-1) x (5-1) = 6x4 となります。
# しかし、cv2.aruco.CharucoBoardはマス目の数を引数に取るため、ここではマス目を指定します。
SQUARES_X = 7        # ボードのX方向のマス数
SQUARES_Y = 5        # ボードのY方向のマス数
SQUARE_LENGTH = 0.032 # 正方形の一辺の長さ (メートル単位)
MARKER_LENGTH = 0.024 # マーカーの一辺の長さ (メートル単位)

# 使用するArUcoマーカーの辞書を指定します (例: DICT_5X5_250)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# -----------------------------------------------------------------------------


# --- 座標系変換のための定義 ---
# ロボットベース座標系（ユーザー定義、左手系）とOpenCV標準座標系（右手系）の間の変換を行います。
# ユーザー定義: X:奥, Y:左, Z:上
# OpenCV標準  : X:右, Y:下, Z:奥
#
# 変換ルール:
#   x_cv = -y_user
#   y_cv = -z_user
#   z_cv =  x_user
#
# この変換を行うための3x3回転行列 C
C = np.array([
    [0., -1.,  0.],
    [0.,  0., -1.],
    [1.,  0.,  0.]
])

def create_charuco_board():
    """ChArUcoボードオブジェクトを作成して返す"""
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), 
        SQUARE_LENGTH, 
        MARKER_LENGTH, 
        ARUCO_DICT
    )

def parse_robot_pose_to_opencv_h_matrix(pose_data):
    """
    JSONから読み取ったユーザー座標系のポーズ情報を、
    OpenCV座標系基準の4x4同次変換行列に変換する。
    オイラー角は内部で回転行列に変換されるため、ロボットの非直感的な
    数値変動（例：ロール角のジャンプ）があっても物理的に等価な姿勢として正しく扱われます。
    """
    # 単位をmmからメートルに変換
    t_user = np.array([
        pose_data['x_mm'] / 1000.0,
        pose_data['y_mm'] / 1000.0,
        pose_data['z_mm'] / 1000.0
    ])
    # 回転をユーザー座標系での回転行列に変換 (オイラー角順序: XYZ)
    r_user_deg = [pose_data['rx_deg'], pose_data['ry_deg'], pose_data['rz_deg']]
    R_user = Rotation.from_euler('xyz', r_user_deg, degrees=True).as_matrix()

    # ユーザー座標系 -> OpenCV座標系 への変換
    # ベクトル変換: t_cv = C @ t_user
    # 回転行列変換: R_cv = C @ R_user @ C.T
    t_cv = C @ t_user
    R_cv = C @ R_user @ C.T

    # 4x4の同次変換行列を作成
    H_cv = np.eye(4)
    H_cv[:3, :3] = R_cv
    H_cv[:3, 3] = t_cv
    return H_cv

def rvec_tvec_to_h_matrix(rvec, tvec):
    """OpenCVから出力される回転ベクトル(rvec)と並進ベクトル(tvec)を
       4x4同次変換行列に変換する"""
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()
    return H

def plot_poses(camera_poses_user, board_pose_user=None):
    """推定されたカメラポーズとボードポーズをユーザー座標系で3D可視化する"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 座標軸を描画するヘルパー関数
    def draw_axis(ax, H, scale=0.1, label_prefix=""):
        origin = H[:3, 3]
        x_axis, y_axis, z_axis = H[:3, 0], H[:3, 1], H[:3, 2]
        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=scale, color='r')
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=scale, color='g')
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=scale, color='b')
        ax.text(origin[0], origin[1], origin[2], label_prefix, color='k')

    # ワールド座標系（ロボットベース）の原点を描画
    draw_axis(ax, np.eye(4), scale=0.2, label_prefix="World (Base)")

    # カメラポーズを描画
    positions = []
    for i, H_cam in enumerate(camera_poses_user):
        draw_axis(ax, H_cam, scale=0.05)
        positions.append(H_cam[:3, 3])
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', color='c', markersize=3, label='Camera Path')

    # ChArUcoボードのポーズを描画
    if board_pose_user is not None:
        draw_axis(ax, board_pose_user, scale=0.1, label_prefix="Board")
        # ボードの外形も描画
        board_corners_local = np.array([
            [0, 0, 0], [SQUARES_X * SQUARE_LENGTH, 0, 0],
            [SQUARES_X * SQUARE_LENGTH, SQUARES_Y * SQUARE_LENGTH, 0],
            [0, SQUARES_Y * SQUARE_LENGTH, 0], [0, 0, 0]
        ])
        R_board, t_board = board_pose_user[:3, :3], board_pose_user[:3, 3]
        board_corners_world = (R_board @ board_corners_local.T).T + t_board
        ax.plot(board_corners_world[:, 0], board_corners_world[:, 1], board_corners_world[:, 2], color='m', label='ChArUco Board')

    # プロット範囲とラベルをユーザー座標系に合わせて設定
    all_points = positions
    if board_pose_user is not None:
        all_points = np.vstack([positions, board_pose_user[:3, 3].reshape(1, 3)])
    
    max_range = np.array([p.max() - p.min() for p in all_points.T]).max() / 2.0
    mid = np.mean(all_points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X (奥) [m]")
    ax.set_ylabel("Y (左) [m]")
    ax.set_zlabel("Z (上) [m]")
    ax.set_title("Estimated Camera Poses in User Coordinate System")
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def main(args):
    """メイン処理"""
    board = create_charuco_board()
    image_dir = Path(args.images)

    # 1. JSONファイルと画像を読み込み、マーカーを検出
    with open(args.json, 'r') as f:
        data = json.load(f)

    all_corners, all_ids, all_H_base_to_ee_cv = [], [], []
    img_size = None

    print(f"--- Step 1: Detecting ChArUco markers in {len(data['captures'])} images ---")
    for capture in data['captures']:
        img_path = image_dir / capture['image_file']
        if not img_path.exists():
            print(f"  [Warning] Image file not found, skipping: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if ids is not None and len(ids) > 4: # 安定のため、最低5マーカー検出を条件とする
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret and len(charuco_corners) > 4:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                H_base_to_ee_cv = parse_robot_pose_to_opencv_h_matrix(capture['pose'])
                all_H_base_to_ee_cv.append(H_base_to_ee_cv)

    print(f"  => Found markers in {len(all_corners)} images.")
    if len(all_corners) < 10: # Hand-Eyeのためには最低10枚以上を推奨
        print("[Error] Not enough valid images to perform reliable calibration. Exiting.")
        return

    # 2. カメラの内部パラメータをキャリブレーション
    print("\n--- Step 2: Calibrating camera intrinsics ---")
    ret, camera_matrix, dist_coeffs, rvecs_marker_to_cam, tvecs_marker_to_cam = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None)
    
    print("  Camera Matrix (fx, fy, cx, cy):")
    print(camera_matrix)
    print("\n  Distortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist_coeffs.flatten())

    # 3. Hand-Eyeキャリブレーション
    print("\n--- Step 3: Performing Hand-Eye calibration ---")
    R_base_to_ee_list = [H[:3, :3] for H in all_H_base_to_ee_cv]
    t_base_to_ee_list = [H[:3, 3] for H in all_H_base_to_ee_cv]

    # マーカー→カメラのポーズを、カメラ→マーカーのポーズに変換（逆変換）
    R_cam_to_marker_list, t_cam_to_marker_list = [], []
    for rvec, tvec in zip(rvecs_marker_to_cam, tvecs_marker_to_cam):
        H_marker_to_cam = rvec_tvec_to_h_matrix(rvec, tvec)
        H_cam_to_marker = np.linalg.inv(H_marker_to_cam)
        R_cam_to_marker_list.append(H_cam_to_marker[:3, :3])
        t_cam_to_marker_list.append(H_cam_to_marker[:3, 3])

    # Hand-Eyeキャリブレーション実行 (Park & Martin法)
    R_ee_to_cam, t_ee_to_cam = cv2.calibrateHandEye(
        R_base_to_ee_list, t_base_to_ee_list,
        R_cam_to_marker_list, t_cam_to_marker_list,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    H_ee_to_cam_cv = np.eye(4)
    H_ee_to_cam_cv[:3, :3] = R_ee_to_cam
    H_ee_to_cam_cv[:3, 3] = t_ee_to_cam.flatten()
    print("  Hand-Eye Transformation Matrix (End-Effector to Camera, OpenCV coords):")
    print(H_ee_to_cam_cv)

    # 4. 各時点でのカメラポーズを算出
    print("\n--- Step 4: Calculating camera poses for each image ---")
    H_base_to_cam_cv_list = [H_b2e @ H_ee_to_cam_cv for H_b2e in all_H_base_to_ee_cv]
    print(f"  => Calculated {len(H_base_to_cam_cv_list)} camera poses.")

    # 5. 可視化のために結果をユーザー座標系に変換
    print("\n--- Step 5: Converting results to user coordinate system and visualizing ---")
    C_inv = C.T # OpenCV -> User への変換は C の転置行列
    
    H_base_to_cam_user_list = []
    for H_cv in H_base_to_cam_cv_list:
        R_cv, t_cv = H_cv[:3, :3], H_cv[:3, 3]
        R_user = C_inv @ R_cv @ C # 回転行列の座標系変換
        t_user = C_inv @ t_cv     # ベクトルの座標系変換
        H_user = np.eye(4)
        H_user[:3, :3], H_user[:3, 3] = R_user, t_user
        H_base_to_cam_user_list.append(H_user)

    # ボードのワールドポーズも計算（最初のカメラポーズを基準とする）
    H_cam_to_marker_1 = np.linalg.inv(rvec_tvec_to_h_matrix(rvecs_marker_to_cam[0], tvecs_marker_to_cam[0]))
    H_base_to_marker_cv = H_base_to_cam_cv_list[0] @ H_cam_to_marker_1
    
    R_board_cv, t_board_cv = H_base_to_marker_cv[:3, :3], H_base_to_marker_cv[:3, 3]
    R_board_user = C_inv @ R_board_cv @ C
    t_board_user = C_inv @ t_board_cv
    H_board_user = np.eye(4)
    H_board_user[:3, :3], H_board_user[:3, 3] = R_board_user, t_board_user

    print("  Displaying 3D plot...")
    plot_poses(H_base_to_cam_user_list, H_board_user)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform Camera Intrinsics and Hand-Eye Calibration using ChArUco board.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--json', type=str, required=True, 
                        help='Path to the JSON file containing robot pose data.')
    parser.add_argument('--images', type=str, required=True, 
                        help='Path to the directory containing the images.')
    args = parser.parse_args()
    main(args)