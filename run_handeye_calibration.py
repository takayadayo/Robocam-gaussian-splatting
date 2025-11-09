import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# -----------------------------------------------------------------------------
### --- ChArUcoボード設定 (※ご自身のボードに合わせて要変更) --- ###
# -----------------------------------------------------------------------------
# ボードのチェッカー柄の、内側の頂点数を指定 (マス数 - 1)
# 例: 8x6マスの場合、(7, 5) を指定
CHARUCO_BOARD_PATTERN = (7, 5)
# チェッカー柄の正方形の一辺の長さ (メートル単位)
SQUARE_LENGTH = 0.032
# ArUcoマーカーの一辺の長さ (メートル単位)
MARKER_LENGTH = 0.024
# 使用したArUcoマーカーの辞書
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# -----------------------------------------------------------------------------


# --- 座標系変換の定義 ---
# ロボットのユーザー座標系 (U: X-奥, Y-左, Z-上) [左手系] と
# OpenCV標準カメラ座標系 (C: X-右, Y-下, Z-奥) [右手系] の間の変換
# p_c = ROT_U_TO_C @ p_u
# x_c(右) = -y_u(左),  y_c(下) = -z_u(上),  z_c(奥) = x_u(奥)
ROT_U_TO_C = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
ROT_C_TO_U = ROT_U_TO_C.T


def create_charuco_board():
    """ChArUcoボードオブジェクトを作成して返す"""
    return cv2.aruco.CharucoBoard(
        CHARUCO_BOARD_PATTERN,
        SQUARE_LENGTH,
        MARKER_LENGTH,
        ARUCO_DICT
    )

def parse_pose_to_h_matrix(pose_data, rot_transform_matrix):
    """
    JSONのポーズデータを読み込み、指定された座標系に変換後の同次変換行列を返す
    """
    t_user = np.array([pose_data['x_mm']/1000.0, pose_data['y_mm']/1000.0, pose_data['z_mm']/1000.0])
    r_user_deg = [pose_data['rx_deg'], pose_data['ry_deg'], pose_data['rz_deg']]
    R_user = Rotation.from_euler('ZYX', r_user_deg, degrees=True).as_matrix()
    t_new = rot_transform_matrix @ t_user
    R_new = rot_transform_matrix @ R_user @ rot_transform_matrix.T
    H_matrix = np.eye(4)
    H_matrix[:3, :3], H_matrix[:3, 3] = R_new, t_new
    return H_matrix

def rvec_tvec_to_h_matrix(rvec, tvec):
    """cv2から出力されるrvecとtvecを4x4同次変換行列に変換する"""
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = R, tvec.flatten()
    return H

def plot_poses_in_user_coords(camera_poses, board_pose):
    """
    推定されたカメラポーズとボードポーズをユーザー座標系で3Dプロットする
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    def draw_axis(ax, H, scale=0.1, label_prefix=""):
        origin = H[:3, 3]
        x_axis, y_axis, z_axis = H[:3, 0], H[:3, 1], H[:3, 2]
        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], length=scale, color='r')
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], length=scale, color='g')
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], length=scale, color='b')
        ax.text(origin[0], origin[1], origin[2], f"  {label_prefix}", color='k')

    draw_axis(ax, np.eye(4), scale=0.2, label_prefix="World(Base)")

    positions = np.array([H[:3, 3] for H in camera_poses])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', color='c', markersize=3, label='Camera Path')
    for H_cam in camera_poses:
        draw_axis(ax, H_cam, scale=0.05)

    # === ボードの厳格な可視化 ===
    draw_axis(ax, board_pose, scale=0.1, label_prefix="Board Origin")
    board_width = CHARUCO_BOARD_PATTERN[0] * SQUARE_LENGTH
    board_height = CHARUCO_BOARD_PATTERN[1] * SQUARE_LENGTH
    board_corners_local = np.array([
        [0, 0, 0], [board_width, 0, 0],
        [board_width, board_height, 0], [0, board_height, 0], [0, 0, 0]
    ])
    R_board, t_board = board_pose[:3, :3], board_pose[:3, 3]
    board_corners_world = (R_board @ board_corners_local.T).T + t_board
    ax.plot(board_corners_world[:, 0], board_corners_world[:, 1], board_corners_world[:, 2], color='m', label='Board Outline')
    
    all_points = np.vstack([positions, board_corners_world])
    max_range = np.array([p.max() - p.min() for p in all_points.T]).max() * 0.6
    mid = np.mean(all_points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X (奥) [m]"), ax.set_ylabel("Y (左) [m]"), ax.set_zlabel("Z (上) [m]")
    ax.set_title("Estimated Camera and Board Poses in World Coordinate System")
    ax.legend(), ax.set_aspect('equal', adjustable='box'), plt.show()

def main(args):
    """メイン処理"""
    board = create_charuco_board()
    image_dir = Path(args.images)
    with open(args.json, 'r') as f:
        pose_data = json.load(f)

    # === Step 1: マーカー検出とデータ準備 ===
    print("--- Step 1: Detecting ChArUco markers and preparing data ---")
    all_charuco_corners, all_charuco_ids, H_base_to_ee_cv_list = [], [], []
    img_size, images_shown_count = None, 0

    for capture in pose_data['captures']:
        img_path = image_dir / capture['image_file']
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None: img_size = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
        if ids is not None and len(ids) > 4:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if ret and len(charuco_corners) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                H_base_to_ee_cv_list.append(parse_pose_to_h_matrix(capture['pose'], ROT_U_TO_C))
                
                # === マーカー認識の途中経過を5枚可視化 ===
                if images_shown_count < 5:
                    print(f"  => Visualizing detection on {img_path.name}. Press any key to continue...")
                    img_drawn = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(img_drawn, charuco_corners, charuco_ids)
                    h, w = img_drawn.shape[:2]
                    scale = 1000 / max(h, w)
                    img_resized = cv2.resize(img_drawn, (int(w*scale), int(h*scale)))
                    cv2.imshow('ChArUco Detection Result', img_resized)
                    cv2.waitKey(0)
                    images_shown_count += 1
                    if images_shown_count == 5:
                        print("  => Reached visualization limit. Continuing without showing images...")
                        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    print(f"\n  => Found markers in {len(all_charuco_corners)} / {len(pose_data['captures'])} images.")
    if len(all_charuco_corners) < 10:
        print("[Error] Not enough valid images for reliable calibration. Exiting.")
        return

    # === Step 2: カメラ内部パラメータのキャリブレーション ===
    print("\n--- Step 2: Calibrating camera intrinsics ---")
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, img_size, None, None)
    if not ret: print("[Error] Camera calibration failed."); return
    print("  Camera Matrix:\n", mtx), print("\n  Distortion Coefficients:\n", dist)

    # === Step 3: Hand-Eyeキャリブレーション ===
    print("\n--- Step 3: Performing Hand-Eye calibration (AX=XB) ---")
    R_g2e = [H[:3,:3] for H in H_base_to_ee_cv_list]
    t_g2e = [H[:3,3] for H in H_base_to_ee_cv_list]
    H_c2t_list = [np.linalg.inv(rvec_tvec_to_h_matrix(r, t)) for r, t in zip(rvecs, tvecs)]
    R_c2t = [H[:3,:3] for H in H_c2t_list]
    t_c2t = [H[:3,3] for H in H_c2t_list]
    R_e2c, t_e2c = cv2.calibrateHandEye(R_g2e, t_g2e, R_c2t, t_c2t, method=cv2.CALIB_HAND_EYE_PARK)
    H_ee_to_cam_cv = rvec_tvec_to_h_matrix(cv2.Rodrigues(R_e2c)[0], t_e2c)
    print("  Hand-Eye Matrix (End-Effector to Camera, OpenCV coords):\n", np.round(H_ee_to_cam_cv, 4))
    t_offset_user = ROT_C_TO_U @ H_ee_to_cam_cv[:3, 3]
    print(f"\n  Offset in User Coords (X:奥,Y:左,Z:上): "
          f"[{t_offset_user[0]*1000:.1f}, {t_offset_user[1]*1000:.1f}, {t_offset_user[2]*1000:.1f}] mm")

    # === Step 4: 全カメラポーズとボードポーズの算出 ===
    print("\n--- Step 4: Calculating all camera poses and board pose ---")
    H_base_to_cam_cv_list = [H_b2e @ H_ee_to_cam_cv for H_b2e in H_base_to_ee_cv_list]
    H_base_to_marker_cv_list = [H_b2c @ H_c2t for H_b2c, H_c2t in zip(H_base_to_cam_cv_list, H_c2t_list)]
    t_avg = np.mean([H[:3, 3] for H in H_base_to_marker_cv_list], axis=0)
    U, _, Vt = np.linalg.svd(np.mean([H[:3, :3] for H in H_base_to_marker_cv_list], axis=0))
    R_avg = U @ Vt
    H_base_to_marker_cv_avg = np.eye(4)
    H_base_to_marker_cv_avg[:3, :3], H_base_to_marker_cv_avg[:3, 3] = R_avg, t_avg
    
    # === Step 5: 結果の可視化 ===
    print("\n--- Step 5: Converting poses to User coordinate system for visualization ---")
    def convert_pose_cv_to_user(H_cv):
        H_user = np.eye(4)
        H_user[:3, :3] = ROT_C_TO_U @ H_cv[:3, :3] @ ROT_C_TO_U.T
        H_user[:3, 3] = ROT_C_TO_U @ H_cv[:3, 3]
        return H_user
    H_base_to_cam_user_list = [convert_pose_cv_to_user(H) for H in H_base_to_cam_cv_list]
    H_base_to_marker_user = convert_pose_cv_to_user(H_base_to_marker_cv_avg)
    
    print("  Ready to plot. Please check the pop-up window.")
    plot_poses_in_user_coords(H_base_to_cam_user_list, H_base_to_marker_user)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Hand-Eye calibration using ChArUco board images and robot poses.')
    parser.add_argument('--json', type=str, required=True, help='Path to the JSON file containing pose data.')
    parser.add_argument('--images', type=str, required=True, help='Path to the directory containing the images.')
    args = parser.parse_args()
    main(args)