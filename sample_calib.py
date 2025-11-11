import numpy as np
import cv2
import json
import argparse
import glob
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pose_data(json_path):
    """JSONファイルからポーズデータを読み込む"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['captures']


def euler_to_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """オイラー角(度)から回転行列を生成
    XYZ軸順でロール・ピッチ・ヨーに対応
    """
    rx_rad = np.deg2rad(rx_deg)
    ry_rad = np.deg2rad(ry_deg)
    rz_rad = np.deg2rad(rz_deg)
    
    # scipy使用でXYZ順のオイラー角から回転行列を生成
    rot = R.from_euler('xyz', [rx_rad, ry_rad, rz_rad])
    return rot.as_matrix()


def pose_to_transformation_matrix(pose):
    """ポーズ情報から4x4変換行列を生成"""
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(pose['rx_deg'], pose['ry_deg'], pose['rz_deg'])
    T[:3, 3] = [pose['x_mm'], pose['y_mm'], pose['z_mm']]
    return T


def detect_charuco_corners(image, aruco_dict, board):
    """ChArUcoボードのコーナーを検出"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ArUcoマーカーを検出
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # ChArUcoコーナーを補間
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        
        if charuco_retval > 0:
            return charuco_corners, charuco_ids
    
    return None, None


def calibrate_camera_charuco(image_files, aruco_dict, board):
    """ChArUcoボードを用いてカメラキャリブレーションを実行"""
    all_corners = []
    all_ids = []
    image_size = None
    
    print("ChArUcoコーナー検出中...")
    for img_file in image_files:
        image = cv2.imread(img_file)
        if image is None:
            print(f"警告: {img_file} を読み込めませんでした")
            continue
            
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        
        charuco_corners, charuco_ids = detect_charuco_corners(image, aruco_dict, board)
        
        if charuco_corners is not None:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print(f"  {img_file}: {len(charuco_corners)} コーナー検出")
        else:
            print(f"  {img_file}: コーナー検出失敗")
    
    if len(all_corners) < 4:
        raise ValueError(f"キャリブレーションには最低4枚の有効な画像が必要です (検出: {len(all_corners)}枚)")
    
    print(f"\nカメラキャリブレーション実行中 ({len(all_corners)}枚の画像)...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )
    
    print(f"キャリブレーション完了 (再投影誤差: {ret:.4f})")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs


def estimate_board_poses(image_files, camera_matrix, dist_coeffs, aruco_dict, board):
    """各画像でのボードポーズを推定"""
    board_poses = []
    valid_indices = []
    
    print("\nボードポーズ推定中...")
    for idx, img_file in enumerate(image_files):
        image = cv2.imread(img_file)
        if image is None:
            continue
            
        charuco_corners, charuco_ids = detect_charuco_corners(image, aruco_dict, board)
        
        if charuco_corners is not None and len(charuco_corners) >= 4:
            # ボードのポーズを推定
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )
            
            if retval:
                board_poses.append((rvec, tvec))
                valid_indices.append(idx)
                print(f"  画像 {idx}: ポーズ推定成功")
            else:
                print(f"  画像 {idx}: ポーズ推定失敗")
        else:
            print(f"  画像 {idx}: コーナー不足")
    
    return board_poses, valid_indices


def solve_hand_eye_calibration(gripper_to_base_transforms, camera_to_board_transforms):
    """Hand-Eye変換行列を推定 (Eye-in-Hand)"""
    R_gripper = []
    t_gripper = []
    R_target = []
    t_target = []
    
    for T_gb in gripper_to_base_transforms:
        R_gripper.append(T_gb[:3, :3])
        t_gripper.append(T_gb[:3, 3])
    
    for rvec, tvec in camera_to_board_transforms:
        R_mat = cv2.Rodrigues(rvec)[0]
        R_target.append(R_mat)
        t_target.append(tvec.flatten())
    
    # OpenCVのHand-Eye calibrationを使用 (Eye-in-Hand)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper, t_gripper,
        R_target, t_target,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # 4x4変換行列に変換
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    return T_cam2gripper


def visualize_camera_poses(gripper_poses, T_cam2gripper, board_poses):
    """カメラポーズを3D可視化"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # カメラポーズを計算して描画
    camera_positions = []
    camera_orientations = []
    
    for T_gb in gripper_poses:
        # カメラのワールド座標系でのポーズ = グリッパーポーズ × Hand-Eye変換
        T_camera_world = T_gb @ T_cam2gripper
        camera_positions.append(T_camera_world[:3, 3])
        camera_orientations.append(T_camera_world[:3, :3])
    
    camera_positions = np.array(camera_positions)
    
    # カメラ位置をプロット
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='blue', marker='o', s=50, label='Camera Positions')
    
    # カメラの向きを矢印で表示(X軸=前方向)
    for pos, ori in zip(camera_positions, camera_orientations):
        direction = ori[:, 0] * 100  # X軸方向(前方)
        ax.quiver(pos[0], pos[1], pos[2],
                 direction[0], direction[1], direction[2],
                 color='red', alpha=0.6, arrow_length_ratio=0.3)
    
    # グリッパー位置もプロット
    gripper_positions = np.array([T[:3, 3] for T in gripper_poses])
    ax.scatter(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
               c='green', marker='^', s=50, label='Gripper Positions')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Camera and Gripper Poses in World Frame')
    ax.legend()
    
    # 軸の範囲を調整
    all_points = np.vstack([camera_positions, gripper_positions])
    max_range = np.ptp(all_points, axis=0).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    plt.show()

import os

def mat_to_colmap_qvec_tvec(R_wc, t_wc):
    """
    R_wc, t_wc: World->Camera の回転行列と並進ベクトル
    戻り: (qw,qx,qy,qz, tx,ty,tz) — COLMAP の Hamilton 規約 (QW QX QY QZ)
    注意: scipy Rotation.as_quat() は (x,y,z,w) を返すため順序入替が必要。
    """
    q_xyzw = R.from_matrix(R_wc).as_quat()  # [qx, qy, qz, qw]
    qx, qy, qz, qw = q_xyzw
    return float(qw), float(qx), float(qy), float(qz), float(t_wc[0]), float(t_wc[1]), float(t_wc[2])

def write_colmap_images_txt(out_path, image_files, valid_indices, gripper_poses, T_cam2gripper, camera_id=1, image_id_start=1):
    """
    image_files: 全画像パスのリスト（sorted）
    valid_indices: ボード姿勢が推定できた画像のインデックス（スクリプトで取得したもの）
    gripper_poses: valid_indices に対応した 4x4 グリッパ→ワールド（T_gb）行列のリスト
                   （※本スクリプトでは pose_to_transformation_matrix(pose) を使って作る）
    T_cam2gripper: 4x4 カメラ->グリッパ行列（solve_hand_eye の出力）
    出力: COLMAP 互換 images.txt（POINTS2D 行は空行）
    """
    lines = []
    lines.append("# Image list with two lines of data per image:")
    lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")
    lines.append("#   POINTS2D[] as (x, y, POINT3D_ID) -- left empty here")

    image_id = int(image_id_start)
    # valid_indices は image_files の index を表す
    for local_i, img_idx in enumerate(valid_indices):
        img_path = image_files[img_idx]
        img_name = os.path.basename(img_path)

        T_gb = gripper_poses[local_i]   # 本スクリプトで作った順に対応している前提
        # スクリプトの可視化コードと同じ：Camera->World = T_gb @ T_cam2gripper
        T_camera_world = T_gb @ T_cam2gripper
        T_world_camera = np.linalg.inv(T_camera_world)  # World -> Camera（COLMAPが期待）

        R_wc = T_world_camera[:3, :3]
        t_wc = T_world_camera[:3, 3]

        qw, qx, qy, qz, tx, ty, tz = mat_to_colmap_qvec_tvec(R_wc, t_wc)
        head = f"{image_id} {qw:.10g} {qx:.10g} {qy:.10g} {qz:.10g} {tx:.10g} {ty:.10g} {tz:.10g} {camera_id} {img_name}"
        lines.append(head)
        lines.append("")  # POINTS2D: 今回は空行
        image_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] COLMAP images.txt -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description='ChArUco Hand-Eye Calibration')
    parser.add_argument('--images', type=str, required=True, help='画像ファイルのパス(ワイルドカード可)')
    parser.add_argument('--poses', type=str, required=True, help='ポーズJSONファイルのパス')
    parser.add_argument('--squares-x', type=int, default=7, help='ChArUcoボードの正方形数(X)')
    parser.add_argument('--squares-y', type=int, default=5, help='ChArUcoボードの正方形数(Y)')
    parser.add_argument('--square-length', type=float, default=32.0, help='正方形のサイズ(mm)')
    parser.add_argument('--marker-length', type=float, default=24.0, help='マーカーのサイズ(mm)')
    parser.add_argument('--dict', type=str, default='4X4_50', help='ArUco辞書 (4X4_50, 5X5_100等)')
    
    parser.add_argument('--images-txt-out', type=str, default=None, help='COLMAP images.txt の出力先パス（任意）')
    parser.add_argument('--camera-id', type=int, default=1, help='images.txt に書く CAMERA_ID')
    parser.add_argument('--image-id-start', type=int, default=1, help='images.txt の IMAGE_ID 開始値')

    
    args = parser.parse_args()
    
    # ArUco辞書とChArUcoボードを作成
    aruco_dict_name = f'DICT_{args.dict}'
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict
    )
    
    # ポーズデータを読み込み
    captures = load_pose_data(args.poses)
    print(f"ポーズデータ読み込み: {len(captures)}件")
    
    # 画像ファイル一覧を取得
    image_files = sorted(glob.glob(args.images))
    print(f"画像ファイル検出: {len(image_files)}枚\n")
    
    # 1. カメラキャリブレーション
    camera_matrix, dist_coeffs, _, _ = calibrate_camera_charuco(image_files, aruco_dict, board)
    
    print("\n=== カメラ内部パラメータ ===")
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    # 2. 各画像でのボードポーズを推定
    board_poses, valid_indices = estimate_board_poses(
        image_files, camera_matrix, dist_coeffs, aruco_dict, board
    )
    
    if len(board_poses) < 3:
        raise ValueError(f"Hand-Eyeキャリブレーションには最低3枚必要です (有効: {len(board_poses)}枚)")
    
    # 3. 有効な画像に対応するグリッパーポーズを取得
    gripper_poses = []
    for idx in valid_indices:
        pose = captures[idx]['pose']
        T = pose_to_transformation_matrix(pose)
        gripper_poses.append(T)
    
    # 4. Hand-Eye変換行列を推定
    print(f"\nHand-Eyeキャリブレーション実行中 ({len(board_poses)}ペア)...")
    T_cam2gripper = solve_hand_eye_calibration(gripper_poses, board_poses)
    
    print("\n=== Hand-Eye変換行列 (Camera to Gripper) ===")
    print(T_cam2gripper)
    
    # 回転行列からオイラー角を計算して表示
    rot = R.from_matrix(T_cam2gripper[:3, :3])
    euler_angles = rot.as_euler('xyz', degrees=True)
    print("\n変換 (並進 [mm], 回転 [度]):")
    print(f"  X: {T_cam2gripper[0, 3]:.3f} mm")
    print(f"  Y: {T_cam2gripper[1, 3]:.3f} mm")
    print(f"  Z: {T_cam2gripper[2, 3]:.3f} mm")
    print(f"  RX: {euler_angles[0]:.3f} deg")
    print(f"  RY: {euler_angles[1]:.3f} deg")
    print(f"  RZ: {euler_angles[2]:.3f} deg")
    
    # 5. カメラポーズを可視化
    print("\nカメラポーズを可視化中...")
    visualize_camera_poses(gripper_poses, T_cam2gripper, board_poses)
    
    print("\n処理完了!")

    # --- images.txt 出力（任意） ---
    if args.images_txt_out:
        write_colmap_images_txt(args.images_txt_out, image_files, valid_indices, gripper_poses, T_cam2gripper,
                                camera_id=args.camera_id, image_id_start=args.image_id_start)


if __name__ == '__main__':
    main()