import numpy as np
import cv2
import json
import argparse
import glob
import os
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
    T_cam2gripper: 4x4 カメラ->グリッパ行列（solve_hand_eye の出力）
    出力: COLMAP 互換 images.txt（POINTS2D 行は空行）
    """
    lines = []
    lines.append("# Image list with two lines of data per image:")
    lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")
    lines.append("#   POINTS2D[] as (x, y, POINT3D_ID) -- left empty here")

    image_id = int(image_id_start)
    for local_i, img_idx in enumerate(valid_indices):
        img_path = image_files[img_idx]
        img_name = os.path.basename(img_path)

        T_gb = gripper_poses[local_i]
        T_camera_world = T_gb @ T_cam2gripper
        T_world_camera = np.linalg.inv(T_camera_world)

        R_wc = T_world_camera[:3, :3]
        t_wc = T_world_camera[:3, 3]

        qw, qx, qy, qz, tx, ty, tz = mat_to_colmap_qvec_tvec(R_wc, t_wc)
        head = f"{image_id} {qw:.10g} {qx:.10g} {qy:.10g} {qz:.10g} {tx:.10g} {ty:.10g} {tz:.10g} {camera_id} {img_name}"
        lines.append(head)
        lines.append("")
        image_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] COLMAP images.txt -> {out_path}")


def extract_features_and_match(image_files, valid_indices, camera_matrix, dist_coeffs):
    """
    特徴点抽出とマッチング
    返り値: features_dict, matches_dict
    features_dict[img_idx] = (keypoints, descriptors)
    matches_dict[(i, j)] = [(query_idx, train_idx), ...]
    """
    print("\n特徴点抽出中...")
    sift = cv2.SIFT_create()
    features_dict = {}
    
    for img_idx in valid_indices:
        img_path = image_files[img_idx]
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 歪み補正を適用
        h, w = gray.shape
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        kp, desc = sift.detectAndCompute(undistorted, None)
        features_dict[img_idx] = (kp, desc)
        print(f"  画像 {img_idx}: {len(kp)} 特徴点")
    
    # マッチング
    print("\n特徴点マッチング中...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_dict = {}
    
    valid_list = sorted(valid_indices)
    for i in range(len(valid_list)):
        for j in range(i+1, len(valid_list)):
            idx_i = valid_list[i]
            idx_j = valid_list[j]
            
            kp_i, desc_i = features_dict[idx_i]
            kp_j, desc_j = features_dict[idx_j]
            
            if desc_i is None or desc_j is None:
                continue
            
            # Lowe's ratio test
            matches = matcher.knnMatch(desc_i, desc_j, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append((m.queryIdx, m.trainIdx))
            
            if len(good_matches) >= 15:
                matches_dict[(idx_i, idx_j)] = good_matches
                print(f"  画像ペア ({idx_i}, {idx_j}): {len(good_matches)} マッチ")
    
    return features_dict, matches_dict


def triangulate_points_fixed_poses(features_dict, matches_dict, camera_matrix, gripper_poses, T_cam2gripper, valid_indices):
    """
    固定されたカメラポーズを用いて三角測量により3D点群を生成
    """
    print("\n三角測量による3D点群生成中...")
    
    # カメラポーズ行列を事前計算
    camera_poses = {}
    projection_matrices = {}
    
    for img_idx in valid_indices:
        local_i = valid_indices.index(img_idx)
        T_gb = gripper_poses[local_i]
        T_camera_world = T_gb @ T_cam2gripper
        T_world_camera = np.linalg.inv(T_camera_world)
        
        camera_poses[img_idx] = T_world_camera
        # Projection matrix: K [R | t]
        P = camera_matrix @ T_world_camera[:3, :]
        projection_matrices[img_idx] = P
    
    # 3D点を格納
    points_3d = []
    point_colors = []
    point_reprojection_errors = []
    
    processed_pairs = 0
    for (idx_i, idx_j), matches in matches_dict.items():
        kp_i, _ = features_dict[idx_i]
        kp_j, _ = features_dict[idx_j]
        
        P_i = projection_matrices[idx_i]
        P_j = projection_matrices[idx_j]
        
        pts_i = np.float32([kp_i[m[0]].pt for m in matches])
        pts_j = np.float32([kp_j[m[1]].pt for m in matches])
        
        # 三角測量
        points_4d = cv2.triangulatePoints(P_i, P_j, pts_i.T, pts_j.T)
        points_3d_homog = points_4d.T
        points_3d_cart = points_3d_homog[:, :3] / points_3d_homog[:, 3:4]
        
        # 再投影誤差を計算してフィルタリング
        for k, pt_3d in enumerate(points_3d_cart):
            # 画像iでの再投影
            pt_3d_homog = np.append(pt_3d, 1.0)
            proj_i = P_i @ pt_3d_homog
            proj_i_2d = proj_i[:2] / proj_i[2]
            error_i = np.linalg.norm(pts_i[k] - proj_i_2d)
            
            # 画像jでの再投影
            proj_j = P_j @ pt_3d_homog
            proj_j_2d = proj_j[:2] / proj_j[2]
            error_j = np.linalg.norm(pts_j[k] - proj_j_2d)
            
            avg_error = (error_i + error_j) / 2.0
            
            # 再投影誤差が閾値以下の点のみ採用
            if avg_error < 5.0:  # ピクセル単位の閾値
                # カメラの前方にある点のみ採用
                T_wc_i = camera_poses[idx_i]
                pt_cam = T_wc_i[:3, :3] @ pt_3d + T_wc_i[:3, 3]
                
                if pt_cam[2] > 0:  # Z座標が正(カメラ前方)
                    points_3d.append(pt_3d)
                    point_reprojection_errors.append(avg_error)
                    # 色情報(簡易的に白色)
                    point_colors.append([200, 200, 200])
        
        processed_pairs += 1
    
    points_3d = np.array(points_3d)
    point_colors = np.array(point_colors)
    point_reprojection_errors = np.array(point_reprojection_errors)
    
    print(f"三角測量完了: {len(points_3d)} 個の3D点を生成")
    if len(points_3d) > 0:
        print(f"  平均再投影誤差: {np.mean(point_reprojection_errors):.3f} ピクセル")
        print(f"  最大再投影誤差: {np.max(point_reprojection_errors):.3f} ピクセル")
    
    return points_3d, point_colors


def visualize_reconstruction(gripper_poses, T_cam2gripper, points_3d, point_colors):
    """カメラポーズと3D点群を可視化"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # カメラポーズを計算
    camera_positions = []
    camera_orientations = []
    
    for T_gb in gripper_poses:
        T_camera_world = T_gb @ T_cam2gripper
        camera_positions.append(T_camera_world[:3, 3])
        camera_orientations.append(T_camera_world[:3, :3])
    
    camera_positions = np.array(camera_positions)
    
    # 3D点群をプロット
    if len(points_3d) > 0:
        # サンプリングして表示(多すぎる場合)
        if len(points_3d) > 10000:
            indices = np.random.choice(len(points_3d), 10000, replace=False)
            points_subset = points_3d[indices]
            colors_subset = point_colors[indices] / 255.0
        else:
            points_subset = points_3d
            colors_subset = point_colors / 255.0
        
        ax.scatter(points_subset[:, 0], points_subset[:, 1], points_subset[:, 2],
                   c=colors_subset, marker='.', s=1, alpha=0.5, label='3D Points')
    
    # カメラ位置をプロット
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='blue', marker='o', s=100, label='Camera Positions', edgecolors='black', linewidths=1)
    
    # カメラの向きを矢印で表示
    arrow_length = 50  # mm
    for pos, ori in zip(camera_positions, camera_orientations):
        direction = ori[:, 0] * arrow_length
        ax.quiver(pos[0], pos[1], pos[2],
                 direction[0], direction[1], direction[2],
                 color='red', alpha=0.8, arrow_length_ratio=0.2)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Reconstruction with Fixed Camera Poses')
    ax.legend()
    
    # 軸の範囲を調整
    if len(points_3d) > 0:
        all_points = np.vstack([camera_positions, points_3d])
    else:
        all_points = camera_positions
    
    max_range = np.ptp(all_points, axis=0).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='ChArUco Hand-Eye Calibration and 3D Reconstruction')
    parser.add_argument('--images', type=str, required=True, help='画像ファイルのパス(ワイルドカード可)')
    parser.add_argument('--poses', type=str, required=True, help='ポーズJSONファイルのパス')
    parser.add_argument('--output-dir', type=str, default='output', help='出力ディレクトリ')
    parser.add_argument('--squares-x', type=int, default=7, help='ChArUcoボードの正方形数(X)')
    parser.add_argument('--squares-y', type=int, default=5, help='ChArUcoボードの正方形数(Y)')
    parser.add_argument('--square-length', type=float, default=32.0, help='正方形のサイズ(mm)')
    parser.add_argument('--marker-length', type=float, default=24.0, help='マーカーのサイズ(mm)')
    parser.add_argument('--dict', type=str, default='4X4_50', help='ArUco辞書 (4X4_50, 5X5_100等)')
    parser.add_argument('--reconstruct', action='store_true', help='3D再構成を実行')
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # 5. COLMAP形式でカメラポーズを出力
    images_txt_path = os.path.join(args.output_dir, 'images.txt')
    write_colmap_images_txt(images_txt_path, image_files, valid_indices, gripper_poses, T_cam2gripper)
    
    # 6. 3D再構成(オプション)
    if args.reconstruct:
        print("\n" + "="*60)
        print("3D再構成処理を開始")
        print("="*60)
        
        # 特徴点抽出とマッチング
        features_dict, matches_dict = extract_features_and_match(
            image_files, valid_indices, camera_matrix, dist_coeffs
        )
        
        if len(matches_dict) == 0:
            print("警告: マッチングペアが見つかりませんでした")
            points_3d = np.array([])
            point_colors = np.array([])
        else:
            # 三角測量による3D点群生成
            points_3d, point_colors = triangulate_points_fixed_poses(
                features_dict, matches_dict, camera_matrix, gripper_poses, T_cam2gripper, valid_indices
            )
        
        # 可視化
        print("\n3D再構成結果を可視化中...")
        visualize_reconstruction(gripper_poses, T_cam2gripper, points_3d, point_colors)
    else:
        print("\nカメラポーズのみを可視化中...")
        # カメラポーズのみを可視化
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        camera_positions = []
        camera_orientations = []
        
        for T_gb in gripper_poses:
            T_camera_world = T_gb @ T_cam2gripper
            camera_positions.append(T_camera_world[:3, 3])
            camera_orientations.append(T_camera_world[:3, :3])
        
        camera_positions = np.array(camera_positions)
        
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                   c='blue', marker='o', s=100, label='Camera Positions', edgecolors='black', linewidths=1)
        
        for pos, ori in zip(camera_positions, camera_orientations):
            direction = ori[:, 0] * 100
            ax.quiver(pos[0], pos[1], pos[2],
                     direction[0], direction[1], direction[2],
                     color='red', alpha=0.8, arrow_length_ratio=0.2)
        
        gripper_positions = np.array([T[:3, 3] for T in gripper_poses])
        ax.scatter(gripper_positions[:, 0], gripper_positions[:, 1], gripper_positions[:, 2],
                   c='green', marker='^', s=100, label='Gripper Positions', edgecolors='black', linewidths=1)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Camera and Gripper Poses in World Frame')
        ax.legend()
        
        all_points = np.vstack([camera_positions, gripper_positions])
        max_range = np.ptp(all_points, axis=0).max() / 2.0
        mid = all_points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.tight_layout()
        plt.show()
    
    print("\n処理完了!")


if __name__ == '__main__':
    main()