import numpy as np
import cv2
import json
import argparse
import glob
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def load_pose_data(json_path):
    """JSONファイルからポーズデータを読み込む"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['captures']


def euler_to_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """オイラー角(度)から回転行列を生成"""
    rx_rad = np.deg2rad(rx_deg)
    ry_rad = np.deg2rad(ry_deg)
    rz_rad = np.deg2rad(rz_deg)
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
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
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
            continue
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        
        charuco_corners, charuco_ids = detect_charuco_corners(image, aruco_dict, board)
        if charuco_corners is not None:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print(f"  {os.path.basename(img_file)}: {len(charuco_corners)} コーナー")
        else:
            print(f"  {os.path.basename(img_file)}: 検出失敗")
    
    if len(all_corners) < 4:
        raise ValueError(f"キャリブレーションには最低4枚必要 (検出: {len(all_corners)}枚)")
    
    print(f"\nカメラキャリブレーション実行中 ({len(all_corners)}枚)...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )
    print(f"完了 (再投影誤差: {ret:.4f})")
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
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )
            if retval:
                board_poses.append((rvec, tvec))
                valid_indices.append(idx)
                print(f"  画像 {idx}: 成功")
    return board_poses, valid_indices


def solve_hand_eye_calibration(gripper_to_base_transforms, camera_to_board_transforms):
    """Hand-Eye変換行列を推定"""
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
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper, t_gripper, R_target, t_target, method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    return T_cam2gripper


def mat_to_colmap_qvec_tvec(R_wc, t_wc):
    """回転行列と並進からCOLMAP形式のクォータニオンと並進を生成"""
    q_xyzw = R.from_matrix(R_wc).as_quat()
    qx, qy, qz, qw = q_xyzw
    return float(qw), float(qx), float(qy), float(qz), float(t_wc[0]), float(t_wc[1]), float(t_wc[2])


def write_colmap_images_txt(out_path, image_files, valid_indices, gripper_poses, T_cam2gripper, camera_id=1, image_id_start=1):
    """COLMAP形式のimages.txtを出力"""
    lines = []
    lines.append("# Image list with two lines of data per image:")
    lines.append("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME")
    lines.append("#   POINTS2D[] as (x, y, POINT3D_ID)")

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
    print(f"[書込] {out_path}")


def extract_and_undistort_sift_features(image_files, valid_indices, camera_matrix, dist_coeffs, max_features=8192):
    """SIFT特徴を抽出し、歪み補正済み座標に変換"""
    print("\n特徴点抽出中 (SIFT)...")
    sift = cv2.SIFT_create(nfeatures=max_features)
    features_dict = {}
    
    for img_idx in valid_indices:
        img_path = image_files[img_idx]
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 特徴抽出
        kp, desc = sift.detectAndCompute(gray, None)
        
        # 特徴点座標を歪み補正
        if len(kp) > 0:
            pts = np.float32([k.pt for k in kp]).reshape(-1, 1, 2)
            pts_undist = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
            pts_undist = pts_undist.reshape(-1, 2)
            
            # KeyPointオブジェクトを更新
            kp_undist = []
            for i, k in enumerate(kp):
                kp_new = cv2.KeyPoint(
                    x=pts_undist[i, 0],
                    y=pts_undist[i, 1],
                    size=k.size,
                    angle=k.angle,
                    response=k.response,
                    octave=k.octave,
                    class_id=k.class_id
                )
                kp_undist.append(kp_new)
            
            features_dict[img_idx] = (kp_undist, desc, image)
            print(f"  画像 {img_idx}: {len(kp)} 特徴点")
        else:
            features_dict[img_idx] = ([], None, image)
    
    return features_dict


def robust_feature_matching(features_dict, valid_indices, ratio_threshold=0.75):
    """ロバストな特徴マッチング (Lowe's ratio test + 対称性チェック)"""
    print("\n特徴点マッチング中...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_dict = {}
    
    valid_list = sorted(valid_indices)
    for i in range(len(valid_list)):
        for j in range(i+1, len(valid_list)):
            idx_i = valid_list[i]
            idx_j = valid_list[j]
            
            kp_i, desc_i, _ = features_dict[idx_i]
            kp_j, desc_j, _ = features_dict[idx_j]
            
            if desc_i is None or desc_j is None or len(kp_i) < 15 or len(kp_j) < 15:
                continue
            
            # 双方向マッチング
            matches_i2j = matcher.knnMatch(desc_i, desc_j, k=2)
            matches_j2i = matcher.knnMatch(desc_j, desc_i, k=2)
            
            # Lowe's ratio test
            good_i2j = {}
            for m_n in matches_i2j:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_threshold * n.distance:
                        good_i2j[m.queryIdx] = m.trainIdx
            
            good_j2i = {}
            for m_n in matches_j2i:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_threshold * n.distance:
                        good_j2i[m.queryIdx] = m.trainIdx
            
            # 対称性チェック (mutual best match)
            symmetric_matches = []
            for q_idx, t_idx in good_i2j.items():
                if t_idx in good_j2i and good_j2i[t_idx] == q_idx:
                    symmetric_matches.append((q_idx, t_idx))
            
            if len(symmetric_matches) >= 15:
                matches_dict[(idx_i, idx_j)] = symmetric_matches
                print(f"  ペア ({idx_i}, {idx_j}): {len(symmetric_matches)} マッチ")
    
    return matches_dict


def calculate_triangulation_angle(pt_3d, cam_center_1, cam_center_2):
    """三角測量角度を計算 (度)"""
    vec1 = pt_3d - cam_center_1
    vec2 = pt_3d - cam_center_2
    
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    
    cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def triangulate_multi_view_optimal(pts_2d_list, P_list):
    """
    多視点三角測量 (COLMAPスタイル)
    最小二乗法で最適な3D点を推定
    """
    n_views = len(pts_2d_list)
    A = np.zeros((2 * n_views, 4))
    
    for i, (pt_2d, P) in enumerate(zip(pts_2d_list, P_list)):
        x, y = pt_2d
        A[2*i] = x * P[2] - P[0]
        A[2*i+1] = y * P[2] - P[1]
    
    # SVD求解
    _, _, Vt = np.linalg.svd(A)
    X_homog = Vt[-1]
    
    # 同次座標から3D点に変換
    if abs(X_homog[3]) < 1e-10:
        return None
    
    X = X_homog[:3] / X_homog[3]
    return X


def triangulate_tracks(track_data, features_dict, projection_matrices, camera_poses, 
                       min_tri_angle=1.5, max_reproj_error=4.0, min_track_length=2):
    """
    トラック単位で三角測量を実行
    COLMAPのロバストな三角測量ロジックを実装
    """
    print("\nトラック三角測量中 (COLMAP風ロバスト処理)...")
    
    points_3d = []
    point_colors = []
    point_track_lengths = []
    point_reproj_errors = []
    
    total_tracks = len(track_data)
    successful = 0
    filtered_angle = 0
    filtered_reproj = 0
    filtered_cheirality = 0
    
    for track_id, observations in track_data.items():
        if len(observations) < min_track_length:
            continue
        
        # 2D点と投影行列を収集
        pts_2d = []
        P_matrices = []
        cam_centers = []
        colors_list = []
        
        for img_idx, feat_idx in observations:
            kp, _, image = features_dict[img_idx]
            pt_2d = np.array(kp[feat_idx].pt)
            pts_2d.append(pt_2d)
            P_matrices.append(projection_matrices[img_idx])
            
            T_wc = camera_poses[img_idx]
            T_cw = np.linalg.inv(T_wc)
            cam_center = T_cw[:3, 3]
            cam_centers.append(cam_center)
            
            # 色情報を取得
            y, x = int(pt_2d[1]), int(pt_2d[0])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                color = image[y, x]
                colors_list.append([int(color[2]), int(color[1]), int(color[0])])  # BGR to RGB
            else:
                colors_list.append([128, 128, 128])
        
        # 多視点三角測量
        pt_3d = triangulate_multi_view_optimal(pts_2d, P_matrices)
        
        if pt_3d is None:
            continue
        
        # === Cheirality Check (正の奥行きチェック) ===
        valid_depth = True
        for img_idx, _ in observations:
            T_wc = camera_poses[img_idx]
            pt_cam = T_wc[:3, :3] @ pt_3d + T_wc[:3, 3]
            if pt_cam[2] <= 0:  # カメラの後ろ
                valid_depth = False
                break
        
        if not valid_depth:
            filtered_cheirality += 1
            continue
        
        # === Triangulation Angle Check ===
        if len(cam_centers) >= 2:
            max_angle = 0
            for i in range(len(cam_centers)):
                for j in range(i+1, len(cam_centers)):
                    angle = calculate_triangulation_angle(pt_3d, cam_centers[i], cam_centers[j])
                    max_angle = max(max_angle, angle)
            
            if max_angle < min_tri_angle:
                filtered_angle += 1
                continue
        
        # === Reprojection Error Check ===
        reproj_errors = []
        for img_idx, feat_idx in observations:
            P = projection_matrices[img_idx]
            kp, _, _ = features_dict[img_idx]
            pt_2d_orig = np.array(kp[feat_idx].pt)
            
            pt_3d_homog = np.append(pt_3d, 1.0)
            proj = P @ pt_3d_homog
            
            if abs(proj[2]) < 1e-10:
                reproj_errors.append(1e6)
                continue
            
            proj_2d = proj[:2] / proj[2]
            error = np.linalg.norm(pt_2d_orig - proj_2d)
            reproj_errors.append(error)
        
        avg_error = np.mean(reproj_errors)
        max_error = np.max(reproj_errors)
        
        if max_error > max_reproj_error:
            filtered_reproj += 1
            continue
        
        # 全チェック通過 → 3D点として採用
        points_3d.append(pt_3d)
        point_colors.append(np.mean(colors_list, axis=0).astype(int))
        point_track_lengths.append(len(observations))
        point_reproj_errors.append(avg_error)
        successful += 1
    
    print(f"  総トラック数: {total_tracks}")
    print(f"  成功: {successful}")
    print(f"  除外 - 三角測量角度: {filtered_angle}")
    print(f"  除外 - 再投影誤差: {filtered_reproj}")
    print(f"  除外 - Cheirality: {filtered_cheirality}")
    
    if len(points_3d) > 0:
        print(f"  平均トラック長: {np.mean(point_track_lengths):.2f}")
        print(f"  平均再投影誤差: {np.mean(point_reproj_errors):.3f} px")
        print(f"  最大再投影誤差: {np.max(point_reproj_errors):.3f} px")
    
    return np.array(points_3d), np.array(point_colors)


def build_tracks(matches_dict):
    """マッチングからトラックを構築"""
    print("\nトラック構築中...")
    
    # Union-Find (Disjoint Set) でトラックを構築
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # マッチングからトラックを構築
    for (img_i, img_j), matches in matches_dict.items():
        for feat_i, feat_j in matches:
            union((img_i, feat_i), (img_j, feat_j))
    
    # トラックをグループ化
    tracks = defaultdict(list)
    for key in parent.keys():
        root = find(key)
        tracks[root].append(key)
    
    # トラックIDを再割り当て
    track_data = {}
    for track_id, observations in enumerate(tracks.values()):
        if len(observations) >= 2:
            track_data[track_id] = observations
    
    print(f"  構築されたトラック数: {len(track_data)}")
    
    # トラック長の分布を表示
    track_lengths = [len(obs) for obs in track_data.values()]
    if len(track_lengths) > 0:
        print(f"  トラック長 - 最小: {min(track_lengths)}, 最大: {max(track_lengths)}, 平均: {np.mean(track_lengths):.2f}")
    
    return track_data


def visualize_matches_sample(features_dict, matches_dict, num_samples=3):
    """マッチングのサンプルを可視化"""
    print("\nマッチング可視化中...")
    
    sample_pairs = list(matches_dict.items())[:num_samples]
    
    for (img_i, img_j), matches in sample_pairs:
        kp_i, _, image_i = features_dict[img_i]
        kp_j, _, image_j = features_dict[img_j]
        
        # cv2.drawMatchesのためにDMatchオブジェクトを作成
        dmatch_list = [cv2.DMatch(m[0], m[1], 0) for m in matches[:50]]  # 最初の50個
        
        img_matches = cv2.drawMatches(
            image_i, kp_i, image_j, kp_j, dmatch_list, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"マッチング: 画像 {img_i} ↔ 画像 {img_j} ({len(matches)} マッチ)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def filter_outlier_points_statistical(points_3d, point_colors, std_multiplier=3.0):
    """統計的外れ値除去 (COLMAPスタイル)"""
    if len(points_3d) < 10:
        return points_3d, point_colors
    
    print(f"\n統計的外れ値フィルタリング中 (前: {len(points_3d)} 点)...")
    
    # 点群の中心と標準偏差を計算
    center = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - center, axis=1)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # 閾値を超える点を除外
    threshold = mean_dist + std_multiplier * std_dist
    mask = distances <= threshold
    
    filtered_points = points_3d[mask]
    filtered_colors = point_colors[mask]
    
    print(f"  後: {len(filtered_points)} 点 (除外: {len(points_3d) - len(filtered_points)} 点)")
    
    return filtered_points, filtered_colors


def visualize_reconstruction(gripper_poses, T_cam2gripper, points_3d, point_colors):
    """カメラポーズと3D点群を可視化"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # カメラポーズを計算
    camera_positions = []
    camera_orientations = []
    
    for T_gb in gripper_poses:
        T_camera_world = T_gb @ T_cam2gripper
        camera_positions.append(T_camera_world[:3, 3])
        camera_orientations.append(T_camera_world[:3, :3])
    
    camera_positions = np.array(camera_positions)
    
    # 3D点群をプロット (明確な色で)
    if len(points_3d) > 0:
        # サンプリング
        if len(points_3d) > 20000:
            indices = np.random.choice(len(points_3d), 20000, replace=False)
            points_subset = points_3d[indices]
            colors_subset = point_colors[indices] / 255.0
        else:
            points_subset = points_3d
            colors_subset = point_colors / 255.0
        
        ax.scatter(points_subset[:, 0], points_subset[:, 1], points_subset[:, 2],
                   c=colors_subset, marker='.', s=3, alpha=0.8, label=f'3D点群 ({len(points_3d)}点)')
    
    # カメラ位置をプロット
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='blue', marker='o', s=150, label='カメラ位置', edgecolors='white', linewidths=2)
    
    # カメラの向きを矢印で表示
    arrow_length = 80
    for idx, (pos, ori) in enumerate(zip(camera_positions, camera_orientations)):
        direction = ori[:, 0] * arrow_length
        ax.quiver(pos[0], pos[1], pos[2],
                 direction[0], direction[1], direction[2],
                 color='red', alpha=0.9, arrow_length_ratio=0.15, linewidths=2)
    
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('3D再構成: 固定カメラポーズ + COLMAP風ロバスト三角測量', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # 軸範囲調整
    if len(points_3d) > 0:
        all_points = np.vstack([camera_positions, points_3d])
    else:
        all_points = camera_positions
    
    max_range = np.ptp(all_points, axis=0).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='ChArUco Hand-Eye Calibration + Robust 3D Reconstruction')
    parser.add_argument('--images', type=str, required=True, help='画像ファイルパス (ワイルドカード可)')
    parser.add_argument('--poses', type=str, required=True, help='ポーズJSONファイルパス')
    parser.add_argument('--output-dir', type=str, default='output', help='出力ディレクトリ')
    parser.add_argument('--squares-x', type=int, default=7, help='ChArUcoボード正方形数(X)')
    parser.add_argument('--squares-y', type=int, default=5, help='ChArUcoボード正方形数(Y)')
    parser.add_argument('--square-length', type=float, default=32.0, help='正方形サイズ(mm)')
    parser.add_argument('--marker-length', type=float, default=24.0, help='マーカーサイズ(mm)')
    parser.add_argument('--dict', type=str, default='4X4_50', help='ArUco辞書')
    parser.add_argument('--reconstruct', action='store_true', help='3D再構成実行')
    parser.add_argument('--min-tri-angle', type=float, default=1.0, help='最小三角測量角度(度)')
    parser.add_argument('--max-reproj-error', type=float, default=4.0, help='最大再投影誤差(px)')
    parser.add_argument('--visualize-matches', action='store_true', help='マッチング可視化')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ArUco辞書とChArUcoボード作成
    aruco_dict_name = f'DICT_{args.dict}'
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict
    )
    
    # ポーズデータ読み込み
    captures = load_pose_data(args.poses)
    print(f"ポーズデータ読み込み: {len(captures)}件")
    
    # 画像ファイル取得
    image_files = sorted(glob.glob(args.images))
    print(f"画像ファイル検出: {len(image_files)}枚\n")
    
    # === Phase 1: カメラキャリブレーション ===
    print("="*70)
    print("Phase 1: カメラキャリブレーション")
    print("="*70)
    camera_matrix, dist_coeffs, _, _ = calibrate_camera_charuco(image_files, aruco_dict, board)
    
    print("\n=== カメラ内部パラメータ ===")
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    # === Phase 2: ボードポーズ推定 ===
    print("\n" + "="*70)
    print("Phase 2: ボードポーズ推定")
    print("="*70)
    board_poses, valid_indices = estimate_board_poses(
        image_files, camera_matrix, dist_coeffs, aruco_dict, board
    )
    
    if len(board_poses) < 3:
        raise ValueError(f"Hand-Eyeキャリブレーションには最低3枚必要 (有効: {len(board_poses)}枚)")
    
    # === Phase 3: Hand-Eye変換推定 ===
    print("\n" + "="*70)
    print("Phase 3: Hand-Eye変換推定")
    print("="*70)
    gripper_poses = []
    for idx in valid_indices:
        pose = captures[idx]['pose']
        T = pose_to_transformation_matrix(pose)
        gripper_poses.append(T)
    
    print(f"Hand-Eyeキャリブレーション実行中 ({len(board_poses)}ペア)...")
    T_cam2gripper = solve_hand_eye_calibration(gripper_poses, board_poses)
    
    print("\n=== Hand-Eye変換行列 (Camera to Gripper) ===")
    print(T_cam2gripper)
    
    rot = R.from_matrix(T_cam2gripper[:3, :3])
    euler_angles = rot.as_euler('xyz', degrees=True)
    print("\n変換 (並進 [mm], 回転 [度]):")
    print(f"  X: {T_cam2gripper[0, 3]:.3f} mm")
    print(f"  Y: {T_cam2gripper[1, 3]:.3f} mm")
    print(f"  Z: {T_cam2gripper[2, 3]:.3f} mm")
    print(f"  RX: {euler_angles[0]:.3f} deg")
    print(f"  RY: {euler_angles[1]:.3f} deg")
    print(f"  RZ: {euler_angles[2]:.3f} deg")
    
    # === Phase 4: カメラポーズ計算・出力 ===
    print("\n" + "="*70)
    print("Phase 4: カメラポーズ計算・COLMAP出力")
    print("="*70)
    
    # カメラポーズと投影行列を事前計算
    camera_poses = {}
    projection_matrices = {}
    
    for local_i, img_idx in enumerate(valid_indices):
        T_gb = gripper_poses[local_i]
        T_camera_world = T_gb @ T_cam2gripper
        T_world_camera = np.linalg.inv(T_camera_world)
        
        camera_poses[img_idx] = T_world_camera
        P = camera_matrix @ T_world_camera[:3, :]
        projection_matrices[img_idx] = P
    
    # COLMAP形式出力
    images_txt_path = os.path.join(args.output_dir, 'images.txt')
    write_colmap_images_txt(images_txt_path, image_files, valid_indices, gripper_poses, T_cam2gripper)
    
    # === Phase 5: 3D再構成 (オプション) ===
    if args.reconstruct:
        print("\n" + "="*70)
        print("Phase 5: 3D再構成 (COLMAP風ロバスト処理)")
        print("="*70)
        
        # 特徴抽出
        features_dict = extract_and_undistort_sift_features(
            image_files, valid_indices, camera_matrix, dist_coeffs, max_features=8192
        )
        
        # マッチング
        matches_dict = robust_feature_matching(features_dict, valid_indices, ratio_threshold=0.75)
        
        if len(matches_dict) == 0:
            print("警告: マッチングペアが見つかりませんでした")
            points_3d = np.array([])
            point_colors = np.array([])
        else:
            # マッチング可視化
            if args.visualize_matches:
                visualize_matches_sample(features_dict, matches_dict, num_samples=2)
            
            # トラック構築
            track_data = build_tracks(matches_dict)
            
            # トラック単位で三角測量
            points_3d, point_colors = triangulate_tracks(
                track_data, features_dict, projection_matrices, camera_poses,
                min_tri_angle=args.min_tri_angle,
                max_reproj_error=args.max_reproj_error,
                min_track_length=2
            )
            
            # 統計的外れ値除去
            if len(points_3d) > 0:
                points_3d, point_colors = filter_outlier_points_statistical(
                    points_3d, point_colors, std_multiplier=2.5
                )
        
        # 可視化
        print("\n3D再構成結果を可視化中...")
        visualize_reconstruction(gripper_poses, T_cam2gripper, points_3d, point_colors)
        
        # 統計情報表示
        if len(points_3d) > 0:
            print("\n=== 最終点群統計 ===")
            print(f"点数: {len(points_3d)}")
            print(f"点群範囲 (mm):")
            print(f"  X: [{np.min(points_3d[:, 0]):.1f}, {np.max(points_3d[:, 0]):.1f}]")
            print(f"  Y: [{np.min(points_3d[:, 1]):.1f}, {np.max(points_3d[:, 1]):.1f}]")
            print(f"  Z: [{np.min(points_3d[:, 2]):.1f}, {np.max(points_3d[:, 2]):.1f}]")
    else:
        print("\nカメラポーズのみを可視化中...")
        visualize_reconstruction(gripper_poses, T_cam2gripper, np.array([]), np.array([]))
    
    print("\n" + "="*70)
    print("処理完了!")
    print("="*70)


if __name__ == '__main__':
    main()