import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation
import open3d as o3d
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# --- グローバル設定・基本関数 (変更なし) ---
SQUARES_X, SQUARES_Y = 7, 5
SQUARE_LENGTH, MARKER_LENGTH = 0.032, 0.024
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, CHARUCO_DICT)
CHARUCO_DETECTOR = cv2.aruco.CharucoDetector(BOARD)

# load_captures_from_json, euler_to_transform_matrix, calibrate_camera_charuco, 
# calibrate_hand_eye は前回のコードと同じです。
def load_captures_from_json(json_path):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        print(f"Successfully loaded {len(data['captures'])} records from {json_path}")
        return data['captures']
    except Exception as e: print(f"Error loading JSON: {e}"); return None

def euler_to_transform_matrix(pose):
    t = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]) / 1000.0
    R = Rotation.from_euler('xyz', [pose['rx_deg'], pose['ry_deg'], pose['rz_deg']], degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T

def calibrate_camera_charuco(captures, image_dir):
    print("\n--- Starting Camera Calibration ---")
    all_corners, all_ids, valid_indices, img_size = [], [], [], None
    for i, cap in enumerate(captures):
        img_path = os.path.join(image_dir, cap['image_file'])
        if not os.path.exists(img_path): continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_size is None: img_size = img.shape[::-1]
        corners, ids, _, _ = CHARUCO_DETECTOR.detectBoard(img)
        if ids is not None and len(ids) > 3:
            all_corners.append(corners); all_ids.append(ids); valid_indices.append(i)
    if len(all_corners) < 4: print("Error: Not enough views for calibration."); return [None]*4
    print(f"Calibrating using {len(all_corners)} valid views...")
    ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, BOARD, img_size, None, None)
    if ret:
        print("Camera calibration successful."); print(f"K:\n{K}\nD:\n{D}")
        return K, D, valid_indices, (rvecs, tvecs)
    else: print("Error: Camera calibration failed."); return [None]*4

def calibrate_hand_eye(robot_poses, target_poses):
    print("\n--- Starting Hand-Eye Calibration ---")
    R_g, t_g = [H[:3, :3] for H in robot_poses], [H[:3, 3] for H in robot_poses]
    R_t, t_t = [H[:3, :3] for H in target_poses], [H[:3, 3] for H in target_poses]
    R, t = cv2.calibrateHandEye(R_g, t_g, R_t, t_t, method=cv2.CALIB_HAND_EYE_TSAI)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.ravel()
    print("Hand-Eye calibration successful."); print(f"T_hand_to_cam:\n{T}")
    return T


def get_robust_depth_map_for_view(
    ref_idx, neighbor_indices, images, camera_poses, K, D, sgbm_params
):
    """
    【NEW CORE LOGIC】
    ある1つの基準ビューのロバストな深度マップを、複数の近傍ビューとの
    幾何学的な一貫性チェックを通じて生成する。
    """
    ref_img_bgr = images[ref_idx]
    h, w, _ = ref_img_bgr.shape
    img_size = (w, h)
    
    # 近傍ビューごとに深度仮説を計算
    depth_hypotheses = []
    
    sgbm_left = cv2.StereoSGBM_create(**sgbm_params)
    sgbm_right = cv2.ximgproc.createRightMatcher(sgbm_left)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm_left)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)

    for neighbor_idx in neighbor_indices:
        T_ref_neighbor = np.linalg.inv(camera_poses[ref_idx]) @ camera_poses[neighbor_idx]
        R, t = T_ref_neighbor[:3, :3], T_ref_neighbor[:3, 3]
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D, K, D, img_size, R, t, alpha=0)
        
        map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, img_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, img_size, cv2.CV_32FC1)

        img1_rect = cv2.remap(cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY), map1x, map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(cv2.cvtColor(images[neighbor_idx], cv2.COLOR_BGR2GRAY), map2x, map2y, cv2.INTER_LINEAR)

        disp_left = sgbm_left.compute(img1_rect, img2_rect)
        disp_right = sgbm_right.compute(img2_rect, img1_rect)
        filtered_disp = wls_filter.filter(disp_left, img1_rect, None, disp_right)

        # BUG FIX: reprojectImageTo3Dにはfloat32が必要
        points_3d = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32), Q)
        depth_map = points_3d[:, :, 2]
        
        # 逆平行化変換を行い、元の基準ビューのピクセルに対応させる
        inv_map_x, inv_map_y = cv2.initInverseRectificationMap(K, D, R1, P1, img_size, cv2.CV_32FC1)
        unrectified_depth = cv2.remap(depth_map, inv_map_x, inv_map_y, cv2.INTER_LINEAR)
        
        depth_hypotheses.append(unrectified_depth)

    if not depth_hypotheses:
        return None

    # --- 幾何学的な一貫性に基づく深度融合 ---
    depths_stack = np.stack(depth_hypotheses, axis=-1)
    
    # 固定の幾何学ノイズ閾値 (シーンスケールに依存しない)
    GEOMETRIC_CONSISTENCY_STD_THRESH = 0.02 # 2cm
    MIN_CONSISTENT_VIEWS = 3 # 少なくとも3ビューが一致することを要求

    valid_mask = (depths_stack > 0.1) & (depths_stack < 5.0)
    
    # 有効な観測がMIN_CONSISTENT_VIEWSに満たないピクセルは除外
    num_valid_views = np.sum(valid_mask, axis=-1)
    sufficient_views_mask = (num_valid_views >= MIN_CONSISTENT_VIEWS)
    
    # 標準偏差を計算
    # 有効でない値は計算から除外するため、一時的にnanに置換
    depths_stack_masked = np.where(valid_mask, depths_stack, np.nan)
    std_devs = np.nanstd(depths_stack_masked, axis=-1)
    
    # ばらつきが小さい（＝一貫性がある）ピクセルのみを選択
    consistent_std_mask = (std_devs < GEOMETRIC_CONSISTENCY_STD_THRESH)
    
    # 両方の条件を満たすピクセルのみが最終的に採用される
    final_mask = sufficient_views_mask & consistent_std_mask
    
    # 信頼できるピクセルの中央値を最終的な深度値とする
    fused_depth_map = np.zeros((h, w), dtype=np.float32)
    median_depths = np.nanmedian(depths_stack_masked, axis=-1)
    fused_depth_map[final_mask] = median_depths[final_mask]

    return fused_depth_map


def reconstruct_dense_point_cloud(captures, image_dir, camera_poses, K, D):
    """【REVISED】COLMAP風のMVSロジックを導入した密な再構成"""
    print("\n--- Starting Dense Reconstruction with COLMAP-inspired MVS Logic ---")
    
    images = [cv2.imread(os.path.join(image_dir, cap['image_file'])) for cap in captures]
    num_views = len(images)
    
    sgbm_params = dict(minDisparity=0, numDisparities=128, blockSize=5,
                       P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
                       uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
                       mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    final_pcd = o3d.geometry.PointCloud()

    for ref_idx in tqdm(range(num_views), desc="Processing each view as reference"):
        # 基準ビューに対する適切な近傍ビューを選択 (ベースラインを考慮)
        neighbors = []
        for i in range(num_views):
            if i == ref_idx: continue
            baseline = np.linalg.norm(camera_poses[ref_idx][:3, 3] - camera_poses[i][:3, 3])
            if 0.05 < baseline < 0.5: # 5cmから50cmのベースラインを持つビューを選択
                neighbors.append(i)
        
        if len(neighbors) < 2: continue # 深度仮説を比較するのに十分な近傍がない

        # 選択した近傍ビューを最大5つに制限
        neighbors = neighbors[:5]

        # この基準ビューに対するロバストな深度マップを計算
        robust_depth = get_robust_depth_map_for_view(ref_idx, neighbors, images, camera_poses, K, D, sgbm_params)
        
        if robust_depth is None: continue
        
        # 深度マップから点群を生成し、ワールド座標系に変換
        h, w = robust_depth.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        # 深度が0より大きい（有効な）ピクセルのみを対象
        valid_pixels = np.where(robust_depth > 0)
        v, u = valid_pixels
        z = robust_depth[v, u]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_cam = np.vstack((x, y, z)).T
        colors = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2RGB)[v, u]
        
        points_hom = np.hstack((points_cam, np.ones((len(points_cam), 1))))
        points_world = (camera_poses[ref_idx] @ points_hom.T).T[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        final_pcd += pcd

    print(f"Fused point cloud has {len(final_pcd.points)} points before final cleanup.")

    # --- 最終クリーンアップ ---
    if not final_pcd.has_points():
        print("Reconstruction resulted in an empty point cloud.")
        return None
        
    final_pcd = final_pcd.voxel_down_sample(voxel_size=0.005)
    cl, ind = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    final_pcd = final_pcd.select_by_index(ind)
    print(f"Final point cloud has {len(final_pcd.points)} points.")
    
    return final_pcd

# --- メイン関数と可視化関数 (変更なし) ---
def visualize_reconstruction(camera_poses, pcd):
    print("\n--- Visualizing Results ---")
    if pcd is None or not pcd.has_points():
        print("Point cloud is empty. Nothing to visualize.")
        return
    camera_frames = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(T) for T in camera_poses]
    world_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    print("Displaying final dense point cloud and camera poses...")
    o3d.visualization.draw_geometries([pcd, world_origin] + camera_frames)

def main(args):
    captures = load_captures_from_json(args.json_file)
    if not captures: return
    
    image_dir = args.image_dir if args.image_dir else Path(args.json_file).parent
    all_robot_poses = [euler_to_transform_matrix(cap['pose']) for cap in captures]

    K, D, valid_indices, target_pose_data = calibrate_camera_charuco(captures, image_dir)
    if K is None: return
        
    valid_robot_poses = [all_robot_poses[i] for i in valid_indices]
    rvecs, tvecs = target_pose_data
    valid_target_poses = []
    for rvec, tvec in zip(rvecs, tvecs):
        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rvec)
        T[:3, 3] = tvec.ravel()
        valid_target_poses.append(T)

    T_hand_cam = calibrate_hand_eye(valid_robot_poses, valid_target_poses)
    all_camera_poses = [robot_pose @ T_hand_cam for robot_pose in all_robot_poses]
    print("\nCalculated all camera poses in world frame.")

    dense_pcd = reconstruct_dense_point_cloud(captures, image_dir, all_camera_poses, K, D)
    visualize_reconstruction(all_camera_poses, dense_pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs dense 3D reconstruction using a COLMAP-inspired MVS logic.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file with poses.")
    parser.add_argument("--image_dir", type=str, help="Path to the image directory.")
    args = parser.parse_args()
    main(args)