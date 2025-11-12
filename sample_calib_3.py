#!/usr/bin/env python3
"""
Hand-Eye Calibration and 3D Reconstruction System
ロボットアームのポーズ情報とChArUcoマーカー画像を用いた高精度3D再構成
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import open3d as o3d


class HandEyeCalibration:
    """Hand-Eyeキャリブレーションシステム"""
    
    def __init__(self, aruco_dict_type=cv2.aruco.DICT_4X4_50, 
                 marker_size_mm=24.0, squares_x=7, squares_y=5):
        """
        Args:
            aruco_dict_type: ArUco辞書タイプ
            marker_size_mm: マーカーサイズ（mm）
            squares_x, squares_y: チェスボードの内部コーナー数
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            squareLength=marker_size_mm * 1.5 / 1000.0,  # m単位
            markerLength=marker_size_mm / 1000.0,
            dictionary=self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # カメラパラメータ
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Hand-Eye変換
        self.R_cam2gripper = None  # カメラ→グリッパー回転
        self.t_cam2gripper = None  # カメラ→グリッパー並進
        
        # データストレージ
        self.captures_data = []
        self.valid_captures = []
        
    def load_captures(self, json_path: str, image_dir: str):
        """キャプチャデータの読み込み"""
        print(f"📂 Loading captures from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.captures_data = []
        image_dir = Path(image_dir)
        
        for cap in data['captures']:
            img_path = image_dir / cap['image_file']
            if img_path.exists():
                self.captures_data.append({
                    'image_path': str(img_path),
                    'pose': cap['pose'],
                    'timestamp': cap['timestamp']
                })
        
        print(f"✅ Loaded {len(self.captures_data)} captures")
        
    def detect_charuco_corners(self, image_path: str) -> Optional[Tuple]:
        """ChArUcoコーナーの検出"""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ArUcoマーカー検出
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 4:
            return None
        
        # ChArUcoコーナー補間
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        
        if num_corners < 8:
            return None
            
        return charuco_corners, charuco_ids, img, corners, ids
    
    def calibrate_camera(self):
        """カメラ内部パラメータのキャリブレーション"""
        print("\n🎯 Camera Intrinsic Calibration")
        
        all_corners = []
        all_ids = []
        img_size = None
        
        for i, cap in enumerate(self.captures_data):
            result = self.detect_charuco_corners(cap['image_path'])
            if result is None:
                continue
                
            charuco_corners, charuco_ids, img, _, _ = result
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            
            if img_size is None:
                img_size = img.shape[:2][::-1]
            
            self.valid_captures.append(cap)
            
            if i < 3:  # 最初の3枚だけ表示
                self._visualize_detection(img, charuco_corners, charuco_ids, i)
        
        print(f"✅ Valid captures: {len(all_corners)}/{len(self.captures_data)}")
        
        if len(all_corners) < 5:
            raise ValueError("有効なキャプチャが不足しています")
        
        # キャリブレーション実行
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = \
            cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, self.board, img_size, None, None
            )
        
        print(f"📐 Reprojection Error: {ret:.4f} pixels")
        print(f"📷 Camera Matrix:\n{self.camera_matrix}")
        print(f"📊 Distortion Coeffs: {self.dist_coeffs.ravel()}")
        
        return rvecs, tvecs
    
    def _visualize_detection(self, img, corners, ids, idx):
        """検出結果の可視化"""
        vis_img = img.copy()
        cv2.aruco.drawDetectedCornersCharuco(vis_img, corners, ids)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f'ChArUco Detection - Image {idx+1}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def estimate_hand_eye_transformation(self):
        """Hand-Eye変換行列の推定（Eye-in-Hand構成）"""
        print("\n🤖 Hand-Eye Transformation Estimation")
        
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []
        
        for cap in self.valid_captures:
            # グリッパー→ベース変換（ロボットアームのポーズ）
            pose = cap['pose']
            R_g2b = self._euler_to_rotation_matrix(
                pose['rx_deg'], pose['ry_deg'], pose['rz_deg']
            )
            t_g2b = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]) / 1000.0
            
            # ターゲット（ChArUcoボード）→カメラ変換
            result = self.detect_charuco_corners(cap['image_path'])
            if result is None:
                continue
            
            charuco_corners, charuco_ids, _, _, _ = result
            
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board,
                self.camera_matrix, self.dist_coeffs, None, None
            )
            
            if not ret:
                continue
            
            R_t2c, _ = cv2.Rodrigues(rvec)
            t_t2c = tvec.ravel()
            
            R_gripper2base_list.append(R_g2b)
            t_gripper2base_list.append(t_g2b)
            R_target2cam_list.append(R_t2c)
            t_target2cam_list.append(t_t2c)
        
        print(f"✅ Valid poses for Hand-Eye: {len(R_gripper2base_list)}")
        
        # OpenCVのHand-Eyeキャリブレーション（Eye-in-Hand）
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_list,
            t_gripper2base_list,
            R_target2cam_list,
            t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        self.R_cam2gripper = R_cam2gripper
        self.t_cam2gripper = t_cam2gripper.ravel()
        
        print(f"🔗 Hand-Eye Rotation:\n{self.R_cam2gripper}")
        print(f"🔗 Hand-Eye Translation: {self.t_cam2gripper}")
        
    def compute_camera_poses(self) -> List[Dict]:
        """各画像のカメラポーズを算出（Hand-Eye変換を使用）"""
        print("\n📸 Computing Camera Poses")
        
        camera_poses = []
        
        for i, cap in enumerate(self.valid_captures):
            # グリッパー→ベース変換
            pose = cap['pose']
            R_g2b = self._euler_to_rotation_matrix(
                pose['rx_deg'], pose['ry_deg'], pose['rz_deg']
            )
            t_g2b = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]) / 1000.0
            
            # カメラ→ベース変換 = (グリッパー→ベース) * (カメラ→グリッパー)
            R_cam2base = R_g2b @ self.R_cam2gripper
            t_cam2base = R_g2b @ self.t_cam2gripper + t_g2b
            
            # ベース→カメラ変換（カメラポーズ）
            R_base2cam = R_cam2base.T
            t_base2cam = -R_base2cam @ t_cam2base
            
            camera_poses.append({
                'image_path': cap['image_path'],
                'R': R_base2cam,
                't': t_base2cam,
                'timestamp': cap['timestamp']
            })
            
            if i < 3:
                print(f"  Camera {i+1}: t={t_base2cam}")
        
        return camera_poses
    
    def _euler_to_rotation_matrix(self, rx_deg, ry_deg, rz_deg):
        """オイラー角から回転行列へ変換（ZYX順）"""
        r = Rotation.from_euler('ZYX', [rz_deg, ry_deg, rx_deg], degrees=True)
        return r.as_matrix()
    
    def visualize_camera_poses(self, camera_poses: List[Dict]):
        """カメラポーズの3D可視化"""
        print("\n🎨 Visualizing Camera Poses")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # カメラ位置
        positions = np.array([pose['t'] for pose in camera_poses])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='royalblue', s=100, marker='o', label='Camera Centers')
        
        # カメラの向き（光軸）
        for i, pose in enumerate(camera_poses):
            # Z軸方向（カメラの前方）
            direction = pose['R'] @ np.array([0, 0, 0.1])
            ax.quiver(pose['t'][0], pose['t'][1], pose['t'][2],
                     direction[0], direction[1], direction[2],
                     color='red', arrow_length_ratio=0.3, linewidth=2)
        
        # 原点（ベースフレーム）
        ax.scatter([0], [0], [0], c='gold', s=200, marker='*', 
                  label='World Origin', edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('Camera Poses in World Frame', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # アスペクト比の調整
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def export_colmap_format(self, camera_poses: List[Dict], output_path: str):
        """COLMAP形式でカメラポーズを出力"""
        print(f"\n💾 Exporting to COLMAP format: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for i, pose in enumerate(camera_poses):
                # 回転行列→クォータニオン
                r = Rotation.from_matrix(pose['R'])
                quat = r.as_quat()  # [x, y, z, w]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                
                tx, ty, tz = pose['t']
                
                img_name = Path(pose['image_path']).name
                
                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_name}\n")
                f.write("\n")  # 空の2D点リスト
        
        print(f"✅ Exported {len(camera_poses)} camera poses")


class SfMReconstructor:
    """Structure from Motion 3D再構成器"""
    
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # SIFT特徴抽出器（高品質設定）
        self.sift = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.03,
            edgeThreshold=15,
            sigma=1.2
        )
        
        # FLANN マッチャー
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.images_data = []
        self.point_cloud = []
        self.point_colors = []
        
    def load_images_with_poses(self, camera_poses: List[Dict]):
        """画像とポーズデータの読み込み"""
        print("\n🖼️  Loading images with known poses")
        
        for pose in camera_poses:
            img = cv2.imread(pose['image_path'])
            if img is None:
                continue
            
            # アンディストーション
            img_undist = cv2.undistort(img, self.camera_matrix, self.dist_coeffs)
            
            # グレースケール変換
            gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
            
            # 特徴点抽出
            kp, desc = self.sift.detectAndCompute(gray, None)
            
            self.images_data.append({
                'image': img_undist,
                'gray': gray,
                'keypoints': kp,
                'descriptors': desc,
                'R': pose['R'],
                't': pose['t'],
                'P': self._compute_projection_matrix(pose['R'], pose['t']),
                'image_path': pose['image_path']
            })
        
        print(f"✅ Loaded {len(self.images_data)} images")
        for i, data in enumerate(self.images_data[:3]):
            print(f"  Image {i+1}: {len(data['keypoints'])} keypoints")
    
    def _compute_projection_matrix(self, R, t):
        """投影行列の計算 P = K[R|t]"""
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = self.camera_matrix @ Rt
        return P
    
    def match_features_robust(self, idx1: int, idx2: int) -> Tuple[np.ndarray, np.ndarray]:
        """ロバストな特徴マッチング"""
        desc1 = self.images_data[idx1]['descriptors']
        desc2 = self.images_data[idx2]['descriptors']
        
        if desc1 is None or desc2 is None:
            return np.array([]), np.array([])
        
        # KNN マッチング
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 20:
            return np.array([]), np.array([])
        
        # 対応点の取得
        pts1 = np.float32([self.images_data[idx1]['keypoints'][m.queryIdx].pt 
                          for m in good_matches])
        pts2 = np.float32([self.images_data[idx2]['keypoints'][m.trainIdx].pt 
                          for m in good_matches])
        
        # RANSAC による外れ値除去
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if mask is None:
            return np.array([]), np.array([])
        
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        
        return pts1, pts2
    
    def triangulate_points(self, idx1: int, idx2: int, pts1: np.ndarray, pts2: np.ndarray):
        """三角測量による3D点の復元"""
        P1 = self.images_data[idx1]['P']
        P2 = self.images_data[idx2]['P']
        
        # 三角測量
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        # カメラの前方にある点のみ保持（チェイラリティチェック）
        valid_points = []
        valid_colors = []
        
        img1 = self.images_data[idx1]['image']
        
        for i, pt3d in enumerate(points_3d):
            # 両方のカメラから見て前方にあるか確認
            pt1_cam = self.images_data[idx1]['R'] @ pt3d + self.images_data[idx1]['t']
            pt2_cam = self.images_data[idx2]['R'] @ pt3d + self.images_data[idx2]['t']
            
            if pt1_cam[2] > 0 and pt2_cam[2] > 0:
                # 再投影誤差チェック
                reproj1 = P1 @ np.append(pt3d, 1)
                reproj1 = reproj1[:2] / reproj1[2]
                error1 = np.linalg.norm(reproj1 - pts1[i])
                
                reproj2 = P2 @ np.append(pt3d, 1)
                reproj2 = reproj2[:2] / reproj2[2]
                error2 = np.linalg.norm(reproj2 - pts2[i])
                
                if error1 < 3.0 and error2 < 3.0:  # 再投影誤差閾値
                    valid_points.append(pt3d)
                    
                    # 色情報の取得
                    x, y = int(pts1[i][0]), int(pts1[i][1])
                    if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
                        color = img1[y, x][::-1] / 255.0  # BGR→RGB, 正規化
                        valid_colors.append(color)
                    else:
                        valid_colors.append([0.5, 0.5, 0.5])
        
        return np.array(valid_points), np.array(valid_colors)
    
    def reconstruct_3d(self):
        """3D再構成の実行"""
        print("\n🏗️  3D Reconstruction")
        
        n_images = len(self.images_data)
        
        # 全ペア間で三角測量
        for i in range(n_images):
            for j in range(i + 1, n_images):
                print(f"  Processing pair ({i+1}, {j+1})...")
                
                pts1, pts2 = self.match_features_robust(i, j)
                
                if len(pts1) < 20:
                    print(f"    ⚠️  Insufficient matches: {len(pts1)}")
                    continue
                
                # マッチング可視化（最初のペアのみ）
                if i == 0 and j == 1:
                    self._visualize_matches(i, j, pts1, pts2)
                
                points_3d, colors = self.triangulate_points(i, j, pts1, pts2)
                
                if len(points_3d) > 0:
                    self.point_cloud.append(points_3d)
                    self.point_colors.append(colors)
                    print(f"    ✅ Triangulated {len(points_3d)} points")
        
        # 点群の結合
        if len(self.point_cloud) > 0:
            self.point_cloud = np.vstack(self.point_cloud)
            self.point_colors = np.vstack(self.point_colors)
            
            # 外れ値除去（統計的フィルタリング）
            self.point_cloud, self.point_colors = self._remove_outliers(
                self.point_cloud, self.point_colors
            )
            
            print(f"\n✅ Total reconstructed points: {len(self.point_cloud)}")
        else:
            print("\n⚠️  No points reconstructed")
    
    def _visualize_matches(self, idx1, idx2, pts1, pts2):
        """特徴マッチングの可視化"""
        img1 = self.images_data[idx1]['image']
        img2 = self.images_data[idx2]['image']
        
        # 画像を横に並べる
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        
        vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
        
        # マッチングを描画（最大50個）
        n_show = min(50, len(pts1))
        for i in range(n_show):
            pt1 = tuple(pts1[i].astype(int))
            pt2 = tuple((pts2[i] + [w1, 0]).astype(int))
            
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
            cv2.line(vis, pt1, pt2, color, 1)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches: Image {idx1+1} ↔ Image {idx2+1} ({len(pts1)} matches)', 
                 fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _remove_outliers(self, points, colors, nb_neighbors=20, std_ratio=2.0):
        """統計的外れ値除去"""
        print("\n🧹 Removing outliers...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 統計的外れ値除去
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                  std_ratio=std_ratio)
        
        points_clean = np.asarray(cl.points)
        colors_clean = colors[ind]
        
        print(f"  Removed {len(points) - len(points_clean)} outliers")
        print(f"  Remaining points: {len(points_clean)}")
        
        return points_clean, colors_clean
    
    def visualize_reconstruction(self):
        """3D再構成結果の可視化"""
        if len(self.point_cloud) == 0:
            print("⚠️  No points to visualize")
            return
        
        print("\n🎨 Visualizing 3D Reconstruction")
        
        # Open3Dで可視化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(self.point_colors)
        
        # カメラフラスタムの追加
        frustums = []
        for i, data in enumerate(self.images_data):
            frustum = self._create_camera_frustum(data['R'], data['t'], scale=0.05)
            frustums.append(frustum)
        
        # 座標軸
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        
        # 可視化
        o3d.visualization.draw_geometries(
            [pcd, coord_frame] + frustums,
            window_name="3D Reconstruction",
            width=1200, height=900,
            point_show_normal=False
        )
    
    def _create_camera_frustum(self, R, t, scale=0.05):
        """カメラフラスタムの作成"""
        # カメラ中心
        center = -R.T @ t
        
        # フラスタムの頂点（カメラ座標系）
        pts_cam = np.array([
            [0, 0, 0],
            [-1, -1, 2],
            [1, -1, 2],
            [1, 1, 2],
            [-1, 1, 2]
        ]) * scale
        
        # ワールド座標系へ変換
        pts_world = (R.T @ pts_cam.T).T + center
        
        # LineSet作成
        lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [[1, 0, 0] for _ in range(len(lines))]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Hand-Eye Calibration and 3D Reconstruction'
    )
    parser.add_argument('--json', required=True, help='Path to captures JSON file')
    parser.add_argument('--images', required=True, help='Path to images directory')
    parser.add_argument('--marker-size', type=float, default=24.0,
                       help='ArUco marker size in mm (default: 50.0)')
    parser.add_argument('--squares-x', type=int, default=7,
                       help='Number of squares in X direction (default: 5)')
    parser.add_argument('--squares-y', type=int, default=5,
                       help='Number of squares in Y direction (default: 7)')
    parser.add_argument('--output-colmap', default='images.txt',
                       help='Output path for COLMAP format (default: images.txt)')
    
    args = parser.parse_args()
    
    print("=" * 30)
    print("  Hand-Eye Calibration & 3D Reconstruction System")
    print("=" * 30)
    
    # ===============================================
    # Phase 1: Hand-Eye Calibration
    # ===============================================
    print("\n" + "=" * 30)
    print("  PHASE 1: Hand-Eye Calibration")
    print("=" * 30)
    
    calib = HandEyeCalibration(
        aruco_dict_type=cv2.aruco.DICT_4X4_50,
        marker_size_mm=args.marker_size,
        squares_x=args.squares_x,
        squares_y=args.squares_y
    )
    
    # データ読み込み
    calib.load_captures(args.json, args.images)
    
    # カメラキャリブレーション
    rvecs, tvecs = calib.calibrate_camera()
    
    # Hand-Eye変換推定
    calib.estimate_hand_eye_transformation()
    
    # カメラポーズ算出
    camera_poses = calib.compute_camera_poses()
    
    # カメラポーズ可視化
    calib.visualize_camera_poses(camera_poses)
    
    # COLMAP形式でエクスポート
    calib.export_colmap_format(camera_poses, args.output_colmap)
    
    # ===============================================
    # Phase 2: 3D Reconstruction
    # ===============================================
    print("\n" + "=" * 30)
    print("  PHASE 2: 3D Reconstruction with Known Poses")
    print("=" * 30)
    
    reconstructor = SfMReconstructor(
        calib.camera_matrix,
        calib.dist_coeffs
    )
    
    # 画像と既知ポーズの読み込み
    reconstructor.load_images_with_poses(camera_poses)
    
    # 3D再構成実行
    reconstructor.reconstruct_3d()
    
    # 結果の可視化
    reconstructor.visualize_reconstruction()
    
    # ===============================================
    # Summary
    # ===============================================
    print("\n" + "=" * 30)
    print("  SUMMARY")
    print("=" * 30)
    print(f"✅ Camera Intrinsics:")
    print(f"   fx={calib.camera_matrix[0,0]:.2f}, fy={calib.camera_matrix[1,1]:.2f}")
    print(f"   cx={calib.camera_matrix[0,2]:.2f}, cy={calib.camera_matrix[1,2]:.2f}")
    print(f"\n✅ Hand-Eye Transform (Camera → Gripper):")
    print(f"   Translation: {calib.t_cam2gripper}")
    print(f"\n✅ Camera Poses: {len(camera_poses)} poses computed")
    print(f"   Exported to: {args.output_colmap}")
    print(f"\n✅ 3D Reconstruction: {len(reconstructor.point_cloud)} points")
    print("=" * 30)
    print("\n🎉 Processing Complete!")


if __name__ == "__main__":
    main()