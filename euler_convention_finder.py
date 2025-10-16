# hand_eye_euler_validator_v3.py

import cv2
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class HandEyeEulerValidator:
    """
    Hand-EyeキャリブレーションのAX=XB問題を解き、
    グローバルな再投影誤差に基づいて最適なオイラー角規約を特定するクラス。
    V3: ターゲットの相対運動(B_rel)の計算式を理論的に正しいものに修正。
    """
    
    EULER_CONVENTIONS = [
        'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx',
        'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'
    ]
    
    def __init__(self, json_path: str, image_dir: str):
        self.json_path = json_path
        self.image_dir = Path(image_dir)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.captures = data['captures']
        
        # Charucoボードのパラメータ
        self.charuco_squares_x = 7
        self.charuco_squares_y = 5
        self.square_length = 0.032  # 40mm
        self.marker_length = 0.024  # 30mm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.charuco_squares_x, self.charuco_squares_y),
            self.square_length, self.marker_length, self.aruco_dict
        )
        self.obj_points = self.charuco_board.getChessboardCorners()
        
        self.camera_matrix = None
        self.dist_coeffs = None

    def _euler_to_matrix(self, pose: Dict, convention: str) -> np.ndarray:
        # (変更なし)
        angles_rad = np.radians([pose['rx_deg'], pose['ry_deg'], pose['rz_deg']])
        rotation = R.from_euler(convention, angles_rad).as_matrix()
        translation = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]) / 1000.0
        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation
        matrix[0:3, 3] = translation
        return matrix

    def calibrate_camera(self) -> bool:
        # (変更なし)
        print("1. カメラ内部パラメータのキャリブレーションを開始...")
        all_corners, all_ids, image_size = [], [], None
        image_files = [self.image_dir / capture['image_file'] for capture in self.captures]
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None: image_size = gray.shape[::-1]
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            if marker_ids is not None and len(marker_ids) > 3:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.charuco_board
                )
                if charuco_corners is not None and len(charuco_corners) > 4:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
        if len(all_corners) < 5: return False
        ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, self.charuco_board, image_size, None, None
        )
        if not ret: return False
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        print("キャリブレーション完了")
        print(f"  再投影誤差: {ret:.4f} pixels")
        print(f"  検出画像数: {len(all_corners)}枚")
        return True

    def _calculate_global_reprojection_error(
        self,
        T_base2gripper_abs_list: List[np.ndarray],
        T_target2cam_abs_list: List[np.ndarray],
        T_gripper2cam: np.ndarray,
        all_img_points: List[np.ndarray],
        all_obj_points_indices: List[np.ndarray]
    ) -> float:
        # (変更なし)
        T_cam2target_abs_list = [np.linalg.inv(T) for T in T_target2cam_abs_list]
        T_base2target_list = [A @ T_gripper2cam @ B_inv for A, B_inv in zip(T_base2gripper_abs_list, T_cam2target_abs_list)]
        rotations = [T[0:3, 0:3] for T in T_base2target_list]
        translations = [T[0:3, 3] for T in T_base2target_list]
        avg_rotation = R.from_matrix(rotations).mean().as_matrix()
        avg_translation = np.mean(translations, axis=0)
        T_base2target_avg = np.eye(4); T_base2target_avg[0:3, 0:3] = avg_rotation; T_base2target_avg[0:3, 3] = avg_translation
        total_error_sq, total_points = 0, 0
        for i in range(len(T_base2gripper_abs_list)):
            T_base2cam_pred = T_base2gripper_abs_list[i] @ T_gripper2cam
            T_cam2base_pred = np.linalg.inv(T_base2cam_pred)
            T_target2cam_pred = T_cam2base_pred @ T_base2target_avg
            rvec_pred, _ = cv2.Rodrigues(T_target2cam_pred[0:3, 0:3])
            tvec_pred = T_target2cam_pred[0:3, 3]
            obj_points_subset = self.obj_points[all_obj_points_indices[i].ravel()]
            projected_points, _ = cv2.projectPoints(obj_points_subset, rvec_pred, tvec_pred, self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(all_img_points[i], projected_points, cv2.NORM_L2)
            total_error_sq += error * error
            total_points += len(all_img_points[i])
        return np.sqrt(total_error_sq / total_points)

    def validate_conventions(self) -> Dict:
        if self.camera_matrix is None and not self.calibrate_camera(): return None
        
        print("\n2. Hand-Eyeキャリブレーション用のデータセットを準備中...")
        T_target2cam_abs_list, valid_captures, all_img_points, all_obj_points_indices = [], [], [], []
        for capture in self.captures:
            img_path = str(self.image_dir / capture['image_file'])
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 4:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
                if charuco_ids is not None and len(charuco_ids) > 4:
                    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.camera_matrix, self.dist_coeffs, None, None)
                    if success:
                        T_target2cam = np.eye(4)
                        T_target2cam[0:3, 0:3], _ = cv2.Rodrigues(rvec)
                        T_target2cam[0:3, 3] = tvec.ravel()
                        T_target2cam_abs_list.append(T_target2cam)
                        valid_captures.append(capture)
                        all_img_points.append(charuco_corners)
                        all_obj_points_indices.append(charuco_ids)
        print(f"  Hand-Eyeキャリブレーションに使用可能な有効ポーズ数: {len(valid_captures)}枚")

        print("\n3. 全てのオイラー角規約を評価中...")
        results = []
        for convention in self.EULER_CONVENTIONS:
            T_base2gripper_abs_list = [self._euler_to_matrix(cap['pose'], convention) for cap in valid_captures]
            
            R_gripper_rel, t_gripper_rel = [], []
            R_target_rel, t_target_rel = [], []

            for i in range(len(T_base2gripper_abs_list) - 1):
                # グリッパの相対運動: A_rel = inv(A_i) * A_{i+1}
                T_g1 = T_base2gripper_abs_list[i]
                T_g2 = T_base2gripper_abs_list[i+1]
                T_g_rel = np.linalg.inv(T_g1) @ T_g2
                R_gripper_rel.append(T_g_rel[0:3, 0:3])
                t_gripper_rel.append(T_g_rel[0:3, 3].reshape(3, 1))

                # ターゲットの相対運動: B_rel = B_i * inv(B_{i+1})
                T_c1 = T_target2cam_abs_list[i]
                T_c2 = T_target2cam_abs_list[i+1]
                # --- 【最重要】ここがv2からの修正箇所 ---
                T_c_rel = T_c1 @ np.linalg.inv(T_c2)
                # --- 修正箇所ここまで ---
                R_target_rel.append(T_c_rel[0:3, 0:3])
                t_target_rel.append(T_c_rel[0:3, 3].reshape(3, 1))

            try:
                R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(
                    R_gripper_rel, t_gripper_rel, R_target_rel, t_target_rel,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )
            except cv2.error as e:
                print(f"  規約 '{convention}': Hand-Eyeキャリブレーションに失敗. スキップします. ({e.msg})")
                continue

            T_gripper2cam = np.eye(4); T_gripper2cam[0:3, 0:3] = R_gripper2cam; T_gripper2cam[0:3, 3] = t_gripper2cam.ravel()
            
            error = self._calculate_global_reprojection_error(
                T_base2gripper_abs_list, T_target2cam_abs_list, T_gripper2cam,
                all_img_points, all_obj_points_indices
            )
            
            print(f"  規約 '{convention:5s}': グローバル再投影誤差 = {error:.4f} pixels")
            results.append({'convention': convention, 'error': error, 'T_gripper2cam': T_gripper2cam.tolist()})
            
        if not results: return None
        results.sort(key=lambda x: x['error'])
        return {'all_results': results}

    def visualize_results(self, results_data: Dict, output_path: str = 'euler_convention_validation.png'):
        # (変更なし)
        if not results_data: return
        all_results = results_data['all_results']
        best_result = all_results[0]
        conventions = [r['convention'] for r in all_results]
        errors = [r['error'] for r in all_results]
        plt.figure(figsize=(14, 7)); bars = plt.bar(conventions, errors, color='skyblue'); bars[conventions.index(best_result['convention'])].set_color('salmon')
        plt.xlabel('オイラー角規約', fontsize=12); plt.ylabel('グローバル再投影誤差 (pixels)', fontsize=12); plt.title('オイラー角規約の妥当性評価 (Hand-Eyeキャリブレーション基準)', fontsize=14, weight='bold')
        plt.xticks(rotation=45); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
        plt.savefig(output_path, dpi=150); print(f"\n結果のグラフを '{output_path}' に保存しました。"); plt.close()

def main():
    json_path = "hand_eye_calibration/json/poses.json"
    image_dir = "hand_eye_calibration/image"
    
    print("="*70); print(" Hand-Eyeキャリブレーションに基づくオイラー角規約の妥当性検証 (v3)"); print("="*70)
    validator = HandEyeEulerValidator(json_path, image_dir)
    results_data = validator.validate_conventions()
    if results_data:
        best_result = results_data['all_results'][0]
        print("\n" + "="*70); print(" 評価結果サマリー"); print("="*70)
        print(f"\n[結論] 最も信頼できるオイラー角規約: '{best_result['convention']}'")
        print(f"       最小グローバル再投影誤差: {best_result['error']:.4f} pixels")
        print("\n--- 上位5つの規約 ---")
        for i, res in enumerate(results_data['all_results'][:5], 1): print(f"{i}. {res['convention']:6s}: {res['error']:.4f} pixels")
        validator.visualize_results(results_data)
        with open('euler_convention_validation_results_v3.json', 'w') as f: json.dump(results_data, f, indent=2)
        print("詳細結果を 'euler_convention_validation_results_v3.json' に保存しました。")
        print("\n--- 次のステップ ---")
        print(f"1. 確定した規約 '{best_result['convention']}' をロボット制御システムで標準として使用してください。")
        print(f"2. 最適な規約の誤差が1.0ピクセル以下であれば、非常に信頼性の高い結果と判断できます。")

if __name__ == "__main__":
    main()