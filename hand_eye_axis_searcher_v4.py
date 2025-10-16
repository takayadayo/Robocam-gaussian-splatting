# hand_eye_axis_searcher_v4.py

import cv2
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import itertools

class HandEyeAxisSearcher:
    """
    ロボットの回転軸の定義(軸の順序・符号)が不明な状況を想定し、
    考えられる全ての軸マッピングとオイラー角規約の組み合わせを総当たりで検証し、
    最適な定義を特定するクラス。
    """
    
    # 評価対象のオイラー角規約
    EULER_CONVENTIONS = [
        'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx',
        'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'
    ]
    
    def __init__(self, json_path: str, image_dir: str):
        # (v3から変更なし)
        self.json_path = json_path; self.image_dir = Path(image_dir)
        with open(json_path, 'r') as f: self.captures = json.load(f)['captures']
        self.charuco_squares_x = 7; self.charuco_squares_y = 5; self.square_length = 0.032; self.marker_length = 0.024
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard((self.charuco_squares_x, self.charuco_squares_y), self.square_length, self.marker_length, self.aruco_dict)
        self.obj_points = self.charuco_board.getChessboardCorners()
        self.camera_matrix = None; self.dist_coeffs = None

    def _generate_axis_mappings(self) -> List[Dict]:
        """48通りの軸マッピング仮説(順列と符号)を生成する"""
        base_indices = [0, 1, 2] # 0:x, 1:y, 2:z
        base_signs = [1, -1]
        
        mappings = []
        
        # 3! = 6通りの順列
        for p in itertools.permutations(base_indices):
            # 2^3 = 8通りの符号
            for sx in base_signs:
                for sy in base_signs:
                    for sz in base_signs:
                        mapping = {
                            "permutation": p, # (i,j,k) -> rxはi番目, ryはj番目, rzはk番目の軸
                            "signs": (sx, sy, sz),
                            "name": f"P({p[0]}{p[1]}{p[2]}) S({sx:+} {sy:+} {sz:+})"
                        }
                        mappings.append(mapping)
        return mappings

    def _apply_axis_mapping(self, angles_deg: List[float], mapping: Dict) -> List[float]:
        """指定された軸マッピングを適用して回転ベクトルを変換する"""
        p = mapping["permutation"]
        s = mapping["signs"]
        
        # 元の (rx, ry, rz)
        rx, ry, rz = angles_deg
        
        # 新しい回転ベクトルを作成
        new_angles = [0, 0, 0]
        new_angles[p[0]] = rx * s[0]
        new_angles[p[1]] = ry * s[1]
        new_angles[p[2]] = rz * s[2]
        
        return new_angles

    def _euler_to_matrix(self, pose: Dict, convention: str, axis_mapping: Dict) -> np.ndarray:
        """軸マッピングを適用後、オイラー角から4x4同次変換行列を生成"""
        original_angles = [pose['rx_deg'], pose['ry_deg'], pose['rz_deg']]
        mapped_angles_deg = self._apply_axis_mapping(original_angles, axis_mapping)
        
        angles_rad = np.radians(mapped_angles_deg)
        rotation = R.from_euler(convention, angles_rad).as_matrix()
        
        translation = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]) / 1000.0
        matrix = np.eye(4); matrix[0:3, 0:3] = rotation; matrix[0:3, 3] = translation
        return matrix

    def calibrate_camera(self) -> bool:
        # (v3から変更なし)
        print("1. カメラ内部パラメータのキャリブレーションを開始...")
        all_corners, all_ids, image_size = [], [], None
        for capture in self.captures:
            img = cv2.imread(str(self.image_dir / capture['image_file']))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None: image_size = gray.shape[::-1]
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            if marker_ids is not None and len(marker_ids) > 3:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, self.charuco_board)
                if charuco_corners is not None and len(charuco_corners) > 4: all_corners.append(charuco_corners); all_ids.append(charuco_ids)
        if len(all_corners) < 5: print("エラー: キャリブレーションに十分な画像がありません"); return False
        ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, self.charuco_board, image_size, None, None)
        if not ret: print("エラー: カメラキャリブレーションに失敗"); return False
        self.camera_matrix = mtx; self.dist_coeffs = dist
        print(f"キャリブレーション完了 (再投影誤差: {ret:.4f} pixels, 画像数: {len(all_corners)}枚)")
        return True

    def _calculate_global_reprojection_error(self, T_base2gripper_abs_list, T_target2cam_abs_list, T_gripper2cam, all_img_points, all_obj_points_indices) -> float:
        # (v3から変更なし)
        T_cam2target_abs_list = [np.linalg.inv(T) for T in T_target2cam_abs_list]
        T_base2target_list = [A @ T_gripper2cam @ B_inv for A, B_inv in zip(T_base2gripper_abs_list, T_cam2target_abs_list)]
        rotations = [T[0:3, 0:3] for T in T_base2target_list]; translations = [T[0:3, 3] for T in T_base2target_list]
        avg_rotation = R.from_matrix(rotations).mean().as_matrix(); avg_translation = np.mean(translations, axis=0)
        T_base2target_avg = np.eye(4); T_base2target_avg[0:3, 0:3] = avg_rotation; T_base2target_avg[0:3, 3] = avg_translation
        total_error_sq, total_points = 0, 0
        for i in range(len(T_base2gripper_abs_list)):
            T_base2cam_pred = T_base2gripper_abs_list[i] @ T_gripper2cam
            T_cam2base_pred = np.linalg.inv(T_base2cam_pred); T_target2cam_pred = T_cam2base_pred @ T_base2target_avg
            rvec_pred, _ = cv2.Rodrigues(T_target2cam_pred[0:3, 0:3]); tvec_pred = T_target2cam_pred[0:3, 3]
            obj_points_subset = self.obj_points[all_obj_points_indices[i].ravel()]
            projected_points, _ = cv2.projectPoints(obj_points_subset, rvec_pred, tvec_pred, self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(all_img_points[i], projected_points, cv2.NORM_L2)
            total_error_sq += error * error; total_points += len(all_img_points[i])
        return np.sqrt(total_error_sq / total_points)

    def search_best_definition(self) -> Dict:
        """全ての軸マッピングとオイラー角規約の組み合わせを検証する"""
        if self.camera_matrix is None and not self.calibrate_camera(): return None
        
        print("\n2. Hand-Eyeキャリブレーション用の観測データ(B)を準備中...")
        T_target2cam_abs_list, valid_captures, all_img_points, all_obj_points_indices = [], [], [], []
        # (データ準備部分はv3から変更なし)
        for capture in self.captures:
            gray = cv2.imread(str(self.image_dir / capture['image_file']), cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 4:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
                if charuco_ids is not None and len(charuco_ids) > 4:
                    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.camera_matrix, self.dist_coeffs, None, None)
                    if success:
                        T = np.eye(4); T[0:3, 0:3], _ = cv2.Rodrigues(rvec); T[0:3, 3] = tvec.ravel()
                        T_target2cam_abs_list.append(T); valid_captures.append(capture); all_img_points.append(charuco_corners); all_obj_points_indices.append(charuco_ids)
        print(f"  有効ポーズ数: {len(valid_captures)}枚")

        axis_mappings = self._generate_axis_mappings()
        all_results = []
        
        print(f"\n3. {len(axis_mappings)}通りの軸定義 x {len(self.EULER_CONVENTIONS)}通りのオイラー規約 = {len(axis_mappings)*len(self.EULER_CONVENTIONS)}通りの総当たり検証を開始...")
        
        # 外側ループ: 軸マッピング
        for i, mapping in enumerate(axis_mappings):
            # 内側ループ: オイラー角規約
            for convention in self.EULER_CONVENTIONS:
                T_base2gripper_abs_list = [self._euler_to_matrix(cap['pose'], convention, mapping) for cap in valid_captures]
                
                R_gripper_rel, t_gripper_rel, R_target_rel, t_target_rel = [], [], [], []
                for j in range(len(T_base2gripper_abs_list) - 1):
                    T_g_rel = np.linalg.inv(T_base2gripper_abs_list[j]) @ T_base2gripper_abs_list[j+1]
                    R_gripper_rel.append(T_g_rel[0:3, 0:3]); t_gripper_rel.append(T_g_rel[0:3, 3].reshape(3, 1))
                    T_c_rel = T_target2cam_abs_list[j] @ np.linalg.inv(T_target2cam_abs_list[j+1])
                    R_target_rel.append(T_c_rel[0:3, 0:3]); t_target_rel.append(T_c_rel[0:3, 3].reshape(3, 1))

                try:
                    R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(R_gripper_rel, t_gripper_rel, R_target_rel, t_target_rel, method=cv2.CALIB_HAND_EYE_TSAI)
                except cv2.error: continue

                T_gripper2cam = np.eye(4); T_gripper2cam[0:3, 0:3] = R_gripper2cam; T_gripper2cam[0:3, 3] = t_gripper2cam.ravel()
                error = self._calculate_global_reprojection_error(T_base2gripper_abs_list, T_target2cam_abs_list, T_gripper2cam, all_img_points, all_obj_points_indices)
                
                if np.isnan(error) or np.isinf(error): continue
                
                print(f"  検証中 ({i*12+self.EULER_CONVENTIONS.index(convention)+1}/{len(axis_mappings)*12}) - 軸: {mapping['name']}, 規約: {convention:5s}, 誤差: {error:.4f} pixels")

                all_results.append({
                    'axis_mapping_name': mapping['name'],
                    'axis_mapping_def': mapping,
                    'convention': convention,
                    'error': error
                })
        
        if not all_results: print("エラー: 有効な結果が得られませんでした。"); return None
        
        all_results.sort(key=lambda x: x['error'])
        return {'all_results': all_results}

def main():
    json_path = "hand_eye_calibration/json/poses.json"
    image_dir = "hand_eye_calibration/image"
    
    print("="*30); print(" Hand-Eyeキャリブレーション: 未知の回転軸定義とオイラー規約の網羅的探索 (v4)"); print("="*30)
    
    searcher = HandEyeAxisSearcher(json_path, image_dir)
    results_data = searcher.search_best_definition()
    
    if results_data:
        best_result = results_data['all_results'][0]
        
        print("\n" + "="*30); print(" 最終評価結果サマリー"); print("="*30)
        
        print("\n[結論] 最も整合性の高いロボット回転軸の定義とオイラー角規約の組み合わせが特定されました。")
        print(f"\n  > 最適な軸マッピング: {best_result['axis_mapping_name']}")
        print(f"    (これは、poses.jsonの(rx,ry,rz)を、permutation={best_result['axis_mapping_def']['permutation']}とsigns={best_result['axis_mapping_def']['signs']}で変換することを意味します)")
        print(f"  > 最適なオイラー角規約: '{best_result['convention']}'")
        print(f"  > 最小グローバル再投影誤差: {best_result['error']:.4f} pixels")
        
        if best_result['error'] < 1.0:
            print("\n[評価] 誤差が1.0ピクセル未満であり、極めて信頼性の高い結果です。")
        elif best_result['error'] < 2.0:
            print("\n[評価] 誤差が2.0ピクセル未満であり、良好で実用的な結果と考えられます。")
        else:
            print("\n[警告] 最小誤差が2.0ピクセルを超えています。TCPのオフセットや同期ズレなど、他の要因がまだ存在する可能性があります。")

        print("\n--- 上位5つの組み合わせ ---")
        for i, res in enumerate(results_data['all_results'][:5], 1):
            print(f"{i}. 軸: {res['axis_mapping_name']:25s} | 規約: {res['convention']:6s} | 誤差: {res['error']:.4f} pixels")
            
        output_file = 'axis_search_results_v4.json'
        with open(output_file, 'w') as f: json.dump(results_data, f, indent=2)
        print(f"\n詳細結果を '{output_file}' に保存しました。")

if __name__ == "__main__":
    main()