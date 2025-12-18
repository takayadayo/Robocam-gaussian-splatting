import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CameraPose:
    """カメラポーズを表すデータクラス"""
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    
    def get_rotation_matrix(self) -> np.ndarray:
        """クォータニオンから回転行列を計算"""
        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def get_translation(self) -> np.ndarray:
        """並進ベクトルを取得"""
        return np.array([self.tx, self.ty, self.tz])
    
    def get_camera_center(self) -> np.ndarray:
        """カメラ中心（ワールド座標）を計算
        C = -R^T * t
        """
        R = self.get_rotation_matrix()
        t = self.get_translation()
        return -R.T @ t


def parse_colmap_images(filepath: str) -> Dict[str, CameraPose]:
    """
    COLMAP images.txtファイルを解析
    
    フォーマット:
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    """
    poses = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # コメント行と空行をスキップ
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) > 10:
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            pose = CameraPose(
                image_id=image_id,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=tx, ty=ty, tz=tz,
                camera_id=camera_id,
                name=name
            )
            poses[name] = pose
    
    return poses


def compute_relative_motion(pose1: CameraPose, pose2: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
    """
    pose1からpose2への相対運動を計算
    
    Returns:
        relative_R: 相対回転行列 (3x3)
        relative_t: 相対並進ベクトル (3,) - 正規化されていない
    """
    R1 = pose1.get_rotation_matrix()
    R2 = pose2.get_rotation_matrix()
    C1 = pose1.get_camera_center()
    C2 = pose2.get_camera_center()
    
    # 相対回転: R_rel = R2 * R1^T
    relative_R = R2 @ R1.T
    
    # 相対並進: t_rel = R1 * (C2 - C1)
    relative_t = R1 @ (C2 - C1)
    
    return relative_R, relative_t


def compute_rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    2つの回転行列の誤差を計算（角度差、度数法）
    
    誤差 = arccos((trace(R1^T * R2) - 1) / 2)
    """
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)


def compute_translation_direction_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """
    2つの並進ベクトルの方向の角度差を計算（度数法）
    スケール不変
    
    Returns:
        angle_error: 方向の角度差（度数法）
    """
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        # 並進がほぼゼロの場合
        return 0.0
    
    # 正規化して方向のみを比較
    t1_normalized = t1 / norm1
    t2_normalized = t2 / norm2
    cos_angle = np.clip(np.dot(t1_normalized, t2_normalized), -1.0, 1.0)
    angle_error = np.degrees(np.arccos(cos_angle))
    
    return angle_error


def estimate_scale_factor(
    poses1: Dict[str, CameraPose],
    poses2: Dict[str, CameraPose],
    sorted_names: List[str]
) -> float:
    """
    2つのポーズセット間のスケール比を推定
    
    各連続画像ペアの相対並進ノルムの比の中央値を使用
    
    Returns:
        scale: poses2のスケールをposes1のスケールに合わせる係数
               (poses2のスケール) / (poses1のスケール)
    """
    if len(sorted_names) < 2:
        return 1.0
    
    scale_ratios = []
    
    for i in range(len(sorted_names) - 1):
        name_prev = sorted_names[i]
        name_curr = sorted_names[i + 1]
        
        # データセット1の相対並進ノルム
        _, t1_rel = compute_relative_motion(poses1[name_prev], poses1[name_curr])
        norm1 = np.linalg.norm(t1_rel)
        
        # データセット2の相対並進ノルム
        _, t2_rel = compute_relative_motion(poses2[name_prev], poses2[name_curr])
        norm2 = np.linalg.norm(t2_rel)
        
        # 両方のノルムが十分大きい場合のみ使用
        if norm1 > 1e-6 and norm2 > 1e-6:
            scale_ratios.append(norm1 / norm2)
    
    if not scale_ratios:
        return 1.0
    
    # 中央値を使用（外れ値に頑健）
    scale = np.median(scale_ratios)
    return scale


def compute_relative_motion_errors(
    images_file1: str,
    images_file2: str,
    scale_invariant: bool = True,
    estimate_scale: bool = True
) -> Tuple[List[Dict], Optional[float]]:
    """
    2つのimages.txtファイルから相対運動の誤差を計算
    
    Args:
        images_file1: 1つ目のimages.txtファイルパス（基準）
        images_file2: 2つ目のimages.txtファイルパス
        scale_invariant: Trueの場合、並進は方向のみを比較（スケール不変）
        estimate_scale: Trueの場合、2つのポーズセット間のスケール比を推定
    
    Returns:
        results: 各画像ペアの誤差情報のリスト
        scale_factor: 推定されたスケール係数（estimate_scale=Falseの場合はNone）
    """
    # ポーズを読み込み
    poses1 = parse_colmap_images(images_file1)
    poses2 = parse_colmap_images(images_file2)
    
    # 共通の画像名を取得してソート（画像ID順）
    common_names = set(poses1.keys()) & set(poses2.keys())
    sorted_names = sorted(common_names, key=lambda x: poses1[x].image_id)
    
    print(f"共通画像数: {len(sorted_names)}")
    
    if len(sorted_names) < 2:
        print("警告: 相対運動を計算するには最低2枚の画像が必要です")
        return [], None
    
    # スケール係数を推定
    scale_factor = None
    if estimate_scale:
        scale_factor = estimate_scale_factor(poses1, poses2, sorted_names)
        print(f"推定スケール係数: {scale_factor:.6f}")
        print(f"  (データセット2の並進を {scale_factor:.6f} 倍してデータセット1に合わせます)")
    
    results = []
    
    # 連続する画像ペア間の相対運動を計算
    for i in range(len(sorted_names) - 1):
        name_prev = sorted_names[i]
        name_curr = sorted_names[i + 1]
        
        # データセット1の相対運動
        pose1_prev = poses1[name_prev]
        pose1_curr = poses1[name_curr]
        R1_rel, t1_rel = compute_relative_motion(pose1_prev, pose1_curr)
        
        # データセット2の相対運動
        pose2_prev = poses2[name_prev]
        pose2_curr = poses2[name_curr]
        R2_rel, t2_rel = compute_relative_motion(pose2_prev, pose2_curr)
        
        # スケール補正を適用
        if estimate_scale and scale_factor is not None:
            t2_rel_scaled = t2_rel * scale_factor
        else:
            t2_rel_scaled = t2_rel
        
        # 回転誤差（スケール不変）
        rotation_error_deg = compute_rotation_error(R1_rel, R2_rel)
        
        # 並進方向誤差（スケール不変）
        trans_direction_error = compute_translation_direction_error(t1_rel, t2_rel_scaled)
        
        result = {
            'image_pair': f"{name_prev} -> {name_curr}",
            'image_id_prev': pose1_prev.image_id,
            'image_id_curr': pose1_curr.image_id,
            'rotation_error_deg': rotation_error_deg,
            'translation_direction_error_deg': trans_direction_error,
            'translation_norm_1': np.linalg.norm(t1_rel),
            'translation_norm_2': np.linalg.norm(t2_rel),
            'translation_norm_2_scaled': np.linalg.norm(t2_rel_scaled) if estimate_scale else None,
        }
        
        # スケール補正後のノルム誤差も計算（参考値）
        if estimate_scale and scale_factor is not None:
            result['translation_magnitude_error_scaled'] = abs(
                np.linalg.norm(t1_rel) - np.linalg.norm(t2_rel_scaled)
            )
        
        results.append(result)
    
    return results, scale_factor


def print_summary_statistics(results: List[Dict], scale_factor: Optional[float] = None):
    """統計情報を表示"""
    if not results:
        print("結果がありません")
        return
    
    rotation_errors = [r['rotation_error_deg'] for r in results]
    trans_direction_errors = [r['translation_direction_error_deg'] for r in results]
    
    print("\n" + "="*80)
    print("統計情報（スケール不変メトリクス）")
    print("="*80)
    
    if scale_factor is not None:
        print(f"\n推定スケール係数: {scale_factor:.6f}")
    
    print("\n【回転誤差（度）】")
    print(f"  平均: {np.mean(rotation_errors):.4f}°")
    print(f"  中央値: {np.median(rotation_errors):.4f}°")
    print(f"  標準偏差: {np.std(rotation_errors):.4f}°")
    print(f"  最小値: {np.min(rotation_errors):.4f}°")
    print(f"  最大値: {np.max(rotation_errors):.4f}°")
    
    print("\n【並進方向誤差（度）- スケール不変】")
    print(f"  平均: {np.mean(trans_direction_errors):.4f}°")
    print(f"  中央値: {np.median(trans_direction_errors):.4f}°")
    print(f"  標準偏差: {np.std(trans_direction_errors):.4f}°")
    print(f"  最小値: {np.min(trans_direction_errors):.4f}°")
    print(f"  最大値: {np.max(trans_direction_errors):.4f}°")
    
    # スケール補正後のノルム誤差統計（参考値）
    if 'translation_magnitude_error_scaled' in results[0] and results[0]['translation_magnitude_error_scaled'] is not None:
        trans_mag_errors = [r['translation_magnitude_error_scaled'] for r in results]
        print("\n【並進ノルム誤差（スケール補正後）- 参考値】")
        print(f"  平均: {np.mean(trans_mag_errors):.6f}")
        print(f"  中央値: {np.median(trans_mag_errors):.6f}")
        print(f"  標準偏差: {np.std(trans_mag_errors):.6f}")
        print(f"  最小値: {np.min(trans_mag_errors):.6f}")
        print(f"  最大値: {np.max(trans_mag_errors):.6f}")


# 使用例
if __name__ == "__main__":
    # ファイルパスを指定
    images_file2 = "C:/Users/takay/robocam_gs/known_pose_BA/images.txt"
    images_file1 = "C:/Users/takay/Downloads/robocam_dataset/colmap/robocam_2/images.txt"
    
    # スケール不変な相対運動誤差を計算（推奨）
    print("\n=== スケール不変モード（推奨） ===")
    results, scale_factor = compute_relative_motion_errors(
        images_file1, 
        images_file2,
        scale_invariant=True,
        estimate_scale=True
    )
    
    # 詳細結果を表示
    print("\n" + "="*80)
    print("画像ペア毎の相対運動誤差")
    print("="*80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['image_pair']}")
        print(f"  画像ID: {result['image_id_prev']} -> {result['image_id_curr']}")
        print(f"  回転誤差: {result['rotation_error_deg']:.4f}°")
        print(f"  並進方向誤差: {result['translation_direction_error_deg']:.4f}°")
        print(f"  並進ノルム1: {result['translation_norm_1']:.6f}")
        print(f"  並進ノルム2: {result['translation_norm_2']:.6f}")
        if result['translation_norm_2_scaled'] is not None:
            print(f"  並進ノルム2(補正後): {result['translation_norm_2_scaled']:.6f}")
    
    # 統計情報を表示
    print_summary_statistics(results, scale_factor)
    
    # CSVに保存する場合
    # import csv
    # with open('relative_motion_errors.csv', 'w', newline='') as f:
    #     if results:
    #         writer = csv.DictWriter(f, fieldnames=results[0].keys())
    #         writer.writeheader()
    #         writer.writerows(results)