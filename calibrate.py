import cv2
import numpy as np
import json
import os
import glob
from scipy.spatial.transform import Rotation as R

# --- 1. 設定値 ---
# ChArUcoボードの仕様
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH_M = 0.032  # 32mm
MARKER_LENGTH_M = 0.024  # 24mm
# ARUCO_DICT = cv2.aruco.DICT_4X4_50 # OpenCV 4.7.0以降
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


# ファイルパス設定
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(ROOT_DIR, "hand_eye_calibration")
IMAGE_DIR = os.path.join(CALIB_DIR, "image")
JSON_PATH = os.path.join(CALIB_DIR, "json", "poses.json") # JSONファイル名を仮定

# ロボットのオイラー角の回転順序（ロボットの仕様書で要確認）
# 'xyz'は一般的だが、'zyx'などもあり得る。大文字は外因性(extrinsic)、小文字は内因性(intrinsic)
EULER_SEQ = 'xyz' 

# --- 2. ChArUcoボードのセットアップ ---
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    ARUCO_DICT
)
# detector = cv2.aruco.CharucoDetector(board) # OpenCV 4.7.0以降
detector_params = cv2.aruco.DetectorParameters()
# charuco_params = cv2.aruco.CharucoParameters()


def calibrate_camera_intrinsics(image_paths):
    """
    全画像を用いてカメラの内部パラメータをキャリブレーションする
    """
    print("--- カメラ内部パラメータのキャリブレーションを開始 ---")
    all_charuco_corners = []
    all_charuco_ids = []
    img_size = None

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 画像ファイルが読み込めません: {img_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        # OpenCV 4.7.0以降
        # charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        # OpenCV 4.7.0以前
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)
        
        if marker_ids is not None and len(marker_ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            if ret > 0:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    if len(all_charuco_corners) < 5:
        print("エラー: キャリブレーションに必要なボードが検出された画像が少なすぎます。")
        return None, None, None

    print(f"{len(all_charuco_corners)} / {len(image_paths)} 枚の画像でボードを検出しました。")

    # キャリブレーション実行
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        img_size,
        None,
        None
    )

    if ret:
        print("カメラ内部パラメータのキャリブレーションが成功しました。")
        print("カメラ行列 (K):")
        print(camera_matrix)
        print("\n歪み係数 (D):")
        print(dist_coeffs)
        return camera_matrix, dist_coeffs, img_size
    else:
        print("エラー: カメラ内部パラメータのキャリブレーションに失敗しました。")
        return None, None, None


def get_robot_poses_from_json(json_path):
    """JSONからロボットのポーズ情報を読み込み、4x4同次変換行列のリストを返す"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # タイムスタンプでソートして、画像とポーズの順序を確実にする
    captures = sorted(data['captures'], key=lambda c: c['timestamp'])
    
    robot_poses_T = []
    image_files = []
    
    for capture in captures:
        pose = capture['pose']
        
        # 単位をメートルとラジアンに変換
        pos_m = np.array([pose['x_mm'], pose['y_mm'], pose['z_mm']]) / 1000.0
        
        # SciPyを使用してオイラー角から回転行列に変換
        rot = R.from_euler(EULER_SEQ, [pose['rx_deg'], pose['ry_deg'], pose['rz_deg']], degrees=True)
        rot_matrix = rot.as_matrix()

        # 4x4同次変換行列を作成
        T = np.identity(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = pos_m
        
        robot_poses_T.append(T)
        image_files.append(capture['image_file'])
        
    return robot_poses_T, image_files


def get_camera_poses_from_images(image_paths, camera_matrix, dist_coeffs):
    """
    画像からChArUcoボードのポーズを検出し、カメラの絶対ポーズ(T_target_to_camera)のリストを返す
    """
    camera_poses_T = []
    valid_image_paths = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)

        if marker_ids is not None and len(marker_ids) > 4:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            
            # ボードのポーズを推定
            # rvec, tvecはターゲット座標系から見たカメラのポーズではなく、
            # カメラ座標系から見たターゲットのポーズを返す
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None, useExtrinsicGuess=False
            )

            if success:
                # T_cam_to_target (カメラから見たターゲット) を作成
                rot_matrix, _ = cv2.Rodrigues(rvec)
                T_cam_to_target = np.identity(4)
                T_cam_to_target[:3, :3] = rot_matrix
                T_cam_to_target[:3, 3] = tvec.flatten()
                
                # 我々が必要なのは逆変換 T_target_to_camera
                T_target_to_camera = np.linalg.inv(T_cam_to_target)
                
                camera_poses_T.append(T_target_to_camera)
                valid_image_paths.append(img_path)
                continue

        print(f"警告: {os.path.basename(img_path)} でボードを検出できませんでした。スキップします。")

    return camera_poses_T, valid_image_paths

def verify_euler_sequence(euler_seq_to_test):
    """_summary_

    Args:
        euler_seq_to_test (_type_): 
    
    指定されたおうらーかくの回転順序が正しいか検証。
    単純な単一軸回転の理論値と、プログラムによる変換結果を比較する。
    """
    
    print(f'\n--- オイラー角の回転機序`{euler_seq_to_test}`の検証 ---')
    
    pos_mm = np.array([0.0, 0.0, 0.0])
    rot_deg = np.array([0.0, 0.0, 90.0])
    
    try:
        r_obj = R.from_euler(euler_seq_to_test, rot_deg, degrees=True)
        computed_matrix = r_obj.as_matrix()
        
    except Exception as e:
        print(f'エラー：Scipyが回転順序`{euler_seq_to_test}` を認識できませんでした。詳細：{e}')
        return False
    
    # 理論値
    theta = np.pi/2
    c, s = np.cos(theta), np.sin(theta)
    theoretical_matrix_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    diff = np.linalg.norm(computed_matrix - theoretical_matrix_z)
    is_correct = np.allclose(computed_matrix, theoretical_matrix_z, atol=1e-6)
    
    print("ケース1: Z軸周り+90度回転 [rx,ry,rz]=[0,0,90]")
    print("プログラムによる計算結果:")
    print(np.round(computed_matrix, 3))
    print("\n理論値:")
    print(np.round(theoretical_matrix_z, 3))
    print(f"行列の差 (ノルム): {diff:.6f}")
    if is_correct:
        print("結果: OK - 理論値と一致しました。")
        # 1つのテストが通れば、そのシーケンスは正しい可能性が非常に高い
        return True
    else:
        print("結果: NG - 理論値と一致しません。")
        # Z軸がダメでも、他の軸が合う可能性も稀にあるため、テストを続ける
    
    # --- ケース2: Y軸周りに+90度回転 ---
    # 仮定: [x,y,z] = [0,0,0], [rx,ry,rz] = [0,90,0]
    rot_deg = np.array([0.0, 90.0, 0.0])
    r_obj = R.from_euler(euler_seq_to_test, rot_deg, degrees=True)
    computed_matrix = r_obj.as_matrix()
    
    # 理論値 (Y軸+90度回転)
    theoretical_matrix_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    is_correct = np.allclose(computed_matrix, theoretical_matrix_y, atol=1e-6)

    print("\nケース2: Y軸周り+90度回転 [rx,ry,rz]=[0,90,0]")
    print("プログラムによる計算結果:")
    print(np.round(computed_matrix, 3))
    print("\n理論値:")
    print(np.round(theoretical_matrix_y, 3))
    if is_correct:
        print("結果: OK - 理論値と一致しました。")
        return True
    else:
        print("結果: NG - 理論値と一致しません。")

    print(f"\n結論: 回転順序 '{euler_seq_to_test}' は、ロボットの仕様と一致していない可能性が高いです。")
    return False

    
def main():
    # グローバル変数を関数冒頭で宣言
    global EULER_SEQ
    
    print("--- Hand-Eye キャリブレーション スクリプト開始 ---")

    # --- Step 0: オイラー角の回転順序を検証 ---
    print("--- Step 0: オイラー角の回転順序を検証 ---")
    
    # 一般的な回転順序の候補
    candidate_sequences = ['xyz', 'zyx', 'XYZ', 'ZYX', 'xzy', 'yxz'] 
    
    verified_seq = None
    for seq in candidate_sequences:
        if verify_euler_sequence(seq):
            verified_seq = seq
            print(f"\n>> 検証成功: 回転順序は '{verified_seq}' であると判断しました。")
            print("-" * 50)
            break
    
    if verified_seq is None:
        print("\n\n[!! 警告 !!]")
        print("どの候補の回転順序も理論値と一致しませんでした。")
        print(f"スクリプト上部で定義された EULER_SEQ = '{EULER_SEQ}' を使用して処理を続行します。")
        print("ロボットのマニュアル等で回転順序を再確認し、スクリプトの EULER_SEQ を直接編集することを強く推奨します。")
        print("キャリブレーション結果は不正確になる可能性が高いです。")
        print("-" * 50)
        # どの候補も合わなかった場合、元々のグローバル変数 EULER_SEQ をそのまま使う
        # verified_seq = EULER_SEQ は不要
    else:
        # 検証に成功した場合のみ、グローバル変数を更新
        EULER_SEQ = verified_seq

    # --- Step 1: カメラ内部パラメータのキャリブレーション ---
    print("\n--- Step 1: カメラ内部パラメータのキャリブレーション ---")
    image_paths_all = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if not image_paths_all:
        print(f"エラー: {IMAGE_DIR} に画像が見つかりません。")
        return

    camera_matrix, dist_coeffs, img_size = calibrate_camera_intrinsics(image_paths_all)
    if camera_matrix is None:
        return

    # --- Step 2: 各時点での絶対ポーズ行列を取得 ---
    print("\n--- Step 2: 絶対ポーズ行列の読み込みと計算 ---")
    # ロボットの絶対ポーズ (T_base_to_hand)
    robot_poses_T_all, json_image_files = get_robot_poses_from_json(JSON_PATH)
    print(f"JSONから {len(robot_poses_T_all)} 個のロボットポーズを読み込みました。")
    
    # 画像からカメラの絶対ポーズ (T_target_to_camera)
    image_paths_for_pose = [os.path.join(IMAGE_DIR, f) for f in json_image_files]
    camera_poses_T_all, valid_image_paths = get_camera_poses_from_images(image_paths_for_pose, camera_matrix, dist_coeffs)
    print(f"画像から {len(camera_poses_T_all)} 個のカメラポーズを計算しました。")

    # ボードが検出できたポーズのみにロボットのポーズをフィルタリング
    valid_indices = [image_paths_for_pose.index(p) for p in valid_image_paths]
    robot_poses_T = [robot_poses_T_all[i] for i in valid_indices]
    camera_poses_T = [camera_poses_T_all[i] for i in valid_indices]
    
    
    print(f"\n最終的に {len(robot_poses_T)} 個の有効なポーズペアを使用します。")
    if len(robot_poses_T) < 3:
        print("エラー: キャリブレーションに必要なポーズ数が足りません (最低3つ必要)。")
        return

    # --- Step 3: 相対移動行列 (A と B) を計算 ---
    print("\n--- Step 3: 相対移動行列 (A, B) の計算 ---")
    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    for i in range(len(robot_poses_T) - 1):
        # ロボットの動き (A) = T_hand(i)_to_hand(i+1)
        T1_base_to_hand = robot_poses_T[i]
        T2_base_to_hand = robot_poses_T[i+1]
        A = np.linalg.inv(T1_base_to_hand) @ T2_base_to_hand
        
        R_gripper2base.append(A[:3, :3])
        t_gripper2base.append(A[:3, 3])

        # カメラの動き (B) = T_camera(i)_to_camera(i+1)
        T1_target_to_camera = camera_poses_T[i]
        T2_target_to_camera = camera_poses_T[i+1]
        B = np.linalg.inv(T1_target_to_camera) @ T2_target_to_camera

        R_target2cam.append(B[:3, :3])
        t_target2cam.append(B[:3, 3])
    print("相対移動行列の計算が完了しました。")

    # --- Step 4: AX=XBを解く ---
    print("\n--- Step 4: Hand-Eyeキャリブレーション (AX=XB) を実行 ---")
    # Tsai-Lenz法を使用
    R_hand2cam, t_hand2cam = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    print("cv2.calibrateHandEye の実行が完了しました。")

    # --- Step 5: 結果の表示と検証 ---
    T_hand_to_camera = np.identity(4)
    T_hand_to_camera[:3, :3] = R_hand2cam
    T_hand_to_camera[:3, 3] = t_hand2cam.flatten()

    print("\n\n--- [最終結果] Hand-to-Camera 変換行列 (X) ---")
    np.set_printoptions(precision=6, suppress=True)
    print(T_hand_to_camera)
    
    print("\nこの行列Xは、ロボットアーム先端(Hand)からカメラ(Eye)への変換を表します。")
    print("使用例: T_base_to_camera = T_base_to_hand @ T_hand_to_camera")

    # --- 簡単な検証 ---
    print("\n--- Step 5: 結果の検証 ---")
    print("異なる2つの視点からターゲットの位置を計算し、その差を確認します...")
    
    # 視点0からターゲットの位置を計算
    T_base_to_hand_0 = robot_poses_T[0]
    T_target_to_camera_0 = camera_poses_T[0]
    T_cam_to_target_0 = np.linalg.inv(T_target_to_camera_0)
    
    T_base_to_target_v0 = T_base_to_hand_0 @ T_hand_to_camera @ T_cam_to_target_0
    target_pos_v0 = T_base_to_target_v0[:3, 3]

    # 最後の視点からターゲットの位置を計算
    T_base_to_hand_last = robot_poses_T[-1]
    T_target_to_camera_last = camera_poses_T[-1]
    T_cam_to_target_last = np.linalg.inv(T_target_to_camera_last)
    
    T_base_to_target_v_last = T_base_to_hand_last @ T_hand_to_camera @ T_cam_to_target_last
    target_pos_v_last = T_base_to_target_v_last[:3, 3]
    
    print(f"視点0からのターゲット位置(m): {target_pos_v0}")
    print(f"視点{len(robot_poses_T)-1}からのターゲット位置(m): {target_pos_v_last}")
    
    error_mm = np.linalg.norm(target_pos_v0 - target_pos_v_last) * 1000
    print(f"\n>> 位置の一致性誤差: {error_mm:.4f} mm")
    if error_mm < 5:
        print(">> 検証結果: 良好 (誤差が5mm未満です)")
    elif error_mm < 20:
        print(">> 検証結果: 注意 (誤差が5mm以上です。キャリブレーション精度が中程度である可能性があります)")
    else:
        print(">> 検証結果: 警告 (誤差が20mm以上です。回転順序、カメラ内部パラメータ、ポーズデータ等の設定に問題がある可能性が高いです)")


if __name__ == "__main__":
    main()