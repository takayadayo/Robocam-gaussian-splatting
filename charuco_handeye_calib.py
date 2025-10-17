# calibrate_robot_charuco.py
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
intrinsics.yaml
    - K: 内部パラメータ行列。カメラ固有の光学的特性（焦点距離 fx, fy、主点 cx, cy）をまとめたもの
    - dist: distortion coefficients (OpenCV plumb-bob order)
    - rms: reprojection error in pixel

handeye.yaml：ハンド（アーム先端, グリッパ）からカメラ座標への剛体変換(hand to camera)（回転Rと並進t）カメラの位置と姿勢を決定. T_camera = T_gripper * X + tg_c
    - Rg_c: rotation (gripper->camera)グリッパー座標系からカメラ座標系への3x3回転行列
    - tg_c: translation (gripper->camera)グリッパー座標系の原点から見たカメラ座標系の原点への3x1並進ベクトル（位置）
    
colmap_sparse
    - cameras.txt: OPENCVモデルで fx fy cx cy k1 k2 p1 p2 k3
    - images.txt: COLMAPが要求する world to camera の回転と並進（四元数 QW QX QY QZ、Hamilton 方式）と翻訳（TX TY TZ）
    こっちが外部パラメータ
"""
# 一般的な回転順序の候補
# candidate_sequences = ['xyz', 'zyx', 'XYZ', 'ZYX', 'xzy', 'yxz'] 

# ---------- Config ----------
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LEN_MM = 32.0
MARKER_LEN_MM = 24.0
DICT = cv2.aruco.DICT_4X4_50
EULER_ORDER = "xyz"
MIN_CORNERS = 20
PNP_FLAG = cv2.SOLVEPNP_SQPNP
REPROJ_REJECT_PX = 2.5
MAX_CALIB_PRUNE = 6


@dataclass
class Capture:
    path: str
    Rb_g: np.ndarray  # (3,3) gripper->base
    tb_g: np.ndarray  # (3,1) meters


@dataclass
class Detection:
    path: str
    corners: np.ndarray        # (N,1,2) float32
    ids: np.ndarray            # (N,1) int32
    corner_count: int
    coverage: float            # normalised board area in the frame
    marker_count: int


def load_robot_json(json_path: str, image_dir: str) -> List[Capture]:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    caps: List[Capture] = []
    for c in js["captures"]:
        img = os.path.join(image_dir, os.path.basename(c["image_file"]))
        t = np.array(
            [c["pose"]["x_mm"], c["pose"]["y_mm"], c["pose"]["z_mm"]],
            dtype=np.float64,
        ) / 1000.0
        angles = np.array(
            [c["pose"]["rx_deg"], c["pose"]["ry_deg"], c["pose"]["rz_deg"]],
            dtype=np.float64,
        )
        Rb_g = R.from_euler(EULER_ORDER, angles, degrees=True).as_matrix()
        
                # ===== デバッグコード挿入 =====
        if c["image_file"] == "特定のファイル名.png": # 最初の画像などで良い
            print(f"DEBUG: EULER_ORDER = {EULER_ORDER}")
            print("DEBUG: Rotation matrix for first pose:")
            print(Rb_g)
        # ============================
        
        caps.append(Capture(img, Rb_g, t.reshape(3, 1)))
    return caps


def _board_coverage(corners: np.ndarray, imsize: Tuple[int, int]) -> float:
    if corners.size == 0:
        return 0.0
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    area = width * height
    full = float(imsize[0] * imsize[1])
    if full <= 0.0:
        return 0.0
    return float(np.clip(area / full, 0.0, 1.0))


def detect_charuco_points(image_paths: List[str], board, dict_obj) -> Tuple[List[Detection], Tuple[int, int]]:
    # OpenCV バージョン差異を吸収しつつ Charuco 検出を行う
    use_charuco_detector = False
    charuco_detector = None
    aruco_detector = None
    det_params = None
    charuco_params = None

    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            det_params = cv2.aruco.DetectorParameters()
            if hasattr(det_params, "cornerRefinementMethod"):
                det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            if hasattr(cv2.aruco, "CharucoParameters"):
                charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, det_params)
            use_charuco_detector = True
        except cv2.error:
            charuco_detector = None
            use_charuco_detector = False

    if not use_charuco_detector:
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            det_params = cv2.aruco.DetectorParameters_create()
        else:
            det_params = cv2.aruco.DetectorParameters()
        if hasattr(det_params, "cornerRefinementMethod"):
            det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            aruco_detector = cv2.aruco.ArucoDetector(dict_obj, det_params)
        else:
            aruco_detector = None
        if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
            raise RuntimeError("This OpenCV build lacks Charuco support.")

    detections: List[Detection] = []
    imsize: Optional[Tuple[int, int]] = None

    subpix_term = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-3,
    )

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] cannot read image: {path}")
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])

        marker_corners = None
        marker_ids = None

        if use_charuco_detector and charuco_detector is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
        else:
            if aruco_detector is not None:
                marker_corners, marker_ids, _ = aruco_detector.detectMarkers(img)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img, dict_obj, parameters=det_params)
            if marker_ids is None or len(marker_ids) == 0:
                print(f"[INFO] no ArUco markers detected in {path}")
                continue
            interpolated = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
            if isinstance(interpolated, tuple):
                if len(interpolated) == 2:
                    charuco_corners, charuco_ids = interpolated
                elif len(interpolated) >= 3:
                    # OpenCV >=4.5 returns (retval, corners, ids, ...). Use the actual arrays.
                    charuco_corners, charuco_ids = interpolated[1], interpolated[2]
                else:
                    charuco_corners, charuco_ids = None, None
            else:
                charuco_corners, charuco_ids = interpolated, None

        if charuco_ids is None or len(charuco_ids) < MIN_CORNERS:
            count = 0 if charuco_ids is None else len(charuco_ids)
            print(f"[INFO] insufficient Charuco corners in {path}: {count}")
            continue

        corners = np.asarray(charuco_corners, dtype=np.float32)
        if corners.size < 2:
            print(f"[INFO] invalid Charuco corner array in {path}, skipping (size={corners.size})")
            continue
        if corners.ndim == 2 and corners.shape[-1] == 2:
            corners = corners.reshape(-1, 1, 2)
        elif corners.ndim == 3 and corners.shape[1:] == (1, 2):
            pass  # already (N,1,2)
        else:
            corners = corners.reshape(-1, 1, 2)
        cv2.cornerSubPix(img, corners, (3, 3), (-1, -1), subpix_term)
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

        coverage = _board_coverage(corners, imsize)
        marker_count = 0 if marker_ids is None else len(marker_ids)

        detections.append(
            Detection(
                path=path,
                corners=corners,
                ids=ids,
                corner_count=int(len(ids)),
                coverage=coverage,
                marker_count=marker_count,
            )
        )

    if imsize is None:
        raise RuntimeError("No images were readable.")

    print(f"[INFO] detected Charuco in {len(detections)}/{len(image_paths)} images.")
    return detections, imsize


def filter_detections(detections: List[Detection]) -> List[Detection]:
    if not detections:
        return []

    counts = np.array([d.corner_count for d in detections], dtype=np.float32)
    coverages = np.array([d.coverage for d in detections], dtype=np.float32)

    count_thresh = max(MIN_CORNERS, int(np.percentile(counts, 40)))
    coverage_baseline = max(0.02, float(np.median(coverages)) * 0.6)

    filtered = [
        d
        for d in detections
        if d.corner_count >= count_thresh and d.coverage >= coverage_baseline
    ]

    if len(filtered) < max(6, len(detections) // 2):
        print("[WARN] filtering would drop too many views; using all detections instead.")
        return detections

    print(
        f"[INFO] filtered detections {len(filtered)}/{len(detections)} "
        f"(corners >= {count_thresh}, coverage >= {coverage_baseline:.3f})"
    )
    return filtered


def calibrate_intrinsics(detections: List[Detection], imsize: Tuple[int, int], board):
    if len(detections) < 4:
        raise RuntimeError("Need at least 4 captured views for calibration.")

    active = list(detections)
    best_result = None
    last_rms = None

    for iteration in range(MAX_CALIB_PRUNE + 1):
        charuco_corners = [d.corners for d in active]
        charuco_ids = [d.ids for d in active]
        retval = cv2.aruco.calibrateCameraCharucoExtended(
            charuco_corners,
            charuco_ids,
            board,
            imsize,
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_LU,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9),
        )

        rms, K, dist, rvecs, tvecs, _, _, per_view = retval
        per_view = per_view.reshape(-1)
        print(f"[INFO] calibration iteration {iteration}: rms={rms:.3f}px, views={len(active)}")
        for det, err in zip(active, per_view):
            print(
                f"   - {os.path.basename(det.path)} : {err:.3f}px "
                f"(corners={det.corner_count}, coverage={det.coverage:.3f})"
            )

        if best_result is None or rms < best_result["rms"]:
            best_result = {
                "rms": float(rms),
                "K": K,
                "dist": dist,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "per_view": per_view.copy(),
                "detections": list(active),
            }

        if rms <= 1.0 or len(active) <= 6:
            break

        worst_idx = int(np.argmax(per_view))
        removed = active.pop(worst_idx)
        print(
            f"[INFO] removing worst view {os.path.basename(removed.path)} "
            f"(per-view error {per_view[worst_idx]:.3f}px)"
        )

        if last_rms is not None and rms >= last_rms * 0.995:
            print("[INFO] calibration no longer improves; stop pruning.")
            break
        last_rms = rms

    if best_result is None:
        raise RuntimeError("Calibration failed; no valid solution found.")

    angles = [np.linalg.norm(rv) for rv in best_result["rvecs"]]
    if angles:
        ang_deg = np.rad2deg(angles)
        print(
            "[INFO] board pose rotation norm (deg) "
            f"min/med/max = {ang_deg.min():.2f}/{np.median(ang_deg):.2f}/{ang_deg.max():.2f}"
        )

    return best_result


def estimate_board_poses(
    captures: List[Capture],
    detections: Dict[str, Detection],
    board,
    K: np.ndarray,
    dist: np.ndarray,
    reproj_threshold: float = REPROJ_REJECT_PX,
) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], Dict[str, Dict[str, float]]]:
    Rt_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    stats: Dict[str, Dict[str, float]] = {}

    for cap in captures:
        det = detections.get(cap.path)
        if det is None:
            Rt_list.append(None)
            continue

        pose_ok = False
        rvec = None
        tvec = None
        reproj_errs = None

        ids = det.ids.flatten().astype(int)
        if hasattr(board, "chessboardCorners"):
            all_obj = board.chessboardCorners
        else:
            all_obj = board.getChessboardCorners()
        obj = all_obj[ids].reshape(-1, 1, 3).astype(np.float32)
        imgp = det.corners.astype(np.float32)

        # 1) try dedicated Charuco pose estimator
        if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
            try:
                pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    det.corners,
                    det.ids,
                    board,
                    K,
                    dist,
                    None,
                    None,
                )
            except cv2.error:
                pose_ok = False

        # 2) fallback to solvePnP with matchImagePoints if needed
        if not pose_ok or rvec is None or tvec is None:
            ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=PNP_FLAG)
            if not ok:
                print(f"[WARN] solvePnP failed for {cap.path}")
                Rt_list.append(None)
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            reproj_errs = np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)
        else:
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            reproj_errs = np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)

        if reproj_errs is None or len(reproj_errs) == 0:
            Rt_list.append(None)
            continue

        errs = reproj_errs
        mean_err = float(errs.mean())
        median_err = float(np.median(errs))

        stats[cap.path] = {
            "mean": mean_err,
            "median": median_err,
            "count": float(len(errs)),
        }

        if mean_err > reproj_threshold:
            print(
                f"[INFO] rejecting board pose {os.path.basename(cap.path)} "
                f"due to mean reprojection {mean_err:.3f}px"
            )
            Rt_list.append(None)
            continue

        Rc_t, _ = cv2.Rodrigues(rvec)
        Rt_list.append((Rc_t, tvec.reshape(3, 1)))

    accepted = sum(1 for r in Rt_list if r is not None)
    print(f"[INFO] accepted {accepted}/{len(captures)} target poses (threshold {reproj_threshold:.2f}px)")

    if stats:
        medians = [v["median"] for v in stats.values()]
        print(
            "[INFO] per-view median reprojection (px) "
            f"min/median/max = {min(medians):.3f}/{np.median(medians):.3f}/{max(medians):.3f}"
        )

    return Rt_list, stats


def run_handeye(captures: List[Capture], target2cam_list: List[Optional[Tuple[np.ndarray, np.ndarray]]]):
    Rg2b, tg2b, Rt2c, tt2c = [], [], [], []
    for cap, Rt in zip(captures, target2cam_list):
        if Rt is None:
            continue
        Rc_t, tc_t = Rt
        Rg2b.append(cap.Rb_g)
        tg2b.append(cap.tb_g)
        Rt2c.append(Rc_t)
        tt2c.append(tc_t)
    if len(Rg2b) < 3:
        raise RuntimeError("Not enough valid board poses for hand-eye calibration.")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        Rg2b, tg2b, Rt2c, tt2c, method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    return R_cam2gripper, t_cam2gripper


def compose_base_to_cam(caps: List[Capture], Rg_c: np.ndarray, tg_c: np.ndarray):
    Tbc_list = []
    for cap in caps:
        Rb_g, tb_g = cap.Rb_g, cap.tb_g        
        Rb_c = Rb_g @ Rg_c
        tb_c = Rb_g @ tg_c + tb_g
        Tbc_list.append((Rb_c, tb_c))
    return Tbc_list


def to_colmap_quat_t(Rbc: np.ndarray, tbc: np.ndarray):
    Rc_b = Rbc.T
    tc_b = -Rc_b @ tbc
    q = R.from_matrix(Rc_b).as_quat()  # returns [qx,qy,qz,qw]
    QW, QX, QY, QZ = q[3], q[0], q[1], q[2]
    return (QW, QX, QY, QZ), tc_b.reshape(-1)


def save_yaml(path: str, data: Dict[str, np.ndarray]):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open {path} for writing.")
    for k, v in data.items():
        fs.write(k, np.array(v))
    fs.release()


def export_colmap(
    cameras_txt: str,
    images_txt: str,
    points_txt: str,
    K: np.ndarray,
    dist: np.ndarray,
    image_paths: List[str],
    Tbc_list: List[Tuple[np.ndarray, np.ndarray] | None],
    width: int,
    height: int,
):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dist_flat = dist.flatten().tolist()
    dist_pad = (dist_flat + [0.0] * 14)[:5]
    k1, k2, p1, p2, k3 = dist_pad

    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 OPENCV {width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}\n")

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (p, bc) in enumerate(zip(image_paths, Tbc_list), start=1):
            if bc is None:
                f.write(f"# {i} {os.path.basename(p)} : pose rejected\n\n")
                continue
            (Rbc, tbc) = bc
            (qw, qx, qy, qz), t = to_colmap_quat_t(Rbc, tbc)
            f.write(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {os.path.basename(p)}\n")
            f.write("\n")

    with open(points_txt, "w") as f:
        f.write("# 3D point list is empty for known-poses bootstrap\n")


def main(json_path: str, image_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    captures = load_robot_json(json_path, image_dir)
    image_paths = [c.path for c in captures]

    dict_obj = cv2.aruco.getPredefinedDictionary(DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LEN_MM / 1000.0,
        MARKER_LEN_MM / 1000.0,
        dict_obj,
    )

    detections, imsize = detect_charuco_points(image_paths, board, dict_obj)
    detection_summary = [
        (os.path.basename(d.path), d.corner_count, d.coverage, d.marker_count)
        for d in detections
    ]
    detection_summary.sort(key=lambda x: x[0])
    for name, count, cov, markers in detection_summary:
        print(f"[TRACE] {name}: corners={count}, coverage={cov:.3f}, markers={markers}")

    detections = filter_detections(detections)
    calib = calibrate_intrinsics(detections, imsize, board)

    K = calib["K"]
    dist = calib["dist"]
    rms = calib["rms"]
    per_view_errors = calib["per_view"]
    used_detections = calib["detections"]

    intrinsics_path = os.path.join(out_dir, "intrinsics.yaml")
    save_yaml(
        intrinsics_path,
        {
            "K": K,
            "dist": dist,
            "rms": np.array([[rms]], dtype=np.float64),
            "per_view_errors": np.array([per_view_errors], dtype=np.float64),
            "num_views": np.array([[len(used_detections)]], dtype=np.int32),
        },
    )
    print(f"[INFO] wrote intrinsics to {intrinsics_path}")

    detection_map = {d.path: d for d in used_detections}
    Rt_list, pose_stats = estimate_board_poses(captures, detection_map, board, K, dist)

    valid_pose_errors = [v["mean"] for v in pose_stats.values() if v["mean"] <= REPROJ_REJECT_PX]
    if valid_pose_errors:
        print(
            "[INFO] accepted pose reprojection mean (px) "
            f"median={np.median(valid_pose_errors):.3f}, max={max(valid_pose_errors):.3f}"
        )

    Rc_g, tc_g = run_handeye(captures, Rt_list)
    
    # Fix version
    Rg_c = Rc_g.T
    tg_c = -Rg_c @ tc_g

    handeye_path = os.path.join(out_dir, "handeye.yaml")
    save_yaml(handeye_path, {"Rg_c": Rg_c, "tg_c": tg_c})
    print(f"[INFO] wrote hand-eye result to {handeye_path}")

    Tbc = compose_base_to_cam(captures, Rg_c, tg_c)
    colmap_dir = os.path.join(out_dir, "colmap_sparse")
    os.makedirs(colmap_dir, exist_ok=True)
    export_colmap(
        os.path.join(colmap_dir, "cameras.txt"),
        os.path.join(colmap_dir, "images.txt"),
        os.path.join(colmap_dir, "points3D.txt"),
        K,
        dist,
        image_paths,
        Tbc,
        imsize[0],
        imsize[1],
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        raise SystemExit("Usage: python calibrate_robot_charuco.py poses.json image_dir output_dir")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
