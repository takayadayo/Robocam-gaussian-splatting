"""
viz_colmap_images.py

COLMAP images.txt を読み込み、カメラ中心と向きを 3D 可視化する。
空白行やコメント行がヘッダーと点群の間に入るケースにも対応しています。

Usage:
    python viz_colmap_images.py /path/to/images.txt --scale 0.2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math

def quaternion_to_rotmat(q):
    qw, qx, qy, qz = q
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    return R

def is_header_like(s):
    """
    その行が images.txt の「ヘッダー行らしいか」を推定する。
    条件（緩め）:
      - トークン数 >= 9
      - 最初のトークンが整数 IMAGE_ID に変換可能
      - 次の4つのトークンが float（四元数）
    """
    if s is None:
        return False
    s = s.strip()
    if s == '' or s.startswith('#'):
        return False
    parts = s.split()
    if len(parts) < 9:
        return False
    try:
        int(parts[0])
        # check qw, qx, qy, qz
        for k in range(1, 5):
            float(parts[k])
    except Exception:
        return False
    return True

def parse_colmap_images(images_txt_path):
    """
    images.txt を堅牢にパース。
    返却: dict name -> {'id','qwqxqyqz','t','R','C'}
    """
    poses = {}
    with open(images_txt_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    # 行をそのまま保持（空行も残す）
    lines = [ln.rstrip('\n') for ln in raw_lines]
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        # コメント・空行はスキップ
        if ln == '' or ln.startswith('#'):
            i += 1
            continue

        # ヘッダーと見なせる行かチェック
        if not is_header_like(ln):
            # ヘッダーではない行は単にスキップ（安全策）
            i += 1
            continue

        parts = ln.split()
        # parse header
        img_id = int(parts[0])
        qw = float(parts[1]); qx = float(parts[2]); qy = float(parts[3]); qz = float(parts[4])
        tx = float(parts[5]); ty = float(parts[6]); tz = float(parts[7])
        name = parts[9] if len(parts) >= 10 else f"img_{img_id}"

        R = quaternion_to_rotmat((qw, qx, qy, qz))
        t = np.array([tx, ty, tz], dtype=np.float64).reshape(3,1)
        C = (-R.T @ t).reshape(3)

        poses[name] = {
            'id': img_id,
            'qwqxqyqz': (qw, qx, qy, qz),
            't': t.reshape(3),
            'R': R,
            'C': C
        }

        # move pointer to next line (possible points2D)
        i += 1
        if i >= n:
            break

        # peek next non-stripped line
        peek = lines[i].strip()

        # case A: next line is empty or comment -> skip it (but DO NOT skip any subsequent header)
        if peek == '' or peek.startswith('#'):
            # skip that blank/comment line, and continue loop - which will process next real header
            i += 1
            continue

        # case B: if next line is header-like, DO NOT consume it (do nothing),
        # so next loop iteration will parse it as a header.
        if is_header_like(peek):
            # do not advance i here; next loop will parse peek as header
            continue

        # otherwise: treat the next line as points2D and skip it (standard COLMAP pattern)
        i += 1
        # then continue loop

    return poses

def set_axes_equal(ax):
    x_lim = ax.get_xlim3d(); y_lim = ax.get_ylim3d(); z_lim = ax.get_zlim3d()
    x_range = abs(x_lim[1] - x_lim[0]); x_mid = np.mean(x_lim)
    y_range = abs(y_lim[1] - y_lim[0]); y_mid = np.mean(y_lim)
    z_range = abs(z_lim[1] - z_lim[0]); z_mid = np.mean(z_lim)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

def plot_poses(poses, scale=0.2, show_labels=False, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    centers = []
    for name, p in poses.items():
        C = p['C']
        centers.append(C)
        R = p['R']
        x_axis = (R.T @ np.array([1,0,0], dtype=np.float64)).reshape(3)
        y_axis = (R.T @ np.array([0,1,0], dtype=np.float64)).reshape(3)
        z_axis = (R.T @ np.array([0,0,1], dtype=np.float64)).reshape(3)

        ax.scatter([C[0]], [C[1]], [C[2]], marker='o', s=20)
        ax.plot([C[0], C[0] + scale * x_axis[0]], [C[1], C[1] + scale * x_axis[1]], [C[2], C[2] + scale * x_axis[2]], linewidth=1)
        ax.plot([C[0], C[0] + scale * y_axis[0]], [C[1], C[1] + scale * y_axis[1]], [C[2], C[2] + scale * y_axis[2]], linewidth=5)
        ax.plot([C[0], C[0] + scale * z_axis[0]], [C[1], C[1] + scale * z_axis[1]], [C[2], C[2] + scale * z_axis[2]], linewidth=10)

        # approximate frustum
        w = scale * 0.4; h = scale * 0.3
        corners_cam = np.array([
            [ w,  h, scale],
            [-w,  h, scale],
            [-w, -h, scale],
            [ w, -h, scale]
        ], dtype=np.float64).T
        corners_world = (R.T @ corners_cam).T + C[np.newaxis,:]
        for corner in corners_world:
            ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]], linewidth=0.5)
        cx = corners_world[:,0]; cy = corners_world[:,1]; cz = corners_world[:,2]
        ax.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), np.append(cz, cz[0]), linewidth=0.5)

        if show_labels:
            ax.text(C[0], C[1], C[2], name, fontsize=6)

    if len(centers) > 0:
        centers = np.vstack(centers)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.auto_scale_xyz(centers[:,0], centers[:,1], centers[:,2])
    set_axes_equal(ax)
    plt.title("COLMAP camera centers & approximate frusta")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_txt', help='path to COLMAP images.txt')
    parser.add_argument('--scale', type=float, default=0.2, help='frustum / axis scale')
    parser.add_argument('--labels', action='store_true', help='show image names as labels')
    args = parser.parse_args()

    poses = parse_colmap_images(args.images_txt)
    if len(poses) == 0:
        print("No poses parsed. Check the images.txt file.")
        return
    print(f"Parsed {len(poses)} poses. Example keys: {list(poses.keys())[:5]}")
    plot_poses(poses, scale=args.scale, show_labels=args.labels)

if __name__ == '__main__':
    main()
