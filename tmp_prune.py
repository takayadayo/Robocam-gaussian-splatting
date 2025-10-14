# -*- coding: utf-8 -*-
from pathlib import Path
path = Path("sfm_minimal.py")
lines = path.read_text(encoding="utf-8").splitlines()
index = next(i for i, line in enumerate(lines) if '    stats = reprojection_stats(K, cams, points3d, obsmap)' in line)
insert_block = [
    '    pruned_obs, pruned_pts = prune_outlier_observations(K, cams, points3d, obsmap, base_thresh=OUTLIER_BASE_THRESH, max_iter=3)',
    '    if pruned_obs > 0 or pruned_pts > 0:',
    '        print(f"[INFO] pruned outliers: observations={pruned_obs}, points={pruned_pts}")'
]
lines = lines[:index] + insert_block + lines[index:]
path.write_text('\n'.join(lines) + '\n', encoding="utf-8")
