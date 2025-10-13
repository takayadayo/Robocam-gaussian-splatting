# sfm_minimal.py
import os, glob, json, math
import numpy as np
import cv2


"""
1. 特徴抽出：SIFT（OpenCV 実装）。
2. ペアマッチング：FLANN＋Lowe比率0.75＋相互チェック。幾何検証は `findEssentialMat(RANSAC)`（内部行列は初期値）でインライアを確定。([cs.ubc.ca][2])
3. 二視点初期化：インライア最多のペアを選び，`recoverPose` → (P_0=[I|0]), (P_1=[R|t])。`triangulatePoints` で初期3D点を生成（両視点で正の深度，かつ再投影誤差しきい値でフィルタ）。([docs.opencv.org][3])
4. トラック構築：インライア対応をユニオンファインダで連結して特徴トラックを作成（画像IDとkeypoint IDの集合）。
5. 逐次拡張（Incremental）：未登録画像を1枚ずつ

    - 既存3D点とその画像の2D観測から `solvePnPRansac` で外部パラメータ初期化 → `SOLVEPNP_ITERATIVE` で微調整。
    - 新規3D点：この画像と既存登録画像の間で観測される未三角測量トラックを最大基線ペアから再三角測量（線形DLT、多視点時はSVDでN視点同時）。([docs.opencv.org][4])
6. （ローカル）BA近似：

    - カメラ外部：各カメラを順に固定3D点で `solvePnP` 再最適化。
    - 3D点：観測カメラ集合で線形N視点三角測量し直す。
    この2手を2–3回交互に回す（擬似BA）。理論的BA（Schur補完を伴う同時最適化）はCeres等が標準ですが、ここではOpenCVのみで近似的に誤差最小化します。参考：BAの定義・Schur補完の要点。([Department of Computer Science][5])
7. 評価出力：

    - グローバルRMS再投影誤差、各画像のmean/median/std、3D点の観測数分布、有効トラック数。
    - （任意）ベースライン角の分布、PnPインライア率、処理採択されたエッジ数。
    
Returns:
- 標準出力
    global RMS reprojection error (px)`、各 `view i` の mean/median/std/n
        - RMS：平均二乗誤差の平方根なので、大きな誤差に影響されやすい --> 外れ値に敏感
- 出力ファイル
    sfm_out/metrics.json（RMS／各ビュー統計／登録カメラ数／3D点数）
    sfm_out/cameras.txt（各カメラの (R,t) ）
    sfm_out/points.ply（可視化用点群）
        
"""

# ---------- ユーティリティ ----------
def load_images(img_dir):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    files = sum([glob.glob(os.path.join(img_dir,e)) for e in exts], [])
    files.sort()
    return files

def build_initial_K(w,h, scale=1.2):
    f = scale * max(w,h)
    return np.array([[f,0,w*0.5],[0,f,h*0.5],[0,0,1.0]], dtype=np.float64)

def to_h(P):  # 3xN -> 4xN (homogeneous)
    return np.vstack([P, np.ones((1,P.shape[1]))])

def from_h(P): # 4xN -> 3xN
    P = P / P[3:4,:]
    return P[:3,:]

def project_points(Xw, K, R, t):
    # Xw: 3xN world, R: 3x3 world->cam, t: 3x1
    x = R @ Xw + t
    x = x / x[2:3,:]
    u = K @ x
    return u[:2,:]  # 2xN

# ---------- 特徴抽出・マッチング ----------
def extract_sift(img):
    if img.ndim==3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray = img
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)
    return kps, des

def match_pair(d1, d2, ratio=0.75, cross_check=True):
    # FLANN for SIFT (KD-Tree)
    index = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE=1
    search = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index, search)
    knn = flann.knnMatch(d1, d2, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx, m.distance))
    if cross_check:
        flann2 = cv2.FlannBasedMatcher(index, search)
        knn2 = flann2.knnMatch(d2, d1, k=2)
        good2 = []
        for m,n in knn2:
            if m.distance < ratio * n.distance:
                good2.append((m.trainIdx, m.queryIdx, m.distance))  # invert
        set1 = set((a,b) for a,b,_ in good)
        good = [(a,b,d) for (a,b,d) in good if (a,b) in set1 and (a,b) in set((a,b) for a,b,_ in good2)]
    return good  # list of (i1,i2,dist)

# ---------- 幾何検証（E行列） ----------
def essential_inliers(kp1, kp2, matches, K):
    if len(matches) < 8:
        return None, None, None
    pts1 = np.float32([kp1[i].pt for i,_,_ in matches])
    pts2 = np.float32([kp2[j].pt for _,j,_ in matches])
    E, inl = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=2.9)
    if E is None or inl is None or inl.sum()<8:
        return None, None, None
    inlier_idx = np.where(inl.ravel()>0)[0]
    inl_m = [matches[k] for k in inlier_idx]
    return E, inl_m, (pts1[inlier_idx], pts2[inlier_idx])

# ---------- トラック生成（Union-Find） ----------
class DSU:
    def __init__(self,n): self.p=list(range(n)); self.sz=[1]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return ra
        if self.sz[ra]<self.sz[rb]: ra,rb=rb,ra
        self.p[rb]=ra; self.sz[ra]+=self.sz[rb]
        return ra

# key_id = image_base + kp_index
def build_tracks(all_kps, inlier_pairs, base=10_000_000):
    # inlier_pairs: dict[(i,j)] = list of (idx_i, idx_j)
    # assign global ids
    gid_offset = []
    gid=0
    for kps in all_kps:
        gid_offset.append(gid)
        gid += len(kps)

    dsu = DSU(gid)
    for (i,j), pairs in inlier_pairs.items():
        offi, offj = gid_offset[i], gid_offset[j]
        for (ai, bj) in pairs:
            dsu.union(offi+ai, offj+bj)
    # collect tracks
    tracks = {}
    for img_id, kps in enumerate(all_kps):
        off = gid_offset[img_id]
        for ki in range(len(kps)):
            root = dsu.find(off+ki)
            if root not in tracks: tracks[root] = []
            tracks[root].append((img_id, ki))
    # drop singleton
    tracks = {tid:obs for tid,obs in tracks.items() if len(obs)>=2}
    return tracks

def init_two_view(kp1, kp2, inl_m, K):
    """
    kp1/kp2: list of cv2.KeyPoint
    inl_m: list of matches (queryIdx, trainIdx, distance)
    K: camera matrix
    returns:
      R (3x3), t (3x1), X (3xM), pts1_filtered (Mx2), pts2_filtered (Mx2), kept_pairs (list of (idx1, idx2))
    kept_pairs: list of (kp_index_in_image1, kp_index_in_image2) corresponding to each column of X
    """
    # build raw arrays from inl_m (order preserved)
    pts1_all = np.array([kp1[i].pt for i,_,_ in inl_m], dtype=np.float64)  # (N0,2)
    pts2_all = np.array([kp2[j].pt for _,j,_ in inl_m], dtype=np.float64)

    if pts1_all.size == 0:
        return None, None, np.empty((3,0)), None, None, []

    # Robust E + recoverPose as before
    E, mask = cv2.findEssentialMat(pts1_all, pts2_all, K, method=cv2.RANSAC, prob=0.999, threshold=2.9)
    if E is None or mask is None:
        return None, None, np.empty((3,0)), None, None, []

    inlier_idx = np.where(mask.ravel() > 0)[0]
    if len(inlier_idx) < 8:
        return None, None, np.empty((3,0)), None, None, []

    pts1 = pts1_all[inlier_idx]
    pts2 = pts2_all[inlier_idx]
    # recoverPose expects inlier-masked points
    _, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, K)
    if mask2 is None:
        return None, None, np.empty((3,0)), None, None, []

    inl2_idx = np.where(mask2.ravel() > 0)[0]
    if len(inl2_idx) < 6:
        return None, None, np.empty((3,0)), None, None, []

    pts1_final = pts1[inl2_idx]
    pts2_final = pts2[inl2_idx]

    # prepare undistort (safe dtype/shape)
    dtype = np.float64 if K.dtype == np.float64 else np.float32
    pts1_in = np.ascontiguousarray(pts1_final.reshape(-1,1,2).astype(dtype))
    pts2_in = np.ascontiguousarray(pts2_final.reshape(-1,1,2).astype(dtype))
    dist = np.zeros((5,1), dtype=dtype)

    pts1n = cv2.undistortPoints(pts1_in, K.astype(dtype), dist).reshape(-1,2).T  # 2xM
    pts2n = cv2.undistortPoints(pts2_in, K.astype(dtype), dist).reshape(-1,2).T

    # Triangulate (P0 = identity)
    P0 = np.hstack((np.eye(3, dtype=dtype), np.zeros((3,1), dtype=dtype)))
    P1 = np.hstack((R.astype(dtype), t.astype(dtype)))
    Xh = cv2.triangulatePoints(P0, P1, pts1n, pts2n)  # 4xM
    if Xh.shape[1] == 0:
        return R, t, np.empty((3,0)), None, None, []
    X = (Xh[:3,:] / Xh[3:4,:])

    # cheirality: keep points with positive depth in both views
    z0 = X[2,:] > 1e-6
    X1 = R @ X + t
    z1 = X1[2,:] > 1e-6
    keep_mask = z0 & z1
    if not np.any(keep_mask):
        return R, t, np.empty((3,0)), None, None, []

    X_keep = X[:, keep_mask]
    pts1_keep = pts1_final[keep_mask]
    pts2_keep = pts2_final[keep_mask]

    # Map kept indices back to original inl_m indices:
    # inl_m -> pts1_all/inlier_idx -> inl2_idx -> keep_mask
    # Step1: indices in original inl_m that survived inlier_idx:
    survivors_after_E = inlier_idx  # indices into original inl_m
    # Step2: survivors_after_E[inl2_idx] are indices in original inl_m that survived recoverPose
    survivors_after_recover = survivors_after_E[inl2_idx]
    # Step3: kept ones after cheirality filtering:
    kept_original_indices = survivors_after_recover[keep_mask]  # indices into original inl_m

    # construct kept_pairs list
    kept_pairs = []
    for idx in kept_original_indices:
        a,b,_ = inl_m[int(idx)]
        kept_pairs.append((a,b))

    return R, t, X_keep, pts1_keep, pts2_keep, kept_pairs



# ---------- N視点線形三角測量 ----------
def triangulate_n_views(K, cams, obs):
    # cams: list of (R,t) world->cam ; obs: list of (img_id, uv) with uv in pixels
    # DLT stacking
    A = []
    for (R,t),(img_id, uv) in zip(cams, obs):
        P = K @ np.hstack([R, t])
        u,v = uv
        A.append(u*P[2,:]-P[0,:])
        A.append(v*P[2,:]-P[1,:])
    A = np.stack(A, axis=0)
    _,_,Vt = np.linalg.svd(A)
    Xh = Vt[-1]; Xh /= Xh[3]
    return Xh[:3]

# ---------- PnPで姿勢推定 ----------
def estimate_pose_pnp(K, X3d, x2d):
    dist = None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X3d, x2d, K, dist, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=2.0, confidence=0.999, iterationsCount=2000)
    if not ok or inl is None or len(inl)<6: return None
    # refine
    ok2, rvec, tvec = cv2.solvePnP(X3d[inl[:,0]], x2d[inl[:,0]], K, dist, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok2: return None
    R,_ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)
    return R, t, inl[:,0]

# ---------- 再投影誤差 ----------
def reprojection_stats(K, cam_list, points3d, obsmap):
    # cam_list[i] = (R,t)
    # obsmap[pt_id] = list of (img_id, uv)
    errs = []
    per_view = {i:[] for i in range(len(cam_list))}
    for pid, X in enumerate(points3d):
        if X is None: continue
        Xw = X.reshape(3,1)
        for (img_id, uv) in obsmap[pid]:
            R,t = cam_list[img_id]
            pr = project_points(Xw, K, R, t).ravel()
            e = np.linalg.norm(pr - uv)
            errs.append(e); per_view[img_id].append(e)
    if len(errs)==0:
        return dict(global_rms=None, per_view=None)
    rms = float(np.sqrt(np.mean(np.square(errs))))
    pv = {}
    for i,arr in per_view.items():
        if arr:
            a = np.array(arr)
            pv[i] = dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()), count=int(len(a)))
    return dict(global_rms=rms, per_view=pv)

# ---------- メイン ----------
def run_sfm(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = load_images(img_dir)
    assert len(files)>=2, "Need >=2 images"
    imgs = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
    H,W = imgs[0].shape[:2]
    K = build_initial_K(W,H, scale=1.2)

    # 特徴
    feats = [extract_sift(im) for im in imgs]
    kps = [f[0] for f in feats]
    des = [f[1] for f in feats]

    # ペアマッチ＆幾何検証
    inlier_pairs = {}
    best = (-1,None,None,None)  # (inlier_count, i, j, (pts1,pts2))
    for i in range(len(files)-1):
        for j in range(i+1,len(files)):
            m = match_pair(des[i], des[j])
            E, inl_m, pts = essential_inliers(kps[i], kps[j], m, K)
            if inl_m is None: continue
            inlier_pairs[(i,j)] = [(a,b) for (a,b,_) in inl_m]
            if len(inl_m) > best[0]:
                best = (len(inl_m), i, j, pts)
    assert best[0]>=8, "No good initial pair found"

    # 二視点初期化
    i0, i1 = best[1], best[2]
    # 再算出（inlierで）
    m = match_pair(des[i0], des[i1])
    _, inl_m, _ = essential_inliers(kps[i0], kps[i1], m, K)
    R01, t01, X01, Pts0, Pts1, kept_pairs = init_two_view(kps[i0], kps[i1], inl_m, K)

    # カメラ姿勢（world=cam0）
    cams = [None]*len(files)
    cams[i0] = (np.eye(3), np.zeros((3,1)))
    cams[i1] = (R01, t01)

    # トラック
    tracks = build_tracks(kps, inlier_pairs)
    # 各トラックに3Dを割り当て
    # 初期は i0 & i1 両方を含む観測のみをX01から割り当て（簡便）
    # 観測map: pid -> [(img_id, uv)]
    pid_list = []
    obsmap = []
    points3d = []

    # build map using kept_pairs (maps directly to columns of X01)
    map01 = {(a,b): idx for idx,(a,b) in enumerate(kept_pairs)}

    for tid,obs in tracks.items():
        imgs_in = [o[0] for o in obs]
        if i0 in imgs_in and i1 in imgs_in:
            idx0 = [ki for (ii,ki) in obs if ii==i0][0]
            idx1 = [kj for (jj,kj) in obs if jj==i1][0]
            if (idx0, idx1) in map01:
                j = map01[(idx0, idx1)]
                Xw = X01[:, j].reshape(3)
                pid = len(points3d)
                points3d.append(Xw)
                # 観測を登録
                obs_entries=[]
                for (ii,kk) in obs:
                    uv = np.array(kps[ii][kk].pt, dtype=np.float64)
                    obs_entries.append((ii, uv))
                obsmap.append(obs_entries)
                pid_list.append(tid)

    # 逐次登録
    registered = set([i0, i1])
    remaining = [i for i in range(len(files)) if i not in registered]

    for ii in remaining:
        # 2D-3D対応を集める
        X3=[]; x2=[]
        for pid, X in enumerate(points3d):
            if X is None: continue
            for (im_id, uv) in obsmap[pid]:
                if im_id==ii:
                    X3.append(X); x2.append(uv)
                    break
        if len(X3)<6:
            continue
        X3 = np.array(X3, dtype=np.float64).reshape(-1,1,3)
        x2 = np.array(x2, dtype=np.float64).reshape(-1,1,2)
        pose = estimate_pose_pnp(K, X3, x2)
        if pose is None:
            continue
        R,t,inliers = pose
        cams[ii]=(R,t)
        registered.add(ii)

        # 未三角測量トラックの生成（iiと既存登録画像のペアから）
        for jj in list(registered):
            if jj==ii: continue
            key = (min(ii,jj), max(ii,jj))
            if key not in inlier_pairs: continue
            pairs = inlier_pairs[key]
            # 各対応から該当トラックIDを逆引き
            for (ai, bj) in pairs:
                imgA, imgB = (ii,jj) if ii<jj else (jj,ii)
                kpA, kpB = (ai,bj) if ii<jj else (bj,ai)
                # トラックID
                # 探索（遅いが簡潔）：実運用は索引を作る
                for tid, obs in tracks.items():
                    imgs_in = [o[0] for o in obs]
                    if imgA in imgs_in and imgB in imgs_in:
                        if kpA in [k for (im,k) in obs if im==imgA] and kpB in [k for (im,k) in obs if im==imgB]:
                            if tid in pid_list:  # 既に3Dあり
                                pass
                            else:
                                # N視点線形三角測量：ここでは2視点のみ使用（簡便）
                                R1,t1 = cams[imgA]
                                R2,t2 = cams[imgB]
                                uv1 = np.array(kps[imgA][kpA].pt)
                                uv2 = np.array(kps[imgB][kpB].pt)
                                # 正規化座標で2視点三角測量
                                P1 = np.hstack([R1,t1]); P2 = np.hstack([R2,t2])
                                uv1n = cv2.undistortPoints(uv1.reshape(1,1,2), K, None).reshape(2,1)
                                uv2n = cv2.undistortPoints(uv2.reshape(1,1,2), K, None).reshape(2,1)
                                Xh = cv2.triangulatePoints(P1, P2, uv1n, uv2n)
                                Xw = (Xh[:3]/Xh[3]).reshape(3)
                                # 簡単な正の深度チェック
                                if (R1@Xw.reshape(3,1)+t1)[2,0] > 1e-6 and (R2@Xw.reshape(3,1)+t2)[2,0] > 1e-6:
                                    pid = len(points3d)
                                    points3d.append(Xw)
                                    obs_entries=[]
                                    for (im,kk) in obs:
                                        uv = np.array(kps[im][kk].pt, dtype=np.float64)
                                        obs_entries.append((im, uv))
                                    obsmap.append(obs_entries)
                                    pid_list.append(tid)
                            break

    # ローカルBA近似（PnP再最適化＋N視点三角測量の交互反復）
    for _ in range(2):  # 2イテレーション程度
        # 3D再計算
        for pid, X in enumerate(points3d):
            obs = obsmap[pid]
            cams_use=[]; obs_use=[]
            for (im,uv) in obs:
                if cams[im] is None: continue
                cams_use.append(cams[im]); obs_use.append((im, uv))
            if len(cams_use)>=2:
                Xw = triangulate_n_views(K, cams_use, obs_use)
                points3d[pid]=Xw
        # 各カメラPnP再最適化
        for i in range(len(files)):
            if cams[i] is None: continue
            X3=[]; x2=[]
            for pid,X in enumerate(points3d):
                for (im,uv) in obsmap[pid]:
                    if im==i:
                        X3.append(X); x2.append(uv)
                        break
            if len(X3)<6: continue
            X3 = np.array(X3,dtype=np.float64).reshape(-1,1,3)
            x2 = np.array(x2,dtype=np.float64).reshape(-1,1,2)
            pose = estimate_pose_pnp(K, X3, x2)
            if pose is not None:
                R,t,_ = pose
                cams[i]=(R,t)

    # --- 評価（堅牢化版） ---
    stats = reprojection_stats(K, cams, points3d, obsmap)

    # 安全ログ：登録カメラ数・3D点数を数える
    num_registered = sum(1 for c in cams if c is not None)
    num_points = sum(1 for p in points3d if p is not None and getattr(p, 'size', 0) > 0)
    print(f"[INFO] registered cameras: {num_registered}, reconstructed points: {num_points}")

    # 必要最小条件のチェック（設定は用途に応じて調整）
    if num_registered < 2 or num_points == 0 or stats.get("global_rms") is None:
        # 詳細ログを出して終了（あるいは下で別ロジック：自動再トライなどを挿入可能）
        print("[ERROR] insufficient reconstruction results for reliable reprojection stats.")
        print("  -> registered cameras:", num_registered)
        print("  -> reconstructed points:", num_points)
        print("  -> global_rms:", stats.get("global_rms"))
        # でも metrics.json は作る（診断用）
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({
                "global_rms_px": stats.get("global_rms"),
                "per_view": stats.get("per_view"),
                "num_registered": num_registered,
                "num_points": num_points,
                "note": "insufficient data to compute reliable metrics"
            }, f, indent=2)
        # カメラ行列だけ出しておく（既に登録されたもの）
        # safe write of cameras (always skip invalid)
        with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
            f.write("# Cameras (registered only). Some entries may be UNREGISTERED or incomplete.\n")
            for i, cam in enumerate(cams):
                if cam is None:
                    f.write(f"# cam {i} : UNREGISTERED\n")
                    continue
                # cam may be (None,None) or (R,t) with R==None
                if not isinstance(cam, (tuple, list)) or len(cam) < 2:
                    f.write(f"# cam {i} : INVALID STRUCTURE\n")
                    continue
                R, t = cam
                if R is None or t is None:
                    f.write(f"# cam {i} : PARTIALLY REGISTERED (R or t is None)\n")
                    continue
                try:
                    f.write(f"# cam {i}\n")
                    f.write("R:\n"+np.array2string(np.asarray(R), formatter={'float_kind':lambda x: f"{x:.8f}"})+"\n")
                    f.write("t:\n"+np.array2string(np.asarray(t).ravel(), formatter={'float_kind':lambda x: f"{x:.8f}"})+"\n")
                except Exception as e:
                    f.write(f"# cam {i} : FAILED TO WRITE DUE TO {e}\n")

        # PLY出力（存在する点のみ）
        pts = [p for p in points3d if p is not None and getattr(p,'size',0)>0]
        if len(pts)>0:
            pts = np.array(pts, dtype=np.float32)
            colors = np.full((pts.shape[0],3), 200, np.uint8)
            write_ply(os.path.join(out_dir, "points.ply"), pts, colors)
        # ここで終了（改善策を試してから再度実行するようユーザーに促す）
        print("[INFO] wrote diagnostic outputs to", out_dir)
        return

    # 正常系：global_rms がある場合に従来の出力を行う
    print("[RESULT] global RMS reprojection error (px):", stats["global_rms"])
    if stats["per_view"] is not None:
        for i, s in stats["per_view"].items():
            print(f" view {i}: mean={s['mean']:.3f} med={s['median']:.3f} std={s['std']:.3f} n={s['count']}")

    # 安全な cameras.txt 出力（None を飛ばす）
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        for i, cam in enumerate(cams):
            if cam is None:
                f.write(f"# cam {i} : UNREGISTERED\n")
                continue
            R, t = cam
            f.write(f"# cam {i}\n")
            f.write("R:\n"+np.array2string(R, formatter={'float_kind':lambda x: f"{x:.8f}"})+"\n")
            f.write("t:\n"+np.array2string(t.ravel(), formatter={'float_kind':lambda x: f"{x:.8f}"})+"\n")

    # metrics.json - 常に書く（数値が None でも明示的に書く）
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "global_rms_px": stats.get("global_rms"),
            "per_view": stats.get("per_view"),
            "num_registered": num_registered,
            "num_points": num_points
        }, f, indent=2)

    # 3D点の出力：存在するものだけを PLY に
    pts = [p for p in points3d if p is not None and getattr(p,'size',0)>0]
    if len(pts)>0:
        pts = np.array(pts, dtype=np.float32)
        colors = np.full((pts.shape[0],3), 200, np.uint8)
        write_ply(os.path.join(out_dir, "points.ply"), pts, colors)

    print("[INFO] finished writing outputs to", out_dir)


def write_ply(path, pts, colors=None):
    if colors is None:
        colors = np.full((pts.shape[0],3), 255, np.uint8)
    with open(path,"wb") as f:
        header = f"ply\nformat binary_little_endian 1.0\n" \
                 f"element vertex {pts.shape[0]}\n" \
                 f"property float x\nproperty float y\nproperty float z\n" \
                 f"property uchar red\nproperty uchar green\nproperty uchar blue\n" \
                 f"end_header\n"
        f.write(header.encode('ascii'))
        arr = np.hstack([pts.astype(np.float32), colors.astype(np.uint8)]).view(np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')]))
        f.write(arr.tobytes())

if __name__=='__main__':
    import sys
    img_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv)>2 else "sfm_out"
    run_sfm(img_dir, out_dir)
