"""
Stage 7 — Multi-view pose estimation + mesh rigging

Three progressive phases, each feeding the next:

  Phase 1 (Easy)   — Multi-view beta averaging
    Run HMR 2.0 on front / 3q_front / side renders + reference photo
    Average shape betas weighted by detection confidence

  Phase 2 (Better) — Silhouette fitting
    Project SMPL mesh orthographically into each of the 5 views
    Optimise betas so the SMPL silhouette matches the TripoSG render mask
    Uses known orthographic camera matrices (exact same params as nvdiffrast)

  Phase 3 (Best)   — Multi-view joint triangulation
    For each view where HMR 2.0 fired, project its 2D keypoints back to 3D
    using the known orthographic camera → set up linear system per joint
    Least-squares triangulation gives world-space joint positions used
    directly as the skeleton, overriding the regressed SMPL joints

Output: rigged GLB (SMPL 24-joint skeleton + skin weights) + FBX via Blender
"""

import os, sys, json, struct, traceback, subprocess, tempfile
# Must be set before any OpenGL/pyrender import (triggered by hmr2)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np

# ── SMPL constants ────────────────────────────────────────────────────────────
SMPL_JOINT_NAMES = [
    "pelvis","left_hip","right_hip","spine1",
    "left_knee","right_knee","spine2",
    "left_ankle","right_ankle","spine3",
    "left_foot","right_foot","neck",
    "left_collar","right_collar","head",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hand","right_hand",
]
SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,
                12,13,14,16,17,18,19,20,21]

# Orthographic camera parameters — must match render_views in triposg_app.py
ORTHO_LEFT, ORTHO_RIGHT = -0.55, 0.55
ORTHO_BOT,  ORTHO_TOP   = -0.55, 0.55
RENDER_W, RENDER_H      = 768, 1024

# Azimuths passed to get_orthogonal_camera: [x-90 for x in [0,45,90,180,315]]
VIEW_AZIMUTHS_DEG = [-90.0, -45.0, 0.0, 90.0, 225.0]
VIEW_NAMES        = ["front", "3q_front", "side", "back", "3q_back"]
VIEW_PATHS        = [f"/tmp/render_{n}.png" for n in VIEW_NAMES]

# Views with a clearly visible front body (used for Phase 1 beta averaging)
FRONT_VIEW_INDICES = [0, 1, 2]   # front, 3q_front, side


# ══════════════════════════════════════════════════════════════════════════════
# Camera utilities
# ══════════════════════════════════════════════════════════════════════════════

def _R_y(deg: float) -> np.ndarray:
    """Rotation matrix around Y axis (right-hand, degrees)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def world_to_cam(pts: np.ndarray, azimuth_deg: float) -> np.ndarray:
    """
    Orthographic projection: world (N,3) → camera (N,2) in world-unit space.
    Convention: camera right = (cos θ, 0, -sin θ),  up = (0,1,0)
    """
    t = np.radians(azimuth_deg)
    right = np.array([np.cos(t),  0.0, -np.sin(t)])
    up    = np.array([0.0,        1.0,  0.0       ])
    return np.stack([pts @ right, pts @ up], axis=-1)   # (N, 2)


def cam_to_pixel(cam_xy: np.ndarray) -> np.ndarray:
    """Camera world-unit coords → pixel coords (u, v) in 768×1024 image."""
    u = (cam_xy[:, 0] - ORTHO_LEFT) / (ORTHO_RIGHT - ORTHO_LEFT) * RENDER_W
    v = (ORTHO_TOP  - cam_xy[:, 1]) / (ORTHO_TOP   - ORTHO_BOT ) * RENDER_H
    return np.stack([u, v], axis=-1)


def pixel_to_cam(uv: np.ndarray) -> np.ndarray:
    """Pixel coords → camera world-unit coords."""
    cx = uv[:, 0] / RENDER_W * (ORTHO_RIGHT - ORTHO_LEFT) + ORTHO_LEFT
    cy = ORTHO_TOP - uv[:, 1] / RENDER_H * (ORTHO_TOP - ORTHO_BOT)
    return np.stack([cx, cy], axis=-1)


def triangulate_joint(obs: list[tuple]) -> np.ndarray:
    """
    Triangulate a single joint from multi-view 2D observations.
    obs: list of (azimuth_deg, pixel_u, pixel_v)
    Returns world (x, y, z).

    For orthographic cameras, Y is directly measured; X and Z satisfy:
      px*cos(θ) - pz*sin(θ) = cx   for each view
    → overdetermined linear system solved with lstsq.
    """
    ys, rows_A, rhs = [], [], []
    for az_deg, pu, pv in obs:
        cx, cy = pixel_to_cam(np.array([[pu, pv]]))[0]
        ys.append(cy)
        t = np.radians(az_deg)
        rows_A.append([np.cos(t), -np.sin(t)])
        rhs.append(cx)

    A  = np.array(rows_A, dtype=np.float64)
    b  = np.array(rhs,    dtype=np.float64)
    wy = float(np.mean(ys))

    if len(obs) >= 2:
        xz, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        wx, wz = xz
    else:
        wx, wz = 0.0, 0.0

    return np.array([wx, wy, wz], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Multi-view HMR 2.0 + beta averaging
# ══════════════════════════════════════════════════════════════════════════════

def _load_hmr2(device):
    from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
    download_models()   # downloads to CACHE_DIR_4DHUMANS (no-op if already done)
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    return model.to(device).eval(), cfg


def _load_detector():
    from detectron2.config import LazyConfig
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    import hmr2
    cfg = LazyConfig.load(str(os.path.join(
        os.path.dirname(hmr2.__file__),
        "configs/cascade_mask_rcnn_vitdet_h_75ep.py")))
    cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl")
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    return DefaultPredictor_Lazy(cfg)


def _run_hmr2_on_image(img_bgr, model, model_cfg, detector, device):
    """
    Run HMR 2.0 on a BGR image. Returns dict or None.
    Keys: betas (10,), body_pose (23,3,3), global_orient (1,3,3),
          kp2d (44,2) in [0,1] normalised, kp3d (44,3), score (float)
    """
    import torch
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset

    det_out = detector(img_bgr)
    instances = det_out["instances"]
    valid = (instances.pred_classes == 0) & (instances.scores > 0.5)
    if not valid.any():
        return None

    boxes = instances.pred_boxes.tensor[valid].cpu().numpy()
    score = float(instances.scores[valid].max().cpu())
    best  = boxes[np.argmax((boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1]))]

    ds = ViTDetDataset(model_cfg, img_bgr, [best])
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    batch = recursive_to(next(iter(dl)), device)

    with torch.no_grad():
        out = model(batch)

    p = out["pred_smpl_params"]
    return {
        "betas":         p["betas"][0].cpu().numpy(),
        "body_pose":     p["body_pose"][0].cpu().numpy(),
        "global_orient": p["global_orient"][0].cpu().numpy(),
        "kp2d":          out["pred_keypoints_2d"][0].cpu().numpy(),  # (44,2) [-1,1]
        "kp3d":          out.get("pred_keypoints_3d", [None]*1)[0],
        "score":         score,
        "detected":      True,
    }


def estimate_betas_multiview(view_paths: list[str],
                              ref_path: str,
                              device: str = "cuda") -> tuple[np.ndarray, list]:
    """
    Phase 1: run HMR 2.0 on reference photo + front/3q/side renders.
    Returns (averaged_betas [10,], list_of_all_results).
    Falls back to zero betas (average body shape) if HMR2 is unavailable.
    """
    import cv2
    print("[rig P1] Loading HMR2 + detector...")
    try:
        model, model_cfg = _load_hmr2(device)
        detector = _load_detector()
    except Exception as e:
        print(f"[rig P1] HMR2 unavailable ({e}) — using zero betas (average body shape)")
        return np.zeros(10, dtype=np.float32), []

    sources = [(ref_path, None)]   # (path, azimuth_deg_or_None)
    for idx in FRONT_VIEW_INDICES:
        if idx < len(view_paths) and os.path.exists(view_paths[idx]):
            sources.append((view_paths[idx], VIEW_AZIMUTHS_DEG[idx]))

    results = []
    weighted_betas, total_w = np.zeros(10, dtype=np.float64), 0.0

    for path, az in sources:
        img = cv2.imread(path)
        if img is None:
            continue
        r = _run_hmr2_on_image(img, model, model_cfg, detector, device)
        if r is None:
            print(f"[rig P1]   {os.path.basename(path)}: no person detected")
            continue
        r["azimuth_deg"] = az
        r["path"]        = path
        results.append(r)
        w = r["score"]
        weighted_betas += r["betas"] * w
        total_w        += w
        print(f"[rig P1]   {os.path.basename(path)}: detected (score={w:.2f}), "
              f"betas[:3]={r['betas'][:3]}")

    avg_betas = (weighted_betas / total_w).astype(np.float32) if total_w > 0 \
                else np.zeros(10, dtype=np.float32)
    print(f"[rig P1] Averaged betas over {len(results)} detections.")
    return avg_betas, results


# ══════════════════════════════════════════════════════════════════════════════
# SMPL helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_smpl_tpose(betas: np.ndarray, smpl_dir: str = "/root/smpl_models"):
    """Returns (verts [N,3], faces [M,3], joints [24,3], lbs_weights [N,24]).
    Uses smplx if SMPL_NEUTRAL.pkl is available, else falls back to a synthetic
    proxy skeleton with proximity-based skinning weights."""
    import torch

    model_path = os.path.join(smpl_dir, "SMPL_NEUTRAL.pkl")
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        # Try download first, silently fall through to synthetic on failure
        try:
            _download_smpl_neutral(smpl_dir)
        except Exception:
            pass

    if os.path.exists(model_path) and os.path.getsize(model_path) > 100_000:
        import smplx
        smpl = smplx.create(smpl_dir, model_type="smpl", gender="neutral", num_betas=10)
        betas_t = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = smpl(betas=betas_t, return_verts=True)
        verts   = out.vertices[0].numpy().astype(np.float32)
        joints  = out.joints[0, :24].numpy().astype(np.float32)
        faces   = smpl.faces.astype(np.int32)
        weights = smpl.lbs_weights.numpy().astype(np.float32)
        return verts, faces, joints, weights

    print("[rig] SMPL_NEUTRAL.pkl unavailable — using synthetic proxy skeleton")
    return _synthetic_smpl_tpose()


def _synthetic_smpl_tpose():
    """Synthetic SMPL substitute: hardcoded T-pose joint positions + proximity weights.
    Gives a rough but functional rig for pipeline testing when SMPL is unavailable.
    For production, provide SMPL_NEUTRAL.pkl from https://smpl.is.tue.mpg.de/."""
    # 24 SMPL T-pose joint positions (metres, Y-up, facing +Z)
    joints = np.array([
        [ 0.00,  0.92,  0.00],  # 0  pelvis
        [-0.09,  0.86,  0.00],  # 1  left_hip
        [ 0.09,  0.86,  0.00],  # 2  right_hip
        [ 0.00,  1.05,  0.00],  # 3  spine1
        [-0.09,  0.52,  0.00],  # 4  left_knee
        [ 0.09,  0.52,  0.00],  # 5  right_knee
        [ 0.00,  1.17,  0.00],  # 6  spine2
        [-0.09,  0.10,  0.00],  # 7  left_ankle
        [ 0.09,  0.10,  0.00],  # 8  right_ankle
        [ 0.00,  1.29,  0.00],  # 9  spine3
        [-0.09,  0.00,  0.07],  # 10 left_foot
        [ 0.09,  0.00,  0.07],  # 11 right_foot
        [ 0.00,  1.46,  0.00],  # 12 neck
        [-0.07,  1.42,  0.00],  # 13 left_collar
        [ 0.07,  1.42,  0.00],  # 14 right_collar
        [ 0.00,  1.62,  0.00],  # 15 head
        [-0.17,  1.40,  0.00],  # 16 left_shoulder
        [ 0.17,  1.40,  0.00],  # 17 right_shoulder
        [-0.42,  1.40,  0.00],  # 18 left_elbow
        [ 0.42,  1.40,  0.00],  # 19 right_elbow
        [-0.65,  1.40,  0.00],  # 20 left_wrist
        [ 0.65,  1.40,  0.00],  # 21 right_wrist
        [-0.72,  1.40,  0.00],  # 22 left_hand
        [ 0.72,  1.40,  0.00],  # 23 right_hand
    ], dtype=np.float32)

    # Build synthetic proxy vertices: ~300 points clustered around each joint
    rng = np.random.default_rng(42)
    n_per_joint = 300
    proxy_v = []
    proxy_w = []
    for ji, jpos in enumerate(joints):
        pts = jpos + rng.normal(0, 0.06, (n_per_joint, 3)).astype(np.float32)
        proxy_v.append(pts)
        w = np.zeros((n_per_joint, 24), np.float32)
        w[:, ji] = 1.0
        proxy_w.append(w)

    proxy_v = np.concatenate(proxy_v, axis=0)   # (7200, 3)
    proxy_w = np.concatenate(proxy_w, axis=0)   # (7200, 24)
    proxy_f = np.zeros((0, 3), dtype=np.int32)  # no faces needed for KNN transfer
    return proxy_v, proxy_f, joints, proxy_w


def _download_smpl_neutral(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    url  = ("https://huggingface.co/spaces/TMElyralab/MusePose/resolve/main"
            "/models/smpl/SMPL_NEUTRAL.pkl")
    dest = os.path.join(out_dir, "SMPL_NEUTRAL.pkl")
    print("[rig] Downloading SMPL_NEUTRAL.pkl...")
    subprocess.run(["wget", "-q", url, "-O", dest], check=True)


def _smpl_to_render_space(verts: np.ndarray, joints: np.ndarray):
    """
    Normalise SMPL vertices to fit inside the [-0.55, 0.55] orthographic
    frustum used by the nvdiffrast renders (same as align_mesh_to_smpl).
    Returns (verts_norm, joints_norm, scale, offset).
    """
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    height = ymax - ymin
    scale  = (ORTHO_TOP - ORTHO_BOT) / max(height, 1e-6)

    # Centre on pelvis (joint 0) horizontally, floor-align vertically
    v = verts  * scale
    j = joints * scale
    cx = (v[:, 0].max() + v[:, 0].min()) * 0.5
    cz = (v[:, 2].max() + v[:, 2].min()) * 0.5
    v[:, 0] -= cx;  j[:, 0] -= cx
    v[:, 2] -= cz;  j[:, 2] -= cz
    v[:, 1] -= v[:, 1].min() + ORTHO_BOT   # floor at ORTHO_BOT
    j[:, 1] -= (verts[:, 1].min() * scale) - ORTHO_BOT
    return v, j, scale, np.array([-cx, -v[:,1].min() + ORTHO_BOT, -cz])


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Silhouette fitting
# ══════════════════════════════════════════════════════════════════════════════

def _extract_silhouette(render_path: str, threshold: int = 20) -> np.ndarray:
    """Binary mask (H×W bool) from a render: foreground = any channel > threshold."""
    import cv2
    img = cv2.imread(render_path)
    if img is None:
        return np.zeros((RENDER_H, RENDER_W), dtype=bool)
    return img.max(axis=2) > threshold


def _render_smpl_silhouette(verts_norm: np.ndarray, faces: np.ndarray,
                              azimuth_deg: float) -> np.ndarray:
    """
    Rasterise SMPL mesh silhouette for given azimuth (orthographic).
    Returns binary mask (H×W bool).
    """
    from PIL import Image, ImageDraw

    cam_xy = world_to_cam(verts_norm, azimuth_deg)
    pix    = cam_to_pixel(cam_xy)  # (N, 2)

    img = Image.new("L", (RENDER_W, RENDER_H), 0)
    draw = ImageDraw.Draw(img)
    for f in faces:
        pts = [(float(pix[i, 0]), float(pix[i, 1])) for i in f]
        draw.polygon(pts, fill=255)
    return np.array(img) > 0


def _sil_loss(betas: np.ndarray, target_masks: list,
               valid_views: list[int], faces: np.ndarray) -> float:
    """1 - mean IoU between SMPL silhouettes and TripoSG render masks."""
    try:
        verts, _, _, _ = get_smpl_tpose(betas.astype(np.float32))
        verts_n, _, _, _ = _smpl_to_render_space(verts, verts.copy())
        iou_sum = 0.0
        for i in valid_views:
            pred = _render_smpl_silhouette(verts_n, faces, VIEW_AZIMUTHS_DEG[i])
            tgt  = target_masks[i]
            inter = (pred & tgt).sum()
            union = (pred | tgt).sum()
            iou_sum += inter / max(union, 1)
        return 1.0 - iou_sum / len(valid_views)
    except Exception:
        return 1.0


def fit_betas_silhouette(betas_init: np.ndarray, view_paths: list[str],
                          max_iter: int = 60) -> np.ndarray:
    """
    Phase 2: optimise SMPL betas to match TripoSG render silhouettes.
    Only uses views whose render file exists.
    """
    from scipy.optimize import minimize

    valid = [i for i, p in enumerate(view_paths) if os.path.exists(p)]
    if not valid:
        print("[rig P2] No render files found — skipping silhouette fit")
        return betas_init

    print(f"[rig P2] Extracting silhouettes from {len(valid)} views...")
    masks = [_extract_silhouette(view_paths[i]) if i in valid
             else np.zeros((RENDER_H, RENDER_W), bool)
             for i in range(len(VIEW_NAMES))]

    # Use only back-facing views for shape, not back (which shows less shape info)
    fit_views = [i for i in valid if i in [0, 1, 2]]
    if not fit_views:
        fit_views = valid

    # Pre-fetch faces (constant across iterations)
    verts0, faces0, _, _ = get_smpl_tpose(betas_init)

    loss0 = _sil_loss(betas_init, masks, fit_views, faces0)
    print(f"[rig P2] Initial silhouette loss: {loss0:.4f}")

    result = minimize(
        fun=lambda b: _sil_loss(b, masks, fit_views, faces0),
        x0=betas_init.astype(np.float64),
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0)] * 10,
        options={"maxiter": max_iter, "ftol": 1e-4, "gtol": 1e-3},
    )

    refined = result.x.astype(np.float32)
    loss1   = _sil_loss(refined, masks, fit_views, faces0)
    print(f"[rig P2] Silhouette fit done: loss {loss0:.4f} → {loss1:.4f}  "
          f"({result.nit} iters, {'converged' if result.success else 'stopped'})")
    return refined


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Multi-view joint triangulation
# ══════════════════════════════════════════════════════════════════════════════

# HMR 2.0 outputs 44 keypoints; first 24 map to SMPL joints
HMR2_TO_SMPL = list(range(24))

def triangulate_joints_multiview(hmr2_results: list) -> np.ndarray | None:
    """
    Phase 3: triangulate world-space SMPL joints from multi-view HMR 2.0 2D keypoints.

    hmr2_results: list of dicts from _run_hmr2_on_image, each with
      kp2d (44,2) in [-1,1] normalised NDC  and  azimuth_deg (float or None).

    Only uses results from rendered views (azimuth_deg is not None).
    Returns (24,3) world joint positions, or None if < 2 valid views.
    """
    view_results = [r for r in hmr2_results
                    if r.get("azimuth_deg") is not None and r.get("kp2d") is not None]

    if len(view_results) < 2:
        print(f"[rig P3] Only {len(view_results)} render views with detections "
              "— need ≥2 for triangulation, skipping")
        return None

    print(f"[rig P3] Triangulating from {len(view_results)} views: "
          + ", ".join(os.path.basename(r["path"]) for r in view_results))

    # Convert HMR2 NDC keypoints → pixel coords
    # kp2d is (44,2) in [-1,1]; pixel = (kp+1)/2 * [W, H]
    joints_world = np.zeros((24, 3), dtype=np.float32)

    for j in range(24):
        obs = []
        for r in view_results:
            kp = r["kp2d"][j]            # (2,) in [-1,1]
            pu = (kp[0] + 1.0) / 2.0 * RENDER_W
            pv = (kp[1] + 1.0) / 2.0 * RENDER_H
            obs.append((r["azimuth_deg"], pu, pv))
        joints_world[j] = triangulate_joint(obs)

    print(f"[rig P3] Triangulated 24 joints. "
          f"Pelvis: {joints_world[0].round(3)}, "
          f"Head: {joints_world[15].round(3)}")
    return joints_world


# ══════════════════════════════════════════════════════════════════════════════
# Skinning weight transfer
# ══════════════════════════════════════════════════════════════════════════════

def transfer_skinning(smpl_verts: np.ndarray, smpl_weights: np.ndarray,
                       target_verts: np.ndarray, k: int = 4) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(smpl_verts)
    dists, idxs = tree.query(target_verts, k=k, workers=-1)
    dists  = np.maximum(dists, 1e-8)
    inv_d  = 1.0 / dists
    inv_d /= inv_d.sum(axis=1, keepdims=True)
    transferred = np.einsum("nk,nkj->nj", inv_d, smpl_weights[idxs])
    row_sums = transferred.sum(axis=1, keepdims=True)
    transferred /= np.where(row_sums > 0, row_sums, 1.0)
    return transferred.astype(np.float32)


def align_mesh_to_smpl(mesh_verts: np.ndarray, smpl_verts: np.ndarray,
                        smpl_joints: np.ndarray) -> np.ndarray:
    smpl_h = smpl_verts[:, 1].max() - smpl_verts[:, 1].min()
    mesh_h = mesh_verts[:, 1].max() - mesh_verts[:, 1].min()
    scale  = smpl_h / max(mesh_h, 1e-6)
    v = mesh_verts * scale
    cx = (v[:, 0].max() + v[:, 0].min()) * 0.5
    cz = (v[:, 2].max() + v[:, 2].min()) * 0.5
    v[:, 0] += smpl_joints[0, 0] - cx
    v[:, 2] += smpl_joints[0, 2] - cz
    v[:, 1] -= v[:, 1].min()
    return v


# ══════════════════════════════════════════════════════════════════════════════
# GLB export
# ══════════════════════════════════════════════════════════════════════════════

def export_rigged_glb(verts, faces, uv, texture_img, joints, skin_weights, out_path):
    import pygltflib
    from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Accessor,
                            BufferView, Buffer, Material, Texture,
                            Image as GImage, Sampler, Skin, Asset)
    from pygltflib import (ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT,
                            UNSIGNED_INT, UNSIGNED_SHORT, LINEAR,
                            LINEAR_MIPMAP_LINEAR, REPEAT, SCALAR, VEC2,
                            VEC3, VEC4, MAT4)

    gltf   = GLTF2()
    gltf.asset = Asset(version="2.0", generator="rig_stage.py")
    blobs  = []

    def _add(data: np.ndarray, comp, acc_type, target=None):
        b   = data.tobytes()
        pad = (4 - len(b) % 4) % 4
        off = sum(len(x) for x in blobs)
        blobs.append(b + b"\x00" * pad)
        bv  = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=off,
                                            byteLength=len(b), target=target))
        ac  = len(gltf.accessors)
        flat = data.flatten()
        gltf.accessors.append(Accessor(
            bufferView=bv, byteOffset=0, componentType=comp,
            type=acc_type, count=len(data),
            min=[float(flat.min())], max=[float(flat.max())]))
        return ac

    pos_acc = _add(verts.astype(np.float32),  FLOAT,          VEC3,  ARRAY_BUFFER)

    v0,v1,v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    fn = np.cross(v1-v0, v2-v0); fn /= (np.linalg.norm(fn,axis=1,keepdims=True)+1e-8)
    vn = np.zeros_like(verts)
    for i in range(3): np.add.at(vn, faces[:,i], fn)
    vn /= (np.linalg.norm(vn,axis=1,keepdims=True)+1e-8)
    nor_acc = _add(vn.astype(np.float32),      FLOAT,          VEC3,  ARRAY_BUFFER)

    if uv is None: uv = np.zeros((len(verts),2), np.float32)
    uv_acc  = _add(uv.astype(np.float32),      FLOAT,          VEC2,  ARRAY_BUFFER)
    idx_acc = _add(faces.astype(np.uint32).flatten(), UNSIGNED_INT, SCALAR, ELEMENT_ARRAY_BUFFER)

    top4_idx = np.argsort(-skin_weights, axis=1)[:,:4].astype(np.uint16)
    top4_w   = np.take_along_axis(skin_weights, top4_idx.astype(np.int64), axis=1).astype(np.float32)
    top4_w  /= top4_w.sum(axis=1,keepdims=True).clip(1e-8,None)
    j_acc   = _add(top4_idx, UNSIGNED_SHORT, "VEC4", ARRAY_BUFFER)
    w_acc   = _add(top4_w,   FLOAT,          "VEC4", ARRAY_BUFFER)

    if texture_img is not None:
        import io
        buf = io.BytesIO(); texture_img.save(buf, format="PNG"); ib = buf.getvalue()
        off = sum(len(x) for x in blobs); pad = (4-len(ib)%4)%4
        blobs.append(ib + b"\x00"*pad)
        gltf.bufferViews.append(BufferView(buffer=0,byteOffset=off,byteLength=len(ib)))
        gltf.images.append(GImage(mimeType="image/png",bufferView=len(gltf.bufferViews)-1))
        gltf.samplers.append(Sampler(magFilter=LINEAR,minFilter=LINEAR_MIPMAP_LINEAR,
                                      wrapS=REPEAT,wrapT=REPEAT))
        gltf.textures.append(Texture(sampler=0,source=0))
        gltf.materials.append(Material(name="body",
            pbrMetallicRoughness={"baseColorTexture":{"index":0},
                                   "metallicFactor":0.0,"roughnessFactor":0.8},
            doubleSided=True))
    else:
        gltf.materials.append(Material(name="body",doubleSided=True))

    prim = Primitive(attributes={"POSITION":pos_acc,"NORMAL":nor_acc,
                                  "TEXCOORD_0":uv_acc,"JOINTS_0":j_acc,"WEIGHTS_0":w_acc},
                     indices=idx_acc, material=0)
    gltf.meshes.append(Mesh(name="body",primitives=[prim]))

    jnodes = []
    for i,(name,parent) in enumerate(zip(SMPL_JOINT_NAMES,SMPL_PARENTS)):
        t = joints[i].tolist() if parent==-1 else (joints[i]-joints[parent]).tolist()
        n = Node(name=name,translation=t,children=[])
        jnodes.append(len(gltf.nodes)); gltf.nodes.append(n)
    for i,p in enumerate(SMPL_PARENTS):
        if p!=-1: gltf.nodes[jnodes[p]].children.append(jnodes[i])

    ibms = np.stack([np.eye(4,dtype=np.float32) for _ in range(len(joints))])
    for i in range(len(joints)): ibms[i,:3,3] = -joints[i]
    ibm_acc = _add(ibms.astype(np.float32), FLOAT, MAT4)
    skin_idx = len(gltf.skins)
    gltf.skins.append(Skin(name="smpl_skin",skeleton=jnodes[0],
                            joints=jnodes,inverseBindMatrices=ibm_acc))

    mesh_node = len(gltf.nodes)
    gltf.nodes.append(Node(name="body_mesh",mesh=0,skin=skin_idx))
    root_node = len(gltf.nodes)
    gltf.nodes.append(Node(name="root",children=[jnodes[0],mesh_node]))
    gltf.scenes.append(Scene(name="Scene",nodes=[root_node]))
    gltf.scene = 0

    bin_data = b"".join(blobs)
    gltf.buffers.append(Buffer(byteLength=len(bin_data)))
    gltf.set_binary_blob(bin_data)
    gltf.save_binary(out_path)
    print(f"[rig] Rigged GLB → {out_path}  ({os.path.getsize(out_path)//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# FBX export via Blender headless
# ══════════════════════════════════════════════════════════════════════════════

_BLENDER_SCRIPT = """\
import bpy, sys
args = sys.argv[sys.argv.index('--') + 1:]
glb_in, fbx_out = args[0], args[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_in)
bpy.ops.export_scene.fbx(
    filepath=fbx_out, use_selection=False,
    add_leaf_bones=False, bake_anim=False,
    path_mode='COPY', embed_textures=True,
)
print('FBX OK:', fbx_out)
"""

def export_fbx(rigged_glb: str, out_path: str) -> bool:
    blender = next((c for c in ["/usr/bin/blender","/usr/local/bin/blender"]
                    if os.path.exists(c)), None)
    if blender is None:
        r = subprocess.run(["which","blender"],capture_output=True,text=True)
        blender = r.stdout.strip() or None
    if blender is None:
        print("[rig] Blender not found — skipping FBX")
        return False
    try:
        with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f:
            f.write(_BLENDER_SCRIPT); script = f.name
        r = subprocess.run([blender,"--background","--python",script,
                             "--",rigged_glb,out_path],
                            capture_output=True,text=True,timeout=120)
        ok = os.path.exists(out_path)
        if not ok: print(f"[rig] Blender stderr:\n{r.stderr[-800:]}")
        return ok
    except Exception:
        print(f"[rig] export_fbx:\n{traceback.format_exc()}")
        return False
    finally:
        try: os.unlink(script)
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# MDM — Motion Diffusion Model
# ══════════════════════════════════════════════════════════════════════════════

MDM_DIR  = "/root/MDM"
MDM_CKPT = f"{MDM_DIR}/save/humanml_trans_enc_512/model000200000.pt"

# HumanML3D 22-joint parent array (matches SMPL joints 0-21)
_MDM_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

def setup_mdm() -> bool:
    """Clone MDM repo, install deps, download checkpoint. Idempotent."""
    if os.path.exists(MDM_CKPT):
        return True
    print("[MDM] First-time setup...")

    if not os.path.exists(MDM_DIR):
        r = subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/GuyTevet/motion-diffusion-model.git", MDM_DIR],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"[MDM] git clone failed:\n{r.stderr}")
            return False

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/openai/CLIP.git",
        "einops", "rotary-embedding-torch", "gdown"], check=False, timeout=300)

    # HumanML3D normalisation stats (small .npy files needed for inference)
    stats_dir = f"{MDM_DIR}/dataset/HumanML3D"
    os.makedirs(stats_dir, exist_ok=True)
    base = "https://github.com/EricGuo5513/HumanML3D/raw/main/HumanML3D"
    for fn in ["Mean.npy", "Std.npy"]:
        dest = f"{stats_dir}/{fn}"
        if not os.path.exists(dest):
            subprocess.run(["wget", "-q", f"{base}/{fn}", "-O", dest],
                           check=False, timeout=60)

    # Checkpoint (~1.3 GB) — try HuggingFace mirror first, then gdown
    ckpt_dir = os.path.dirname(MDM_CKPT)
    os.makedirs(ckpt_dir, exist_ok=True)
    hf = ("https://huggingface.co/Mathux/motion-diffusion-model/resolve/main/"
          "humanml_trans_enc_512/model000200000.pt")
    r = subprocess.run(["wget", "-q", "--show-progress", hf, "-O", MDM_CKPT],
                        capture_output=True, timeout=3600)
    if r.returncode != 0 or not os.path.exists(MDM_CKPT) or \
            os.path.getsize(MDM_CKPT) < 10_000_000:
        print("[MDM] HF download failed — trying gdown (official Google Drive)...")
        subprocess.run([sys.executable, "-m", "gdown",
                        "--id", "1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821",
                        "-O", MDM_CKPT], check=False, timeout=3600)

    ok = os.path.exists(MDM_CKPT) and os.path.getsize(MDM_CKPT) > 10_000_000
    print(f"[MDM] Setup {'OK' if ok else 'FAILED'}")
    return ok


def generate_motion_mdm(text_prompt: str, n_frames: int = 120,
                          fps: int = 20, device: str = "cuda") -> dict | None:
    """
    Run MDM text-to-motion. Returns {'positions': (n_frames,22,3), 'fps': fps}
    or None on failure. First call runs setup_mdm() which may take ~10 min.
    """
    if not setup_mdm():
        return None

    out_dir = tempfile.mkdtemp(prefix="mdm_")
    motion_len = round(n_frames / fps, 2)

    # Minimal inline driver — avoids MDM's argparse setup entirely
    driver_src = f"""
import sys, os
sys.path.insert(0, {repr(MDM_DIR)})
os.chdir({repr(MDM_DIR)})
import numpy as np, torch

from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import clip as clip_lib

fixseed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dist_util.dev = lambda: device

import argparse
args = argparse.Namespace(
    arch='trans_enc', emb_trans_dec=False,
    layers=8, latent_dim=512, ff_size=1024, num_heads=4,
    dropout=0.1, activation='gelu', data_rep='rot6d',
    dataset='humanml', cond_mode='text', cond_mask_prob=0.1,
    lambda_rcxyz=0, lambda_vel=0, lambda_fc=0,
    njoints=263, nfeats=1,
    num_actions=1, translation=True, pose_rep='rot6d',
    glob=True, glob_rot=True, npose=315,
    device=0, seed=42, batch_size=1, num_samples=1,
    num_repetitions=1, motion_length={motion_len!r},
    input_text='', text_prompt='', action_file='', action_name='',
    output_dir={repr(out_dir)}, guidance_param=2.5,
    unconstrained=False,
    # additional args required by get_model_args / create_gaussian_diffusion
    text_encoder_type='clip',
    pos_embed_max_len=5000,
    mask_frames=False,
    pred_len=0,
    context_len=0,
    diffusion_steps=1000,
    noise_schedule='cosine',
    sigma_small=True,
    lambda_target_loc=0,
)

class _MockData:
    class dataset:
        pass
model, diffusion = create_model_and_diffusion(args, _MockData())
state = torch.load({repr(MDM_CKPT)}, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(state, strict=False)
model.eval().to(device)

max_frames = int({n_frames})
shape = (1, model.njoints, model.nfeats, max_frames)
clip_model, _ = clip_lib.load('ViT-B/32', device=device, jit=False)
clip_model.eval()
tokens = clip_lib.tokenize([{repr(text_prompt)}]).to(device)
with torch.no_grad():
    text_emb = clip_model.encode_text(tokens).float()

model_kwargs = {{
    'y': {{
        'mask': torch.ones(1, 1, 1, max_frames).to(device),
        'lengths': torch.tensor([max_frames]).to(device),
        'text': [{repr(text_prompt)}],
        'tokens': [''],
        'scale': torch.ones(1).to(device) * 2.5,
    }}
}}

with torch.no_grad():
    sample = diffusion.p_sample_loop(
        model, shape, clip_denoised=False,
        model_kwargs=model_kwargs, skip_timesteps=0,
        init_image=None, progress=False, dump_steps=None,
        noise=None, const_noise=False,
    )  # (1, 263, 1, n_frames)

# Convert HumanML3D features → joint XYZ using recover_from_ric (no SMPL needed)
# sample: (1, 263, 1, n_frames) → (1, n_frames, 263)
sample_ric = sample[:, :, 0, :].permute(0, 2, 1)
xyz = recover_from_ric(sample_ric, 22)  # (1, n_frames, 22, 3)
positions = xyz[0].cpu().numpy()        # (n_frames, 22, 3)
np.save(os.path.join({repr(out_dir)}, 'positions.npy'), positions)
print('MDM_DONE')
"""
    driver_f = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(driver_src)
            driver_f = f.name

        r = subprocess.run(
            [sys.executable, driver_f],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": MDM_DIR, "CUDA_VISIBLE_DEVICES": "0"},
        )
        print(f"[MDM] stdout: {r.stdout[-400:]}")
        if r.returncode != 0:
            print(f"[MDM] FAILED:\n{r.stderr[-600:]}")
            return None

        npy = os.path.join(out_dir, "positions.npy")
        if not os.path.exists(npy):
            print("[MDM] positions.npy not found")
            return None

        arr = np.load(npy)                       # (n_frames, 22, 3)
        positions = arr                          # already (n_frames, 22, 3)
        print(f"[MDM] Motion: {positions.shape}, fps={fps}")
        return {"positions": positions, "fps": fps, "n_frames": positions.shape[0]}

    except Exception:
        print(f"[MDM] Exception:\n{traceback.format_exc()}")
        return None
    finally:
        if driver_f:
            try: os.unlink(driver_f)
            except: pass


# ══════════════════════════════════════════════════════════════════════════════
# FK Inversion — joint world-positions → local quaternions per frame
# ══════════════════════════════════════════════════════════════════════════════

def _quat_between(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion [x,y,z,w] that rotates unit vector v0 → v1."""
    cross = np.cross(v0, v1)
    dot   = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
    cn    = np.linalg.norm(cross)
    if cn < 1e-8:
        return np.array([0., 0., 0., 1.], np.float32) if dot > 0 \
               else np.array([1., 0., 0., 0.], np.float32)
    axis  = cross / cn
    angle = np.arctan2(cn, dot)
    s     = np.sin(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle*0.5)], np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two [x,y,z,w] quaternions."""
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], np.float32)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.array([v[0], v[1], v[2], 0.], np.float32)
    return _quat_mul(_quat_mul(q, qv), _quat_inv(q))[:3]


def positions_to_local_quats(positions: np.ndarray,
                               t_pose_joints: np.ndarray,
                               parents: list) -> np.ndarray:
    """
    Derive per-joint local quaternions from world-space joint positions.
    positions   : (n_frames, n_joints, 3)
    t_pose_joints : (n_joints, 3)  — SMPL T-pose joints in same scale/space
    parents     : list of length n_joints, parent index (-1 for root)
    Returns     : (n_frames, n_joints, 4) XYZW local quaternions
    """
    n_frames, n_joints, _ = positions.shape
    quats = np.zeros((n_frames, n_joints, 4), np.float32)
    quats[:, :, 3] = 1.0  # default identity

    # Compute global quats first, then convert to local
    global_quats = np.zeros_like(quats)
    global_quats[:, :, 3] = 1.0

    for j in range(n_joints):
        p = parents[j]
        if p < 0:
            # Root: no rotation relative to world (translation handles it)
            global_quats[:, j] = [0, 0, 0, 1]
            continue

        # T-pose parent→child bone direction
        tp_dir = t_pose_joints[j] - t_pose_joints[p]
        tp_len = np.linalg.norm(tp_dir)
        if tp_len < 1e-6:
            continue
        tp_dir /= tp_len

        for f in range(n_frames):
            an_dir = positions[f, j] - positions[f, p]
            an_len = np.linalg.norm(an_dir)
            if an_len < 1e-6:
                global_quats[f, j] = global_quats[f, p]
                continue
            an_dir /= an_len
            # Global rotation = parent_global ∘ local
            # We want global bone direction to match an_dir
            # global_bone_tpose = rotate(global_parent, tp_dir_in_parent_space)
            # For SMPL T-pose, bone dirs are in world space already
            gq = _quat_between(tp_dir, an_dir)
            global_quats[f, j] = gq

    # Convert global → local (local = inv_parent_global ∘ global)
    for j in range(n_joints):
        p = parents[j]
        if p < 0:
            quats[:, j] = global_quats[:, j]
        else:
            for f in range(n_frames):
                quats[f, j] = _quat_mul(_quat_inv(global_quats[f, p]),
                                         global_quats[f, j])

    return quats


# ══════════════════════════════════════════════════════════════════════════════
# Animated GLB export
# ══════════════════════════════════════════════════════════════════════════════

def export_animated_glb(verts, faces, uv, texture_img,
                          joints,         # (24, 3) T-pose joint world positions
                          skin_weights,   # (N_verts, 24)
                          joint_quats,    # (n_frames, 24, 4) XYZW local quaternions
                          root_trans,     # (n_frames, 3) world translation of root
                          fps: int,
                          out_path: str):
    """
    Export fully animated rigged GLB.
    Skeleton + skin weights identical to export_rigged_glb;
    adds a GLTF animation with per-joint rotation channels + root translation.
    """
    import pygltflib
    from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Accessor,
                            BufferView, Buffer, Material, Texture,
                            Image as GImage, Sampler, Skin, Asset,
                            Animation, AnimationChannel, AnimationChannelTarget,
                            AnimationSampler)
    from pygltflib import (ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT,
                            UNSIGNED_INT, UNSIGNED_SHORT, LINEAR,
                            LINEAR_MIPMAP_LINEAR, REPEAT, SCALAR, VEC2,
                            VEC3, VEC4, MAT4)

    n_frames, n_joints_anim, _ = joint_quats.shape
    n_joints = len(joints)

    gltf  = GLTF2()
    gltf.asset = Asset(version="2.0", generator="rig_stage.py/animated")
    blobs = []

    def _add(data: np.ndarray, comp, acc_type, target=None,
             set_min_max=False):
        b   = data.tobytes()
        pad = (4 - len(b) % 4) % 4
        off = sum(len(x) for x in blobs)
        blobs.append(b + b"\x00" * pad)
        bv  = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=off,
                                            byteLength=len(b), target=target))
        ac  = len(gltf.accessors)
        flat = data.flatten().astype(np.float32)
        kw = {}
        if set_min_max:
            kw = {"min": [float(flat.min())], "max": [float(flat.max())]}
        gltf.accessors.append(Accessor(
            bufferView=bv, byteOffset=0, componentType=comp,
            type=acc_type, count=len(data), **kw))
        return ac

    # ── Mesh geometry ──────────────────────────────────────────────────────
    pos_acc = _add(verts.astype(np.float32), FLOAT, VEC3, ARRAY_BUFFER)

    v0,v1,v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    fn = np.cross(v1-v0, v2-v0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    vn = np.zeros_like(verts)
    for i in range(3): np.add.at(vn, faces[:,i], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)
    nor_acc = _add(vn.astype(np.float32), FLOAT, VEC3, ARRAY_BUFFER)

    if uv is None: uv = np.zeros((len(verts), 2), np.float32)
    uv_acc  = _add(uv.astype(np.float32), FLOAT, VEC2, ARRAY_BUFFER)
    idx_acc = _add(faces.astype(np.uint32).flatten(), UNSIGNED_INT,
                   SCALAR, ELEMENT_ARRAY_BUFFER)

    top4_idx = np.argsort(-skin_weights, axis=1)[:, :4].astype(np.uint16)
    top4_w   = np.take_along_axis(skin_weights, top4_idx.astype(np.int64), axis=1).astype(np.float32)
    top4_w  /= top4_w.sum(axis=1, keepdims=True).clip(1e-8, None)
    j_acc = _add(top4_idx, UNSIGNED_SHORT, "VEC4", ARRAY_BUFFER)
    w_acc = _add(top4_w,   FLOAT,          "VEC4", ARRAY_BUFFER)

    # ── Texture ────────────────────────────────────────────────────────────
    if texture_img is not None:
        import io
        buf = io.BytesIO(); texture_img.save(buf, format="PNG"); ib = buf.getvalue()
        off = sum(len(x) for x in blobs); pad2 = (4 - len(ib) % 4) % 4
        blobs.append(ib + b"\x00" * pad2)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=len(ib)))
        gltf.images.append(GImage(mimeType="image/png", bufferView=len(gltf.bufferViews)-1))
        gltf.samplers.append(Sampler(magFilter=LINEAR, minFilter=LINEAR_MIPMAP_LINEAR,
                                      wrapS=REPEAT, wrapT=REPEAT))
        gltf.textures.append(Texture(sampler=0, source=0))
        gltf.materials.append(Material(name="body",
            pbrMetallicRoughness={"baseColorTexture": {"index": 0},
                                   "metallicFactor": 0.0, "roughnessFactor": 0.8},
            doubleSided=True))
    else:
        gltf.materials.append(Material(name="body", doubleSided=True))

    prim = Primitive(
        attributes={"POSITION": pos_acc, "NORMAL": nor_acc,
                    "TEXCOORD_0": uv_acc, "JOINTS_0": j_acc, "WEIGHTS_0": w_acc},
        indices=idx_acc, material=0)
    gltf.meshes.append(Mesh(name="body", primitives=[prim]))

    # ── Skeleton nodes ─────────────────────────────────────────────────────
    jnodes = []
    for i, (name, parent) in enumerate(zip(SMPL_JOINT_NAMES, SMPL_PARENTS)):
        t = joints[i].tolist() if parent == -1 else (joints[i] - joints[parent]).tolist()
        n = Node(name=name, translation=t, children=[])
        jnodes.append(len(gltf.nodes)); gltf.nodes.append(n)
    for i, p in enumerate(SMPL_PARENTS):
        if p != -1: gltf.nodes[jnodes[p]].children.append(jnodes[i])

    ibms = np.stack([np.eye(4, dtype=np.float32) for _ in range(n_joints)])
    for i in range(n_joints): ibms[i, :3, 3] = -joints[i]
    ibm_acc  = _add(ibms.astype(np.float32), FLOAT, MAT4)
    skin_idx = len(gltf.skins)
    gltf.skins.append(Skin(name="smpl_skin", skeleton=jnodes[0],
                            joints=jnodes, inverseBindMatrices=ibm_acc))

    mesh_node = len(gltf.nodes)
    gltf.nodes.append(Node(name="body_mesh", mesh=0, skin=skin_idx))
    root_node = len(gltf.nodes)
    gltf.nodes.append(Node(name="root", children=[jnodes[0], mesh_node]))
    gltf.scenes.append(Scene(name="Scene", nodes=[root_node]))
    gltf.scene = 0

    # ── Animation ──────────────────────────────────────────────────────────
    dt   = 1.0 / fps
    times = np.arange(n_frames, dtype=np.float32) * dt   # (n_frames,)
    time_acc = _add(times, FLOAT, SCALAR, set_min_max=True)

    channels, samplers = [], []

    # Per-joint rotation tracks
    for j in range(min(n_joints_anim, n_joints)):
        q = joint_quats[:, j, :].astype(np.float32)   # (n_frames, 4) XYZW
        q_acc = _add(q, FLOAT, VEC4)
        si = len(samplers)
        samplers.append(AnimationSampler(input=time_acc, output=q_acc,
                                          interpolation="LINEAR"))
        channels.append(AnimationChannel(
            sampler=si,
            target=AnimationChannelTarget(node=jnodes[j], path="rotation")))

    # Root translation track
    if root_trans is not None:
        tr = root_trans.astype(np.float32)   # (n_frames, 3)
        tr_acc = _add(tr, FLOAT, VEC3)
        si = len(samplers)
        samplers.append(AnimationSampler(input=time_acc, output=tr_acc,
                                          interpolation="LINEAR"))
        channels.append(AnimationChannel(
            sampler=si,
            target=AnimationChannelTarget(node=jnodes[0], path="translation")))

    gltf.animations.append(Animation(name="mdm_motion",
                                      channels=channels, samplers=samplers))

    # ── Finalise ───────────────────────────────────────────────────────────
    bin_data = b"".join(blobs)
    gltf.buffers.append(Buffer(byteLength=len(bin_data)))
    gltf.set_binary_blob(bin_data)
    gltf.save_binary(out_path)
    dur = times[-1] if len(times) else 0
    print(f"[rig] Animated GLB → {out_path}  "
          f"({os.path.getsize(out_path)//1024} KB, {n_frames} frames @ {fps}fps = {dur:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_rig_pipeline(glb_path: str, reference_image_path: str,
                      out_dir: str, device: str = "cuda",
                      export_fbx_flag: bool = True,
                      mdm_prompt: str = "",
                      mdm_n_frames: int = 120,
                      mdm_fps: int = 20) -> dict:
    import trimesh
    os.makedirs(out_dir, exist_ok=True)
    result = {"rigged_glb": None, "animated_glb": None, "fbx": None,
              "smpl_params": None, "status": "", "phases": {}}

    try:
        # ── load TripoSG mesh ─────────────────────────────────────────────
        print("[rig] Loading TripoSG mesh...")
        scene = trimesh.load(glb_path, force="scene")
        if isinstance(scene, trimesh.Scene):
            geom = list(scene.geometry.values())
            mesh = trimesh.util.concatenate(geom) if len(geom)>1 else geom[0]
        else:
            mesh = scene
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces,    dtype=np.int32)

        # UV + texture: try source geoms before concatenation (more reliable)
        uv, tex = None, None
        src_geoms = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
        for g in src_geoms:
            if not hasattr(g.visual, "uv") or g.visual.uv is None:
                continue
            try:
                candidate_uv = np.array(g.visual.uv, dtype=np.float32)
                if len(candidate_uv) == len(verts):
                    uv = candidate_uv
                    mat = getattr(g.visual, "material", None)
                    if mat is not None:
                        for attr in ("image", "baseColorTexture", "diffuse"):
                            img = getattr(mat, attr, None)
                            if img is not None:
                                from PIL import Image as _PILImage
                                tex = img if isinstance(img, _PILImage.Image) else None
                                break
                    break
            except Exception:
                pass
        if uv is None:
            print("[rig] WARNING: UV not found or vertex count mismatch — mesh will be untextured")
        print(f"[rig] Mesh: {len(verts)} verts, {len(faces)} faces, "
              f"UV={'yes' if uv is not None else 'no'}, "
              f"texture={'yes' if tex is not None else 'no'}")

        # ── Phase 1: multi-view beta averaging ───────────────────────────
        print("\n[rig] ── Phase 1: multi-view beta averaging ──")
        betas, hmr2_results = estimate_betas_multiview(VIEW_PATHS, reference_image_path, device)
        result["phases"]["p1_betas"] = betas.tolist()

        # ── Phase 2: silhouette fitting ───────────────────────────────────
        print("\n[rig] ── Phase 2: silhouette fitting ──")
        betas = fit_betas_silhouette(betas, VIEW_PATHS)
        result["phases"]["p2_betas"] = betas.tolist()

        # ── Phase 3: multi-view joint triangulation ───────────────────────
        print("\n[rig] ── Phase 3: multi-view joint triangulation ──")
        tri_joints = triangulate_joints_multiview(hmr2_results)
        result["phases"]["p3_triangulated"] = tri_joints is not None

        # ── build SMPL T-pose with refined betas ──────────────────────────
        print("\n[rig] Building SMPL T-pose...")
        smpl_v, smpl_f, smpl_j, smpl_w = get_smpl_tpose(betas)

        # Override with triangulated joints if available
        if tri_joints is not None:
            # Triangulated joints are in render-normalised space; convert to SMPL scale
            _, _, scale, _ = _smpl_to_render_space(smpl_v.copy(), smpl_j.copy())
            smpl_j = tri_joints / scale          # back to SMPL metric space
            print("[rig] Using triangulated skeleton joints.")

        # ── align TripoSG mesh to SMPL ────────────────────────────────────
        verts_aligned = align_mesh_to_smpl(verts, smpl_v, smpl_j)

        # ── skinning weight transfer ──────────────────────────────────────
        print("[rig] Transferring skinning weights...")
        skin_w = transfer_skinning(smpl_v, smpl_w, verts_aligned)

        # ── export rigged GLB ─────────────────────────────────────────────
        rigged_glb = os.path.join(out_dir, "rigged.glb")
        export_rigged_glb(verts_aligned, faces, uv, tex, smpl_j, skin_w, rigged_glb)
        result["rigged_glb"] = rigged_glb

        # ── export FBX ────────────────────────────────────────────────────
        if export_fbx_flag:
            fbx = os.path.join(out_dir, "rigged.fbx")
            result["fbx"] = fbx if export_fbx(rigged_glb, fbx) else None

        # ── MDM animation ─────────────────────────────────────────────────
        if mdm_prompt.strip():
            print(f"\n[rig] ── MDM animation: {mdm_prompt!r} ({mdm_n_frames} frames) ──")
            mdm_out = generate_motion_mdm(mdm_prompt, n_frames=mdm_n_frames,
                                           fps=mdm_fps, device=device)
            if mdm_out is not None:
                pos = mdm_out["positions"]   # (n_frames, 22, 3)
                actual_frames = pos.shape[0]

                # Align MDM joint positions to SMPL scale/space
                # MDM outputs in metres roughly matching SMPL metric
                # Scale so pelvis height matches our SMPL pelvis
                mdm_pelvis_h = float(np.median(pos[:, 0, 1]))
                smpl_pelvis_h = float(smpl_j[0, 1])
                if abs(mdm_pelvis_h) > 1e-4:
                    pos = pos * (smpl_pelvis_h / mdm_pelvis_h)

                # FK inversion: positions → local quaternions for joints 0-21
                t_pose_22 = smpl_j[:22]
                quats_22  = positions_to_local_quats(pos, t_pose_22, _MDM_PARENTS)
                # Pad to 24 joints (SMPL hands = identity)
                quats_24  = np.zeros((actual_frames, 24, 4), np.float32)
                quats_24[:, :, 3] = 1.0
                quats_24[:, :22, :] = quats_22

                # Root translation: MDM root XZ + SMPL Y offset
                root_trans = pos[:, 0, :].copy()   # (n_frames, 3)

                anim_glb = os.path.join(out_dir, "animated.glb")
                export_animated_glb(
                    verts_aligned, faces, uv, tex,
                    smpl_j, skin_w,
                    quats_24, root_trans, mdm_fps, anim_glb
                )
                result["animated_glb"] = anim_glb
                print(f"[rig] MDM animation complete → {anim_glb}")
            else:
                print("[rig] MDM generation failed — static GLB only")

        result["smpl_params"] = {
            "betas": betas.tolist(),
            "p1_sources": len(hmr2_results),
            "p3_triangulated": tri_joints is not None,
        }
        p3_note  = " + triangulated skeleton" if tri_joints is not None else ""
        fbx_note = " + FBX" if result["fbx"] else ""
        anim_note = f" + MDM({mdm_n_frames}f)" if result.get("animated_glb") else ""
        result["status"] = (
            f"Rigged ({len(hmr2_results)} views used{p3_note}{fbx_note}{anim_note}). "
            f"{len(verts)} verts, 24 joints."
        )

    except Exception:
        err = traceback.format_exc()
        print(f"[rig] FAILED:\n{err}")
        result["status"] = f"Rigging failed:\n{err[-600:]}"

    return result
