"""
face_swap_render.py — Paint reference face onto TripoSG UV texture using
MV-Adapter's UV-baking pipeline.

Pipeline:
  1. Load mesh with same params as triposg_app.py render stage
  2. Create orthographic camera matching render_front.png (azimuth=-90)
  3. Detect face landmarks in render_front.png + reference photo via InsightFace
  4. norm_crop reference → canonical 512×512 frontal face
  5. Estimate 4-DOF similarity (canonical → render) and warpAffine
     → produces face_on_render.png: reference face at correct render-space coords
  6. uv_render_attr(images=face_on_render) → projects render image into UV space
     No inverse transform, no scale mismatch — the render-space coordinate system
     is shared between the camera projection and the UV lookup.
  7. Blend projected face into original texture with geometry mask guard.
  8. Save updated GLB

Usage:
    python face_swap_render.py \
        --body  /tmp/triposg_textured.glb \
        --face  /tmp/triposg_face_ref.png \
        --render /tmp/render_front.png \
        --out   /tmp/face_swapped.glb \
        [--blend 0.93] [--uv_size 4096] [--debug_dir /tmp]
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial
from insightface.utils import face_align as insightface_align

sys.path.insert(0, '/root/MV-Adapter')
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper, load_mesh, get_orthogonal_camera,
)
from mvadapter.utils.mesh_utils.uv import (
    uv_precompute, uv_render_geometry, uv_render_attr,
)
from insightface.app import FaceAnalysis


def _detect_largest_face(img_bgr, app):
    faces = app.get(img_bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0]


def _build_front_face_uv_mask(mesh_t, tex_H, tex_W, neck_frac=0.84):
    """
    Build a UV-space mask covering only the front-facing face triangles.
    Excludes back-of-head, hair, and ears (lateral vertices).
    """
    verts  = np.array(mesh_t.vertices, dtype=np.float64)
    faces  = np.array(mesh_t.faces,    dtype=np.int32)
    uvs    = np.array(mesh_t.visual.uv, dtype=np.float64)

    # Head vertices above neck
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    neck_y   = float(y_min + (y_max - y_min) * neck_frac)
    head_idx = np.where(verts[:, 1] > neck_y)[0]
    hv = verts[head_idx]

    # Front half: z >= 40th percentile — generous to include jaw/cheek toward ears
    # No lateral exclusion — it splits UV islands through the eyes/mouth → duplicates
    z_thresh = float(np.percentile(hv[:, 2], 40))
    front    = hv[:, 2] >= z_thresh
    if front.sum() < 30:
        front = np.ones(len(hv), bool)

    face_vert_idx  = head_idx[front]
    face_vert_mask = np.zeros(len(verts), bool)
    face_vert_mask[face_vert_idx] = True

    face_tri_mask = face_vert_mask[faces].all(axis=1)
    face_tris     = faces[face_tri_mask]
    print(f'  Geometry mask: {face_tri_mask.sum()} front-face triangles selected '
          f'(neck_y={neck_y:.3f}, z_thresh={z_thresh:.3f})')

    # Rasterize into UV-space mask (trimesh UV: y=0 is bottom-left → flip V)
    geom_mask = np.zeros((tex_H, tex_W), dtype=np.float32)
    pts_list = []
    for tri in face_tris:
        uv = uvs[tri]  # (3, 2)
        px = uv[:, 0] * tex_W
        py = (1.0 - uv[:, 1]) * tex_H
        pts_list.append(np.column_stack([px, py]).astype(np.int32))
    if pts_list:
        cv2.fillPoly(geom_mask, pts_list, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    geom_mask = cv2.dilate(geom_mask, kernel, iterations=2)   # close intra-tri gaps
    geom_mask = cv2.erode(geom_mask,  kernel, iterations=1)   # retreat from island edges
    geom_mask = cv2.GaussianBlur(geom_mask, (31, 31), 8)      # soft transition
    return geom_mask


def face_swap_render(body_glb, face_img_path, render_img_path, out_glb,
                     blend=0.93, uv_size=4096, neck_frac=0.76, debug_dir=None):

    device = 'cuda'

    # ── Step 1: Load mesh ─────────────────────────────────────────────────────
    print(f'[fsr] Loading mesh: {body_glb}')
    ctx = NVDiffRastContextWrapper(device=device, context_type='cuda')
    mesh_mv = load_mesh(body_glb, rescale=True, device=device)

    scene_t = trimesh.load(body_glb)
    if isinstance(scene_t, trimesh.Scene):
        geom_name = list(scene_t.geometry.keys())[0]
        mesh_t = scene_t.geometry[geom_name]
    else:
        mesh_t = scene_t; geom_name = None

    orig_tex = np.array(mesh_t.visual.material.baseColorTexture, dtype=np.float32) / 255.0
    uvs = np.array(mesh_t.visual.uv, dtype=np.float64)
    tex_H, tex_W = orig_tex.shape[:2]
    print(f'  UV size: {tex_W}×{tex_H}')

    # ── Step 1b: Geometry mask (front-face UV islands only) ───────────────────
    print('[fsr] Building geometry front-face UV mask ...')
    geom_uv_mask = _build_front_face_uv_mask(mesh_t, tex_H, tex_W, neck_frac)

    # ── Step 2: Orthographic camera matching render_front.png ─────────────────
    render_img = cv2.imread(render_img_path)
    H_r, W_r = render_img.shape[:2]
    print(f'  Render size: {W_r}×{H_r}')

    camera = get_orthogonal_camera(
        elevation_deg=[0], distance=[1.8],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[-90], device=device,
    )

    print(f'[fsr] Precomputing UV geometry ({uv_size}×{uv_size}) ...')
    uv_pre  = uv_precompute(ctx, mesh_mv, height=uv_size, width=uv_size)
    uv_geom = uv_render_geometry(
        ctx, mesh_mv, camera,
        view_height=H_r, view_width=W_r,
        uv_precompute_output=uv_pre,
        compute_depth_grad=False,
    )

    # ── Step 3: Face landmark detection ───────────────────────────────────────
    print('[fsr] Detecting face landmarks ...')
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    ref_bgr = cv2.imread(face_img_path)
    render_face = _detect_largest_face(render_img, app)
    if render_face is None:
        raise RuntimeError(f'No face detected in render: {render_img_path}')
    ref_face = _detect_largest_face(ref_bgr, app)
    if ref_face is None:
        raise RuntimeError(f'No face detected in reference: {face_img_path}')

    render_kps = render_face.kps   # (5, 2)
    ref_kps    = ref_face.kps
    print(f'  render kps: x={render_kps[:,0].min():.0f}-{render_kps[:,0].max():.0f}'
          f'  y={render_kps[:,1].min():.0f}-{render_kps[:,1].max():.0f}')

    # ── Step 4: norm_crop → canonical 512×512 frontal face ───────────────────
    CANONICAL_SIZE = 512
    aligned_bgr = insightface_align.norm_crop(ref_bgr, ref_kps, image_size=CANONICAL_SIZE)

    # Fixed ARCFACE 5-point positions scaled to CANONICAL_SIZE
    ARCFACE_112 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
    canonical_kps = ARCFACE_112 * (CANONICAL_SIZE / 112.0)

    # ── Step 5: Forward warp: canonical → render space ────────────────────────
    # 4-DOF similarity (scale + rotation + translation) with all 5 kps.
    # FORWARD direction: canonical_kps → render_kps so that warpAffine places
    # the face at exactly the render-space coordinates, downsampling cleanly.
    fwd_M, inliers = cv2.estimateAffinePartial2D(
        canonical_kps.astype(np.float32),
        render_kps.astype(np.float32),
        method=cv2.LMEDS,
    )
    print(f'  Forward warp M:\n{fwd_M}')

    face_on_render_bgr = cv2.warpAffine(
        aligned_bgr, fwd_M, (W_r, H_r),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    face_on_render_rgb = cv2.cvtColor(face_on_render_bgr,
                                      cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ── Step 6: Render-space face hull mask ───────────────────────────────────
    # Only paint UV texels that correspond to pixels inside the face region.
    hull_pts = cv2.convexHull(render_kps.astype(np.float32)).squeeze(1)
    hull_cx, hull_cy = hull_pts.mean(axis=0)
    hull_expanded = (hull_pts - [hull_cx, hull_cy]) * 4.0 + [hull_cx, hull_cy]
    face_mask_render = np.zeros((H_r, W_r), dtype=np.float32)
    cv2.fillPoly(face_mask_render, [hull_expanded.astype(np.int32)], 1.0)
    # Restrict to where the warped face actually has content
    face_content = (face_on_render_bgr.mean(axis=2) > 3.0 / 255.0).astype(np.float32)
    face_mask_render = face_mask_render * face_content
    face_mask_render = cv2.GaussianBlur(face_mask_render, (51, 51), 15)

    # ── Step 7: Project face-on-render into UV space ──────────────────────────
    # uv_render_attr uses uv_pos_ndc as a lookup: for each UV texel, sample the
    # render-space image at that texel's render NDC position.
    # Since face_on_render is already in render-space coords, this is exact.
    print('[fsr] Projecting face into UV space via uv_render_attr ...')
    face_t = torch.tensor(face_on_render_rgb, device=device).unsqueeze(0)  # (1,H,W,3)
    mask_t = torch.tensor(face_mask_render[None], device=device)

    uv_attr_out = uv_render_attr(
        images=face_t,
        masks=mask_t,
        uv_render_geometry_output=uv_geom,
    )
    uv_face_img  = uv_attr_out.uv_attr_proj[0].cpu().numpy()   # (uv, uv, 3)
    uv_face_mask = uv_attr_out.uv_mask_proj[0].cpu().numpy()   # (uv, uv)

    # Rescale to tex resolution if needed
    if uv_size != tex_H or uv_size != tex_W:
        uv_face_img_rs  = cv2.resize(uv_face_img,  (tex_W, tex_H), interpolation=cv2.INTER_LINEAR)
        uv_face_mask_rs = cv2.resize(uv_face_mask, (tex_W, tex_H), interpolation=cv2.INTER_LINEAR)
    else:
        uv_face_img_rs  = uv_face_img
        uv_face_mask_rs = uv_face_mask

    # ── Step 7b: Apply geometry mask — kill back-of-head / ear UV islands ────
    uv_face_mask_rs = uv_face_mask_rs * geom_uv_mask

    # Final blend alpha — use full blend=1.0 inside the face region so no
    # original texture leaks through and creates duplicate features
    alpha = np.clip(uv_face_mask_rs, 0, 1)[..., None]
    painted_px = int((alpha[..., 0] > 0.01).sum())
    print(f'  Painted texels: {painted_px}')

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'fsr_aligned_ref.png'), aligned_bgr)
        cv2.imwrite(os.path.join(debug_dir, 'fsr_face_on_render.png'), face_on_render_bgr)
        cv2.imwrite(os.path.join(debug_dir, 'fsr_face_mask_render.png'),
                    (face_mask_render * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, 'fsr_geom_mask.png'),
                    (geom_uv_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, 'fsr_uv_mask.png'),
                    (uv_face_mask_rs * 255).astype(np.uint8))
        Image.fromarray((uv_face_img_rs * 255).clip(0, 255).astype(np.uint8)).save(
            os.path.join(debug_dir, 'fsr_uv_face.png'))
        print(f'  Debug files saved to {debug_dir}')

    # ── Step 8: Blend into original texture ───────────────────────────────────
    print(f'[fsr] Blending (blend={blend}) ...')
    new_tex = uv_face_img_rs * alpha + orig_tex * (1.0 - alpha)

    # ── Step 9: Save GLB ──────────────────────────────────────────────────────
    new_pil = Image.fromarray((new_tex * 255).clip(0, 255).astype(np.uint8))
    mesh_t.visual = TextureVisuals(uv=uvs, material=PBRMaterial(baseColorTexture=new_pil))

    if geom_name and isinstance(scene_t, trimesh.Scene):
        scene_t.geometry[geom_name] = mesh_t
        scene_t.export(out_glb)
    else:
        mesh_t.export(out_glb)

    print(f'[fsr] Saved: {out_glb}  ({os.path.getsize(out_glb)//1024} KB)')
    return out_glb


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--body',      required=True)
    ap.add_argument('--face',      required=True)
    ap.add_argument('--render',    required=True, help='Front render (e.g. render_front.png)')
    ap.add_argument('--out',       required=True)
    ap.add_argument('--blend',     type=float, default=0.93)
    ap.add_argument('--uv_size',   type=int,   default=4096)
    ap.add_argument('--neck_frac', type=float, default=0.76)
    ap.add_argument('--debug_dir', default=None)
    args = ap.parse_args()
    face_swap_render(args.body, args.face, args.render, args.out,
                     blend=args.blend, uv_size=args.uv_size,
                     neck_frac=args.neck_frac, debug_dir=args.debug_dir)
