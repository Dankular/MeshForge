"""
face_inswap_bake.py — Proper face swap on rendered views, then UV-bake.

Pipeline:
  1. Render the mesh from multiple views (front + L/R 3-quarter)
  2. Run inswapper_128 to swap reference face onto each rendered view
  3. uv_render_attr() bakes each swapped render directly into UV texture
     (render-space coords shared with UV lookup — no coordinate transforms)
  4. Composite multiple views (front takes priority, sides fill gaps)
  5. Save updated GLB

Usage:
    python face_inswap_bake.py \
        --body   /tmp/triposg_textured.glb \
        --face   /tmp/triposg_face_ref.png \
        --out    /tmp/face_swapped.glb \
        [--uv_size 4096] [--debug_dir /tmp]
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

sys.path.insert(0, '/root/MV-Adapter')
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper, load_mesh, get_orthogonal_camera, render,
)
from mvadapter.utils.mesh_utils.uv import (
    uv_precompute, uv_render_geometry, uv_render_attr,
)
from insightface.app import FaceAnalysis
import insightface
from gfpgan import GFPGANer


GFPGAN_PATH = '/root/MV-Adapter/checkpoints/GFPGANv1.4.pth'


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_front_face_uv_mask(mesh_t, tex_H, tex_W, neck_frac=0.76):
    """UV-space mask covering only front-facing head triangles (no back-of-head)."""
    verts = np.array(mesh_t.vertices, dtype=np.float64)
    faces = np.array(mesh_t.faces,    dtype=np.int32)
    uvs   = np.array(mesh_t.visual.uv, dtype=np.float64)

    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    neck_y   = float(y_min + (y_max - y_min) * neck_frac)
    head_idx = np.where(verts[:, 1] > neck_y)[0]
    hv = verts[head_idx]

    z_thresh = float(np.percentile(hv[:, 2], 40))
    front    = hv[:, 2] >= z_thresh
    if front.sum() < 30:
        front = np.ones(len(hv), bool)

    face_vert_idx  = head_idx[front]
    face_vert_mask = np.zeros(len(verts), bool)
    face_vert_mask[face_vert_idx] = True
    face_tri_mask  = face_vert_mask[faces].all(axis=1)
    face_tris      = faces[face_tri_mask]
    print(f'  Geometry mask: {face_tri_mask.sum()} front-face triangles '
          f'(neck_y={neck_y:.3f}, z_thresh={z_thresh:.3f})')

    geom_mask = np.zeros((tex_H, tex_W), dtype=np.float32)
    pts_list = []
    for tri in face_tris:
        uv = uvs[tri]
        px = uv[:, 0] * tex_W
        py = (1.0 - uv[:, 1]) * tex_H
        pts_list.append(np.column_stack([px, py]).astype(np.int32))
    if pts_list:
        cv2.fillPoly(geom_mask, pts_list, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    geom_mask = cv2.dilate(geom_mask, kernel, iterations=2)
    geom_mask = cv2.erode(geom_mask,  kernel, iterations=1)
    geom_mask = cv2.GaussianBlur(geom_mask, (31, 31), 8)
    return geom_mask


def _detect_largest_face(img_bgr, app):
    faces = app.get(img_bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))


def _render_view(ctx, mesh_mv, uv_pre, azimuth_deg, H, W, device):
    """Render the mesh from a given azimuth; return (camera, uv_geom)."""
    camera = get_orthogonal_camera(
        elevation_deg=[0], distance=[1.8],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[azimuth_deg], device=device,
    )
    uv_geom = uv_render_geometry(
        ctx, mesh_mv, camera,
        view_height=H, view_width=W,
        uv_precompute_output=uv_pre,
        compute_depth_grad=False,
    )
    return camera, uv_geom


def face_inswap_bake(body_glb, face_img_path, out_glb,
                     uv_size=4096, debug_dir=None):

    device = 'cuda'
    INSWAPPER_PATH = '/root/MV-Adapter/checkpoints/inswapper_128.onnx'

    # ── Load GFPGAN enhancer ──────────────────────────────────────────────────
    print('[fib] Loading GFPGANv1.4 ...')
    enhancer = GFPGANer(
        model_path=GFPGAN_PATH,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
    )

    # ── Load mesh ─────────────────────────────────────────────────────────────
    print(f'[fib] Loading mesh: {body_glb}')
    ctx = NVDiffRastContextWrapper(device=device, context_type='cuda')
    mesh_mv = load_mesh(body_glb, rescale=True, device=device)

    scene_t = trimesh.load(body_glb)
    if isinstance(scene_t, trimesh.Scene):
        geom_name = list(scene_t.geometry.keys())[0]
        mesh_t = scene_t.geometry[geom_name]
    else:
        mesh_t = scene_t; geom_name = None

    orig_tex_np = np.array(mesh_t.visual.material.baseColorTexture, dtype=np.float32) / 255.0
    uvs = np.array(mesh_t.visual.uv, dtype=np.float64)
    tex_H, tex_W = orig_tex_np.shape[:2]
    print(f'  Texture: {tex_W}×{tex_H}')

    # Build geometry mask (front-face head triangles only) at UV resolution
    print('[fib] Building front-face geometry UV mask ...')
    geom_uv_mask = _build_front_face_uv_mask(mesh_t, uv_size, uv_size)

    # Render dimensions (match triposg_app.py)
    H_r, W_r = 1024, 768

    # ── Precompute UV geometry ─────────────────────────────────────────────────
    print(f'[fib] Precomputing UV geometry ({uv_size}×{uv_size}) ...')
    uv_pre = uv_precompute(ctx, mesh_mv, height=uv_size, width=uv_size)

    # ── Load face swap model + face detector ──────────────────────────────────
    print('[fib] Loading inswapper_128 ...')
    swapper = insightface.model_zoo.get_model(
        INSWAPPER_PATH, download=False,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    ref_bgr = cv2.imread(face_img_path)
    ref_face = _detect_largest_face(ref_bgr, app)
    if ref_face is None:
        raise RuntimeError(f'No face detected in reference: {face_img_path}')
    print(f'  Reference face detected: bbox={ref_face.bbox.astype(int).tolist()}')

    # ── Process each view ─────────────────────────────────────────────────────
    # Views: front (azimuth=-90), slight left (-60), slight right (-120)
    # Azimuth convention from MV-Adapter: -90 = front-facing
    views = [
        ('front',       -90,   1.0),   # (name, azimuth_deg, priority_weight)
        ('threequarter_r', -60,  0.7),
        ('threequarter_l', -120, 0.7),
    ]

    # Accumulators for weighted UV compositing
    uv_colour_acc = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
    uv_weight_acc = np.zeros((uv_size, uv_size),    dtype=np.float32)

    for view_name, azimuth, weight in views:
        print(f'\n[fib] View: {view_name} (azimuth={azimuth}°)')

        # Create camera + UV geometry for this view
        camera, uv_geom = _render_view(ctx, mesh_mv, uv_pre, azimuth, H_r, W_r, device)

        # Render textured mesh from this view
        render_out = render(ctx, mesh_mv, camera, height=H_r, width=W_r,
                            render_attr=True, render_depth=False, render_normal=False,
                            attr_background=0.0)
        # render_out.attr: (1, H, W, 3) float in [0,1]
        rendered_np = (render_out.attr[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f'fib_render_{view_name}.png'), rendered_bgr)

        # Detect face in this rendered view
        tgt_face = _detect_largest_face(rendered_bgr, app)
        if tgt_face is None:
            print(f'  No face in {view_name} render — skipping')
            continue
        print(f'  Target face: bbox={tgt_face.bbox.astype(int).tolist()}')

        # Swap face
        swapped_bgr = swapper.get(rendered_bgr.copy(), tgt_face, ref_face, paste_back=True)

        # Enhance face detail with GFPGAN
        _, _, enhanced_bgr = enhancer.enhance(
            swapped_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        if enhanced_bgr is not None:
            swapped_bgr = enhanced_bgr
            print(f'  GFPGAN enhanced')

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f'fib_swapped_{view_name}.png'), swapped_bgr)

        swapped_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Build render-space face hull mask
        kps = tgt_face.kps
        hull_pts = cv2.convexHull(kps.astype(np.float32)).squeeze(1)
        hull_cx, hull_cy = hull_pts.mean(axis=0)
        hull_exp = (hull_pts - [hull_cx, hull_cy]) * 3.5 + [hull_cx, hull_cy]
        face_mask = np.zeros((H_r, W_r), dtype=np.float32)
        cv2.fillPoly(face_mask, [hull_exp.astype(np.int32)], 1.0)
        face_mask = cv2.GaussianBlur(face_mask, (61, 61), 20)

        # Bake swapped render into UV space
        swapped_t = torch.tensor(swapped_rgb, device=device).unsqueeze(0)  # (1,H,W,3)
        mask_t    = torch.tensor(face_mask[None], device=device)

        uv_out = uv_render_attr(
            images=swapped_t,
            masks=mask_t,
            uv_render_geometry_output=uv_geom,
        )
        uv_img  = uv_out.uv_attr_proj[0].cpu().numpy()   # (uv, uv, 3)
        uv_mask = uv_out.uv_mask_proj[0].cpu().numpy()   # (uv, uv)

        # Kill back-of-head UV islands
        uv_mask = uv_mask * geom_uv_mask

        # Weighted accumulate
        w = uv_mask * weight
        uv_colour_acc += uv_img * w[..., None]
        uv_weight_acc += w
        print(f'  Painted texels: {(uv_mask > 0.05).sum()}')

    # ── Composite ──────────────────────────────────────────────────────────────
    print('\n[fib] Compositing views ...')
    valid = uv_weight_acc > 0.01
    uv_final = np.where(valid[..., None],
                        uv_colour_acc / np.maximum(uv_weight_acc[..., None], 1e-6),
                        orig_tex_np[:uv_size, :uv_size] if uv_size <= tex_H else orig_tex_np)

    # Resize to texture resolution if needed
    if uv_size != tex_H or uv_size != tex_W:
        uv_final_rs = cv2.resize(uv_final,         (tex_W, tex_H), interpolation=cv2.INTER_LINEAR)
        weight_rs   = cv2.resize(uv_weight_acc,    (tex_W, tex_H), interpolation=cv2.INTER_LINEAR)
    else:
        uv_final_rs = uv_final
        weight_rs   = uv_weight_acc

    # Blend with original texture: use face-swap result where painted, orig elsewhere
    alpha = np.clip(weight_rs, 0, 1)[..., None]
    new_tex = uv_final_rs * alpha + orig_tex_np * (1.0 - alpha)
    print(f'  Total painted texels (tex res): {(weight_rs > 0.05).sum()}')

    if debug_dir:
        Image.fromarray((uv_final_rs * 255).clip(0,255).astype(np.uint8)).save(
            os.path.join(debug_dir, 'fib_uv_composite.png'))

    # ── Save GLB ──────────────────────────────────────────────────────────────
    new_pil = Image.fromarray((new_tex * 255).clip(0, 255).astype(np.uint8))
    mesh_t.visual = TextureVisuals(uv=uvs, material=PBRMaterial(baseColorTexture=new_pil))

    if geom_name and isinstance(scene_t, trimesh.Scene):
        scene_t.geometry[geom_name] = mesh_t
        scene_t.export(out_glb)
    else:
        mesh_t.export(out_glb)

    print(f'[fib] Saved: {out_glb}  ({os.path.getsize(out_glb)//1024} KB)')
    return out_glb


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--body',      required=True)
    ap.add_argument('--face',      required=True)
    ap.add_argument('--out',       required=True)
    ap.add_argument('--uv_size',   type=int, default=4096)
    ap.add_argument('--debug_dir', default=None)
    args = ap.parse_args()
    face_inswap_bake(args.body, args.face, args.out,
                     uv_size=args.uv_size, debug_dir=args.debug_dir)
