"""
face_project.py — Project reference face image onto TripoSG mesh UV texture.

Keeps geometry 100% intact. Paints the face-region UV triangles using
barycentric rasterization — never interpolates across UV island boundaries.

Usage:
    python face_project.py --body /tmp/triposg_textured.glb \
                           --face /tmp/triposg_face_ref.png \
                           --out  /tmp/face_projected.glb \
                           [--blend 0.9] [--neck_frac 0.84] [--debug_tex /tmp/tex.png]
"""

import os, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
import trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial


# ── Face alignment ─────────────────────────────────────────────────────────────

def _aligned_face_bgr(face_img_bgr, target_size=512):
    """Detect + align face via InsightFace 5-pt warp; falls back to square crop."""
    try:
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(face_img_bgr)
        if faces:
            faces.sort(
                key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
                reverse=True)
            aligned = face_align.norm_crop(face_img_bgr, faces[0].kps,
                                           image_size=target_size)
            print(f'  InsightFace aligned: {aligned.shape}')
            return aligned
    except Exception as e:
        print(f'  InsightFace unavailable ({e}), using centre-crop')
    h, w = face_img_bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(face_img_bgr[y0:y0+side, x0:x0+side], (target_size, target_size))


# ── Triangle rasterizer ────────────────────────────────────────────────────────

def _rasterize_triangles(face_tri_uvs_px, face_tri_img_xy,
                          face_img_rgb, tex, blend,
                          max_uv_span=300):
    """
    Paint face_img_rgb colour into tex at UV locations, triangle by triangle.

    face_tri_uvs_px  : (M, 3, 2)  UV pixel coords of M triangles
    face_tri_img_xy  : (M, 3, 2)  projected image coords of M triangles
    face_img_rgb     : (H, W, 3)  reference face image
    tex              : (texH, texW, 3) float32 texture (modified in-place)
    blend            : float 0–1
    max_uv_span      : skip triangles whose UV bounding box exceeds this (UV seams)
    """
    H_f, W_f   = face_img_rgb.shape[:2]
    tex_H, tex_W = tex.shape[:2]
    painted = 0

    for fi in range(len(face_tri_uvs_px)):
        uv  = face_tri_uvs_px[fi]    # (3, 2) in texture pixel coords
        img = face_tri_img_xy[fi]    # (3, 2) in face-image pixel coords

        # Skip UV-seam triangles (vertices far apart in UV space)
        if (uv[:, 0].max() - uv[:, 0].min() > max_uv_span or
                uv[:, 1].max() - uv[:, 1].min() > max_uv_span):
            continue

        # Bounding box in texture space
        u_lo = max(0,      int(uv[:, 0].min()))
        u_hi = min(tex_W,  int(uv[:, 0].max()) + 2)
        v_lo = max(0,      int(uv[:, 1].min()))
        v_hi = min(tex_H,  int(uv[:, 1].max()) + 2)
        if u_hi <= u_lo or v_hi <= v_lo:
            continue

        # Grid of texel centres in this bounding box
        gu, gv = np.meshgrid(np.arange(u_lo, u_hi), np.arange(v_lo, v_hi))
        pts = np.column_stack([gu.ravel().astype(np.float32),
                               gv.ravel().astype(np.float32)])  # (K, 2)

        # Barycentric coordinates (in UV pixel space)
        A   = uv[0].astype(np.float64)
        AB  = (uv[1] - uv[0]).astype(np.float64)
        AC  = (uv[2] - uv[0]).astype(np.float64)
        denom = AB[0] * AC[1] - AB[1] * AC[0]
        if abs(denom) < 0.5:
            continue
        P  = pts.astype(np.float64) - A
        b1 = (P[:, 0] * AC[1] - P[:, 1] * AC[0]) / denom
        b2 = (P[:, 1] * AB[0] - P[:, 0] * AB[1]) / denom
        b0 = 1.0 - b1 - b2

        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not inside.any():
            continue

        # Interpolate reference-face image coordinates
        ix_f = (b0[inside] * img[0, 0] +
                b1[inside] * img[1, 0] +
                b2[inside] * img[2, 0])
        iy_f = (b0[inside] * img[0, 1] +
                b1[inside] * img[1, 1] +
                b2[inside] * img[2, 1])

        valid = ((ix_f >= 0) & (ix_f < W_f) & (iy_f >= 0) & (iy_f < H_f))
        if not valid.any():
            continue

        ix = np.clip(ix_f[valid].astype(int), 0, W_f - 1)
        iy = np.clip(iy_f[valid].astype(int), 0, H_f - 1)
        colours = face_img_rgb[iy, ix].astype(np.float32)   # (P, 3)

        tu = pts[inside][valid, 0].astype(int)
        tv = pts[inside][valid, 1].astype(int)
        in_tex = (tu >= 0) & (tu < tex_W) & (tv >= 0) & (tv < tex_H)

        tex[tv[in_tex], tu[in_tex]] = (
            blend * colours[in_tex] +
            (1.0 - blend) * tex[tv[in_tex], tu[in_tex]]
        )
        painted += int(in_tex.sum())

    return painted


# ── Main ───────────────────────────────────────────────────────────────────────

def project_face(body_glb, face_img_path, out_glb,
                 blend=0.90, neck_frac=0.84, debug_tex=None):
    """
    Project reference face onto TripoSG UV texture via per-triangle rasterization.
    """

    # ── Load mesh ─────────────────────────────────────────────────────────────
    print(f'[face_project] Loading {body_glb}')
    scene = trimesh.load(body_glb)
    if isinstance(scene, trimesh.Scene):
        geom_name = list(scene.geometry.keys())[0]
        mesh = scene.geometry[geom_name]
    else:
        mesh = scene
        geom_name = None

    verts    = np.array(mesh.vertices,  dtype=np.float64)   # (N, 3)
    faces    = np.array(mesh.faces,     dtype=np.int32)     # (F, 3)
    uvs      = np.array(mesh.visual.uv, dtype=np.float64)   # (N, 2)
    mat      = mesh.visual.material
    orig_tex = np.array(mat.baseColorTexture, dtype=np.float32)  # (H, W, 3) RGB
    tex_H, tex_W = orig_tex.shape[:2]
    print(f'  {len(verts)} verts | {len(faces)} faces | texture {orig_tex.shape}')

    # ── Identify face-region vertices ─────────────────────────────────────────
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    neck_y = float(y_min + (y_max - y_min) * neck_frac)

    head_mask = verts[:, 1] > neck_y
    head_idx  = np.where(head_mask)[0]
    hv = verts[head_idx]

    # Front half only (z >= median — face faces +Z)
    z_med  = float(np.median(hv[:, 2]))
    front  = hv[:, 2] >= z_med
    if front.sum() < 30:
        front = np.ones(len(hv), bool)

    face_vert_idx = head_idx[front]         # indices into the full vertex array

    # Build boolean mask for fast triangle selection
    face_vert_mask = np.zeros(len(verts), bool)
    face_vert_mask[face_vert_idx] = True

    # Select triangles where ALL 3 vertices are in the face region
    face_tri_mask = face_vert_mask[faces].all(axis=1)
    face_tris     = faces[face_tri_mask]    # (M, 3)
    print(f'  neck_y={neck_y:.4f} | head={len(head_idx)} '
          f'| face-front={front.sum()} | face triangles={len(face_tris)}')

    # ── Load and align reference face ─────────────────────────────────────────
    print(f'[face_project] Reference face: {face_img_path}')
    raw_bgr     = cv2.imread(face_img_path)
    aligned_bgr = _aligned_face_bgr(raw_bgr, target_size=512)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    H_f, W_f    = aligned_rgb.shape[:2]

    # ── Compute face projection axes from actual face normal ─────────────────
    fv  = verts[face_vert_idx]

    # Average normal of the front-facing face triangles defines projection dir
    face_tri_normals = np.array(mesh.face_normals)[face_tri_mask]
    face_fwd = face_tri_normals.mean(axis=0)
    face_fwd /= np.linalg.norm(face_fwd)

    # Build orthonormal right/up axes in the face plane
    world_up   = np.array([0., 1., 0.])
    face_right = np.cross(face_fwd, world_up)
    face_right /= np.linalg.norm(face_right)
    face_up    = np.cross(face_right, face_fwd)
    face_up    /= np.linalg.norm(face_up)
    print(f'  Face normal: {face_fwd.round(3)}')

    # Project face vertices onto local (right, up) plane
    fv_centroid = fv.mean(axis=0)
    fv_c  = fv - fv_centroid
    lx    = fv_c @ face_right
    ly    = fv_c @ face_up
    x_span = float(lx.max() - lx.min())
    y_span = float(ly.max() - ly.min())

    # InsightFace norm_crop places eyes at ~37% from top of the 512px image.
    # In 3D the eyes are ~78% up from neck → 28% above centroid.
    # Shift the vertical origin up by 0.112*y_span so eye level → 37% in image.
    cy_shift = 0.112 * y_span
    pad = 0.10   # tighter crop so face features fill more of the image

    def vert_to_img(v):
        """Project 3D vertex to reference-face image using the face normal."""
        c  = v - fv_centroid                          # (N, 3)
        lx = c @ face_right
        ly = c @ face_up
        pu = lx / (x_span * (1 + 2*pad)) + 0.5
        pv = -(ly - cy_shift) / (y_span * (1 + 2*pad)) + 0.5
        return np.column_stack([pu * W_f, pv * H_f])  # (N, 2)

    def vert_to_uv_px(v_idx):
        """Convert vertex UV coords to texture pixel coordinates."""
        uv = uvs[v_idx]
        # trimesh loads GLB UV with (0,0)=bottom-left; flip V for image row
        col = uv[:, 0] * tex_W
        row = (1.0 - uv[:, 1]) * tex_H
        return np.column_stack([col, row])   # (N, 2)

    # Pre-compute image + UV pixel coords for every vertex
    all_img_px = vert_to_img(verts)           # (N, 2)
    all_uv_px  = vert_to_uv_px(np.arange(len(verts)))  # (N, 2)

    # Gather per-triangle arrays
    face_tri_uvs_px  = all_uv_px[face_tris]   # (M, 3, 2)
    face_tri_img_xy  = all_img_px[face_tris]  # (M, 3, 2)

    print(f'  UV pixel range: u={face_tri_uvs_px[:,:,0].min():.0f}→'
          f'{face_tri_uvs_px[:,:,0].max():.0f} '
          f'v={face_tri_uvs_px[:,:,1].min():.0f}→'
          f'{face_tri_uvs_px[:,:,1].max():.0f}')
    print(f'  Image coord range: x={face_tri_img_xy[:,:,0].min():.1f}→'
          f'{face_tri_img_xy[:,:,0].max():.1f} '
          f'y={face_tri_img_xy[:,:,1].min():.1f}→'
          f'{face_tri_img_xy[:,:,1].max():.1f}')

    # ── Rasterize face triangles into UV texture ──────────────────────────────
    print(f'[face_project] Rasterizing {len(face_tris)} triangles into texture...')
    new_tex = orig_tex.copy()
    painted = _rasterize_triangles(
        face_tri_uvs_px, face_tri_img_xy,
        aligned_rgb, new_tex, blend,
        max_uv_span=300
    )
    print(f'  Painted {painted} texels across {len(face_tris)} triangles')

    # ── Save debug texture if requested ──────────────────────────────────────
    if debug_tex:
        dbg = np.clip(new_tex, 0, 255).astype(np.uint8)
        Image.fromarray(dbg).save(debug_tex)
        print(f'  Debug texture: {debug_tex}')

    # ── Write modified texture back to mesh ───────────────────────────────────
    new_pil     = Image.fromarray(np.clip(new_tex, 0, 255).astype(np.uint8))
    new_mat     = PBRMaterial(baseColorTexture=new_pil)
    mesh.visual = TextureVisuals(uv=uvs, material=new_mat)

    os.makedirs(os.path.dirname(os.path.abspath(out_glb)), exist_ok=True)
    if geom_name and isinstance(scene, trimesh.Scene):
        scene.geometry[geom_name] = mesh
        scene.export(out_glb)
    else:
        mesh.export(out_glb)

    print(f'[face_project] Saved: {out_glb}  ({os.path.getsize(out_glb)//1024} KB)')
    return out_glb


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--body',      required=True)
    ap.add_argument('--face',      required=True)
    ap.add_argument('--out',       required=True)
    ap.add_argument('--blend',     type=float, default=0.90)
    ap.add_argument('--neck_frac', type=float, default=0.84)
    ap.add_argument('--debug_tex', default=None)
    args = ap.parse_args()
    project_face(args.body, args.face, args.out,
                 blend=args.blend, neck_frac=args.neck_frac,
                 debug_tex=args.debug_tex)
