"""
rig_yolo.py — Rig a humanoid mesh using YOLO-pose joint detection.

Instead of estimating T-pose rotations (which failed), detect where joints
actually ARE in the mesh's current pose and use those positions as the bind pose.

Pipeline:
  1. Render front view (azimuth=-90, same camera as triposg_app.py views)
  2. YOLOv8x-pose → COCO-17 2D keypoints
  3. Unproject to 3D in original mesh coordinate space
  4. Map COCO-17 → SMPL-24 (interpolate spine, collar, hand, foot joints)
  5. LBS weights: proximity-based (k=4 nearest joints per vertex)
  6. Export rigged GLB — bind pose = current pose

Usage:
    python rig_yolo.py --body /tmp/triposg_textured.glb \
                       --out  /tmp/rig_out/rigged.glb \
                       [--debug_dir /tmp/rig_debug]
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, '/root/MV-Adapter')

# ── Camera constants — MUST match triposg_app.py ──────────────────────────────
ORTHO_LEFT, ORTHO_RIGHT = -0.55, 0.55
ORTHO_BOT,  ORTHO_TOP   = -0.55, 0.55
RENDER_W, RENDER_H      = 768, 1024
FRONT_AZ                = -90       # azimuth that gives front view
# Orthographic proj scale: 2/(right-left) = 1.818...
PROJ_SCALE = 2.0 / (ORTHO_RIGHT - ORTHO_LEFT)

SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,
                12,13,14,16,17,18,19,20,21]
SMPL_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1',
    'left_knee','right_knee','spine2',
    'left_ankle','right_ankle','spine3',
    'left_foot','right_foot','neck',
    'left_collar','right_collar','head',
    'left_shoulder','right_shoulder',
    'left_elbow','right_elbow',
    'left_wrist','right_wrist',
    'left_hand','right_hand',
]

# COCO-17 order
COCO_NAMES = ['nose','L_eye','R_eye','L_ear','R_ear',
              'L_shoulder','R_shoulder','L_elbow','R_elbow','L_wrist','R_wrist',
              'L_hip','R_hip','L_knee','R_knee','L_ankle','R_ankle']


# ── Step 0: Load mesh directly from GLB (correct UV channel) ─────────────────

def load_mesh_from_gltf(body_glb):
    """
    Load mesh from GLB using pygltflib, reading the UV channel the material
    actually references (TEXCOORD_0 or TEXCOORD_1).
    Returns: verts (N,3) float64, faces (F,3) int32,
             uv (N,2) float32 or None, texture_pil PIL.Image or None
    """
    import pygltflib
    from PIL import Image as PILImage
    import io

    gltf = pygltflib.GLTF2().load(body_glb)
    blob = gltf.binary_blob()

    # componentType → (numpy dtype, bytes per element)
    _DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
              5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    _NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}

    def read_accessor(idx):
        if idx is None:
            return None
        acc = gltf.accessors[idx]
        bv  = gltf.bufferViews[acc.bufferView]
        dtype  = _DTYPE[acc.componentType]
        n_comp = _NCOMP[acc.type]
        bv_off  = bv.byteOffset  or 0
        acc_off = acc.byteOffset or 0
        elem_bytes = np.dtype(dtype).itemsize * n_comp
        stride = bv.byteStride if (bv.byteStride and bv.byteStride != elem_bytes) else elem_bytes

        if stride == elem_bytes:
            start = bv_off + acc_off
            size  = acc.count * elem_bytes
            arr   = np.frombuffer(blob[start:start + size], dtype=dtype)
        else:
            # interleaved buffer
            rows = []
            for i in range(acc.count):
                start = bv_off + acc_off + i * stride
                rows.append(np.frombuffer(blob[start:start + elem_bytes], dtype=dtype))
            arr = np.concatenate(rows)

        return arr.reshape(acc.count, n_comp) if n_comp > 1 else arr

    # ── Find which texCoord index the material references ──────────────────────
    texcoord_idx = 0
    if gltf.materials:
        pbr = gltf.materials[0].pbrMetallicRoughness
        if pbr and pbr.baseColorTexture:
            texcoord_idx = getattr(pbr.baseColorTexture, 'texCoord', 0) or 0
    print(f'  material uses TEXCOORD_{texcoord_idx}')

    # ── Read primitive ─────────────────────────────────────────────────────────
    prim  = gltf.meshes[0].primitives[0]
    attrs = prim.attributes

    verts = read_accessor(attrs.POSITION).astype(np.float64)

    idx_data = read_accessor(prim.indices).flatten()
    faces = idx_data.reshape(-1, 3).astype(np.int32)

    # Read the correct UV channel; fall back to TEXCOORD_0
    uv_acc_idx = getattr(attrs, f'TEXCOORD_{texcoord_idx}', None)
    if uv_acc_idx is None and texcoord_idx != 0:
        uv_acc_idx = getattr(attrs, 'TEXCOORD_0', None)
    uv_raw = read_accessor(uv_acc_idx)
    uv = uv_raw.astype(np.float32) if uv_raw is not None else None

    print(f'  verts={len(verts)}  faces={len(faces)}  uv={len(uv) if uv is not None else None}')

    # ── Extract embedded texture ───────────────────────────────────────────────
    texture_pil = None
    try:
        pbr = gltf.materials[0].pbrMetallicRoughness
        if pbr and pbr.baseColorTexture is not None:
            tex_idx = pbr.baseColorTexture.index
            if tex_idx is not None and tex_idx < len(gltf.textures):
                src_idx = gltf.textures[tex_idx].source
                if src_idx is not None and src_idx < len(gltf.images):
                    img_obj = gltf.images[src_idx]
                    if img_obj.bufferView is not None:
                        bv = gltf.bufferViews[img_obj.bufferView]
                        bv_off = bv.byteOffset or 0
                        img_bytes = blob[bv_off:bv_off + bv.byteLength]
                        texture_pil = PILImage.open(io.BytesIO(img_bytes)).convert('RGBA')
                        print(f'  texture: {texture_pil.size}')
    except Exception as e:
        print(f'  texture extraction failed: {e}')

    return verts, faces, uv, texture_pil


# ── Step 1: Render front view ─────────────────────────────────────────────────

def render_front(body_glb, debug_dir=None):
    """
    Render front view using MV-Adapter.
    Returns (img_bgr, scale_factor) where scale_factor = max_abs / 0.5
    (used to convert std-space back to original mesh space).
    """
    from mvadapter.utils.mesh_utils import (
        NVDiffRastContextWrapper, load_mesh, get_orthogonal_camera, render,
    )
    ctx = NVDiffRastContextWrapper(device='cuda', context_type='cuda')
    mesh_mv, _offset, scale_factor = load_mesh(
        body_glb, rescale=True, return_transform=True, device='cuda')
    camera = get_orthogonal_camera(
        elevation_deg=[0], distance=[1.8],
        left=ORTHO_LEFT, right=ORTHO_RIGHT,
        bottom=ORTHO_BOT, top=ORTHO_TOP,
        azimuth_deg=[FRONT_AZ], device='cuda')
    out = render(ctx, mesh_mv, camera,
                 height=RENDER_H, width=RENDER_W,
                 render_attr=True, render_depth=False, render_normal=False,
                 attr_background=0.5)
    img_np  = (out.attr[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'front_render.png'), img_bgr)
    print(f'  render: {RENDER_W}x{RENDER_H}, scale_factor={scale_factor:.4f}')
    return img_bgr, scale_factor


# ── Step 2: YOLO-pose keypoints ───────────────────────────────────────────────

def detect_keypoints(img_bgr, debug_dir=None):
    """
    Run YOLOv8x-pose on the rendered image.
    Returns (17, 3) array: [pixel_x, pixel_y, confidence] for COCO-17 joints.
    Picks the largest detected bounding box (the character body).
    """
    from ultralytics import YOLO
    model = YOLO('yolov8x-pose.pt')
    results = model(img_bgr, verbose=False)

    if not results or results[0].keypoints is None or len(results[0].boxes) == 0:
        raise RuntimeError('YOLO: no person detected in front render')

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    idx   = int(areas.argmax())

    kp_xy   = r.keypoints[idx].xy[0].cpu().numpy()    # (17, 2) pixel
    kp_conf = r.keypoints[idx].conf[0].cpu().numpy()  # (17,) confidence
    kp      = np.concatenate([kp_xy, kp_conf[:,None]], axis=1)  # (17, 3)

    print('  YOLO detections: %d boxes, using largest' % len(boxes))
    for i, name in enumerate(COCO_NAMES):
        if kp_conf[i] > 0.3:
            print('    [%d] %-14s  px=(%.0f, %.0f)  conf=%.2f' % (
                i, name, kp_xy[i,0], kp_xy[i,1], kp_conf[i]))

    if debug_dir:
        vis = img_bgr.copy()
        for i in range(17):
            if kp_conf[i] > 0.3:
                x, y = int(kp_xy[i,0]), int(kp_xy[i,1])
                cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(vis, COCO_NAMES[i][:4], (x+4, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
        cv2.imwrite(os.path.join(debug_dir, 'yolo_keypoints.png'), vis)

    return kp


# ── Step 3: Unproject 2D → 3D ────────────────────────────────────────────────

def unproject_to_3d(kp_2d_conf, scale_factor, mesh_verts_orig):
    """
    Convert COCO-17 pixel positions to 3D positions in original mesh space.

    MV-Adapter orthographic camera at azimuth=-90 maps:
      pixel_x  →  orig_x   (character lateral axis)
      pixel_y  →  orig_y   (character height axis, flipped from pixel)
      orig_z   estimated from k-nearest mesh vertices in image space

    Forward projection (for reference):
      std_x = orig_x / scale_factor
      NDC_x = PROJ_SCALE * std_x
      pixel_x = (NDC_x + 1) / 2 * W

      std_z = orig_y / scale_factor        (mesh Y ↔ std Z ↔ image vertical)
      NDC_y = -PROJ_SCALE * std_z          (Y-flipped by proj matrix)
      pixel_y = (NDC_y + 1) / 2 * H

    Inverse:
      orig_x = (2*px/W - 1) / PROJ_SCALE * scale_factor
      orig_y = -(2*py/H - 1) / PROJ_SCALE * scale_factor
    """
    W, H = RENDER_W, RENDER_H

    # Project all mesh vertices to image space (for Z lookup)
    verts_px_x = ((mesh_verts_orig[:,0] / scale_factor * PROJ_SCALE) + 1.0) / 2.0 * W
    verts_px_y = ((-mesh_verts_orig[:,1] / scale_factor * PROJ_SCALE) + 1.0) / 2.0 * H

    joints_3d = np.full((17, 3), np.nan)
    for i in range(17):
        px, py, conf = kp_2d_conf[i]
        if conf < 0.15 or px < 1 or py < 1:
            continue

        orig_x = (2.0*px/W - 1.0) / PROJ_SCALE * scale_factor
        orig_y = -(2.0*py/H - 1.0) / PROJ_SCALE * scale_factor

        # Z: median of k-nearest mesh vertices in image space
        dist_2d = np.hypot(verts_px_x - px, verts_px_y - py)
        k = 30
        near_idx = np.argpartition(dist_2d, k-1)[:k]
        orig_z   = float(np.median(mesh_verts_orig[near_idx, 2]))

        joints_3d[i] = [orig_x, orig_y, orig_z]

    return joints_3d


# ── Step 4: COCO-17 → SMPL-24 ────────────────────────────────────────────────

def coco17_to_smpl24(coco_3d, mesh_verts):
    """
    Build 24 SMPL joint positions from COCO-17 detections.
    Spine / collar / hand / foot joints are interpolated.
    Low-confidence (NaN) COCO joints fall back to mesh geometry.
    """
    def lerp(a, b, t):
        return a + t * (b - a)

    def valid(i):
        return not np.any(np.isnan(coco_3d[i]))

    # Fill NaN joints from mesh geometry (centroid fallback)
    c = coco_3d.copy()
    centroid = mesh_verts.mean(axis=0)
    for i in range(17):
        if not valid(i):
            c[i] = centroid

    # Key anchor points
    L_shoulder = c[5]
    R_shoulder = c[6]
    L_hip      = c[11]
    R_hip      = c[12]

    pelvis = lerp(L_hip, R_hip, 0.5)
    mid_shoulder = lerp(L_shoulder, R_shoulder, 0.5)
    # Neck: midpoint of shoulders, raised slightly (~ collar bone level)
    neck   = mid_shoulder + np.array([0.0, 0.04 * (mid_shoulder[1] - pelvis[1]), 0.0])

    J = np.zeros((24, 3), dtype=np.float64)

    J[0]  = pelvis                         # pelvis
    J[1]  = L_hip                          # left_hip
    J[2]  = R_hip                          # right_hip
    J[3]  = lerp(pelvis, neck, 0.25)       # spine1
    J[4]  = c[13]                          # left_knee
    J[5]  = c[14]                          # right_knee
    J[6]  = lerp(pelvis, neck, 0.5)        # spine2
    J[7]  = c[15]                          # left_ankle
    J[8]  = c[16]                          # right_ankle
    J[9]  = lerp(pelvis, neck, 0.75)       # spine3
    J[12] = neck                           # neck

    # Feet: project ankle downward toward mesh floor
    mesh_floor_y = mesh_verts[:,1].min()
    foot_y = mesh_floor_y + 0.02 * (c[15][1] - mesh_floor_y)  # 2% above floor
    J[10] = np.array([c[15][0], foot_y, c[15][2]])  # left_foot
    J[11] = np.array([c[16][0], foot_y, c[16][2]])  # right_foot

    J[13] = lerp(neck, L_shoulder, 0.5)   # left_collar
    J[14] = lerp(neck, R_shoulder, 0.5)   # right_collar
    J[15] = c[0]                           # head (nose as proxy)
    J[16] = L_shoulder                    # left_shoulder
    J[17] = R_shoulder                    # right_shoulder
    J[18] = c[7]                           # left_elbow
    J[19] = c[8]                           # right_elbow
    J[20] = c[9]                           # left_wrist
    J[21] = c[10]                          # right_wrist

    # Hands: extrapolate one step beyond wrist in elbow→wrist direction
    for side, (elbow_i, wrist_i, hand_i) in enumerate([(7,9,22), (8,10,23)]):
        elbow = c[elbow_i]; wrist = c[wrist_i]
        bone  = wrist - elbow
        blen  = np.linalg.norm(bone)
        if blen > 1e-3:
            J[hand_i] = wrist + bone / blen * 0.05
        else:
            J[hand_i] = wrist

    print('  SMPL-24 joints:')
    print('    pelvis   : (%.3f, %.3f, %.3f)' % tuple(J[0]))
    print('    L_hip    : (%.3f, %.3f, %.3f)' % tuple(J[1]))
    print('    R_hip    : (%.3f, %.3f, %.3f)' % tuple(J[2]))
    print('    neck     : (%.3f, %.3f, %.3f)' % tuple(J[12]))
    print('    L_shoulder: (%.3f, %.3f, %.3f)' % tuple(J[16]))
    print('    R_shoulder: (%.3f, %.3f, %.3f)' % tuple(J[17]))
    print('    head     : (%.3f, %.3f, %.3f)' % tuple(J[15]))

    return J.astype(np.float32)


# ── Step 5: LBS skinning weights ─────────────────────────────────────────────

def compute_skinning_weights(mesh_verts, joints, k=4):
    """
    Proximity-based LBS weights: each vertex gets k-nearest joint weights
    via inverse-distance weighting.
    Returns (N, 24) float32 full weight matrix.
    """
    N = len(mesh_verts)
    tree = cKDTree(joints)
    dists, idxs = tree.query(mesh_verts, k=k, workers=-1)

    # Clamp minimum distance to avoid division by zero
    inv_d = 1.0 / np.maximum(dists, 1e-6)
    inv_d /= inv_d.sum(axis=1, keepdims=True)

    W_full = np.zeros((N, 24), dtype=np.float32)
    for ki in range(k):
        W_full[np.arange(N), idxs[:, ki]] += inv_d[:, ki].astype(np.float32)

    # Normalize (should already be normalized, but just in case)
    row_sum = W_full.sum(axis=1, keepdims=True)
    W_full /= np.where(row_sum > 0, row_sum, 1.0)

    print('  weights: max_joint=%d  mean_support=%.2f joints/vert' % (
        W_full.argmax(axis=1).max(),
        (W_full > 0.01).sum(axis=1).mean()))

    return W_full


# ── Skeleton mesh builder ─────────────────────────────────────────────────────

def make_skeleton_mesh(joints, radius=0.008):
    """
    Build a mesh of hexagonal-prism cylinders connecting parent→child joints.
    Returns (verts, faces) as float32 / int32 numpy arrays.
    """
    SEG = 6  # hexagonal cross-section
    angles = np.linspace(0, 2 * np.pi, SEG, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (SEG, 2)

    all_verts, all_faces = [], []
    vert_offset = 0

    for i, parent in enumerate(SMPL_PARENTS):
        if parent == -1:
            continue
        p0 = joints[parent].astype(np.float64)
        p1 = joints[i].astype(np.float64)
        bone_vec = p1 - p0
        length = np.linalg.norm(bone_vec)
        if length < 1e-4:
            continue

        z_axis = bone_vec / length
        ref = np.array([0., 1., 0.]) if abs(z_axis[1]) < 0.9 else np.array([1., 0., 0.])
        x_axis = np.cross(ref, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Bottom ring at p0, top ring at p1
        offsets = radius * (circle[:, 0:1] * x_axis + circle[:, 1:2] * y_axis)
        bottom  = p0 + offsets   # (SEG, 3)
        top     = p1 + offsets   # (SEG, 3)

        all_verts.append(np.vstack([bottom, top]).astype(np.float32))

        for j in range(SEG):
            j1 = (j + 1) % SEG
            b0, b1 = vert_offset + j,       vert_offset + j1
            t0, t1 = vert_offset + SEG + j, vert_offset + SEG + j1
            all_faces.extend([[b0, b1, t0], [b1, t1, t0]])

        vert_offset += 2 * SEG

    if not all_verts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    return np.vstack(all_verts), np.array(all_faces, dtype=np.int32)


# ── Step 6: Export rigged GLB ─────────────────────────────────────────────────

def export_rigged_glb(verts, faces, uv, texture_pil, joints, skin_weights,
                      out_path, skel_verts=None, skel_faces=None):
    """
    Export skinned GLB using pygltflib.
    bind pose = current pose (joints at detected positions).
    IBM[j] = Translation(-J_world[j])  (pure offset, no rotation).

    If skel_verts/skel_faces are provided, a second mesh (bright green skeleton
    sticks) is embedded alongside the body mesh.
    """
    import pygltflib
    from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Accessor,
                            BufferView, Buffer, Material, Texture,
                            Image as GImage, Sampler, Skin, Asset)
    from pygltflib import (ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT,
                            UNSIGNED_INT, UNSIGNED_SHORT, LINEAR,
                            LINEAR_MIPMAP_LINEAR, REPEAT, SCALAR, VEC2,
                            VEC3, VEC4, MAT4)

    gltf  = GLTF2()
    gltf.asset = Asset(version='2.0', generator='rig_yolo.py')
    blobs = []

    def _add(data, comp, acc_type, target=None):
        b   = data.tobytes()
        pad = (4 - len(b) % 4) % 4
        off = sum(len(x) for x in blobs)
        blobs.append(b + b'\x00' * pad)
        bv = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(
            buffer=0, byteOffset=off, byteLength=len(b), target=target))
        ac = len(gltf.accessors)
        flat = data.flatten()
        gltf.accessors.append(Accessor(
            bufferView=bv, byteOffset=0, componentType=comp,
            type=acc_type, count=len(data),
            min=[float(flat.min())], max=[float(flat.max())]))
        return ac

    # Geometry
    pos_acc = _add(verts.astype(np.float32), FLOAT, VEC3, ARRAY_BUFFER)

    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    fn = np.cross(v1-v0, v2-v0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:,i], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)
    nor_acc = _add(vn.astype(np.float32), FLOAT, VEC3, ARRAY_BUFFER)

    if uv is None:
        uv = np.zeros((len(verts), 2), np.float32)
    uv_acc  = _add(uv.astype(np.float32), FLOAT, VEC2, ARRAY_BUFFER)
    idx_acc = _add(faces.astype(np.uint32).flatten(), UNSIGNED_INT, SCALAR,
                   ELEMENT_ARRAY_BUFFER)

    # Skinning: top-4 joints per vertex
    top4_idx = np.argsort(-skin_weights, axis=1)[:, :4].astype(np.uint16)
    top4_w   = np.take_along_axis(skin_weights, top4_idx.astype(np.int64), axis=1)
    top4_w   = top4_w.astype(np.float32)
    top4_w  /= top4_w.sum(axis=1, keepdims=True).clip(1e-8, None)
    j_acc   = _add(top4_idx, UNSIGNED_SHORT, VEC4, ARRAY_BUFFER)
    w_acc   = _add(top4_w,   FLOAT,          VEC4, ARRAY_BUFFER)

    # Texture
    if texture_pil is not None:
        import io
        buf = io.BytesIO()
        texture_pil.save(buf, format='PNG')
        ib  = buf.getvalue()
        off = sum(len(x) for x in blobs)
        pad = (4 - len(ib) % 4) % 4
        blobs.append(ib + b'\x00' * pad)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=off, byteLength=len(ib)))
        gltf.images.append(
            GImage(mimeType='image/png', bufferView=len(gltf.bufferViews)-1))
        gltf.samplers.append(
            Sampler(magFilter=LINEAR, minFilter=LINEAR_MIPMAP_LINEAR,
                    wrapS=REPEAT, wrapT=REPEAT))
        gltf.textures.append(Texture(sampler=0, source=0))
        gltf.materials.append(Material(
            name='body',
            pbrMetallicRoughness={
                'baseColorTexture': {'index': 0},
                'metallicFactor': 0.0,
                'roughnessFactor': 0.8},
            doubleSided=True))
    else:
        gltf.materials.append(Material(name='body', doubleSided=True))

    body_prim = Primitive(
        attributes={'POSITION': pos_acc, 'NORMAL': nor_acc,
                    'TEXCOORD_0': uv_acc, 'JOINTS_0': j_acc, 'WEIGHTS_0': w_acc},
        indices=idx_acc, material=0)
    gltf.meshes.append(Mesh(name='body', primitives=[body_prim]))

    # ── Optional skeleton mesh ─────────────────────────────────────────────────
    skel_mesh_idx = None
    if skel_verts is not None and len(skel_verts) > 0:
        sv = skel_verts.astype(np.float32)
        sf = skel_faces.astype(np.int32)

        sv0, sv1, sv2 = sv[sf[:,0]], sv[sf[:,1]], sv[sf[:,2]]
        sfn = np.cross(sv1-sv0, sv2-sv0)
        sfn /= (np.linalg.norm(sfn, axis=1, keepdims=True) + 1e-8)
        svn = np.zeros_like(sv)
        for i in range(3):
            np.add.at(svn, sf[:,i], sfn)
        svn /= (np.linalg.norm(svn, axis=1, keepdims=True) + 1e-8)

        s_pos_acc = _add(sv,                  FLOAT,        VEC3, ARRAY_BUFFER)
        s_nor_acc = _add(svn.astype(np.float32), FLOAT,     VEC3, ARRAY_BUFFER)
        s_idx_acc = _add(sf.astype(np.uint32).flatten(), UNSIGNED_INT, SCALAR,
                         ELEMENT_ARRAY_BUFFER)

        # Lime-green unlit material for skeleton sticks
        mat_idx = len(gltf.materials)
        gltf.materials.append(Material(
            name='skeleton',
            pbrMetallicRoughness={
                'baseColorFactor': [0.2, 1.0, 0.3, 1.0],
                'metallicFactor': 0.0,
                'roughnessFactor': 0.5},
            doubleSided=True))

        skel_mesh_idx = len(gltf.meshes)
        skel_prim = Primitive(
            attributes={'POSITION': s_pos_acc, 'NORMAL': s_nor_acc},
            indices=s_idx_acc, material=mat_idx)
        gltf.meshes.append(Mesh(name='skeleton', primitives=[skel_prim]))

    # ── Skeleton nodes ─────────────────────────────────────────────────────────
    jnodes = []
    for i, (name, parent) in enumerate(zip(SMPL_JOINT_NAMES, SMPL_PARENTS)):
        t = joints[i].tolist() if parent == -1 else (joints[i] - joints[parent]).tolist()
        n = Node(name=name, translation=t, children=[])
        jnodes.append(len(gltf.nodes))
        gltf.nodes.append(n)
    for i, p in enumerate(SMPL_PARENTS):
        if p != -1:
            gltf.nodes[jnodes[p]].children.append(jnodes[i])

    # Inverse bind matrices: IBM[j] = Translation(-J_world[j])
    # glTF MAT4 is column-major; numpy .tobytes() is row-major.
    # glTF reads the numpy buffer as the TRANSPOSE of what numpy stores.
    # So we set the translation in the last ROW of the numpy matrix — glTF
    # reads that as the last COLUMN (translation column) of a 4x4 mat.
    ibms = np.stack([np.eye(4, dtype=np.float32) for _ in range(len(joints))])
    for i in range(len(joints)):
        ibms[i, 3, :3] = -joints[i]
    ibm_acc = _add(ibms.astype(np.float32), FLOAT, MAT4)

    skin_idx = len(gltf.skins)
    gltf.skins.append(Skin(
        name='smpl_skin', skeleton=jnodes[0],
        joints=jnodes, inverseBindMatrices=ibm_acc))

    mesh_node = len(gltf.nodes)
    gltf.nodes.append(Node(name='body_mesh', mesh=0, skin=skin_idx))

    root_children = [jnodes[0], mesh_node]

    if skel_mesh_idx is not None:
        skel_node_idx = len(gltf.nodes)
        gltf.nodes.append(Node(name='skeleton_mesh', mesh=skel_mesh_idx))
        root_children.append(skel_node_idx)

    root_node = len(gltf.nodes)
    gltf.nodes.append(Node(name='root', children=root_children))
    gltf.scenes.append(Scene(name='Scene', nodes=[root_node]))
    gltf.scene = 0

    bin_data = b''.join(blobs)
    gltf.buffers.append(Buffer(byteLength=len(bin_data)))
    gltf.set_binary_blob(bin_data)
    gltf.save_binary(out_path)
    print('  rigged GLB -> %s  (%d KB)' % (out_path, os.path.getsize(out_path) // 1024))


# ── Main ──────────────────────────────────────────────────────────────────────

def rig_yolo(body_glb, out_glb, debug_dir=None):
    """
    Rig body_glb and write to out_glb.
    Returns (out_glb, out_skel_glb) where out_skel_glb includes visible
    skeleton bone sticks alongside the body mesh.
    """
    os.makedirs(os.path.dirname(out_glb) or '.', exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print('[rig_yolo] Rendering front view ...')
    img_bgr, scale_factor = render_front(body_glb, debug_dir)

    print('[rig_yolo] Running YOLO-pose ...')
    kp = detect_keypoints(img_bgr, debug_dir)

    print('[rig_yolo] Loading original mesh (pygltflib, correct UV channel) ...')
    verts, faces, uv, texture_pil = load_mesh_from_gltf(body_glb)

    print('[rig_yolo] Unprojecting YOLO keypoints to 3D ...')
    coco_3d = unproject_to_3d(kp, scale_factor, verts)

    print('[rig_yolo] Building SMPL-24 skeleton ...')
    joints = coco17_to_smpl24(coco_3d, verts)

    print('[rig_yolo] Computing skinning weights ...')
    skin_weights = compute_skinning_weights(verts, joints, k=4)

    print('[rig_yolo] Exporting rigged GLB (no skeleton) ...')
    export_rigged_glb(verts, faces, uv, texture_pil, joints, skin_weights, out_glb)

    print('[rig_yolo] Building skeleton mesh ...')
    skel_verts, skel_faces = make_skeleton_mesh(joints)
    out_skel_glb = out_glb.replace('.glb', '_skel.glb')
    print('[rig_yolo] Exporting rigged GLB (with skeleton) ...')
    export_rigged_glb(verts, faces, uv, texture_pil, joints, skin_weights,
                      out_skel_glb, skel_verts=skel_verts, skel_faces=skel_faces)

    print('[rig_yolo] Done.')
    return out_glb, out_skel_glb


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--body',      required=True, help='Input textured GLB')
    ap.add_argument('--out',       required=True, help='Output rigged GLB')
    ap.add_argument('--debug_dir', default=None,  help='Save debug renders here')
    args = ap.parse_args()
    rigged, rigged_skel = rig_yolo(args.body, args.out, args.debug_dir)
    print('Rigged:        ', rigged)
    print('Rigged + skel: ', rigged_skel)
