"""
tpose_smpl.py -- T-pose a humanoid GLB via inverse Linear Blend Skinning.

Pipeline:
  1. Render front view and run HMR2 -> SMPL body_pose + betas
  2. Read rigged.glb: mesh verts (rig world space), skinning weights, T-pose joints
  3. Compute FK transforms in rig world space using HMR2 body_pose
  4. Apply inverse LBS: v_tpose = (Sum_j W_j * A_j)^-1 * v_posed
  5. Map T-posed verts back to original mesh coordinate space, preserve UV/texture
  6. Optionally export SKEL bone mesh in T-pose

Usage:
    python tpose_smpl.py --body /tmp/triposg_textured.glb \
                         --rig  /tmp/rig_out/rigged.glb \
                         --out  /tmp/tposed_surface.glb \
                         [--skel_out /tmp/tposed_bones.glb] \
                         [--debug_dir /tmp/tpose_debug]
"""

import os, sys, argparse, struct, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, '/root/MV-Adapter')
SMPL_NEUTRAL = '/root/body_models/smpl/SMPL_NEUTRAL.pkl'
SKEL_DIR     = '/root/body_models/skel'

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                12, 13, 14, 16, 17, 18, 19, 20, 21]


# ---- Step 1: Render front view -----------------------------------------------

def render_front(body_glb, H=1024, W=768, device='cuda'):
    from mvadapter.utils.mesh_utils import (
        NVDiffRastContextWrapper, load_mesh, get_orthogonal_camera, render,
    )
    ctx     = NVDiffRastContextWrapper(device=device, context_type='cuda')
    mesh_mv = load_mesh(body_glb, rescale=True, device=device)
    camera  = get_orthogonal_camera(
        elevation_deg=[0], distance=[1.8],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[-90], device=device,
    )
    out = render(ctx, mesh_mv, camera, height=H, width=W,
                 render_attr=True, render_depth=False, render_normal=False,
                 attr_background=0.5)
    img_np = (out.attr[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


# ---- Step 2: HMR2 pose estimation --------------------------------------------

def run_hmr2(img_bgr, device='cuda'):
    from pathlib import Path
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT, download_models
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2 as hmr2_pkg

    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    cfg_path = Path(hmr2_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    det_cfg  = LazyConfig.load(str(cfg_path))
    det_cfg.train.init_checkpoint = (
        'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h'
        '/f328730692/model_final_f05665.pkl'
    )
    for i in range(3):
        det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(det_cfg)

    det_out   = detector(img_bgr)
    instances = det_out['instances']
    valid     = (instances.pred_classes == 0) & (instances.scores > 0.5)
    boxes     = instances.pred_boxes.tensor[valid].cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError('HMR2: no person detected in render')

    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    boxes = boxes[areas.argmax():areas.argmax()+1]

    dataset    = ViTDetDataset(model_cfg, img_bgr, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        sp = out['pred_smpl_params']
        return {
            'body_pose': sp['body_pose'][0].cpu(),    # (23, 3, 3)
            'betas':     sp['betas'][0].cpu(),         # (10,)
        }


# ---- Step 3: Read all data from rigged.glb -----------------------------------

def read_rigged_glb(rig_glb):
    """
    Returns dict with:
      verts        : (N, 3) mesh vertices in rig world space
      j_idx        : (N, 4) joint indices
      w_arr        : (N, 4) skinning weights
      J_bind       : (24, 3) T-pose joint world positions
    """
    with open(rig_glb, 'rb') as fh:
        raw = fh.read()
    ch_len, _ = struct.unpack_from('<II', raw, 12)
    gltf = json.loads(raw[20:20+ch_len])
    bin_data = raw[20+ch_len+8:]

    def _read(acc_i):
        acc = gltf['accessors'][acc_i]
        bv  = gltf['bufferViews'][acc['bufferView']]
        off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        cnt = acc['count']
        n   = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[acc['type']]
        fmt = {5121:'B',5123:'H',5125:'I',5126:'f'}[acc['componentType']]
        nb  = {'B':1,'H':2,'I':4,'f':4}[fmt]
        return np.frombuffer(bin_data[off:off+cnt*n*nb],
                             dtype=np.dtype(fmt)).reshape(cnt, n)

    prim  = gltf['meshes'][0]['primitives'][0]['attributes']
    verts = _read(prim['POSITION']).astype(np.float64)   # (N, 3)
    j_idx = _read(prim['JOINTS_0']).astype(int)          # (N, 4)
    w_arr = _read(prim['WEIGHTS_0']).astype(np.float64)  # (N, 4)
    row_sum = w_arr.sum(axis=1, keepdims=True)
    w_arr /= np.where(row_sum > 0, row_sum, 1.0)

    # Read T-pose joint world positions by accumulating node translations
    nodes   = gltf['nodes']
    skin    = gltf['skins'][0]
    j_nodes = skin['joints']                             # [0, 1, ..., 23]
    J_bind  = np.zeros((24, 3), dtype=np.float64)
    for ji, ni in enumerate(j_nodes):
        t_local = np.array(nodes[ni].get('translation', [0, 0, 0]))
        p = SMPL_PARENTS[ji]
        J_bind[ji] = (J_bind[p] if p >= 0 else np.zeros(3)) + t_local

    print('  Rig verts: %d  Y: [%.3f, %.3f]  X: [%.3f, %.3f]' % (
        len(verts),
        verts[:,1].min(), verts[:,1].max(),
        verts[:,0].min(), verts[:,0].max()))
    print('  J_bind pelvis: (%.3f, %.3f, %.3f)  L_shoulder: (%.3f, %.3f, %.3f)' % (
        *J_bind[0], *J_bind[16]))
    return {'verts': verts, 'j_idx': j_idx, 'w_arr': w_arr, 'J_bind': J_bind}


# ---- Step 4: FK in rig world space -> A matrices -----------------------------

_FLIP_X = np.diag([-1.0, 1.0, 1.0])   # X-axis mirror matrix


def _adapt_rotmat_to_flipped_x(R_smpl):
    """
    Convert an SO(3) rotation matrix from SMPL convention (left=+X)
    to rig convention (left=-X).  F @ R @ F  where F = diag(-1,1,1).
    """
    return _FLIP_X @ R_smpl @ _FLIP_X


def compute_rig_fk_transforms(J_bind, body_pose_rotmats):
    """
    Compute A_j = G_j_posed * IBM_j in rig world space.
    A_j maps T-pose -> posed, so A_j^{-1} maps posed -> T-pose.

    HMR2 returns rotations in SMPL convention (left shoulder at +X).
    The rig uses the opposite convention (left shoulder at -X).
    We convert by conjugating with the X-flip matrix before building FK.

    J_bind          : (24, 3) T-pose joint world positions from rig
    body_pose_rotmats: (23, 3, 3) HMR2 body pose rotation matrices (joints 1-23)
    Returns A: (24, 4, 4)
    """
    G = [None] * 24
    for j in range(24):
        p = SMPL_PARENTS[j]
        # Convert rotation from SMPL (+X=left) to rig (-X=left) convention
        R_smpl = body_pose_rotmats[j-1].numpy() if j >= 1 else np.eye(3)
        R_j    = _adapt_rotmat_to_flipped_x(R_smpl)

        if p < 0:
            t_j = J_bind[j]           # root: absolute world position
        else:
            t_j = J_bind[j] - J_bind[p]

        L = np.eye(4, dtype=np.float64)
        L[:3, :3] = R_j
        L[:3, 3]  = t_j

        G[j] = L if p < 0 else G[p] @ L

    G = np.stack(G)

    A = np.zeros((24, 4, 4), dtype=np.float64)
    for j in range(24):
        IBM = np.eye(4, dtype=np.float64)
        IBM[:3, 3] = -J_bind[j]
        A[j] = G[j] @ IBM

    return A


# ---- Step 5: Inverse LBS -----------------------------------------------------

def inverse_lbs(verts, j_idx, w_arr, A):
    """
    v_tpose = (Sum_j W_j * A_j)^{-1} * v_posed
    All inputs in rig world space.
    Returns (N, 3) T-posed vertices.
    """
    N = len(verts)
    # Blend forward transforms
    T_fwd = np.zeros((N, 4, 4), dtype=np.float64)
    for k in range(4):
        ji   = j_idx[:, k]
        w    = w_arr[:, k]
        mask = w > 1e-6
        if mask.any():
            T_fwd[mask] += w[mask, None, None] * A[ji[mask]]

    T_inv = np.linalg.inv(T_fwd)
    v_h   = np.concatenate([verts, np.ones((N, 1))], axis=1)
    v_tp  = np.einsum('nij,nj->ni', T_inv, v_h)[:, :3]

    disp  = np.linalg.norm(v_tp - verts, axis=1)
    print('  inverse LBS: mean_disp=%.4f  max_disp=%.4f' % (disp.mean(), disp.max()))
    return v_tp


# ---- Step 6: Map T-posed rig verts back to original mesh space ---------------

def rig_to_original_space(rig_verts_tposed, rig_verts_original, orig_mesh_verts):
    """
    Rig verts are a scaled + translated version of the original mesh verts.
    Recover the (scale, offset) from the mapping:
      rig_vert = orig_vert * scale + offset

    Estimates scale from height ratio, offset from floor alignment.
    Returns T-posed vertices in original mesh coordinate space.
    """
    rig_h  = rig_verts_original[:, 1].max() - rig_verts_original[:, 1].min()
    orig_h = orig_mesh_verts[:, 1].max()    - orig_mesh_verts[:, 1].min()
    scale  = rig_h / max(orig_h, 1e-6)

    # The rig aligns: orig * scale, then v[:,1] -= v[:,1].min() (floor at 0)
    # and v[:,0] += smpl_joints[0,0] - cx; v[:,2] += smpl_joints[0,2] - cz
    # We can recover offset from comparing means/floors
    # offset = rig_floor_Y - (orig_floor_Y * scale)
    rig_floor  = rig_verts_original[:, 1].min()
    orig_floor = orig_mesh_verts[:, 1].min()
    y_offset   = rig_floor - orig_floor * scale

    # X, Z: center offset
    rig_cx  = (rig_verts_original[:, 0].max() + rig_verts_original[:, 0].min()) * 0.5
    orig_cx = (orig_mesh_verts[:, 0].max()    + orig_mesh_verts[:, 0].min())    * 0.5
    x_offset = rig_cx - orig_cx * scale

    rig_cz  = (rig_verts_original[:, 2].max() + rig_verts_original[:, 2].min()) * 0.5
    orig_cz = (orig_mesh_verts[:, 2].max()    + orig_mesh_verts[:, 2].min())    * 0.5
    z_offset = rig_cz - orig_cz * scale

    print('  rig->orig: scale=%.4f  offset=[%.3f, %.3f, %.3f]' % (scale, x_offset, y_offset, z_offset))

    # Invert: orig_vert = (rig_vert - offset) / scale
    # For T-posed verts: they're in rig space but T-posed, so same inversion
    tposed_orig = np.zeros_like(rig_verts_tposed)
    tposed_orig[:, 0] = (rig_verts_tposed[:, 0] - x_offset) / scale
    tposed_orig[:, 1] = (rig_verts_tposed[:, 1] - y_offset) / scale
    tposed_orig[:, 2] = (rig_verts_tposed[:, 2] - z_offset) / scale
    return tposed_orig


# ---- SKEL bone geometry ------------------------------------------------------

def export_skel_bones(betas, out_path, gender='male'):
    try:
        from skel.skel_model import SKEL
    except ImportError:
        print('  [skel] Not installed')
        return None
    skel_file = os.path.join(SKEL_DIR, 'skel_%s.pkl' % gender)
    if not os.path.exists(skel_file):
        print('  [skel] Weights not found: %s' % skel_file)
        return None
    try:
        skel_model = SKEL(gender=gender, model_path=SKEL_DIR)
        betas_t    = betas.unsqueeze(0)[:, :10]
        poses_zero = torch.zeros(1, 46)
        trans_zero = torch.zeros(1, 3)
        with torch.no_grad():
            out = skel_model(poses=poses_zero, betas=betas_t, trans=trans_zero, skelmesh=True)
        bone_verts = out.skel_verts[0].numpy()
        bone_faces = skel_model.skel_f.numpy()
        mesh = trimesh.Trimesh(vertices=bone_verts, faces=bone_faces, process=False)
        mesh.export(out_path)
        print('  [skel] Bone mesh -> %s  (%d verts)' % (out_path, len(bone_verts)))
        return out_path
    except Exception as e:
        print('  [skel] Export failed: %s' % e)
        return None


# ---- Main --------------------------------------------------------------------

def tpose_smpl(body_glb, out_glb, rig_glb=None, debug_dir=None, skel_out=None):
    device = 'cuda'

    if not rig_glb or not os.path.exists(rig_glb):
        raise RuntimeError('--rig is required: provide the rigged.glb from the Rig step.')

    print('[tpose_smpl] Rendering front view ...')
    img_bgr = render_front(body_glb, device=device)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, 'tpose_render.png'), img_bgr)

    print('[tpose_smpl] Running HMR2 pose estimation ...')
    hmr2_out = run_hmr2(img_bgr, device=device)
    print('  betas: %s' % hmr2_out['betas'].numpy().round(3))

    print('[tpose_smpl] Reading rigged GLB (rig world space) ...')
    rig_data = read_rigged_glb(rig_glb)

    print('[tpose_smpl] Loading original mesh for UV/texture ...')
    scene = trimesh.load(body_glb)
    if isinstance(scene, trimesh.Scene):
        geom_name = list(scene.geometry.keys())[0]
        orig_mesh  = scene.geometry[geom_name]
    else:
        orig_mesh = scene; geom_name = None

    orig_verts = np.array(orig_mesh.vertices, dtype=np.float64)
    uvs        = np.array(orig_mesh.visual.uv, dtype=np.float64)
    orig_tex   = orig_mesh.visual.material.baseColorTexture
    print('  Orig mesh: %d verts  Y: [%.3f, %.3f]  X: [%.3f, %.3f]' % (
        len(orig_verts),
        orig_verts[:,1].min(), orig_verts[:,1].max(),
        orig_verts[:,0].min(), orig_verts[:,0].max()))

    print('[tpose_smpl] Computing FK transforms in rig world space ...')
    body_pose_rotmats = hmr2_out['body_pose']   # (23, 3, 3)
    A = compute_rig_fk_transforms(rig_data['J_bind'], body_pose_rotmats)

    # Verify zero-pose gives identity (sanity check)
    A_zero = compute_rig_fk_transforms(rig_data['J_bind'],
                                        torch.zeros(23, 3, 3) + torch.eye(3))
    v_test = rig_data['verts'][:3]
    v_h = np.concatenate([v_test, np.ones((3,1))], axis=1)
    T_fwd_test = np.zeros((3, 4, 4))
    for k in range(4):
        ji = rig_data['j_idx'][:3, k]; w = rig_data['w_arr'][:3, k]
        T_fwd_test += w[:, None, None] * A_zero[ji]
    identity_err = np.abs(T_fwd_test - np.eye(4)).max()
    print('  zero-pose identity check: max_err=%.6f (expect ~0)' % identity_err)

    print('[tpose_smpl] Applying inverse LBS ...')
    rig_verts_tposed = inverse_lbs(
        rig_data['verts'], rig_data['j_idx'], rig_data['w_arr'], A)

    print('[tpose_smpl] T-posed rig verts: Y: [%.3f, %.3f]  X: [%.3f, %.3f]' % (
        rig_verts_tposed[:,1].min(), rig_verts_tposed[:,1].max(),
        rig_verts_tposed[:,0].min(), rig_verts_tposed[:,0].max()))

    print('[tpose_smpl] Mapping back to original mesh coordinate space ...')
    tposed_orig = rig_to_original_space(
        rig_verts_tposed, rig_data['verts'], orig_verts)

    print('[tpose_smpl] T-posed orig: Y: [%.3f, %.3f]  X: [%.3f, %.3f]' % (
        tposed_orig[:,1].min(), tposed_orig[:,1].max(),
        tposed_orig[:,0].min(), tposed_orig[:,0].max()))

    orig_mesh.vertices = tposed_orig
    orig_mesh.visual = TextureVisuals(uv=uvs,
                                      material=PBRMaterial(baseColorTexture=orig_tex))

    if geom_name and isinstance(scene, trimesh.Scene):
        scene.geometry[geom_name] = orig_mesh
        scene.export(out_glb)
    else:
        orig_mesh.export(out_glb)

    print('[tpose_smpl] Saved: %s  (%d KB)' % (out_glb, os.path.getsize(out_glb)//1024))

    if skel_out:
        print('[tpose_smpl] Exporting SKEL bone geometry ...')
        export_skel_bones(hmr2_out['betas'], skel_out)

    return out_glb


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--body',      required=True)
    ap.add_argument('--out',       required=True)
    ap.add_argument('--rig',       required=True, help='Rigged GLB from rig step')
    ap.add_argument('--skel_out',  default=None,  help='SKEL BSM bone mesh output')
    ap.add_argument('--debug_dir', default=None)
    args = ap.parse_args()
    os.makedirs(args.debug_dir, exist_ok=True) if args.debug_dir else None
    tpose_smpl(args.body, args.out, rig_glb=args.rig,
               debug_dir=args.debug_dir, skel_out=args.skel_out)
