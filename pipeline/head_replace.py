"""
head_replace.py — Replace TripoSG head with DECA-reconstructed head at mesh level.

Requires: trimesh, numpy, scipy, cv2, torch (+ face-alignment via DECA deps)
Optional: pymeshlab (for mesh clean-up)

Usage (standalone):
    python head_replace.py --body /tmp/triposg_textured.glb \
                           --face /path/to/face.jpg \
                           --out  /tmp/head_replaced.glb

Returns combined GLB with DECA head geometry + TripoSG body.
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image

# ──────────────────────────────────────────────────────────────────
# Patch DECA before importing it to avoid pytorch3d dependency
# ──────────────────────────────────────────────────────────────────
DECA_ROOT = '/root/DECA'
sys.path.insert(0, DECA_ROOT)

# Stub out the rasterizer so DECA doesn't try to import pytorch3d
import importlib, types
_fake_renderer = types.ModuleType('decalib.utils.renderer')
_fake_renderer.set_rasterizer = lambda t='pytorch3d': None

class _FakeRender:
    """No-op renderer — we only need the mesh, not rendered images."""
    def __init__(self, *a, **kw): pass
    def to(self, *a, **kw): return self
    def __call__(self, *a, **kw): return {'images': None, 'alpha_images': None,
                                          'normal_images': None, 'grid': None,
                                          'transformed_normals': None, 'normals': None}
    def render_shape(self, *a, **kw): return None, None, None, None
    def world2uv(self, *a, **kw): return None
    def add_SHlight(self, *a, **kw): return None

_fake_renderer.SRenderY = _FakeRender
sys.modules['decalib.utils.renderer'] = _fake_renderer

# Patch deca.py: make _setup_renderer a no-op when renderer not available
from decalib import deca as _deca_mod
_orig_setup = _deca_mod.DECA._setup_renderer

def _patched_setup(self, model_cfg):
    try:
        _orig_setup(self, model_cfg)
    except Exception as e:
        print(f'[head_replace] Renderer disabled ({e})')
        self.render = _FakeRender()
        # Still load mask / displacement data we need for UV baking
        from skimage.io import imread
        import torch, torch.nn.functional as F
        try:
            mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.
            mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
            self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size])
            mask2 = imread(model_cfg.face_mask_path).astype(np.float32) / 255.
            mask2 = torch.from_numpy(mask2[:, :, 0])[None, None, :, :].contiguous()
            self.uv_face_mask = F.interpolate(mask2, [model_cfg.uv_size, model_cfg.uv_size])
        except Exception:
            pass
        try:
            fixed_dis = np.load(model_cfg.fixed_displacement_path)
            self.fixed_uv_dis = torch.tensor(fixed_dis).float()
        except Exception:
            pass
        try:
            mean_tex_np = imread(model_cfg.mean_tex_path).astype(np.float32) / 255.
            mean_tex = torch.from_numpy(mean_tex_np.transpose(2, 0, 1))[None]
            self.mean_texture = F.interpolate(mean_tex, [model_cfg.uv_size, model_cfg.uv_size])
        except Exception:
            pass
        try:
            self.dense_template = np.load(model_cfg.dense_template_path,
                                          allow_pickle=True, encoding='latin1').item()
        except Exception:
            pass

_deca_mod.DECA._setup_renderer = _patched_setup


# ──────────────────────────────────────────────────────────────────
# FLAME mesh: parse head_template.obj for UV map
# ──────────────────────────────────────────────────────────────────
def _load_flame_template(obj_path=os.path.join(DECA_ROOT, 'data', 'head_template.obj')):
    """Return (verts, faces, uv_verts, uv_faces) from head_template.obj."""
    verts, uv_verts = [], []
    faces_v, faces_uv = [], []
    for line in open(obj_path):
        t = line.split()
        if not t:
            continue
        if t[0] == 'v':
            verts.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == 'vt':
            uv_verts.append([float(t[1]), float(t[2])])
        elif t[0] == 'f':
            vi, uvi = [], []
            for tok in t[1:]:
                parts = tok.split('/')
                vi.append(int(parts[0]) - 1)
                uvi.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else 0)
            if len(vi) == 3:
                faces_v.append(vi)
                faces_uv.append(uvi)
    return (np.array(verts, dtype=np.float32),
            np.array(faces_v, dtype=np.int32),
            np.array(uv_verts, dtype=np.float32),
            np.array(faces_uv, dtype=np.int32))


# ──────────────────────────────────────────────────────────────────
# UV texture baking (software rasteriser, no pytorch3d needed)
# ──────────────────────────────────────────────────────────────────
def _bake_uv_texture(verts3d, faces_v, uv_verts, faces_uv, cam, face_img_bgr, tex_size=256):
    """
    Project face_img_bgr onto the FLAME UV map using orthographic camera.
    verts3d : (N,3) FLAME vertices in world space
    cam     : (3,) = [scale, tx, ty] orthographic camera
    Returns : (tex_size, tex_size, 3) uint8 texture (BGR)
    """
    H, W = face_img_bgr.shape[:2]
    scale, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])

    # Orthographic project: DECA formula = (vert_2D + [tx,ty]) * scale, then flip y
    proj = np.zeros((len(verts3d), 2), dtype=np.float32)
    proj[:, 0] = (verts3d[:, 0] + tx) * scale
    proj[:, 1] = -((verts3d[:, 1] + ty) * scale)  # y-flip matches DECA convention

    # Map to pixel coords: image spans proj ∈ [-1,1] → pixel [0, WH]
    img_pts = (proj + 1.0) * 0.5 * np.array([W, H], dtype=np.float32)  # (N, 2)

    # UV pixel coords
    uv_px = uv_verts * tex_size  # (K, 2)

    # Output buffers
    tex_acc = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    tex_cnt = np.zeros((tex_size, tex_size), dtype=np.float64)
    z_buf   = np.full((tex_size, tex_size), -1e9, dtype=np.float64)

    # Vectorised rasteriser in UV space:
    # For each face, scatter samples from img_pts into uv_px coords.
    # Use scipy.interpolate.griddata as a fast splat.
    from scipy.interpolate import griddata

    # Front-facing mask (z > threshold) — only bake visible faces
    z_face = verts3d[faces_v, 2].mean(axis=1)           # (M,) mean z per face
    front_mask = z_face >= -0.02                          # keep front and side faces

    # For each face corner, record (uv_px, img_pts) sample
    corners_uv  = uv_px[faces_uv[front_mask]]            # (K, 3, 2)
    corners_img = img_pts[faces_v[front_mask]]            # (K, 3, 2)

    # Flatten to (K*3, 2)
    src_uv  = corners_uv.reshape(-1, 2)                  # UV pixel destination
    src_img = corners_img.reshape(-1, 2)                  # image pixel source

    # Remove out-of-bounds image samples
    valid = ((src_img[:, 0] >= 0) & (src_img[:, 0] < W) &
             (src_img[:, 1] >= 0) & (src_img[:, 1] < H))
    src_uv  = src_uv[valid]
    src_img = src_img[valid]

    # Sample face image at src_img positions
    ix = np.clip(src_img[:, 0].astype(int), 0, W - 1)
    iy = np.clip(src_img[:, 1].astype(int), 0, H - 1)
    colours = face_img_bgr[iy, ix].astype(np.float32)    # (P, 3)

    # Clip UV destinations to texture bounds
    uv_dest = np.clip(src_uv, 0, tex_size - 1 - 1e-6).astype(np.float32)

    # Build query grid for griddata interpolation
    grid_u, grid_v = np.meshgrid(np.arange(tex_size), np.arange(tex_size))
    grid_pts = np.column_stack([grid_u.ravel(), grid_v.ravel()])

    # Interpolate each colour channel
    tex_baked = np.zeros((tex_size * tex_size, 3), dtype=np.float32)
    for ch in range(3):
        ch_vals = griddata(uv_dest, colours[:, ch], grid_pts,
                           method='linear', fill_value=np.nan)
        tex_baked[:, ch] = ch_vals
    tex_baked = tex_baked.reshape(tex_size, tex_size, 3)
    face_baked_mask = ~np.isnan(tex_baked[:, :, 0])

    # Base texture: mean_texture (skin tone fallback for unsampled regions)
    mean_tex_path = os.path.join(DECA_ROOT, 'data', 'mean_texture.jpg')
    if os.path.exists(mean_tex_path):
        mt = cv2.resize(cv2.imread(mean_tex_path), (tex_size, tex_size)).astype(np.float32)
    else:
        mt = np.full((tex_size, tex_size, 3), 180.0, dtype=np.float32)

    # Blend: baked face over mean texture
    result = mt.copy()
    result[face_baked_mask] = np.clip(tex_baked[face_baked_mask], 0, 255)
    return result.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────
# DECA inference
# ──────────────────────────────────────────────────────────────────
def run_deca(face_img_path, device='cuda'):
    """
    Run DECA on face_img_path.
    Returns (verts_np, cam_np, faces_v, uv_verts, faces_uv, tex_img_bgr)
    """
    import torch
    from decalib.deca import DECA
    from decalib.utils import config as cfg_module
    from decalib.datasets import datasets

    cfg = cfg_module.get_cfg_defaults()
    cfg.model.use_tex = False

    print('[DECA] Loading model...')
    deca = DECA(config=cfg, device=device)
    deca.eval()

    print('[DECA] Preprocessing image...')
    testdata = datasets.TestData(face_img_path)
    img_tensor = testdata[0]['image'].to(device)[None, ...]

    print('[DECA] Encoding...')
    with torch.no_grad():
        codedict = deca.encode(img_tensor, use_detail=False)
        verts, _, _ = deca.flame(
            shape_params=codedict['shape'],
            expression_params=codedict['exp'],
            pose_params=codedict['pose']
        )

    verts_np = verts[0].cpu().numpy()  # (5023, 3)
    cam_np   = codedict['cam'][0].cpu().numpy()  # (3,)
    print(f'[DECA] Mesh: {verts_np.shape}, cam={cam_np}')

    # Load FLAME UV map
    _, faces_v, uv_verts, faces_uv = _load_flame_template()

    # Get face image for texture baking (use the cropped/aligned 224x224)
    img_np = (img_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    print('[DECA] Baking UV texture...')
    tex_bgr = _bake_uv_texture(verts_np, faces_v, uv_verts, faces_uv, cam_np, img_bgr, tex_size=256)

    return verts_np, cam_np, faces_v, uv_verts, faces_uv, tex_bgr


# ──────────────────────────────────────────────────────────────────
# Mesh helpers
# ──────────────────────────────────────────────────────────────────
def _find_neck_height(mesh):
    """
    Find the best neck cut height in a body mesh.
    Strategy: in the top 40% of the mesh, find the local minimum of
    cross-sectional area (the neck is narrower than the head).
    Returns the y-value of the cut plane.
    """
    verts = mesh.vertices
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min

    # Scan [80%, 87%] to find the neck-base narrowing below the face.
    # The range [83%, 91%] was picking the crown taper instead of the neck.
    y_start = y_min + y_range * 0.80
    y_end   = y_min + y_range * 0.87
    steps   = 20
    ys      = np.linspace(y_start, y_end, steps)
    band    = y_range * 0.015

    r10_vals = []
    for y in ys:
        pts = verts[(verts[:, 1] >= y - band) & (verts[:, 1] <= y + band)]
        if len(pts) < 6:
            r10_vals.append(1.0); continue
        xz = pts[:, [0, 2]]
        cx, cz = xz.mean(0)
        radii = np.sqrt((xz[:, 0] - cx)**2 + (xz[:, 1] - cz)**2)
        r10_vals.append(float(np.percentile(radii, 10)))

    from scipy.ndimage import uniform_filter1d
    r10 = uniform_filter1d(np.array(r10_vals), size=3)
    neck_idx = int(np.argmin(r10[2:-2])) + 2
    neck_y = float(ys[neck_idx])
    frac = (neck_y - y_min) / y_range
    print(f'[neck] Cut height: {neck_y:.4f} (y_range {y_min:.3f}–{y_max:.3f}, frac={frac:.2f})')
    return neck_y


def _weld_mesh(mesh):
    """
    Merge duplicate vertices (UV-split mesh → geometric mesh).
    Returns a new trimesh with welded vertices.
    """
    import trimesh
    from scipy.spatial import cKDTree
    verts = mesh.vertices
    tree = cKDTree(verts)
    # Build mapping: each vertex → canonical representative
    N = len(verts)
    mapping = np.arange(N, dtype=np.int64)
    pairs = tree.query_pairs(r=1e-5)
    for a, b in pairs:
        root_a = int(mapping[a])
        root_b = int(mapping[b])
        while mapping[root_a] != root_a:
            root_a = int(mapping[root_a])
        while mapping[root_b] != root_b:
            root_b = int(mapping[root_b])
        if root_a != root_b:
            mapping[root_b] = root_a
    # Flatten chains
    for i in range(N):
        root = int(mapping[i])
        while mapping[root] != root:
            root = int(mapping[root])
        mapping[i] = root
    # Compact the mapping
    unique_ids = np.unique(mapping)
    compact = np.full(N, -1, dtype=np.int64)
    compact[unique_ids] = np.arange(len(unique_ids))
    new_faces = compact[mapping[mesh.faces]]
    new_verts = verts[unique_ids]
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)


def _cut_mesh_below(mesh, y_cut):
    """Keep only faces where all vertices are at or below y_cut. Preserves UV/texture."""
    import trimesh
    from trimesh.visual.texture import TextureVisuals
    v_mask = mesh.vertices[:, 1] <= y_cut
    f_keep = np.all(v_mask[mesh.faces], axis=1)
    faces_kept = mesh.faces[f_keep]
    used_verts = np.unique(faces_kept)
    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int64)
    old_to_new[used_verts] = np.arange(len(used_verts))
    new_faces = old_to_new[faces_kept]
    new_verts = mesh.vertices[used_verts]
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    # Preserve UV + texture if present
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        new_mesh.visual = TextureVisuals(
            uv=mesh.visual.uv[used_verts],
            material=mesh.visual.material)
    return new_mesh


def _extract_neck_ring_geometric(mesh, neck_y, n_pts=64, band_frac=0.02):
    """
    Extract a neck ring using topological boundary edges near neck_y.
    Falls back to angle-sorted vertices if topology is non-manifold.
    Works on welded (geometric) meshes.
    """
    verts = mesh.vertices
    y_range = verts[:, 1].max() - verts[:, 1].min()
    band = y_range * band_frac

    # --- Try topological boundary near neck_y first ---
    edges = np.sort(mesh.edges, axis=1)
    u, c2 = np.unique(edges, axis=0, return_counts=True)
    be = u[c2 == 1]   # boundary edges

    # Keep boundary edges where BOTH endpoints are near neck_y
    v_near = np.abs(verts[:, 1] - neck_y) <= band * 2
    neck_be = be[v_near[be[:, 0]] & v_near[be[:, 1]]]

    if len(neck_be) >= 8:
        # Build adjacency and walk loop
        adj = {}
        for e in neck_be:
            adj.setdefault(int(e[0]), []).append(int(e[1]))
            adj.setdefault(int(e[1]), []).append(int(e[0]))
        # Find the largest connected loop
        visited = set()
        loops = []
        for start in adj:
            if start in visited: continue
            loop = [start]; visited.add(start); prev = -1; cur = start
            for _ in range(len(neck_be) + 1):
                nbrs = [v for v in adj.get(cur, []) if v != prev]
                if not nbrs: break
                nxt = nbrs[0]
                if nxt == start: break
                if nxt in visited: break
                visited.add(nxt); prev = cur; cur = nxt; loop.append(cur)
            loops.append(loop)
        if loops:
            best = max(loops, key=len)
            if len(best) >= 8:
                ring_pts = verts[best]
                # Snap all ring points to neck_y (smooth the cut plane)
                ring_pts = ring_pts.copy()
                ring_pts[:, 1] = neck_y
                return _resample_loop(ring_pts, n_pts)

    # --- Fallback: use inner-cluster (neck column) vertices in the band ---
    mask = (verts[:, 1] >= neck_y - band) & (verts[:, 1] <= neck_y + band)
    pts = verts[mask]
    if len(pts) < 8:
        raise ValueError(f'Too few vertices near neck_y={neck_y:.4f}: {len(pts)}')

    # Keep only inner-ring vertices (below 35th percentile radius from centroid)
    # This excludes the outer face/head surface and keeps only the neck column
    xz = pts[:, [0, 2]]
    cx, cz = xz.mean(0)
    radii = np.sqrt((xz[:, 0] - cx)**2 + (xz[:, 1] - cz)**2)
    thresh = np.percentile(radii, 35)
    inner_mask = radii <= thresh
    if inner_mask.sum() >= 8:
        pts = pts[inner_mask]
        # Recompute centroid on inner pts
        cx, cz = pts[:, [0, 2]].mean(0)

    # Sort by angle in XZ plane
    angles = np.arctan2(pts[:, 2] - cz, pts[:, 0] - cx)
    pts_sorted = pts[np.argsort(angles)]
    pts_sorted = pts_sorted.copy()
    pts_sorted[:, 1] = neck_y  # snap to cut plane
    return _resample_loop(pts_sorted, n_pts)


def _extract_boundary_loop(mesh):
    """
    Extract the boundary edge loop (ordered) from a welded mesh.
    Returns (N, 3) ordered vertex positions.
    """
    # Find boundary edges (edges used by exactly one face)
    edges = np.sort(mesh.edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]

    if len(boundary_edges) == 0:
        raise ValueError('No boundary edges found — mesh may be closed')

    # Build adjacency for boundary edges
    adj = {}
    for e in boundary_edges:
        adj.setdefault(int(e[0]), []).append(int(e[1]))
        adj.setdefault(int(e[1]), []).append(int(e[0]))

    # Walk the longest connected loop
    # Find all loops
    visited = set()
    loops = []
    for start_v in adj:
        if start_v in visited:
            continue
        loop = [start_v]
        visited.add(start_v)
        prev = -1
        cur = start_v
        for _ in range(len(boundary_edges) + 1):
            nbrs = [v for v in adj.get(cur, []) if v != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start_v:
                break
            if nxt in visited:
                break
            visited.add(nxt)
            prev = cur
            cur = nxt
            loop.append(cur)
        loops.append(loop)

    # Use the longest loop
    best = max(loops, key=len)
    return mesh.vertices[best]


def _resample_loop(loop_pts, N):
    """Resample an ordered set of 3D points to exactly N evenly-spaced points."""
    from scipy.interpolate import interp1d
    # Arc-length parameterisation
    diffs = np.diff(loop_pts, axis=0, prepend=loop_pts[-1:])
    seg_lens = np.linalg.norm(diffs, axis=1)
    t = np.cumsum(seg_lens)
    t = np.insert(t, 0, 0)
    t /= t[-1]
    # Close the loop
    t[-1] = 1.0
    loop_closed = np.vstack([loop_pts, loop_pts[0]])
    interp = interp1d(t, loop_closed, axis=0)
    t_new = np.linspace(0, 1, N, endpoint=False)
    return interp(t_new)


def _bridge_loops(loop_a, loop_b):
    """
    Create a triangle strip bridging two ordered loops of equal length N.
    loop_a, loop_b: (N, 3) vertex positions
    Returns (verts, faces) — just the bridge strip as a trimesh-ready array.
    """
    N = len(loop_a)
    verts = np.vstack([loop_a, loop_b])  # (2N, 3) — a:0..N-1, b:N..2N-1
    faces = []
    for i in range(N):
        j = (i + 1) % N
        ai, aj = i, j
        bi, bj = i + N, j + N
        faces.append([ai, aj, bi])
        faces.append([aj, bj, bi])
    return verts, np.array(faces, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────
# DECA head → trimesh
# ──────────────────────────────────────────────────────────────────
def deca_to_trimesh(verts_np, faces_v, uv_verts, faces_uv, tex_bgr):
    """
    Assemble a trimesh.Trimesh from DECA outputs with UV texture.
    Uses per-vertex UV (averaged over face corners sharing each vertex).
    """
    import trimesh
    from trimesh.visual.texture import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    # Average face-corner UVs per vertex
    N = len(verts_np)
    uv_sum = np.zeros((N, 2), dtype=np.float64)
    uv_cnt = np.zeros(N, dtype=np.int32)
    for fi in range(len(faces_v)):
        for ci in range(3):
            vi = faces_v[fi, ci]
            uvi = faces_uv[fi, ci]
            uv_sum[vi] += uv_verts[uvi]
            uv_cnt[vi] += 1
    uv_cnt = np.maximum(uv_cnt, 1)
    uv_per_vert = (uv_sum / uv_cnt[:, None]).astype(np.float32)

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_v, process=False)

    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)
    tex_pil = Image.fromarray(tex_rgb)

    try:
        mat = PBRMaterial(baseColorTexture=tex_pil)
        mesh.visual = TextureVisuals(uv=uv_per_vert, material=mat)
        print(f'[deca_to_trimesh] UV attached: {uv_per_vert.shape}, tex={tex_rgb.shape}')
    except Exception as e:
        print(f'[deca_to_trimesh] UV attach failed ({e}) — using vertex colours')
        mesh.visual.vertex_colors = np.tile([200, 175, 155, 255], (len(verts_np), 1))

    return mesh


# ──────────────────────────────────────────────────────────────────
# Main head-replacement function
# ──────────────────────────────────────────────────────────────────
def replace_head(body_glb: str, face_img_path: str, out_glb: str,
                 device: str = 'cuda', bridge_n: int = 64):
    """
    Main entry point.
    body_glb       : path to TripoSG textured GLB
    face_img_path  : path to reference face image
    out_glb        : output path for combined GLB
    bridge_n       : number of vertices in the stitching ring
    """
    import trimesh
    import torch

    # ── 1. Load body GLB ──────────────────────────────────────────
    print('[replace_head] Loading body GLB...')
    scene = trimesh.load(body_glb)
    if isinstance(scene, trimesh.Scene):
        body_mesh = trimesh.util.concatenate(
            [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        body_mesh = scene

    print(f'  Body: {len(body_mesh.vertices)} verts, {len(body_mesh.faces)} faces')

    # ── 1b. Weld body mesh (UV-split → geometric) ─────────────────
    print('[replace_head] Welding mesh vertices...')
    body_welded = _weld_mesh(body_mesh)
    print(f'  Welded: {len(body_welded.vertices)} verts (was {len(body_mesh.vertices)})')

    # ── 2. Find neck cut height ───────────────────────────────────
    neck_y = _find_neck_height(body_welded)

    # ── 3. Cut body at neck ───────────────────────────────────────
    print('[replace_head] Cutting body at neck...')
    # Work on welded mesh for topology; keep original mesh for geometry export
    body_lower_welded = _cut_mesh_below(body_welded, neck_y)
    body_lower = _cut_mesh_below(body_mesh, neck_y)  # keeps original UV/texture
    print(f'  Body lower: {len(body_lower.vertices)} verts')

    # Extract neck ring geometrically (robust for non-manifold UV-split meshes)
    body_neck_loop = _extract_neck_ring_geometric(body_welded, neck_y, n_pts=bridge_n)
    print(f'  Body neck ring: {len(body_neck_loop)} pts (geometric)')

    # ── 4. Run DECA ───────────────────────────────────────────────
    print('[replace_head] Running DECA...')
    verts_np, cam_np, faces_v, uv_verts, faces_uv, tex_bgr = run_deca(face_img_path, device=device)

    # ── 5. Align DECA head to body coordinate system ─────────────
    # TripoSG body is roughly in [-1,1] world space (y-up)
    # DECA/FLAME space: head centered around origin, scale ≈ 1.5-2.5 units for full head
    # We need to:
    #   a) Scale the FLAME head to match body scale
    #   b) Position the FLAME head so its neck base aligns with body neck ring

    # Get the bottom of the FLAME head (neck area)
    # FLAME template: bottom vertices are the neck boundary ring
    flame_mesh_tmp = __import__('trimesh').Trimesh(vertices=verts_np, faces=faces_v, process=False)
    try:
        flame_neck_loop = _extract_boundary_loop(flame_mesh_tmp)
        print(f'  FLAME neck ring (topology): {len(flame_neck_loop)} verts')
    except Exception as e:
        print(f'  FLAME boundary loop failed ({e}), using geometric extraction')
        # Geometric fallback: bottom 5% of head vertices
        flame_neck_y = verts_np[:, 1].min() + (verts_np[:, 1].max() - verts_np[:, 1].min()) * 0.08
        flame_neck_loop = _extract_neck_ring_geometric(flame_mesh_tmp, flame_neck_y, n_pts=bridge_n)
        print(f'  FLAME neck ring (geometric): {len(flame_neck_loop)} pts')

    # ── 5b. Compute head position using NECK RING centroid ───────────────
    # Directly align FLAME neck ring center → body neck ring center in all 3 axes.
    # This is robust regardless of body pose or tilt.
    body_neck_center = body_neck_loop.mean(axis=0)

    # Estimate head height from WELDED mesh crown (more reliable than UV-split mesh)
    welded_y_max = float(body_welded.vertices[:, 1].max())
    body_head_height = welded_y_max - neck_y

    flame_neck_center_unscaled = flame_neck_loop.mean(axis=0)
    flame_y_min = verts_np[:, 1].min()
    flame_y_max = verts_np[:, 1].max()
    flame_head_height = flame_y_max - flame_y_min

    print(f'  Body neck center: {body_neck_center.round(4)}')
    print(f'  Body head space: {body_head_height:.4f} (neck_y={neck_y:.4f}, crown_y={welded_y_max:.4f})')
    print(f'  FLAME head height (unscaled): {flame_head_height:.4f}')
    print(f'  FLAME neck center (unscaled): {flame_neck_center_unscaled.round(4)}')

    # Scale FLAME head to match body head height
    if flame_head_height > 1e-5:
        head_scale = body_head_height / flame_head_height
    else:
        head_scale = 1.0
    print(f'  Head scale: {head_scale:.4f}')

    # Translate: FLAME neck ring center → body neck ring center in XZ,
    # FLAME mesh bottom (flame_y_min) → neck_y in Y.
    # This ensures the head fills the full space from neck_y to body crown.
    translate = np.array([
        body_neck_center[0] - flame_neck_center_unscaled[0] * head_scale,
        neck_y              - flame_y_min                  * head_scale,
        body_neck_center[2] - flame_neck_center_unscaled[2] * head_scale,
    ])
    print(f'  Translate: {translate.round(4)}')
    verts_aligned = verts_np * head_scale + translate
    print(f'  FLAME aligned y={verts_aligned[:,1].min():.4f}→{verts_aligned[:,1].max():.4f}'
          f'  x={verts_aligned[:,0].min():.4f}→{verts_aligned[:,0].max():.4f}'
          f'  z={verts_aligned[:,2].min():.4f}→{verts_aligned[:,2].max():.4f}')

    # Extract FLAME neck loop after alignment (at the cut plane y=neck_y)
    flame_verts_aligned = verts_aligned
    flame_mesh_aligned  = __import__('trimesh').Trimesh(
        vertices=flame_verts_aligned, faces=faces_v, process=False)
    try:
        flame_neck_loop_aligned = _extract_boundary_loop(flame_mesh_aligned)
        print(f'  FLAME neck ring (topology): {len(flame_neck_loop_aligned)} verts')
    except Exception:
        flame_neck_y_aligned = flame_verts_aligned[:, 1].min() + (
            flame_verts_aligned[:, 1].max() - flame_verts_aligned[:, 1].min()) * 0.05
        flame_neck_loop_aligned = _extract_neck_ring_geometric(
            flame_mesh_aligned, flame_neck_y_aligned, n_pts=bridge_n)
        print(f'  FLAME neck ring (geometric): {len(flame_neck_loop_aligned)} pts')

    flame_neck_r = np.linalg.norm(flame_neck_loop_aligned - flame_neck_loop_aligned.mean(0), axis=1).mean()
    body_neck_r  = np.linalg.norm(body_neck_loop       - body_neck_loop.mean(0),       axis=1).mean()
    print(f'  Body neck radius: {body_neck_r:.4f}  FLAME neck radius (scaled): {flame_neck_r:.4f}')

    # ── 6. Resample both neck loops to bridge_n points ────────────
    body_loop_r  = _resample_loop(body_neck_loop, bridge_n)
    flame_loop_r = _resample_loop(flame_neck_loop_aligned, bridge_n)

    # Ensure loops are oriented consistently (both CW or both CCW)
    # Compute signed area to check orientation
    def _loop_orientation(loop):
        c = loop.mean(0)
        t = loop - c
        cross = np.cross(t[:-1], t[1:])
        return float(np.sum(cross[:, 1]))  # y-component

    o_body  = _loop_orientation(body_loop_r)
    o_flame = _loop_orientation(flame_loop_r)
    if (o_body > 0) != (o_flame > 0):
        flame_loop_r = flame_loop_r[::-1]

    # ── 7. Align loop starting points (minimise bridge twist) ─────
    # Match starting vertex: find flame loop point closest to body loop start
    dists = np.linalg.norm(flame_loop_r - body_loop_r[0], axis=1)
    best_offset = int(np.argmin(dists))
    flame_loop_r = np.roll(flame_loop_r, -best_offset, axis=0)

    # ── 8. Build bridge strip ─────────────────────────────────────
    bridge_verts, bridge_faces = _bridge_loops(body_loop_r, flame_loop_r)
    bridge_mesh = __import__('trimesh').Trimesh(vertices=bridge_verts, faces=bridge_faces, process=False)

    # ── 9. Combine: body_lower + bridge + FLAME head ──────────────
    # Build FLAME head mesh with texture
    head_mesh = deca_to_trimesh(flame_verts_aligned, faces_v, uv_verts, faces_uv, tex_bgr)

    # Combine all parts
    combined = __import__('trimesh').util.concatenate([body_lower, bridge_mesh, head_mesh])
    combined = __import__('trimesh').Trimesh(
        vertices=combined.vertices,
        faces=combined.faces,
        process=False
    )

    # Try to copy body texture to combined if available
    try:
        if hasattr(body_lower.visual, 'material'):
            pass  # Keep per-mesh materials — export as scene
    except Exception:
        pass

    # ── 10. Export ────────────────────────────────────────────────
    print(f'[replace_head] Exporting combined mesh: {len(combined.vertices)} verts...')
    os.makedirs(os.path.dirname(out_glb) or '.', exist_ok=True)

    # Export as GLB scene with separate submeshes (preserves textures)
    try:
        import trimesh
        scene_out = trimesh.Scene()
        scene_out.add_geometry(body_lower, geom_name='body')
        scene_out.add_geometry(bridge_mesh, geom_name='bridge')
        scene_out.add_geometry(head_mesh, geom_name='head')
        scene_out.export(out_glb)
        print(f'[replace_head] Saved scene GLB: {out_glb}  ({os.path.getsize(out_glb)//1024} KB)')
    except Exception as e:
        print(f'[replace_head] Scene export failed ({e}), trying single mesh...')
        combined.export(out_glb)
        print(f'[replace_head] Saved GLB: {out_glb}  ({os.path.getsize(out_glb)//1024} KB)')

    return out_glb


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--body',   required=True, help='TripoSG body GLB path')
    ap.add_argument('--face',   required=True, help='Reference face image path')
    ap.add_argument('--out',    required=True, help='Output GLB path')
    ap.add_argument('--bridge', type=int, default=64, help='Bridge ring vertex count')
    ap.add_argument('--cpu',    action='store_true', help='Use CPU instead of CUDA')
    args = ap.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    replace_head(args.body, args.face, args.out, device=device, bridge_n=args.bridge)
