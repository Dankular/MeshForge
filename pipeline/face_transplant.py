"""
face_transplant.py
==================
Replace the face/head region of a rigged UniRig GLB with a higher-detail
PSHuman mesh, while preserving the skeleton, rig, and skinning weights.

Algorithm
---------
1. Parse rigged GLB  → vertices, faces, UVs, JOINTS_0, WEIGHTS_0, bone list
2. Identify head vertices  →  any vert whose dominant bone is in HEAD_BONES
3. Load PSHuman mesh (OBJ or GLB, no rig)
4. Align PSHuman head to UniRig head bounding box  (scale + translate)
5. Transfer skinning weights to PSHuman verts via K-nearest-neighbour from
   UniRig head verts  (scipy KDTree, weighted average)
6. Retract UniRig face verts slightly inward so PSHuman sits on top cleanly
7. Rebuild the GLB with two mesh primitives:
     - Primitive 0  : UniRig body  (face verts retracted)
     - Primitive 1  : PSHuman face (new, with transferred weights)
8. Write output GLB

Usage
-----
python -m pipeline.face_transplant \\
    --body   rigged_body.glb \\
    --face   pshuman_output.obj \\
    --output rigged_body_with_pshuman_face.glb

Optionally supply --head-bones as comma-separated bone-name substrings
(default: head,Head,skull).  Any bone whose name contains one of these
substrings is treated as a head bone.

Requires:  pygltflib  numpy  scipy  trimesh  (pip install each)
"""

from __future__ import annotations

import argparse
import base64
import struct
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree
import trimesh


# ---------------------------------------------------------------------------
# GLB low-level helpers (subset of Retarget/io/gltf_io.py re-used here)
# ---------------------------------------------------------------------------

try:
    import pygltflib
except ImportError:
    raise ImportError("pip install pygltflib")


def _read_accessor_raw(gltf: pygltflib.GLTF2, accessor_idx: int) -> np.ndarray:
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[bv.buffer]

    if buf.uri and buf.uri.startswith("data:"):
        _, b64 = buf.uri.split(",", 1)
        raw = base64.b64decode(b64)
    elif buf.uri:
        base_dir = (
            Path(gltf._path).parent if getattr(gltf, "_path", None) else Path(".")
        )
        raw = (base_dir / buf.uri).read_bytes()
    else:
        raw = bytes(gltf.binary_blob())

    type_nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    fmt_map = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}

    n_comp = type_nc[acc.type]
    fmt = fmt_map[acc.componentType]
    item_sz = struct.calcsize(fmt) * n_comp
    stride = bv.byteStride or item_sz
    start = bv.byteOffset + (acc.byteOffset or 0)

    items = []
    for i in range(acc.count):
        offset = start + i * stride
        vals = struct.unpack_from(f"{n_comp}{fmt}", raw, offset)
        items.append(vals)

    arr = np.array(items)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def _accessor_dtype(gltf: pygltflib.GLTF2, accessor_idx: int):
    fmt_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    return fmt_map[gltf.accessors[accessor_idx].componentType]


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------


class GLBMesh:
    """
    All data from the first skin's first mesh primitive in a GLB.
    """

    def __init__(self, path: str):
        self.path = path
        gltf = pygltflib.GLTF2().load(path)
        gltf._path = path
        self.gltf = gltf

        if not gltf.skins:
            raise ValueError("No skin found in GLB — is this a rigged file?")
        self.skin = gltf.skins[0]
        self.joint_names: List[str] = [
            gltf.nodes[j].name or f"joint_{k}" for k, j in enumerate(self.skin.joints)
        ]

        # Find a mesh node that uses this skin
        self.mesh_prim, self.mesh_node_idx = self._find_skinned_prim()

        attrs = self.mesh_prim.attributes
        self.verts = _read_accessor_raw(gltf, attrs.POSITION).astype(np.float32)
        self.normals = (
            _read_accessor_raw(gltf, attrs.NORMAL).astype(np.float32)
            if attrs.NORMAL is not None
            else None
        )
        self.uvs = (
            _read_accessor_raw(gltf, attrs.TEXCOORD_0).astype(np.float32)
            if attrs.TEXCOORD_0 is not None
            else None
        )
        self.faces = (
            _read_accessor_raw(gltf, self.mesh_prim.indices)
            .astype(np.int32)
            .reshape(-1, 3)
        )

        # Skinning — may be JOINTS_0 / WEIGHTS_0 (uint8/uint16 + float)
        self.joints4 = None
        self.weights4 = None
        if attrs.JOINTS_0 is not None:
            self.joints4 = _read_accessor_raw(gltf, attrs.JOINTS_0).astype(np.int32)
            self.weights4 = _read_accessor_raw(gltf, attrs.WEIGHTS_0).astype(np.float32)

        # Material index (carry over to output)
        self.material_idx = self.mesh_prim.material

    def _find_skinned_prim(self):
        skin_node_indices = set(self.skin.joints)
        # find mesh node that references this skin
        for ni, node in enumerate(self.gltf.nodes):
            if node.skin == 0 and node.mesh is not None:
                mesh = self.gltf.meshes[node.mesh]
                return mesh.primitives[0], ni
        # fallback: first mesh node
        for ni, node in enumerate(self.gltf.nodes):
            if node.mesh is not None:
                mesh = self.gltf.meshes[node.mesh]
                return mesh.primitives[0], ni
        raise ValueError("No mesh primitive found")

    def head_bone_indices(
        self, substrings=("head", "Head", "skull", "Skull", "neck", "Neck")
    ) -> List[int]:
        """Return joint indices (into self.joint_names) matching any substring.
        Falls back to positional heuristic (highest-Y dominant bone) when no
        bone names match (e.g. generic bone_0/bone_1 naming from UniRig)."""
        result = []
        for i, name in enumerate(self.joint_names):
            if any(s in name for s in substrings):
                result.append(i)
        if not result and self.joints4 is not None and self.weights4 is not None:
            # Positional fallback: pick bone whose dominant vertices have highest avg Y.
            n_bones = len(self.joint_names)
            bone_y_sum = np.zeros(n_bones)
            bone_y_cnt = np.zeros(n_bones, dtype=np.int32)
            for vi in range(len(self.verts)):
                dom = int(self.joints4[vi, np.argmax(self.weights4[vi])])
                bone_y_sum[dom] += self.verts[vi, 1]
                bone_y_cnt[dom] += 1
            with np.errstate(invalid="ignore"):
                bone_y_avg = np.where(bone_y_cnt > 0, bone_y_sum / bone_y_cnt, -np.inf)
            top = int(np.argmax(bone_y_avg))
            print(
                f"[face_transplant] No named head bones; positional fallback: "
                f"bone {top} ({self.joint_names[top]}, avg_y={bone_y_avg[top]:.3f})"
            )
            result = [top]
        return result


# ---------------------------------------------------------------------------
# Face-region identification
# ---------------------------------------------------------------------------


def find_face_verts(
    glb_mesh: GLBMesh, head_joint_indices: List[int], weight_threshold: float = 0.35
) -> np.ndarray:
    """
    Return boolean mask of face/head vertices:
    any vert whose total weight on head joints exceeds weight_threshold.
    """
    if glb_mesh.joints4 is None:
        raise ValueError("Mesh has no skinning weights — cannot identify face region")

    n = len(glb_mesh.verts)
    mask = np.zeros(n, dtype=bool)
    head_set = set(head_joint_indices)

    for vi in range(n):
        total_head_w = 0.0
        for c in range(4):
            j = glb_mesh.joints4[vi, c]
            w = glb_mesh.weights4[vi, c]
            if j in head_set:
                total_head_w += w
        if total_head_w >= weight_threshold:
            mask[vi] = True

    return mask


# ---------------------------------------------------------------------------
# PSHuman mesh loading + alignment
# ---------------------------------------------------------------------------


def _crop_to_head(
    mesh: trimesh.Trimesh, head_fraction: float = 0.17
) -> trimesh.Trimesh:
    """
    Keep only the top head_fraction of the PSHuman body mesh by Y coordinate,
    then apply a secondary X-width restriction to exclude T-pose arms that sit
    at the same Y level as the neck/shoulders.
    0.17 = top 17% ≈ head only; 0.22 was too large and included shoulder content.
    """
    y = mesh.vertices[:, 1]
    threshold = y.max() - (y.max() - y.min()) * head_fraction
    vert_keep_y = y >= threshold

    # T-pose arms extend horizontally at shoulder/neck Y level.
    # Restrict X to ±(head_height * 0.7) from the X center so arms are excluded.
    y_range_kept = y[vert_keep_y]
    head_height = (
        float(y_range_kept.max() - y_range_kept.min()) if len(y_range_kept) > 0 else 0.3
    )
    x_center = float(mesh.vertices[vert_keep_y, 0].mean()) if vert_keep_y.any() else 0.0
    x_half = head_height * 0.75  # allow slightly more than head width for ears/hair
    vert_keep_x = np.abs(mesh.vertices[:, 0] - x_center) <= x_half

    vert_keep = vert_keep_y & vert_keep_x
    n_y = int(vert_keep_y.sum())
    n_both = int(vert_keep.sum())
    print(
        f"[face_transplant] PSHuman crop: Y-threshold={threshold:.3f} kept {n_y} verts; "
        f"X±{x_half:.3f} from {x_center:.3f} further reduced to {n_both} verts"
    )
    face_keep = vert_keep[mesh.faces].all(axis=1)
    kept_faces = mesh.faces[face_keep]
    used = np.unique(kept_faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    new_verts = mesh.vertices[used].astype(np.float32)
    new_faces = remap[kept_faces]
    result = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        result.visual = trimesh.visual.TextureVisuals(uv=np.array(mesh.visual.uv)[used])
    else:
        # Preserve vertex colors (PSHuman outputs v x y z r g b OBJ format)
        try:
            vc = np.array(mesh.visual.vertex_colors)
            if vc is not None and len(vc) == len(mesh.vertices):
                result.visual = trimesh.visual.ColorVisuals(
                    mesh=result, vertex_colors=vc[used]
                )
        except Exception:
            pass
    print(
        f"[face_transplant] PSHuman head crop ({head_fraction * 100:.0f}%): "
        f"{len(mesh.vertices)} → {len(new_verts)} verts  (Y ≥ {threshold:.3f})"
    )
    return result


def load_and_align_pshuman(
    pshuman_path: str, target_verts: np.ndarray
) -> trimesh.Trimesh:
    """
    Load PSHuman mesh (OBJ/GLB/PLY), crop to head region, then scale+translate
    to fit the bounding box of target_verts (UniRig head verts).
    """
    mesh: trimesh.Trimesh = trimesh.load(pshuman_path, force="mesh", process=False)
    print(
        f"[face_transplant] PSHuman mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces"
    )

    # PSHuman is full-body — crop to just the head before aligning
    mesh = _crop_to_head(mesh)

    # Target bbox from UniRig head region
    tgt_min = target_verts.min(axis=0)
    tgt_max = target_verts.max(axis=0)
    tgt_ctr = (tgt_min + tgt_max) * 0.5
    tgt_ext = tgt_max - tgt_min

    src_min = mesh.vertices.min(axis=0).astype(np.float32)
    src_max = mesh.vertices.max(axis=0).astype(np.float32)
    src_ctr = (src_min + src_max) * 0.5
    src_ext = src_max - src_min

    # Uniform scale: match the largest axis of the target
    dominant = np.argmax(tgt_ext)
    scale = float(tgt_ext[dominant]) / float(src_ext[dominant] + 1e-9)

    verts = mesh.vertices.astype(np.float32).copy()
    verts = (verts - src_ctr) * scale + tgt_ctr

    mesh.vertices = verts
    print(f"[face_transplant] PSHuman aligned: scale={scale:.4f}, center={tgt_ctr}")
    return mesh


# ---------------------------------------------------------------------------
# Weight transfer via KDTree
# ---------------------------------------------------------------------------


def transfer_weights(
    donor_verts: np.ndarray,  # (M, 3) UniRig face verts
    donor_joints: np.ndarray,  # (M, 4) uint16
    donor_weights: np.ndarray,  # (M, 4) float32
    recipient_verts: np.ndarray,  # (N, 3) PSHuman face verts
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-nearest-neighbour weight transfer.
    Returns (joints4, weights4) for recipient_verts.
    """
    tree = KDTree(donor_verts)
    dists, idxs = tree.query(recipient_verts, k=k)  # (N, k)

    N = len(recipient_verts)
    n_joints_total = int(donor_joints.max()) + 1

    # Build dense per-recipient weight vector
    dense = np.zeros((N, n_joints_total), dtype=np.float64)
    for ki in range(k):
        w_dist = 1.0 / (dists[:, ki] + 1e-8)  # inverse-distance
        for vi in range(N):
            di = idxs[vi, ki]
            for c in range(4):
                j = donor_joints[di, c]
                w = donor_weights[di, c]
                dense[vi, j] += w * w_dist[vi]

    # Re-normalise rows
    row_sum = dense.sum(axis=1, keepdims=True) + 1e-12
    dense /= row_sum

    # Pack back into 4-bone format (top-4 by weight)
    out_joints = np.zeros((N, 4), dtype=np.uint16)
    out_weights = np.zeros((N, 4), dtype=np.float32)
    for vi in range(N):
        top4 = np.argsort(dense[vi])[-4:][::-1]
        total = dense[vi, top4].sum() + 1e-12
        for c, j in enumerate(top4):
            out_joints[vi, c] = j
            out_weights[vi, c] = dense[vi, j] / total

    return out_joints, out_weights


# ---------------------------------------------------------------------------
# Neck bridge helpers
# ---------------------------------------------------------------------------


def _extract_boundary_loop_indices(mesh: trimesh.Trimesh):
    """Return ordered boundary vertex indices, or None if mesh is closed."""
    edges = np.sort(mesh.edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    if len(boundary_edges) == 0:
        return None
    adj = {}
    for e in boundary_edges:
        adj.setdefault(int(e[0]), []).append(int(e[1]))
        adj.setdefault(int(e[1]), []).append(int(e[0]))
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
    best = max(loops, key=len)
    return np.array(best, dtype=np.int32)


def _resample_loop(loop_pts: np.ndarray, N: int) -> np.ndarray:
    """Resample ordered 3D points to N evenly-spaced points."""
    from scipy.interpolate import interp1d

    diffs = np.diff(loop_pts, axis=0, prepend=loop_pts[-1:])
    seg_lens = np.linalg.norm(diffs, axis=1)
    t = np.cumsum(seg_lens)
    t = np.insert(t, 0, 0)
    t /= t[-1]
    t[-1] = 1.0
    loop_closed = np.vstack([loop_pts, loop_pts[0]])
    interp = interp1d(t, loop_closed, axis=0, kind="linear")
    t_new = np.linspace(0, 1, N, endpoint=False)
    return interp(t_new)


def _laplacian_smooth_verts(
    V: np.ndarray, F: np.ndarray, blend_indices: np.ndarray, iterations: int = 25
):
    """Umbrella smoothing on blend_indices; other verts stay fixed."""
    adj = {}
    for f in F:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj.setdefault(int(f[i]), set()).add(int(f[j]))
    blend_set = set(int(x) for x in blend_indices)
    for _ in range(iterations):
        new_V = V.copy()
        for vi in blend_set:
            nbrs = list(adj.get(vi, set()))
            if nbrs:
                new_V[vi] = V[nbrs].mean(axis=0)
        V = new_V
    return V


# ---------------------------------------------------------------------------
# GLB rebuild
# ---------------------------------------------------------------------------


def _pack_buffer_view(
    data_bytes: bytes, target: list, byte_offset: int, byte_stride: Optional[int] = None
) -> Tuple[int, int]:
    """
    Append data_bytes to target buffer, return (buffer_view_index, new_offset).
    """
    bv = pygltflib.BufferView(
        buffer=0,
        byteOffset=byte_offset,
        byteLength=len(data_bytes),
    )
    if byte_stride:
        bv.byteStride = byte_stride
    return bv, byte_offset + len(data_bytes)


def _make_accessor(
    component_type: int,
    type_str: str,
    count: int,
    bv_idx: int,
    min_vals=None,
    max_vals=None,
) -> pygltflib.Accessor:
    acc = pygltflib.Accessor(
        bufferView=bv_idx,
        byteOffset=0,
        componentType=component_type,
        count=count,
        type=type_str,
    )
    if min_vals is not None:
        acc.min = [float(v) for v in min_vals]
    if max_vals is not None:
        acc.max = [float(v) for v in max_vals]
    return acc


FLOAT32 = pygltflib.FLOAT  # 5126
UINT16 = pygltflib.UNSIGNED_SHORT  # 5123
UINT32 = pygltflib.UNSIGNED_INT  # 5125
UBYTE = pygltflib.UNSIGNED_BYTE  # 5121


def transplant_face(
    body_glb_path: str,
    pshuman_mesh_path: str,
    output_path: str,
    head_bone_substrings: Tuple[str, ...] = ("head", "Head", "skull", "Skull"),
    weight_threshold: float = 0.35,
    retract_amount: float = 0.004,  # metres — how far to push face verts inward
    knn: int = 5,
):
    """
    Main entry point.

    Parameters
    ----------
    body_glb_path          : rigged UniRig GLB
    pshuman_mesh_path      : PSHuman output mesh (OBJ / GLB / PLY)
    output_path            : result GLB path
    head_bone_substrings   : bone name fragments that identify head joints
    weight_threshold       : head-weight sum above which a vertex is "face"
    retract_amount         : metres to push face verts inward to avoid z-fight
    knn                    : neighbours for weight transfer
    """
    print(f"[face_transplant] Loading rigged GLB: {body_glb_path}")
    glb = GLBMesh(body_glb_path)
    print(f"  Verts: {len(glb.verts)}  Faces: {len(glb.faces)}")
    print(f"  Bones ({len(glb.joint_names)}): {', '.join(glb.joint_names[:8])} ...")

    # 1. Identify head joints
    head_ji = glb.head_bone_indices(substrings=head_bone_substrings)
    if not head_ji:
        raise RuntimeError(
            f"No head bones found with substrings {head_bone_substrings}.\n"
            f"Available bones: {glb.joint_names}"
        )
    print(f"  Head joints ({len(head_ji)}): {[glb.joint_names[i] for i in head_ji]}")

    # 2. Find face/head vertices
    face_mask = find_face_verts(glb, head_ji, weight_threshold=weight_threshold)
    print(f"  Face verts: {face_mask.sum()} / {len(glb.verts)}")

    min_face_verts = max(3, min(10, len(glb.verts) // 4))
    if face_mask.sum() < min_face_verts:
        raise RuntimeError(
            f"Only {face_mask.sum()} face vertices found (need >= {min_face_verts}) — "
            f"try lowering --weight-threshold (current: {weight_threshold})"
        )

    # 3. Load + align PSHuman mesh
    face_verts_ur = glb.verts[face_mask]
    ps_mesh = load_and_align_pshuman(pshuman_mesh_path, face_verts_ur)

    # Post-alignment bbox crop: restrict PSHuman to TripoSG face_mask bounding box
    # + small margin for hair/ears.  Grounded in diagnostic data:
    #   face_mask X: -0.133..0.088, PSHuman X after alignment: -0.248..0.208
    #   → ~0.11m shoulder overhang on each side creates visible artifacts.
    face_bb_min = face_verts_ur.min(axis=0)
    face_bb_max = face_verts_ur.max(axis=0)
    x_margin = 0.04  # 4 cm lateral for ears
    y_margin = 0.02  # 2 cm headroom above; -0.04 below (clamp at shoulder to avoid arm overlap)
    z_margin = 0.10  # 10 cm front-to-back (face has depth)
    # Pull the bottom of PSHuman up by 4 cm to avoid overlapping with the T-pose arm area
    y_min_override = face_bb_min[1] + 0.04
    ps_v_pre = np.array(ps_mesh.vertices, dtype=np.float32)
    in_bb = (
        (ps_v_pre[:, 0] >= face_bb_min[0] - x_margin)
        & (ps_v_pre[:, 0] <= face_bb_max[0] + x_margin)
        & (ps_v_pre[:, 1] >= y_min_override)
        & (ps_v_pre[:, 1] <= face_bb_max[1] + y_margin)
        & (ps_v_pre[:, 2] >= face_bb_min[2] - z_margin)
        & (ps_v_pre[:, 2] <= face_bb_max[2] + z_margin)
    )
    keep_f_bb = in_bb[ps_mesh.faces].all(axis=1)
    kept_f_bb = ps_mesh.faces[keep_f_bb]
    used_bb = np.unique(kept_f_bb)
    remap_bb = np.full(len(ps_v_pre), -1, dtype=np.int32)
    remap_bb[used_bb] = np.arange(len(used_bb))
    bb_verts = ps_v_pre[used_bb]
    bb_faces = remap_bb[kept_f_bb]
    print(
        f"[face_transplant] Post-alignment bbox crop: {len(ps_v_pre)} → {len(bb_verts)} verts "
        f"(X±{x_margin * 100:.0f}cm, Y±{y_margin * 100:.0f}cm, Z±{z_margin * 100:.0f}cm from face bbox)"
    )
    try:
        vc_pre = np.array(ps_mesh.visual.vertex_colors)
        ps_mesh_bb = trimesh.Trimesh(vertices=bb_verts, faces=bb_faces, process=False)
        if vc_pre is not None and len(vc_pre) == len(ps_v_pre):
            ps_mesh_bb.visual = trimesh.visual.ColorVisuals(
                mesh=ps_mesh_bb, vertex_colors=vc_pre[used_bb]
            )
        ps_mesh = ps_mesh_bb
    except Exception as e:
        print(f"[face_transplant] Post-crop color preservation failed: {e}")
        ps_mesh = trimesh.Trimesh(vertices=bb_verts, faces=bb_faces, process=False)

    ps_verts = np.array(ps_mesh.vertices, dtype=np.float32)
    ps_faces = np.array(ps_mesh.faces, dtype=np.int32)
    ps_uvs = None
    ps_colors = None
    if hasattr(ps_mesh.visual, "uv") and ps_mesh.visual.uv is not None:
        ps_uvs = np.array(ps_mesh.visual.uv, dtype=np.float32)
    else:
        # PSHuman uses vertex colors — extract RGBA float (0-1)
        try:
            vc = np.array(ps_mesh.visual.vertex_colors, dtype=np.float32)
            if vc is not None and len(vc) == len(ps_verts):
                if vc.max() > 1.5:  # uint8 range 0-255 → normalise
                    vc = vc / 255.0
                if vc.shape[1] == 3:  # RGB → RGBA
                    vc = np.concatenate(
                        [vc, np.ones((len(vc), 1), dtype=np.float32)], axis=1
                    )
                ps_colors = vc[:, :4].astype(np.float32)
                print(
                    f"[face_transplant] PSHuman vertex colors: {len(ps_colors)} verts"
                )
        except Exception as e:
            print(f"[face_transplant] Could not extract vertex colors: {e}")

    # 4. Transfer weights: donor = UniRig face verts, recipient = PSHuman verts
    print("[face_transplant] Transferring skinning weights via KNN ...")
    ps_joints, ps_weights = transfer_weights(
        donor_verts=glb.verts[face_mask].astype(np.float64),
        donor_joints=glb.joints4[face_mask],
        donor_weights=glb.weights4[face_mask],
        recipient_verts=ps_verts.astype(np.float64),
        k=knn,
    )
    print(
        f"  Done. Head joint coverage in PSHuman: "
        f"{(np.isin(ps_joints[:, 0], head_ji)).mean() * 100:.1f}% primary bone is head"
    )

    # 5. Remove body triangles in the face region so the PSHuman face shows through.
    #    Diagnostic from isolated render confirmed PSHuman face Z-max (0.052) is
    #    behind the TripoSG face even after 4mm retraction — the TripoSG face was
    #    occluding it. Deleting interior face-region triangles creates a real hole.
    body_verts = glb.verts.copy()
    body_faces = glb.faces.copy()

    # Remove triangles where ALL 3 verts are face verts (interior of head region).
    # Boundary triangles (mixed face/body verts) are kept to avoid seam gaps.
    interior = (
        face_mask[body_faces[:, 0]]
        & face_mask[body_faces[:, 1]]
        & face_mask[body_faces[:, 2]]
    )
    body_faces_trimmed = body_faces[~interior]
    n_removed = int(interior.sum())
    print(
        f"[face_transplant] Removed {n_removed} body face-region triangles "
        f"(of {len(body_faces)}) to expose hole for PSHuman face"
    )

    # Retract only interior face verts (no faces remain on them anyway).
    # Boundary verts are kept at original position for bridge attachment.
    is_boundary_face = (
        face_mask[body_faces_trimmed[:, 0]]
        | face_mask[body_faces_trimmed[:, 1]]
        | face_mask[body_faces_trimmed[:, 2]]
    )
    boundary_vert_candidates = np.unique(body_faces_trimmed[is_boundary_face].flatten())
    is_boundary_vert = np.zeros(len(body_verts), dtype=bool)
    is_boundary_vert[boundary_vert_candidates] = True
    seam_mask = face_mask & ~is_boundary_vert
    if glb.normals is not None and seam_mask.any():
        body_verts[seam_mask] -= glb.normals[seam_mask] * retract_amount
    print(
        f"[face_transplant] Retracted {seam_mask.sum()} interior face verts by {retract_amount * 1000:.1f}mm"
    )

    # --- NECK BRIDGE: close gap between body boundary and PSHuman boundary ---
    body_mesh_tmp = trimesh.Trimesh(
        vertices=body_verts, faces=body_faces_trimmed, process=False
    )
    body_loop_idx = _extract_boundary_loop_indices(body_mesh_tmp)
    ps_loop_idx = _extract_boundary_loop_indices(ps_mesh)

    body_normals = glb.normals
    body_uvs = glb.uvs
    body_joints = glb.joints4
    body_weights = glb.weights4

    if (
        body_loop_idx is not None
        and ps_loop_idx is not None
        and len(body_loop_idx) >= 6
        and len(ps_loop_idx) >= 6
    ):
        N = len(body_loop_idx)
        print(
            f"[face_transplant] Building neck bridge: body boundary={N} ps boundary={len(ps_loop_idx)}"
        )

        body_loop = body_verts[body_loop_idx]
        ps_loop = ps_mesh.vertices[ps_loop_idx]

        # Resample PSHuman boundary to match body boundary count
        ps_loop_r = _resample_loop(ps_loop, N)

        # Orient consistently
        def _loop_orientation(loop):
            c = loop.mean(0)
            t = loop - c
            cross = np.cross(t[:-1], t[1:])
            return float(np.sum(cross[:, 1]))

        if (_loop_orientation(body_loop) > 0) != (_loop_orientation(ps_loop_r) > 0):
            ps_loop_r = ps_loop_r[::-1]

        # Align starting points
        dists = np.linalg.norm(ps_loop_r - body_loop[0], axis=1)
        best_offset = int(np.argmin(dists))
        ps_loop_r = np.roll(ps_loop_r, -best_offset, axis=0)

        # Build bridge faces: reuse existing body boundary verts, add new PSHuman-side verts
        n_body = len(body_verts)
        bridge_verts = ps_loop_r.astype(np.float32)
        bridge_faces = []
        for i in range(N):
            j = (i + 1) % N
            ai = int(body_loop_idx[i])
            aj = int(body_loop_idx[j])
            bi = n_body + i
            bj = n_body + j
            bridge_faces.append([ai, aj, bi])
            bridge_faces.append([aj, bj, bi])
        bridge_faces = np.array(bridge_faces, dtype=np.int32)

        # Transfer skinning to bridge verts from nearest original PSHuman boundary vert
        tree_ps = KDTree(ps_loop)
        _, nearest_ps = tree_ps.query(ps_loop_r, k=1)
        bridge_joints = ps_joints[ps_loop_idx[nearest_ps]]
        bridge_weights = ps_weights[ps_loop_idx[nearest_ps]]

        # Merge bridge into body
        body_verts = np.vstack([body_verts, bridge_verts])
        body_faces = np.vstack([body_faces_trimmed, bridge_faces])

        # Extend body attribute arrays for bridge verts
        if glb.joints4 is not None:
            body_joints = np.vstack([glb.joints4, bridge_joints])
            body_weights = np.vstack([glb.weights4, bridge_weights])

        # Laplacian smooth bridge verts
        bridge_vert_indices = np.arange(n_body, n_body + N)
        body_verts = _laplacian_smooth_verts(
            body_verts, body_faces, bridge_vert_indices, iterations=25
        )

        # Recompute normals on combined mesh
        combined_tmp = trimesh.Trimesh(
            vertices=body_verts, faces=body_faces, process=False
        )
        combined_tmp.fix_normals()
        body_normals = combined_tmp.vertex_normals.astype(np.float32)

        if glb.uvs is not None:
            tree_body = KDTree(body_loop)
            _, nearest_body = tree_body.query(ps_loop_r, k=1)
            bridge_uvs = glb.uvs[body_loop_idx[nearest_body]]
            body_uvs = np.vstack([glb.uvs, bridge_uvs])

        print(
            f"[face_transplant] Bridge merged: {len(body_verts)} verts, {len(body_faces)} faces"
        )
    else:
        print(
            f"[face_transplant] WARNING: Could not extract boundary loops "
            f"(body={body_loop_idx is not None}, ps={ps_loop_idx is not None}) — skipping bridge"
        )

    # 6. Rebuild GLB
    print("[face_transplant] Rebuilding GLB ...")
    _write_transplanted_glb(
        source_gltf=glb,
        body_verts=body_verts,
        body_faces=body_faces,
        ps_verts=ps_verts,
        ps_faces=ps_faces,
        ps_uvs=ps_uvs,
        ps_colors=ps_colors,
        ps_joints=ps_joints,
        ps_weights=ps_weights,
        output_path=output_path,
        body_normals=body_normals,
        body_uvs=body_uvs,
        body_joints=body_joints,
        body_weights=body_weights,
    )
    print(f"[face_transplant] Saved -> {output_path}")


# ---------------------------------------------------------------------------
# GLB writer
# ---------------------------------------------------------------------------


def _write_transplanted_glb(
    source_gltf: GLBMesh,
    body_verts: np.ndarray,
    body_faces: np.ndarray,
    ps_verts: np.ndarray,
    ps_faces: np.ndarray,
    ps_uvs: Optional[np.ndarray],
    ps_colors: Optional[np.ndarray],
    ps_joints: np.ndarray,
    ps_weights: np.ndarray,
    output_path: str,
    body_normals: Optional[np.ndarray] = None,
    body_uvs: Optional[np.ndarray] = None,
    body_joints: Optional[np.ndarray] = None,
    body_weights: Optional[np.ndarray] = None,
):
    """
    Copy the source GLB structure, replace mesh primitive 0 vertex data,
    and append a new primitive for the PSHuman face.
    """
    import copy

    gltf = pygltflib.GLTF2().load(source_gltf.path)
    gltf._path = source_gltf.path

    # ------------------------------------------------------------------
    # Read the original binary blob ONCE so we can extract embedded data.
    # ------------------------------------------------------------------
    try:
        blob = bytes(gltf.binary_blob())
    except Exception:
        blob = b""

    # Collect embedded image bytes BEFORE we wipe buffer views.
    # We re-embed them into the rebuilt binary blob (NOT as data URIs — model-viewer
    # handles a 17 MB binary image fine but struggles with a 23 MB base64 string).
    saved_images: List[Tuple[object, bytes]] = []  # (img_node, raw_bytes)
    for img in gltf.images:
        if img.bufferView is not None and img.uri is None and blob:
            bv = gltf.bufferViews[img.bufferView]
            img_bytes = blob[bv.byteOffset : bv.byteOffset + bv.byteLength]
            saved_images.append((img, img_bytes))
            img.bufferView = None  # will be assigned a new index after buffer rebuild

    # Extract inverse bind matrices BEFORE wiping accessors.
    # CRITICAL: the skin's inverseBindMatrices accessor must be preserved —
    # without it model-viewer applies identity matrices and the mesh explodes.
    ibm_bytes = None
    ibm_count = 0
    if gltf.skins:
        skin0 = gltf.skins[0]
        if skin0.inverseBindMatrices is not None:
            ibm_acc = gltf.accessors[skin0.inverseBindMatrices]
            ibm_bv = gltf.bufferViews[ibm_acc.bufferView]
            ibm_off = ibm_bv.byteOffset + (ibm_acc.byteOffset or 0)
            ibm_count = ibm_acc.count
            ibm_bytes = blob[
                ibm_off : ibm_off + ibm_count * 64
            ]  # MAT4 = 16 × float32 = 64 bytes
            print(
                f"[face_transplant] Saved {ibm_count} inverse-bind matrices ({len(ibm_bytes)} bytes)"
            )

    # ------------------------------------------------------------------
    # We will rebuild the entire binary buffer from scratch.
    # Collect all data chunks; track buffer views + accessors.
    # ------------------------------------------------------------------
    chunks: List[bytes] = []
    bviews: List[pygltflib.BufferView] = []
    accors: List[pygltflib.Accessor] = []
    byte_offset = 0

    def add_chunk(
        data: bytes,
        component_type: int,
        type_str: str,
        count: int,
        min_v=None,
        max_v=None,
        stride: int = None,
    ) -> int:
        """Append data, create buffer view + accessor, return accessor index."""
        nonlocal byte_offset
        bv = pygltflib.BufferView(
            buffer=0, byteOffset=byte_offset, byteLength=len(data)
        )
        if stride:
            bv.byteStride = stride
        bviews.append(bv)
        bv_idx = len(bviews) - 1

        acc = pygltflib.Accessor(
            bufferView=bv_idx,
            byteOffset=0,
            componentType=component_type,
            count=count,
            type=type_str,
        )
        if min_v is not None:
            acc.min = [float(x) for x in np.atleast_1d(min_v)]
        if max_v is not None:
            acc.max = [float(x) for x in np.atleast_1d(max_v)]
        accors.append(acc)
        acc_idx = len(accors) - 1

        chunks.append(data)
        byte_offset += len(data)
        return acc_idx

    # ------------------------------------------------------------------
    # Primitive 0 — UniRig body (retracted face verts)
    # ------------------------------------------------------------------
    body_v = body_verts.astype(np.float32)
    body_i = body_faces.astype(np.uint32).flatten()
    body_n = (
        body_normals.astype(np.float32)
        if body_normals is not None
        else (
            source_gltf.normals.astype(np.float32)
            if source_gltf.normals is not None
            else None
        )
    )
    body_uv = (
        body_uvs.astype(np.float32)
        if body_uvs is not None
        else (
            source_gltf.uvs.astype(np.float32) if source_gltf.uvs is not None else None
        )
    )
    body_j = (
        body_joints.astype(np.uint16)
        if body_joints is not None
        else source_gltf.joints4.astype(np.uint16)
    )
    body_w = (
        body_weights.astype(np.float32)
        if body_weights is not None
        else source_gltf.weights4.astype(np.float32)
    )

    # indices
    bi_idx = add_chunk(
        body_i.tobytes(),
        UINT32,
        "SCALAR",
        len(body_i),
        min_v=[int(body_i.min())],
        max_v=[int(body_i.max())],
    )
    # positions
    bv_idx = add_chunk(
        body_v.tobytes(),
        FLOAT32,
        "VEC3",
        len(body_v),
        min_v=body_v.min(axis=0),
        max_v=body_v.max(axis=0),
    )
    body_attrs = pygltflib.Attributes(POSITION=bv_idx)
    if body_n is not None:
        body_attrs.NORMAL = add_chunk(body_n.tobytes(), FLOAT32, "VEC3", len(body_n))
    if body_uv is not None:
        body_attrs.TEXCOORD_0 = add_chunk(
            body_uv.tobytes(), FLOAT32, "VEC2", len(body_uv)
        )
    if body_j is not None:
        body_attrs.JOINTS_0 = add_chunk(body_j.tobytes(), UINT16, "VEC4", len(body_j))
        body_attrs.WEIGHTS_0 = add_chunk(body_w.tobytes(), FLOAT32, "VEC4", len(body_w))

    prim0 = pygltflib.Primitive(
        attributes=body_attrs,
        indices=bi_idx,
        material=source_gltf.material_idx,
        mode=4,  # TRIANGLES
    )

    # ------------------------------------------------------------------
    # Primitive 1 — PSHuman face
    # ------------------------------------------------------------------
    ps_v = ps_verts.astype(np.float32)
    ps_i = ps_faces.astype(np.uint32).flatten()
    ps_j4 = ps_joints.astype(np.uint16)
    ps_w4 = ps_weights.astype(np.float32)

    fi_idx = add_chunk(
        ps_i.tobytes(),
        UINT32,
        "SCALAR",
        len(ps_i),
        min_v=[int(ps_i.min())],
        max_v=[int(ps_i.max())],
    )
    fv_idx = add_chunk(
        ps_v.tobytes(),
        FLOAT32,
        "VEC3",
        len(ps_v),
        min_v=ps_v.min(axis=0),
        max_v=ps_v.max(axis=0),
    )
    face_attrs = pygltflib.Attributes(POSITION=fv_idx)
    if ps_uvs is not None:
        face_attrs.TEXCOORD_0 = add_chunk(
            ps_uvs.tobytes(), FLOAT32, "VEC2", len(ps_uvs)
        )
    if ps_colors is not None:
        face_attrs.COLOR_0 = add_chunk(
            ps_colors.astype(np.float32).tobytes(), FLOAT32, "VEC4", len(ps_colors)
        )
    face_attrs.JOINTS_0 = add_chunk(ps_j4.tobytes(), UINT16, "VEC4", len(ps_j4))
    face_attrs.WEIGHTS_0 = add_chunk(ps_w4.tobytes(), FLOAT32, "VEC4", len(ps_w4))

    # PSHuman material: if vertex colors, add a simple unlit-style PBR material
    # (white baseColor so COLOR_0 renders as-is); otherwise reuse body material.
    if ps_colors is not None:
        face_mat = pygltflib.Material(
            name="pshuman_vertex_color",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.85,
            ),
            doubleSided=True,
        )
        gltf.materials.append(face_mat)
        face_mat_idx = len(gltf.materials) - 1
    else:
        face_mat_idx = source_gltf.material_idx

    prim1 = pygltflib.Primitive(
        attributes=face_attrs,
        indices=fi_idx,
        material=face_mat_idx,
        mode=4,
    )

    # ------------------------------------------------------------------
    # Patch gltf structure
    # ------------------------------------------------------------------
    # Find or create the mesh that uses our skin
    mesh_node = gltf.nodes[source_gltf.mesh_node_idx]
    old_mesh_idx = mesh_node.mesh

    new_mesh = pygltflib.Mesh(
        name="body_with_pshuman_face",
        primitives=[prim0, prim1],
    )

    # Replace or append
    if old_mesh_idx is not None and old_mesh_idx < len(gltf.meshes):
        gltf.meshes[old_mesh_idx] = new_mesh
        target_mesh_idx = old_mesh_idx
    else:
        gltf.meshes.append(new_mesh)
        target_mesh_idx = len(gltf.meshes) - 1

    mesh_node.mesh = target_mesh_idx

    # ------------------------------------------------------------------
    # Add inverse bind matrices back into the new buffer BEFORE finalising.
    # Without this the skin has no inverseBindMatrices and model-viewer
    # applies identity matrices → mesh explodes.
    # ------------------------------------------------------------------
    if ibm_bytes and ibm_count > 0:
        ibm_acc_idx = add_chunk(ibm_bytes, FLOAT32, "MAT4", ibm_count)
        for skin in gltf.skins:
            skin.inverseBindMatrices = ibm_acc_idx
        print(
            f"[face_transplant] Restored inverseBindMatrices → accessor {ibm_acc_idx}"
        )

    # ------------------------------------------------------------------
    # Re-embed texture images into the binary blob (not as data URIs).
    # A buffer view for an image has no corresponding accessor.
    # ------------------------------------------------------------------
    for img, img_bytes in saved_images:
        # Pad image bytes to 4-byte alignment
        padded = img_bytes
        if len(padded) % 4:
            padded = padded + b"\x00" * (4 - len(padded) % 4)
        bv = pygltflib.BufferView(
            buffer=0, byteOffset=byte_offset, byteLength=len(img_bytes)
        )
        bviews.append(bv)
        img.bufferView = len(bviews) - 1
        chunks.append(padded)
        byte_offset += len(padded)
        print(
            f"[face_transplant] Re-embedded image ({len(img_bytes) / 1024:.0f} KB) → bufferView {img.bufferView}"
        )

    # Replace buffer views and accessors
    gltf.bufferViews = bviews
    gltf.accessors = accors

    # Rewrite buffer
    combined = b"".join(chunks)
    # Pad to 4-byte alignment
    if len(combined) % 4:
        combined += b"\x00" * (4 - len(combined) % 4)

    gltf.buffers = [pygltflib.Buffer(byteLength=len(combined))]
    gltf.set_binary_blob(combined)

    # Drop stale animation (it referenced old accessor indices)
    gltf.animations = []

    gltf.save(output_path)


# ---------------------------------------------------------------------------
# Pre-rig variant — no skin required
# ---------------------------------------------------------------------------


def transplant_face_prerig(
    body_glb_path: str,
    pshuman_mesh_path: str,
    output_path: str,
    head_fraction: float = 0.22,
    retract_amount: float = 0.004,
) -> str:
    """
    Stitch PSHuman face into an UNRIGGED TripoSG GLB.

    Head region is detected positionally — top *head_fraction* of the mesh's
    Y-axis extent (TripoSG produces Y-up meshes with the head at the top).
    Output is an unrigged combined GLB; feed it to UniRig afterwards so the
    entire combined mesh gets rigged in one pass.

    Parameters
    ----------
    body_glb_path    : unrigged TripoSG textured GLB
    pshuman_mesh_path: PSHuman output OBJ / GLB
    output_path      : result GLB path
    head_fraction    : fraction of mesh height to treat as head (default 0.22 = top 22%)
    retract_amount   : metres to push head verts inward to avoid z-fighting
    """
    print(f"[face_transplant_prerig] Loading unrigged GLB: {body_glb_path}")
    body: trimesh.Trimesh = trimesh.load(body_glb_path, force="mesh", process=False)
    print(f"  Verts: {len(body.vertices)}  Faces: {len(body.faces)}")

    # Detect head verts by Y-position (top head_fraction of total height)
    verts = np.array(body.vertices, dtype=np.float64)
    y = verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_thresh = y_max - (y_max - y_min) * head_fraction
    head_mask = y >= y_thresh
    print(
        f"  Head verts (top {head_fraction * 100:.0f}%, Y >= {y_thresh:.4f}): "
        f"{head_mask.sum()} / {len(verts)}"
    )

    if head_mask.sum() < 3:
        raise RuntimeError(
            f"Only {head_mask.sum()} head verts found — mesh may be oriented incorrectly "
            f"or head_fraction is too small."
        )

    # Retract head verts inward to avoid z-fighting when PSHuman sits on top
    if body.vertex_normals is not None and len(body.vertex_normals) == len(verts):
        verts[head_mask] -= (
            np.array(body.vertex_normals, dtype=np.float64)[head_mask] * retract_amount
        )
    else:
        centroid = verts[head_mask].mean(axis=0)
        dirs = centroid - verts[head_mask]
        norms_len = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        verts[head_mask] += (dirs / norms_len) * retract_amount

    # Load + align PSHuman face to the TripoSG head bounding box
    head_verts_f32 = verts[head_mask].astype(np.float32)
    ps_mesh = load_and_align_pshuman(pshuman_mesh_path, head_verts_f32)

    # Rebuild body with retracted verts
    body_mod = trimesh.Trimesh(
        vertices=verts.astype(np.float32),
        faces=np.array(body.faces, dtype=np.int32),
        process=False,
    )
    # Carry over visual if available
    try:
        body_mod.visual = body.visual
    except Exception:
        pass

    # Concatenate: body (retracted head) + PSHuman face
    combined = trimesh.util.concatenate([body_mod, ps_mesh])
    print(
        f"[face_transplant_prerig] Combined: {len(combined.vertices)} verts, "
        f"{len(combined.faces)} faces"
    )

    combined.export(output_path)
    print(f"[face_transplant_prerig] Saved -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Transplant PSHuman face into UniRig GLB"
    )
    parser.add_argument("--body", required=True, help="Rigged UniRig GLB")
    parser.add_argument("--face", required=True, help="PSHuman mesh (OBJ/GLB/PLY)")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument(
        "--head-bones",
        default="head,Head,skull,Skull",
        help="Comma-separated bone name substrings for head detection",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.35,
        help="Minimum head-bone weight sum to classify a vert as face",
    )
    parser.add_argument(
        "--retract",
        type=float,
        default=0.004,
        help="Metres to retract UniRig face verts inward (default 0.004)",
    )
    parser.add_argument(
        "--knn", type=int, default=5, help="K nearest neighbours for weight transfer"
    )
    args = parser.parse_args()

    subs = tuple(s.strip() for s in args.head_bones.split(","))
    transplant_face(
        body_glb_path=args.body,
        pshuman_mesh_path=args.face,
        output_path=args.output,
        head_bone_substrings=subs,
        weight_threshold=args.weight_threshold,
        retract_amount=args.retract,
        knn=args.knn,
    )


if __name__ == "__main__":
    main()
