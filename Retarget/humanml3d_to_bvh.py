#!/usr/bin/env python3
"""
humanml3d_to_bvh.py
Convert HumanML3D .npy motion files → BVH animation.
When a UniRig-rigged GLB (or ASCII FBX) is supplied via --rig, the BVH is
built using the UniRig skeleton's own bone names and hierarchy, with
automatic bone-to-SMPL-joint mapping — no Blender required.

Dependencies
  numpy           (always required)
  pygltflib       pip install pygltflib   (required for --rig GLB files)

Usage
  # SMPL-named BVH (no rig needed)
  python humanml3d_to_bvh.py 000001.npy

  # Retargeted to UniRig skeleton
  python humanml3d_to_bvh.py 000001.npy --rig rigged_mesh.glb

  # Explicit output + fps
  python humanml3d_to_bvh.py 000001.npy --rig rigged_mesh.glb -o anim.bvh --fps 20
"""

from __future__ import annotations
import argparse, re, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# SMPL 22-joint skeleton definition
# ══════════════════════════════════════════════════════════════════════════════

SMPL_NAMES = [
    "Hips",          # 0  pelvis / root
    "LeftUpLeg",     # 1  left_hip
    "RightUpLeg",    # 2  right_hip
    "Spine",         # 3  spine1
    "LeftLeg",       # 4  left_knee
    "RightLeg",      # 5  right_knee
    "Spine1",        # 6  spine2
    "LeftFoot",      # 7  left_ankle
    "RightFoot",     # 8  right_ankle
    "Spine2",        # 9  spine3
    "LeftToeBase",   # 10 left_foot
    "RightToeBase",  # 11 right_foot
    "Neck",          # 12 neck
    "LeftShoulder",  # 13 left_collar
    "RightShoulder", # 14 right_collar
    "Head",          # 15 head
    "LeftArm",       # 16 left_shoulder
    "RightArm",      # 17 right_shoulder
    "LeftForeArm",   # 18 left_elbow
    "RightForeArm",  # 19 right_elbow
    "LeftHand",      # 20 left_wrist
    "RightHand",     # 21 right_wrist
]

SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
NUM_SMPL = 22

SMPL_TPOSE = np.array([
    [ 0.000,  0.920,  0.000],  # 0  Hips
    [-0.095,  0.920,  0.000],  # 1  LeftUpLeg
    [ 0.095,  0.920,  0.000],  # 2  RightUpLeg
    [ 0.000,  0.980,  0.000],  # 3  Spine
    [-0.095,  0.495,  0.000],  # 4  LeftLeg
    [ 0.095,  0.495,  0.000],  # 5  RightLeg
    [ 0.000,  1.050,  0.000],  # 6  Spine1
    [-0.095,  0.075,  0.000],  # 7  LeftFoot
    [ 0.095,  0.075,  0.000],  # 8  RightFoot
    [ 0.000,  1.120,  0.000],  # 9  Spine2
    [-0.095,  0.000, -0.020],  # 10 LeftToeBase
    [ 0.095,  0.000, -0.020],  # 11 RightToeBase
    [ 0.000,  1.370,  0.000],  # 12 Neck
    [-0.130,  1.290,  0.000],  # 13 LeftShoulder
    [ 0.130,  1.290,  0.000],  # 14 RightShoulder
    [ 0.000,  1.500,  0.000],  # 15 Head
    [-0.330,  1.290,  0.000],  # 16 LeftArm
    [ 0.330,  1.290,  0.000],  # 17 RightArm
    [-0.630,  1.290,  0.000],  # 18 LeftForeArm
    [ 0.630,  1.290,  0.000],  # 19 RightForeArm
    [-0.910,  1.290,  0.000],  # 20 LeftHand
    [ 0.910,  1.290,  0.000],  # 21 RightHand
], dtype=np.float32)

_SMPL_CHILDREN: list[list[int]] = [[] for _ in range(NUM_SMPL)]
for _j, _p in enumerate(SMPL_PARENT):
    if _p >= 0:
        _SMPL_CHILDREN[_p].append(_j)


def _smpl_dfs() -> list[int]:
    order, stack = [], [0]
    while stack:
        j = stack.pop()
        order.append(j)
        for c in reversed(_SMPL_CHILDREN[j]):
            stack.append(c)
    return order


SMPL_DFS = _smpl_dfs()

# ══════════════════════════════════════════════════════════════════════════════
# Quaternion helpers  (numpy, WXYZ)
# ══════════════════════════════════════════════════════════════════════════════

def qnorm(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1, -1, -1, -1], dtype=np.float32)


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.concatenate([np.zeros((*v.shape[:-1], 1), dtype=v.dtype), v], axis=-1)
    return qmul(qmul(q, vq), qinv(q))[..., 1:]


def qbetween(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Swing quaternion rotating unit-vectors a to b.  [..., 3] to [..., 4]."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    dot   = np.clip((a * b).sum(axis=-1, keepdims=True), -1.0, 1.0)
    cross = np.cross(a, b)
    w     = np.sqrt(np.maximum((1.0 + dot) * 0.5, 0.0))
    xyz   = cross / (2.0 * w + 1e-9)
    anti  = (dot[..., 0] < -0.9999)
    if anti.any():
        perp = np.where(
            np.abs(a[anti, 0:1]) < 0.9,
            np.tile([1, 0, 0], (anti.sum(), 1)),
            np.tile([0, 1, 0], (anti.sum(), 1)),
        ).astype(np.float32)
        ax_f       = np.cross(a[anti], perp)
        ax_f       = ax_f / (np.linalg.norm(ax_f, axis=-1, keepdims=True) + 1e-9)
        w[anti]    = 0.0
        xyz[anti]  = ax_f
    return qnorm(np.concatenate([w, xyz], axis=-1))


def quat_to_euler_ZXY(q: np.ndarray) -> np.ndarray:
    """WXYZ quaternions to ZXY Euler degrees (rz, rx, ry) for BVH."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sin_x = np.clip(2.0*(w*x - y*z), -1.0, 1.0)
    return np.stack([
        np.degrees(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(x*x + z*z))),
        np.degrees(np.arcsin(sin_x)),
        np.degrees(np.arctan2(2.0*(w*y + x*z), 1.0 - 2.0*(x*x + y*y))),
    ], axis=-1)

# ══════════════════════════════════════════════════════════════════════════════
# HumanML3D 263-dim recovery
#
# Layout per frame:
#   [0]       root Y-axis angular velocity (rad/frame)
#   [1]       root height Y (m)
#   [2:4]     root XZ velocity in local frame
#   [4:67]    local positions of joints 1-21  (21 x 3 = 63)
#   [67:193]  6-D rotations for joints 1-21  (21 x 6 = 126, unused here)
#   [193:259] joint velocities               (22 x 3 = 66,  unused here)
#   [259:263] foot contact                   (4,             unused here)
# ══════════════════════════════════════════════════════════════════════════════

def _recover_root(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T     = data.shape[0]
    theta = np.cumsum(data[:, 0])
    half  = theta * 0.5
    r_rot = np.zeros((T, 4), dtype=np.float32)
    r_rot[:, 0] = np.cos(half)   # W
    r_rot[:, 2] = np.sin(half)   # Y
    vel_local = np.stack([data[:, 2], np.zeros(T, dtype=np.float32), data[:, 3]], -1)
    vel_world = qrot(r_rot, vel_local)
    r_pos = np.zeros((T, 3), dtype=np.float32)
    r_pos[:, 0] = np.cumsum(vel_world[:, 0])
    r_pos[:, 1] = data[:, 1]
    r_pos[:, 2] = np.cumsum(vel_world[:, 2])
    return r_rot, r_pos


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """263-dim features to world-space positions [T, joints_num, 3]."""
    data = data.astype(np.float32)
    r_rot, r_pos = _recover_root(data)
    loc  = data[:, 4:4 + (joints_num-1)*3].reshape(-1, joints_num-1, 3)
    rinv = np.broadcast_to(qinv(r_rot)[:, None], (*loc.shape[:2], 4)).copy()
    wloc = qrot(rinv, loc) + r_pos[:, None]
    return np.concatenate([r_pos[:, None], wloc], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# SMPL geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _scale_smpl_tpose(positions: np.ndarray) -> np.ndarray:
    data_h = positions[:, :, 1].max() - positions[:, :, 1].min()
    ref_h  = SMPL_TPOSE[:, 1].max() - SMPL_TPOSE[:, 1].min()
    scale  = (data_h / ref_h) if (ref_h > 1e-6 and data_h > 1e-6) else 1.0
    return SMPL_TPOSE * scale


def _rest_dirs(tpose: np.ndarray, children: list[list[int]],
               parent: list[int]) -> np.ndarray:
    N = tpose.shape[0]
    dirs = np.zeros((N, 3), dtype=np.float32)
    for j in range(N):
        ch = children[j]
        if ch:
            avg  = np.stack([tpose[c] - tpose[j] for c in ch]).mean(0)
            dirs[j] = avg / (np.linalg.norm(avg) + 1e-9)
        else:
            v = tpose[j] - tpose[parent[j]]
            dirs[j] = v / (np.linalg.norm(v) + 1e-9)
    return dirs


def positions_to_local_quats(positions: np.ndarray,
                              tpose: np.ndarray) -> np.ndarray:
    """World-space joint positions [T, 22, 3] to local quaternions [T, 22, 4]."""
    T  = positions.shape[0]
    rd = _rest_dirs(tpose, _SMPL_CHILDREN, SMPL_PARENT)

    world_q = np.zeros((T, NUM_SMPL, 4), dtype=np.float32)
    world_q[:, :, 0] = 1.0

    for j in range(NUM_SMPL):
        ch = _SMPL_CHILDREN[j]
        if ch:
            vecs = np.stack([positions[:, c] - positions[:, j] for c in ch], 1).mean(1)
        else:
            vecs = positions[:, j] - positions[:, SMPL_PARENT[j]]
        cur     = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-9)
        rd_b    = np.broadcast_to(rd[j], cur.shape).copy()
        world_q[:, j] = qbetween(rd_b, cur)

    local_q = np.zeros_like(world_q)
    local_q[:, :, 0] = 1.0
    for j in SMPL_DFS:
        p = SMPL_PARENT[j]
        if p < 0:
            local_q[:, j] = world_q[:, j]
        else:
            local_q[:, j] = qmul(qinv(world_q[:, p]), world_q[:, j])

    return qnorm(local_q)

# ══════════════════════════════════════════════════════════════════════════════
# UniRig skeleton data structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Bone:
    name:           str
    parent:         Optional[str]
    world_rest_pos: np.ndarray
    children:       list[str] = field(default_factory=list)
    smpl_idx:       Optional[int] = None


class UnirigSkeleton:
    def __init__(self, bones: dict[str, Bone]):
        self.bones = bones
        self.root  = next(b for b in bones.values() if b.parent is None)

    def dfs_order(self) -> list[str]:
        order, stack = [], [self.root.name]
        while stack:
            n = stack.pop()
            order.append(n)
            for c in reversed(self.bones[n].children):
                stack.append(c)
        return order

    def local_offsets(self) -> dict[str, np.ndarray]:
        offsets = {}
        for name, bone in self.bones.items():
            if bone.parent is None:
                offsets[name] = bone.world_rest_pos.copy()
            else:
                offsets[name] = bone.world_rest_pos - self.bones[bone.parent].world_rest_pos
        return offsets

    def rest_direction(self, name: str) -> np.ndarray:
        bone = self.bones[name]
        if bone.children:
            vecs = np.stack([self.bones[c].world_rest_pos - bone.world_rest_pos
                             for c in bone.children])
            avg  = vecs.mean(0)
            return avg / (np.linalg.norm(avg) + 1e-9)
        if bone.parent is None:
            return np.array([0, 1, 0], dtype=np.float32)
        v = bone.world_rest_pos - self.bones[bone.parent].world_rest_pos
        return v / (np.linalg.norm(v) + 1e-9)

# ══════════════════════════════════════════════════════════════════════════════
# GLB skeleton parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_glb_skeleton(path: str) -> UnirigSkeleton:
    """Extract skeleton from a UniRig-rigged GLB (uses pygltflib)."""
    try:
        import pygltflib
    except ImportError:
        sys.exit("[ERROR] pygltflib not installed.  pip install pygltflib")

    import base64

    gltf = pygltflib.GLTF2().load(path)
    if not gltf.skins:
        sys.exit(f"[ERROR] No skin found in {path}")

    skin          = gltf.skins[0]
    joint_indices = skin.joints

    def get_buffer_bytes(buf_idx: int) -> bytes:
        buf = gltf.buffers[buf_idx]
        if buf.uri is None:
            return bytes(gltf.binary_blob())
        if buf.uri.startswith("data:"):
            return base64.b64decode(buf.uri.split(",", 1)[1])
        return (Path(path).parent / buf.uri).read_bytes()

    def read_accessor(acc_idx: int) -> np.ndarray:
        acc  = gltf.accessors[acc_idx]
        bv   = gltf.bufferViews[acc.bufferView]
        raw  = get_buffer_bytes(bv.buffer)
        COMP = {5120: ('b',1), 5121: ('B',1), 5122: ('h',2),
                5123: ('H',2), 5125: ('I',4), 5126: ('f',4)}
        DIMS = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}
        fmt, sz = COMP[acc.componentType]
        dim     = DIMS[acc.type]
        start   = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        stride  = bv.byteStride
        if stride is None or stride == 0 or stride == sz * dim:
            chunk = raw[start: start + acc.count * sz * dim]
            return np.frombuffer(chunk, dtype=fmt).reshape(acc.count, dim).astype(np.float32)
        rows = []
        for i in range(acc.count):
            off = start + i * stride
            rows.append(np.frombuffer(raw[off: off + sz * dim], dtype=fmt))
        return np.stack(rows).astype(np.float32)

    ibm     = read_accessor(skin.inverseBindMatrices).reshape(-1, 4, 4)
    joint_set = set(joint_indices)
    ni_name   = {ni: (gltf.nodes[ni].name or f"bone_{ni}") for ni in joint_indices}

    bones: dict[str, Bone] = {}
    for i, ni in enumerate(joint_indices):
        name      = ni_name[ni]
        world_mat = np.linalg.inv(ibm[i])
        bones[name] = Bone(name=name, parent=None,
                           world_rest_pos=world_mat[:3, 3].astype(np.float32))

    for ni in joint_indices:
        for ci in (gltf.nodes[ni].children or []):
            if ci in joint_set:
                p, c = ni_name[ni], ni_name[ci]
                bones[c].parent = p
                bones[p].children.append(c)

    print(f"[GLB] {len(bones)} bones from skin '{gltf.skins[0].name or 'Armature'}'")
    return UnirigSkeleton(bones)

# ══════════════════════════════════════════════════════════════════════════════
# ASCII FBX skeleton parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_fbx_ascii_skeleton(path: str) -> UnirigSkeleton:
    """Parse ASCII-format FBX for LimbNode / Root bones."""
    raw = Path(path).read_bytes()
    if raw[:4] == b"Kayd":
        sys.exit(
            f"[ERROR] {path} is binary FBX.\n"
            "Convert to GLB first, e.g.:\n"
            "  gltf-pipeline -i rigged.fbx -o rigged.glb"
        )
    text = raw.decode("utf-8", errors="replace")

    model_pat = re.compile(
        r'Model:\s*(\d+),\s*"Model::([^"]+)",\s*"(LimbNode|Root|Null)"'
        r'.*?Properties70:\s*\{(.*?)\}',
        re.DOTALL
    )
    trans_pat = re.compile(
        r'P:\s*"Lcl Translation".*?(-?[\d.e+\-]+),\s*(-?[\d.e+\-]+),\s*(-?[\d.e+\-]+)'
    )

    uid_name:  dict[str, str]        = {}
    uid_local: dict[str, np.ndarray] = {}

    for m in model_pat.finditer(text):
        uid, name = m.group(1), m.group(2)
        uid_name[uid] = name
        tm = trans_pat.search(m.group(4))
        uid_local[uid] = (np.array([float(tm.group(i)) for i in (1,2,3)], dtype=np.float32)
                          if tm else np.zeros(3, dtype=np.float32))

    if not uid_name:
        sys.exit("[ERROR] No LimbNode/Root bones found in FBX")

    conn_pat    = re.compile(r'C:\s*"OO",\s*(\d+),\s*(\d+)')
    uid_parent: dict[str, str] = {}
    for m in conn_pat.finditer(text):
        child, par = m.group(1), m.group(2)
        if child in uid_name and par in uid_name:
            uid_parent[child] = par

    # Detect cm vs m
    all_y = np.array([t[1] for t in uid_local.values()])
    scale = 0.01 if all_y.max() > 10.0 else 1.0
    if scale != 1.0:
        print(f"[FBX] Centimetre units detected — scaling by {scale}")
    for uid in uid_local:
        uid_local[uid] *= scale

    # Accumulate world translations (topological order)
    def topo(uid_to_par):
        visited, order = set(), []
        def visit(u):
            if u in visited: return
            visited.add(u)
            if u in uid_to_par: visit(uid_to_par[u])
            order.append(u)
        for u in uid_to_par: visit(u)
        for u in uid_name:
            if u not in visited: order.append(u)
        return order

    world: dict[str, np.ndarray] = {}
    for uid in topo(uid_parent):
        loc = uid_local.get(uid, np.zeros(3, dtype=np.float32))
        world[uid] = (world.get(uid_parent[uid], np.zeros(3, dtype=np.float32)) + loc
                      if uid in uid_parent else loc.copy())

    bones: dict[str, Bone] = {}
    for uid, name in uid_name.items():
        bones[name] = Bone(name=name, parent=None, world_rest_pos=world[uid])

    for uid, p_uid in uid_parent.items():
        c, p = uid_name[uid], uid_name[p_uid]
        bones[c].parent = p
        if c not in bones[p].children:
            bones[p].children.append(c)

    print(f"[FBX] {len(bones)} bones parsed from ASCII FBX")
    return UnirigSkeleton(bones)

# ══════════════════════════════════════════════════════════════════════════════
# Auto bone mapping: UniRig bones to SMPL joints
# ══════════════════════════════════════════════════════════════════════════════

# Keyword table: normalised name fragments -> SMPL joint index
_NAME_HINTS: list[tuple[list[str], int]] = [
    (["hips","pelvis","root","hip"],                                               0),
    (["leftupleg","l_upleg","lupleg","leftthigh","lefthip",
      "left_upper_leg","l_thigh","thigh_l","upperleg_l","j_bip_l_upperleg"],       1),
    (["rightupleg","r_upleg","rupleg","rightthigh","righthip",
      "right_upper_leg","r_thigh","thigh_r","upperleg_r","j_bip_r_upperleg"],      2),
    (["spine","spine0","spine_01","j_bip_c_spine"],                                3),
    (["leftleg","leftknee","l_leg","lleg","leftlowerleg",
      "left_lower_leg","lowerleg_l","knee_l","j_bip_l_lowerleg"],                  4),
    (["rightleg","rightknee","r_leg","rleg","rightlowerleg",
      "right_lower_leg","lowerleg_r","knee_r","j_bip_r_lowerleg"],                 5),
    (["spine1","spine_02","j_bip_c_spine1"],                                       6),
    (["leftfoot","left_foot","l_foot","lfoot","foot_l","j_bip_l_foot"],            7),
    (["rightfoot","right_foot","r_foot","rfoot","foot_r","j_bip_r_foot"],          8),
    (["spine2","spine_03","j_bip_c_spine2","chest"],                               9),
    (["lefttoebase","lefttoe","l_toe","ltoe","toe_l"],                            10),
    (["righttoebase","righttoe","r_toe","rtoe","toe_r"],                           11),
    (["neck","j_bip_c_neck"],                                                     12),
    (["leftshoulder","leftcollar","l_shoulder","leftclavicle",
      "clavicle_l","j_bip_l_shoulder"],                                           13),
    (["rightshoulder","rightcollar","r_shoulder","rightclavicle",
      "clavicle_r","j_bip_r_shoulder"],                                           14),
    (["head","j_bip_c_head"],                                                     15),
    (["leftarm","leftupper","l_arm","larm","leftupperarm",
      "upperarm_l","j_bip_l_upperarm"],                                           16),
    (["rightarm","rightupper","r_arm","rarm","rightupperarm",
      "upperarm_r","j_bip_r_upperarm"],                                           17),
    (["leftforearm","leftlower","l_forearm","lforearm",
      "lowerarm_l","j_bip_l_lowerarm"],                                           18),
    (["rightforearm","rightlower","r_forearm","rforearm",
      "lowerarm_r","j_bip_r_lowerarm"],                                           19),
    (["lefthand","l_hand","lhand","hand_l","j_bip_l_hand"],                       20),
    (["righthand","r_hand","rhand","hand_r","j_bip_r_hand"],                      21),
]


def _strip_name(name: str) -> str:
    """Remove common rig namespace prefixes, then lower-case, remove separators."""
    name = re.sub(r'^(mixamorig:|j_bip_[lcr]_|cc_base_|bip01_|rig:|chr:)',
                  "", name, flags=re.IGNORECASE)
    return re.sub(r'[_\-\s.]', "", name).lower()


def _normalise_positions(pos: np.ndarray) -> np.ndarray:
    """Normalise [N, 3] to [0,1] in Y, [-1,1] in X and Z."""
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    h = (y_max - y_min) or 1.0
    xr = (pos[:, 0].max() - pos[:, 0].min()) or 1.0
    zr = (pos[:, 2].max() - pos[:, 2].min()) or 1.0
    out = pos.copy()
    out[:, 0] /= xr
    out[:, 1]  = (out[:, 1] - y_min) / h
    out[:, 2] /= zr
    return out


def auto_map(skel: UnirigSkeleton, verbose: bool = True) -> None:
    """
    Assign skel.bones[name].smpl_idx for each UniRig bone that best matches
    an SMPL joint.  Score = 0.6 * position_proximity + 0.4 * name_hint.
    Greedy: each SMPL joint taken by at most one UniRig bone.
    Bones with combined score < 0.35 are left unmapped (identity in BVH).
    """
    names   = list(skel.bones.keys())
    ur_pos  = np.stack([skel.bones[n].world_rest_pos for n in names])   # [M, 3]

    # Scale SMPL T-pose to match UniRig height
    ur_h   = ur_pos[:, 1].max() - ur_pos[:, 1].min()
    sm_h   = SMPL_TPOSE[:, 1].max() - SMPL_TPOSE[:, 1].min()
    sm_pos = SMPL_TPOSE * ((ur_h / sm_h) if sm_h > 1e-6 else 1.0)

    all_norm = _normalise_positions(np.concatenate([ur_pos, sm_pos]))
    ur_norm  = all_norm[:len(names)]
    sm_norm  = all_norm[len(names):]

    # Distance score [M, 22]
    dist     = np.linalg.norm(ur_norm[:, None] - sm_norm[None], axis=-1)
    dist_sc  = 1.0 - np.clip(dist / (dist.max() + 1e-9), 0, 1)

    # Name score [M, 22]
    norm_names = [_strip_name(n) for n in names]
    name_sc    = np.array(
        [[1.0 if norm in kws else 0.0
          for kws, _ in _NAME_HINTS]
         for norm in norm_names],
        dtype=np.float32,
    )  # [M, 22]

    combined = 0.6 * dist_sc + 0.4 * name_sc    # [M, 22]

    # Greedy assignment
    THRESHOLD   = 0.35
    taken_smpl: set[int] = set()
    pairs = sorted(
        ((i, j, combined[i, j])
         for i in range(len(names)) for j in range(NUM_SMPL)),
        key=lambda x: -x[2],
    )
    for bi, si, score in pairs:
        if score < THRESHOLD:
            break
        name = names[bi]
        if skel.bones[name].smpl_idx is not None or si in taken_smpl:
            continue
        skel.bones[name].smpl_idx = si
        taken_smpl.add(si)

    if verbose:
        mapped   = [(n, b.smpl_idx) for n, b in skel.bones.items() if b.smpl_idx is not None]
        unmapped = [n for n, b in skel.bones.items() if b.smpl_idx is None]
        print(f"\n[MAP] {len(mapped)}/{len(skel.bones)} bones mapped to SMPL joints:")
        for ur_name, si in sorted(mapped, key=lambda x: x[1]):
            sc = combined[names.index(ur_name), si]
            print(f"       {ur_name:40s} -> {SMPL_NAMES[si]:16s}  score={sc:.2f}")
        if unmapped:
            print(f"[MAP] {len(unmapped)} unmapped (identity rotation): "
                  + ", ".join(unmapped[:8])
                  + (" ..." if len(unmapped) > 8 else ""))
        print()

# ══════════════════════════════════════════════════════════════════════════════
# BVH writers
# ══════════════════════════════════════════════════════════════════════════════

def _smpl_offsets(tpose: np.ndarray) -> np.ndarray:
    offsets = np.zeros_like(tpose)
    for j, p in enumerate(SMPL_PARENT):
        offsets[j] = tpose[j] if p < 0 else tpose[j] - tpose[p]
    return offsets


def write_bvh_smpl(output_path: str, positions: np.ndarray, fps: int = 20) -> None:
    """BVH with standard SMPL bone names (no rig file needed)."""
    T       = positions.shape[0]
    tpose   = _scale_smpl_tpose(positions)
    offsets = _smpl_offsets(tpose)
    tp_w    = tpose + (positions[0, 0] - tpose[0])
    local_q = positions_to_local_quats(positions, tp_w)
    euler   = quat_to_euler_ZXY(local_q)

    with open(output_path, "w") as f:
        f.write("HIERARCHY\n")

        def wj(j, ind):
            off = offsets[j]
            f.write(f"{'ROOT' if SMPL_PARENT[j]<0 else ind+'JOINT'} {SMPL_NAMES[j]}\n")
            f.write(f"{ind}{{\n")
            f.write(f"{ind}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
            if SMPL_PARENT[j] < 0:
                f.write(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition "
                        "Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{ind}\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
            for c in _SMPL_CHILDREN[j]:
                wj(c, ind + "\t")
            if not _SMPL_CHILDREN[j]:
                f.write(f"{ind}\tEnd Site\n{ind}\t{{\n"
                        f"{ind}\t\tOFFSET 0.000000 0.050000 0.000000\n{ind}\t}}\n")
            f.write(f"{ind}}}\n")

        wj(0, "")
        f.write(f"MOTION\nFrames: {T}\nFrame Time: {1.0/fps:.8f}\n")
        for t in range(T):
            rp  = positions[t, 0]
            row = [f"{rp[0]:.6f}", f"{rp[1]:.6f}", f"{rp[2]:.6f}"]
            for j in SMPL_DFS:
                rz, rx, ry = euler[t, j]
                row += [f"{rz:.6f}", f"{rx:.6f}", f"{ry:.6f}"]
            f.write(" ".join(row) + "\n")

    print(f"[OK] {T} frames @ {fps} fps -> {output_path}  (SMPL skeleton)")


def write_bvh_unirig(output_path: str,
                     positions: np.ndarray,
                     skel: UnirigSkeleton,
                     fps: int = 20) -> None:
    """
    BVH using UniRig bone names and hierarchy.
    Mapped bones receive SMPL-derived local rotations with rest-pose correction.
    Unmapped bones (fingers, face bones, etc.) are set to identity.
    """
    T = positions.shape[0]

    # Compute SMPL local quaternions
    tpose   = _scale_smpl_tpose(positions)
    tp_w    = tpose + (positions[0, 0] - tpose[0])
    smpl_q  = positions_to_local_quats(positions, tp_w)    # [T, 22, 4]
    smpl_rd = _rest_dirs(tp_w, _SMPL_CHILDREN, SMPL_PARENT)  # [22, 3]

    # Rest-pose correction quaternions per bone:
    #   q_corr = qbetween(unirig_rest_dir, smpl_rest_dir)
    #   unirig_local_q = smpl_local_q @ q_corr
    # This ensures: when applied to unirig_rest_dir, the result matches
    # the SMPL animated direction — accounting for any difference in
    # rest-pose bone orientations between the two skeletons.
    corrections: dict[str, np.ndarray] = {}
    for name, bone in skel.bones.items():
        if bone.smpl_idx is None:
            continue
        ur_rd = skel.rest_direction(name).astype(np.float32)
        sm_rd = smpl_rd[bone.smpl_idx].astype(np.float32)
        corrections[name] = qbetween(ur_rd[None], sm_rd[None])[0]  # [4]

    # Scale root translation from SMPL proportions to UniRig proportions
    ur_h    = (max(b.world_rest_pos[1] for b in skel.bones.values())
               - min(b.world_rest_pos[1] for b in skel.bones.values()))
    sm_h    = tp_w[:, 1].max() - tp_w[:, 1].min()
    pos_sc  = (ur_h / sm_h) if sm_h > 1e-6 else 1.0

    dfs     = skel.dfs_order()
    offsets = skel.local_offsets()

    # Pre-compute euler per bone [T, 3]
    ID_EUL = np.zeros((T, 3), dtype=np.float32)
    bone_euler: dict[str, np.ndarray] = {}
    for name, bone in skel.bones.items():
        if bone.smpl_idx is not None:
            q = smpl_q[:, bone.smpl_idx].copy()               # [T, 4]
            c = corrections.get(name)
            if c is not None:
                q = qnorm(qmul(q, np.broadcast_to(c[None], q.shape).copy()))
            bone_euler[name] = quat_to_euler_ZXY(q)           # [T, 3]
        else:
            bone_euler[name] = ID_EUL

    with open(output_path, "w") as f:
        f.write("HIERARCHY\n")

        def wj(name, ind):
            off  = offsets[name]
            bone = skel.bones[name]
            f.write(f"{'ROOT' if bone.parent is None else ind+'JOINT'} {name}\n")
            f.write(f"{ind}{{\n")
            f.write(f"{ind}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
            if bone.parent is None:
                f.write(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition "
                        "Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{ind}\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
            for c in bone.children:
                wj(c, ind + "\t")
            if not bone.children:
                f.write(f"{ind}\tEnd Site\n{ind}\t{{\n"
                        f"{ind}\t\tOFFSET 0.000000 0.050000 0.000000\n{ind}\t}}\n")
            f.write(f"{ind}}}\n")

        wj(skel.root.name, "")
        f.write(f"MOTION\nFrames: {T}\nFrame Time: {1.0/fps:.8f}\n")

        for t in range(T):
            rp  = positions[t, 0] * pos_sc
            row = [f"{rp[0]:.6f}", f"{rp[1]:.6f}", f"{rp[2]:.6f}"]
            for name in dfs:
                rz, rx, ry = bone_euler[name][t]
                row += [f"{rz:.6f}", f"{rx:.6f}", f"{ry:.6f}"]
            f.write(" ".join(row) + "\n")

    n_mapped = sum(1 for b in skel.bones.values() if b.smpl_idx is not None)
    print(f"[OK] {T} frames @ {fps} fps -> {output_path}  "
          f"(UniRig: {n_mapped} driven, {len(skel.bones)-n_mapped} identity)")

# ══════════════════════════════════════════════════════════════════════════════
# Motion loader
# ══════════════════════════════════════════════════════════════════════════════

def load_motion(npy_path: str) -> tuple[np.ndarray, int]:
    """Return (positions [T, 22, 3], fps).  Auto-detects HumanML3D format."""
    data = np.load(npy_path).astype(np.float32)
    print(f"[INFO] {npy_path}  shape={data.shape}")

    if data.ndim == 3 and data.shape[1] == 22 and data.shape[2] == 3:
        print("[INFO] Format: new_joints [T, 22, 3]")
        return data, 20

    if data.ndim == 2 and data.shape[1] == 263:
        print("[INFO] Format: new_joint_vecs [T, 263]")
        pos = recover_from_ric(data, 22)
        print(f"[INFO] Recovered positions {pos.shape}")
        return pos, 20

    if data.ndim == 2 and data.shape[1] == 272:
        print("[INFO] Format: 272-dim (30 fps)")
        return recover_from_ric(data[:, :263], 22), 30

    if (data.ndim == 2 and data.shape[1] == 251) or \
       (data.ndim == 3 and data.shape[1] == 21):
        sys.exit("[ERROR] KIT-ML (21-joint) format not yet supported.")

    sys.exit(f"[ERROR] Unrecognised shape {data.shape}. "
             "Expected [T,22,3] or [T,263].")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="HumanML3D .npy -> BVH, optionally retargeted to UniRig skeleton",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python humanml3d_to_bvh.py 000001.npy
      Standard SMPL-named BVH (no rig file needed)

  python humanml3d_to_bvh.py 000001.npy --rig rigged_mesh.glb
      BVH retargeted to UniRig bone names, auto-mapped by position + name

  python humanml3d_to_bvh.py 000001.npy --rig rigged_mesh.glb -o anim.bvh --fps 20

Supported --rig formats
  .glb / .gltf   UniRig merge.sh output (requires: pip install pygltflib)
  .fbx            ASCII FBX only (binary FBX: convert to GLB first)
""")
    ap.add_argument("input",          help="HumanML3D .npy motion file")
    ap.add_argument("--rig",          default=None,
                    help="UniRig-rigged mesh .glb or ASCII .fbx for auto-mapping")
    ap.add_argument("-o", "--output", default=None, help="Output .bvh path")
    ap.add_argument("--fps",          type=int, default=0,
                    help="Override FPS (default: auto from format)")
    ap.add_argument("--quiet",        action="store_true",
                    help="Suppress mapping table")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_suffix(".bvh")

    positions, auto_fps = load_motion(str(inp))
    fps = args.fps if args.fps > 0 else auto_fps

    if args.rig:
        ext = Path(args.rig).suffix.lower()
        if ext in (".glb", ".gltf"):
            skel = parse_glb_skeleton(args.rig)
        elif ext == ".fbx":
            skel = parse_fbx_ascii_skeleton(args.rig)
        else:
            sys.exit(f"[ERROR] Unsupported rig format: {ext}  (use .glb or .fbx)")

        auto_map(skel, verbose=not args.quiet)
        write_bvh_unirig(str(out), positions, skel, fps=fps)
    else:
        write_bvh_smpl(str(out), positions, fps=fps)


if __name__ == "__main__":
    main()
