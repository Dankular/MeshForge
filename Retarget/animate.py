"""
animate.py
──────────────────────────────────────────────────────────────────────────────
Bake SMPL motion (from HumanML3D [T, 263] features) onto a UniRig-rigged GLB.

Retargeting method: world-direction matching
────────────────────────────────────────────
Commercial retargeters (Mixamo, Rokoko, MotionBuilder) avoid rest-pose
convention mismatches by matching WORLD BONE DIRECTIONS, not local rotations.

Algorithm (per frame, per bone):
  1. Run t2m FK with HumanML3D 6D rotations → world bone direction d_t2m
  2. Flip X axis: t2m +X = character's LEFT; SMPL/UniRig +X = character's RIGHT
     So d_desired = (-d_t2m_x, d_t2m_y, d_t2m_z) in SMPL/UniRig world frame
  3. d_rest = normalize(ur_pos[bone] - ur_pos[parent]) from GLB inverse bind matrices
  4. R_world = R_between(d_rest, d_desired)  -- minimal rotation in world space
  5. local_rot = inv(R_world[parent]) @ R_world[bone]
  6. pose_rot_delta = inv(rest_r) @ local_rot  -- composing with glTF rest rotation

This avoids all rest-pose convention issues:
  - t2m canonical arms point DOWN: handled automatically
  - t2m canonical hips/shoulders have inverted X: handled by the X-flip
  - UniRig non-identity rest rotations: handled by inv(rest_r) composition

Key bugs fixed vs previous version:
  - IBM column-major: glTF IBMs are column-major; was using inv(ibm)[:3,3] (zeros).
    Fixed to inv(ibm.T)[:3,3] which gives correct world-space bone positions.
  - Normalisation: was mixing ur/smpl Y ranges, causing wrong height alignment.
    Fixed with independent per-skeleton Y normalisation.
  - Rotation convention: was applying t2m rotations directly without X-flip.
    Fixed by world-direction matching with coordinate-frame conversion.
"""
from __future__ import annotations
import os
import re
import numpy as np
from typing import Union

from .smpl import SMPLMotion, hml3d_to_smpl_motion


# ──────────────────────────────────────────────────────────────────────────────
# T2M (HumanML3D) skeleton constants
# Source: HumanML3D/common/paramUtil.py
# ──────────────────────────────────────────────────────────────────────────────

T2M_RAW_OFFSETS = np.array([
    [ 0, 0, 0],   # 0  Hips          (root)
    [ 1, 0, 0],   # 1  LeftUpLeg     +X = character LEFT in t2m convention
    [-1, 0, 0],   # 2  RightUpLeg
    [ 0, 1, 0],   # 3  Spine
    [ 0,-1, 0],   # 4  LeftLeg
    [ 0,-1, 0],   # 5  RightLeg
    [ 0, 1, 0],   # 6  Spine1
    [ 0,-1, 0],   # 7  LeftFoot
    [ 0,-1, 0],   # 8  RightFoot
    [ 0, 1, 0],   # 9  Spine2
    [ 0, 0, 1],   # 10 LeftToeBase
    [ 0, 0, 1],   # 11 RightToeBase
    [ 0, 1, 0],   # 12 Neck
    [ 1, 0, 0],   # 13 LeftShoulder  +X = character LEFT
    [-1, 0, 0],   # 14 RightShoulder
    [ 0, 0, 1],   # 15 Head
    [ 0,-1, 0],   # 16 LeftArm       arms hang DOWN in t2m canonical
    [ 0,-1, 0],   # 17 RightArm
    [ 0,-1, 0],   # 18 LeftForeArm
    [ 0,-1, 0],   # 19 RightForeArm
    [ 0,-1, 0],   # 20 LeftHand
    [ 0,-1, 0],   # 21 RightHand
], dtype=np.float64)

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],        # Hips -> RightUpLeg -> RightLeg -> RightFoot -> RightToe
    [0, 1, 4, 7, 10],        # Hips -> LeftUpLeg  -> LeftLeg  -> LeftFoot  -> LeftToe
    [0, 3, 6, 9, 12, 15],    # Hips -> Spine -> Spine1 -> Spine2 -> Neck -> Head
    [9, 14, 17, 19, 21],      # Spine2 -> RightShoulder -> RightArm -> RightForeArm -> RightHand
    [9, 13, 16, 18, 20],      # Spine2 -> LeftShoulder  -> LeftArm  -> LeftForeArm  -> LeftHand
]

# Parent joint index for each of the 22 t2m joints
T2M_PARENTS = [-1] * 22
for _chain in T2M_KINEMATIC_CHAIN:
    for _k in range(1, len(_chain)):
        T2M_PARENTS[_chain[_k]] = _chain[_k - 1]

# ──────────────────────────────────────────────────────────────────────────────
# SMPL joint names / T-pose (for bone mapping reference)
# ──────────────────────────────────────────────────────────────────────────────

SMPL_NAMES = [
    "Hips",         "LeftUpLeg",    "RightUpLeg",   "Spine",
    "LeftLeg",      "RightLeg",     "Spine1",       "LeftFoot",
    "RightFoot",    "Spine2",       "LeftToeBase",  "RightToeBase",
    "Neck",         "LeftShoulder", "RightShoulder","Head",
    "LeftArm",      "RightArm",     "LeftForeArm",  "RightForeArm",
    "LeftHand",     "RightHand",
]

# Approximate T-pose joint world positions in metres (Y-up, facing +Z)
# +X = character's RIGHT (standard SMPL/UniRig convention)
SMPL_TPOSE = np.array([
    [ 0.000,  0.920,  0.000],  # 0  Hips
    [-0.095,  0.920,  0.000],  # 1  LeftUpLeg   (character's left = -X)
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

# Name hint table: lowercase substrings -> SMPL joint index
_NAME_HINTS: list[tuple[list[str], int]] = [
    (["hips","pelvis","root"],                                          0),
    (["leftupleg","l_upleg","leftthigh","lefthip","thigh_l"],           1),
    (["rightupleg","r_upleg","rightthigh","righthip","thigh_r"],        2),
    (["spine","spine0","spine_01"],                                     3),
    (["leftleg","leftknee","lowerleg_l","knee_l"],                      4),
    (["rightleg","rightknee","lowerleg_r","knee_r"],                    5),
    (["spine1","spine_02"],                                             6),
    (["leftfoot","l_foot","foot_l"],                                    7),
    (["rightfoot","r_foot","foot_r"],                                   8),
    (["spine2","spine_03","chest"],                                     9),
    (["lefttoebase","lefttoe","l_toe","toe_l"],                        10),
    (["righttoebase","righttoe","r_toe","toe_r"],                      11),
    (["neck"],                                                         12),
    (["leftshoulder","leftcollar","clavicle_l"],                       13),
    (["rightshoulder","rightcollar","clavicle_r"],                     14),
    (["head"],                                                         15),
    (["leftarm","upperarm_l","l_arm"],                                 16),
    (["rightarm","upperarm_r","r_arm"],                                17),
    (["leftforearm","lowerarm_l","l_forearm"],                         18),
    (["rightforearm","lowerarm_r","r_forearm"],                        19),
    (["lefthand","hand_l","l_hand"],                                   20),
    (["righthand","hand_r","r_hand"],                                  21),
]


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion helpers  (scalar-first WXYZ convention throughout)
# ──────────────────────────────────────────────────────────────────────────────

_ID_QUAT = np.array([1., 0., 0., 0.], dtype=np.float32)
_ID_MAT3 = np.eye(3, dtype=np.float64)

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float32)

def _qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return (q / n) if n > 1e-12 else _ID_QUAT.copy()

def _qinv(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse for unit quaternion."""
    return q * np.array([1., -1., -1., -1.], dtype=np.float32)

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """WXYZ quaternion -> 3x3 rotation matrix (float64)."""
    w, x, y, z = q.astype(np.float64)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)

def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> WXYZ quaternion (float32, positive-W)."""
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(m.astype(np.float64)).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    if wxyz[0] < 0:
        wxyz = -wxyz
    return wxyz

def _r_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Minimal rotation matrix (3x3) that maps unit vector u to unit vector v.
    Uses the Rodrigues formula; handles parallel/antiparallel cases.
    """
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.dot(u, v))
    if c >= 1.0 - 1e-7:
        return _ID_MAT3.copy()
    if c <= -1.0 + 1e-7:
        # 180 degree rotation: pick any perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
        ax = np.cross(u, perp)
        ax /= np.linalg.norm(ax)
        return 2.0 * np.outer(ax, ax) - _ID_MAT3
    ax = np.cross(u, v)                    # sin(theta) * rotation axis
    s  = np.linalg.norm(ax)
    K  = np.array([[    0, -ax[2],  ax[1]],
                   [ ax[2],     0, -ax[0]],
                   [-ax[1],  ax[0],     0]], dtype=np.float64)
    return _ID_MAT3 + K + K @ K * ((1.0 - c) / (s * s + 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# GLB skin reader
# ──────────────────────────────────────────────────────────────────────────────

def _read_glb_skin(rigged_glb: str):
    """
    Return (gltf, skin, ibm[n,4,4], node_trs{name->(t,r_wxyz,s)},
            bone_names[], bone_parent_map{name->parent_name_or_None}).

    ibm is stored as-read from the binary blob (column-major from glTF spec).
    Callers must use inv(ibm[i].T)[:3,3] to get correct world positions.
    """
    import base64
    import pygltflib

    gltf = pygltflib.GLTF2().load(rigged_glb)
    if not gltf.skins:
        raise ValueError(f"No skin found in {rigged_glb}")
    skin = gltf.skins[0]

    def _raw_bytes(buf):
        if buf.uri is None:
            return bytes(gltf.binary_blob())
        if buf.uri.startswith("data:"):
            return base64.b64decode(buf.uri.split(",", 1)[1])
        from pathlib import Path
        return (Path(rigged_glb).parent / buf.uri).read_bytes()

    acc   = gltf.accessors[skin.inverseBindMatrices]
    bv    = gltf.bufferViews[acc.bufferView]
    raw   = _raw_bytes(gltf.buffers[bv.buffer])
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    n     = acc.count
    ibm   = np.frombuffer(raw[start: start + n * 64], dtype=np.float32).reshape(n, 4, 4)

    # Build node parent map (node_index -> parent_node_index)
    node_parent: dict[int, int] = {}
    for ni, node in enumerate(gltf.nodes):
        for child_idx in (node.children or []):
            node_parent[child_idx] = ni

    joint_set     = set(skin.joints)
    bone_names    = []
    node_trs: dict[str, tuple] = {}
    bone_parent_map: dict[str, str | None] = {}

    for i, j_idx in enumerate(skin.joints):
        node = gltf.nodes[j_idx]
        name = node.name or f"bone_{i}"
        bone_names.append(name)

        t      = np.array(node.translation or [0., 0., 0.], dtype=np.float32)
        r_xyzw = np.array(node.rotation    or [0., 0., 0., 1.], dtype=np.float32)
        s      = np.array(node.scale       or [1., 1., 1.], dtype=np.float32)
        r_wxyz = np.array([r_xyzw[3], r_xyzw[0], r_xyzw[1], r_xyzw[2]], dtype=np.float32)
        node_trs[name] = (t, r_wxyz, s)

        # Find parent bone (walk up node hierarchy to nearest joint)
        parent_node = node_parent.get(j_idx)
        parent_name: str | None = None
        while parent_node is not None:
            if parent_node in joint_set:
                pnode = gltf.nodes[parent_node]
                parent_name = pnode.name or f"bone_{skin.joints.index(parent_node)}"
                break
            parent_node = node_parent.get(parent_node)
        bone_parent_map[name] = parent_name

    print(f"[GLB] {len(bone_names)} bones from skin '{skin.name or 'Armature'}'")
    return gltf, skin, ibm, node_trs, bone_names, bone_parent_map


# ──────────────────────────────────────────────────────────────────────────────
# Bone mapping
# ──────────────────────────────────────────────────────────────────────────────

def _strip_name(name: str) -> str:
    name = re.sub(r'^(mixamorig:|j_bip_[lcr]_|cc_base_|bip01_|rig:|chr:)',
                  "", name, flags=re.IGNORECASE)
    return re.sub(r'[_\-\s.]', "", name).lower()


def build_bone_map(
    rigged_glb: str,
    verbose: bool = True,
) -> tuple[dict, dict, float, dict, dict]:
    """
    Map UniRig bone names -> SMPL joint index by spatial proximity + name hints.

    Returns
    -------
    bone_to_smpl    : {bone_name: smpl_joint_index}
    node_trs        : {bone_name: (t[3], r_wxyz[4], s[3])}
    height_scale    : float  (UniRig height / SMPL reference height)
    bone_parent_map : {bone_name: parent_bone_name_or_None}
    ur_pos_by_name  : {bone_name: world_pos[3]}
    """
    _gltf, _skin, ibm, node_trs, bone_names, bone_parent_map = _read_glb_skin(rigged_glb)

    # FIX: glTF IBMs are stored column-major.
    # numpy reads as row-major, so the stored data is the TRANSPOSE of the actual matrix.
    # Correct world position = inv(actual_IBM)[:3,3] = inv(ibm[i].T)[:3,3]
    ur_pos = np.array([
        np.linalg.inv(ibm[i].T)[:3, 3] for i in range(len(bone_names))
    ], dtype=np.float32)

    ur_pos_by_name = {name: ur_pos[i] for i, name in enumerate(bone_names)}

    # Scale SMPL T-pose to match character height
    ur_h = ur_pos[:, 1].max() - ur_pos[:, 1].min()
    sm_h = SMPL_TPOSE[:, 1].max() - SMPL_TPOSE[:, 1].min()
    h_sc = (ur_h / sm_h) if sm_h > 1e-6 else 1.0
    sm_pos = SMPL_TPOSE * h_sc

    # FIX: Normalise ur and smpl Y ranges independently (floor=0, top=1 for each).
    # The old code used a shared reference which caused floor offsets to misalign.
    def _norm_independent(pos, own_range_min, own_range_max, x_range, z_range):
        p = pos.copy().astype(np.float64)
        y_range = (own_range_max - own_range_min) or 1.0
        p[:, 0] /= (x_range or 1.0)
        p[:, 1]  = (p[:, 1] - own_range_min) / y_range
        p[:, 2] /= (z_range or 1.0)
        return p

    # Common X/Z scale (use both skeletons' width for reference)
    x_range = max(
        abs(ur_pos[:, 0].max()  - ur_pos[:, 0].min()),
        abs(sm_pos[:, 0].max()  - sm_pos[:, 0].min()),
    ) or 1.0
    z_range = max(
        abs(ur_pos[:, 2].max()  - ur_pos[:, 2].min()),
        abs(sm_pos[:, 2].max()  - sm_pos[:, 2].min()),
    ) or 1.0

    ur_n = _norm_independent(ur_pos, ur_pos[:, 1].min(), ur_pos[:, 1].max(), x_range, z_range)
    sm_n = _norm_independent(sm_pos, sm_pos[:, 1].min(), sm_pos[:, 1].max(), x_range, z_range)

    dist  = np.linalg.norm(ur_n[:, None] - sm_n[None], axis=-1)   # [M, 22]
    d_sc  = 1.0 - np.clip(dist / (dist.max() + 1e-9), 0, 1)

    # Name hint score
    n_sc = np.zeros((len(bone_names), 22), dtype=np.float32)
    for mi, bname in enumerate(bone_names):
        stripped = _strip_name(bname)
        for kws, ji in _NAME_HINTS:
            if any(kw in stripped for kw in kws):
                n_sc[mi, ji] = 1.0

    combined = 0.6 * d_sc + 0.4 * n_sc   # [M, 22]

    # Greedy assignment
    THRESHOLD = 0.35
    pairs = sorted(
        ((mi, ji, combined[mi, ji])
         for mi in range(len(bone_names))
         for ji in range(22)),
        key=lambda x: -x[2],
    )
    bone_to_smpl: dict[str, int] = {}
    taken: set[int] = set()
    for mi, ji, score in pairs:
        if score < THRESHOLD:
            break
        bname = bone_names[mi]
        if bname in bone_to_smpl or ji in taken:
            continue
        bone_to_smpl[bname] = ji
        taken.add(ji)

    if verbose:
        n_mapped = len(bone_to_smpl)
        print(f"\n[MAP] {n_mapped}/{len(bone_names)} bones mapped to SMPL joints:")
        for bname, ji in sorted(bone_to_smpl.items(), key=lambda x: x[1]):
            print(f"       {bname:<40} -> {SMPL_NAMES[ji]}")
        unmapped = [n for n in bone_names if n not in bone_to_smpl]
        if unmapped:
            preview = ", ".join(unmapped[:8])
            print(f"[MAP] {len(unmapped)} unmapped (identity): {preview}"
                  + (" ..." if len(unmapped) > 8 else ""))
        print()

    return bone_to_smpl, node_trs, h_sc, bone_parent_map, ur_pos_by_name


# ──────────────────────────────────────────────────────────────────────────────
# T2M forward kinematics (world rotation matrices)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_t2m_world_rots(
    root_rot_wxyz: np.ndarray,      # [4] WXYZ
    local_rots_wxyz: np.ndarray,    # [21, 4] WXYZ (joints 1-21)
) -> np.ndarray:
    """
    Compute accumulated world rotation matrices for all 22 t2m joints at one frame.
    Matches skeleton.py's forward_kinematics_cont6d_np: each chain RESETS to R_root.

    Returns [22, 3, 3] world rotation matrices.
    """
    R_root = _quat_to_mat(root_rot_wxyz)
    world_rots = np.zeros((22, 3, 3), dtype=np.float64)
    world_rots[0] = R_root

    for chain in T2M_KINEMATIC_CHAIN:
        R = R_root.copy()              # always start from R_root (matches skeleton.py)
        for i in range(1, len(chain)):
            j = chain[i]
            R_local = _quat_to_mat(local_rots_wxyz[j - 1])  # j-1: joints 1-21
            R = R @ R_local
            world_rots[j] = R

    return world_rots


# ──────────────────────────────────────────────────────────────────────────────
# Keyframe builder — world-direction matching
# ──────────────────────────────────────────────────────────────────────────────

def build_keyframes(
    motion:          SMPLMotion,
    bone_to_smpl:    dict[str, int],
    node_trs:        dict[str, tuple],
    height_scale:    float,
    bone_parent_map: dict[str, str | None],
    ur_pos_by_name:  dict[str, np.ndarray],
) -> list[dict]:
    """
    Convert SMPLMotion -> List[Dict[bone_name -> (loc, rot_delta, scale)]]
    using world-direction matching retargeting.
    """
    T      = motion.num_frames
    zeros3 = np.zeros(3, dtype=np.float32)
    ones3  = np.ones(3,  dtype=np.float32)

    # Topological order: root joints (si==0) first, then by SMPL joint index
    # (parents always have lower SMPL indices in the kinematic chain)
    sorted_bones = sorted(bone_to_smpl.keys(), key=lambda b: bone_to_smpl[b])

    keyframes: list[dict] = []

    for ti in range(T):
        frame: dict = {}

        # T2M world rotation matrices for this frame
        world_rots_t2m = _compute_t2m_world_rots(
            motion.root_rot[ti].astype(np.float64),
            motion.local_rot[ti].astype(np.float64),
        )

        # Track UniRig world rotations per bone (needed for child local rotations)
        world_rot_ur: dict[str, np.ndarray] = {}

        for bname in sorted_bones:
            si = bone_to_smpl[bname]
            rest_t, rest_r, _rest_s = node_trs[bname]
            rest_t = rest_t.astype(np.float32)
            rest_r_mat = _quat_to_mat(rest_r)

            # ── Root bone (si == 0): drive world translation + facing rotation ──
            if si == 0:
                world_pos = motion.root_pos[ti].astype(np.float64) * height_scale
                pose_loc  = (world_pos - rest_t.astype(np.float64)).astype(np.float32)

                # Root world rotation = t2m root rotation (Y-axis only)
                R_world_root = _quat_to_mat(motion.root_rot[ti])
                world_rot_ur[bname] = R_world_root

                # pose_rot_delta = inv(rest_r) @ target_world_rot
                pose_rot_mat = rest_r_mat.T @ R_world_root
                pose_rot     = _mat_to_quat(pose_rot_mat)
                frame[bname] = (pose_loc, pose_rot, ones3)
                continue

            # ── Non-root bone: world-direction matching ──────────────────────

            # T2M world bone direction (in t2m coordinate frame)
            raw_dir_t2m = world_rots_t2m[si] @ T2M_RAW_OFFSETS[si]  # [3]

            # COORDINATE FRAME CONVERSION: t2m +X = character LEFT; SMPL +X = character RIGHT
            # Flip X to convert t2m world directions -> SMPL/UniRig world directions
            d_desired = np.array([-raw_dir_t2m[0], raw_dir_t2m[1], raw_dir_t2m[2]])
            d_desired_norm = d_desired / (np.linalg.norm(d_desired) + 1e-12)

            # UniRig rest bone direction (from inverse bind matrices, world space)
            parent_b = bone_parent_map.get(bname)
            if parent_b and parent_b in ur_pos_by_name:
                d_rest = (ur_pos_by_name[bname] - ur_pos_by_name[parent_b]).astype(np.float64)
            else:
                d_rest = ur_pos_by_name[bname].astype(np.float64)
            d_rest_norm = d_rest / (np.linalg.norm(d_rest) + 1e-12)

            # Minimal world-space rotation: rest direction -> desired direction
            R_world_desired = _r_between(d_rest_norm, d_desired_norm)  # [3, 3]
            world_rot_ur[bname] = R_world_desired

            # Local rotation = inv(parent_world) @ R_world_desired
            if parent_b and parent_b in world_rot_ur:
                R_parent = world_rot_ur[parent_b]
            else:
                R_parent = _ID_MAT3

            local_rot_mat = R_parent.T @ R_world_desired   # R_parent^-1 @ R_world

            # pose_rot_delta = inv(rest_r) @ local_rot
            # (glTF applies: final = rest_r @ pose_rot_delta = local_rot)
            pose_rot_mat = rest_r_mat.T @ local_rot_mat
            pose_rot     = _mat_to_quat(pose_rot_mat)

            frame[bname] = (zeros3, pose_rot, ones3)

        keyframes.append(frame)

    return keyframes


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def animate_glb(
    motion:      Union[np.ndarray, list, SMPLMotion],
    rigged_glb:  str,
    output_glb:  str,
    fps:         float = 20.0,
    start_frame: int   = 0,
    num_frames:  int   = -1,
) -> str:
    """
    Bake a HumanML3D motion clip onto a UniRig-rigged GLB.

    Parameters
    ----------
    motion       : [T, 263] ndarray, list, or pre-parsed SMPLMotion
    rigged_glb   : path to UniRig merge output (.glb with a skin)
    output_glb   : destination path for animated GLB
    fps          : frame rate embedded in the animation track
    start_frame / num_frames : optional clip range (-1 = all frames)

    Returns str absolute path to output_glb.
    """
    from .io.gltf_io import write_gltf_animation

    # 1. Parse motion
    if isinstance(motion, SMPLMotion):
        smpl = motion
    else:
        data = np.asarray(motion, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 193:
            raise ValueError(f"Expected [T, 263] HumanML3D features, got {data.shape}")
        smpl = hml3d_to_smpl_motion(data, fps=fps)

    # 2. Slice
    end  = (start_frame + num_frames) if num_frames > 0 else smpl.num_frames
    smpl = smpl.slice(start_frame, end)
    print(f"[animate] {smpl.num_frames} frames @ {fps:.0f} fps  ->  {output_glb}")

    # 3. Build bone map (now returns parent map and world positions too)
    bone_to_smpl, node_trs, h_sc, bone_parent_map, ur_pos_by_name = \
        build_bone_map(rigged_glb, verbose=True)
    if not bone_to_smpl:
        raise RuntimeError(
            "build_bone_map returned 0 matches. "
            "Ensure the GLB has a valid skin with readable inverse bind matrices."
        )

    # 4. Build keyframes using world-direction matching
    keyframes = build_keyframes(smpl, bone_to_smpl, node_trs, h_sc,
                                bone_parent_map, ur_pos_by_name)

    # 5. Write GLB
    out_dir = os.path.dirname(os.path.abspath(output_glb))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_gltf_animation(
        source_filepath=rigged_glb,
        dest_armature=None,
        keyframes=keyframes,
        output_filepath=output_glb,
        fps=float(fps),
    )

    return output_glb


# Backwards-compatibility alias
def animate_glb_from_hml3d(
    motion, rigged_glb, output_glb, fps=20, start_frame=0, num_frames=-1
):
    return animate_glb(
        motion, rigged_glb, output_glb,
        fps=fps, start_frame=start_frame, num_frames=num_frames,
    )
