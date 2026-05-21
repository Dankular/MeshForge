"""
test_rotation_validation.py
────────────────────────────────────────────────────────────────────────────────
Diagnostic: prove whether the HumanML3D 6D rotations are in t2m canonical
frame or SMPL T-pose frame, by FK-reconstructing ground-truth positions.

Expected result
───────────────
  PASS: FK on t2m skeleton (arms canonical = [0,-1,0]) reproduces dims[4:67]
        with near-zero error (~1mm or less)
  FAIL: FK on SMPL/UniRig skeleton (arms = [±1,0,0]) shows large arm errors

If both pass → rotations are frame-agnostic (e.g. identity at T-pose)
If t2m passes and SMPL fails → rotations are t2m-canonical-frame  ← expected
This directly identifies the root cause of the mangled animation retargeting.

Usage
─────
    python test_rotation_validation.py
    python test_rotation_validation.py --verbose    # per-joint per-frame detail
    python test_rotation_validation.py --frames 10  # validate first N frames
"""
from __future__ import annotations
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# t2m canonical skeleton  (from HumanML3D/common/paramUtil.py)
# ──────────────────────────────────────────────────────────────────────────────

T2M_RAW_OFFSETS = np.array([
    [ 0, 0, 0],   # 0  Hips
    [ 1, 0, 0],   # 1  LeftUpLeg
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
    [ 1, 0, 0],   # 13 LeftShoulder (collar)
    [-1, 0, 0],   # 14 RightShoulder (collar)
    [ 0, 0, 1],   # 15 Head
    [ 0,-1, 0],   # 16 LeftArm       ← arms point DOWN, not sideways
    [ 0,-1, 0],   # 17 RightArm      ← arms point DOWN, not sideways
    [ 0,-1, 0],   # 18 LeftForeArm
    [ 0,-1, 0],   # 19 RightForeArm
    [ 0,-1, 0],   # 20 LeftHand
    [ 0,-1, 0],   # 21 RightHand
], dtype=np.float64)

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],          # Hips → RightUpLeg → RightLeg → RightFoot → RightToeBase
    [0, 1, 4, 7, 10],          # Hips → LeftUpLeg  → LeftLeg  → LeftFoot  → LeftToeBase
    [0, 3, 6, 9, 12, 15],      # Hips → Spine → Spine1 → Spine2 → Neck → Head
    [9, 14, 17, 19, 21],        # Spine2 → RightShoulder → RightArm → RightForeArm → RightHand
    [9, 13, 16, 18, 20],        # Spine2 → LeftShoulder  → LeftArm  → LeftForeArm  → LeftHand
]

# Build parent array from kinematic chains
T2M_PARENTS = [-1] * 22
for chain in T2M_KINEMATIC_CHAIN:
    for k in range(1, len(chain)):
        T2M_PARENTS[chain[k]] = chain[k - 1]

SMPL_JOINT_NAMES = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine",
    "LeftLeg", "RightLeg", "Spine1", "LeftFoot",
    "RightFoot", "Spine2", "LeftToeBase", "RightToeBase",
    "Neck", "LeftShoulder", "RightShoulder", "Head",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand",
]

# SMPL/UniRig T-pose offsets: arms are HORIZONTAL (±X), not down
# These match the bone directions seen in UniRig auto-rigged GLBs
SMPL_RAW_OFFSETS = np.array([
    [ 0, 0, 0],   # 0  Hips
    [-1, 0, 0],   # 1  LeftUpLeg   (pelvis to left hip)
    [ 1, 0, 0],   # 2  RightUpLeg
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
    [-1, 0, 0],   # 13 LeftShoulder  (spine2 to left collar)
    [ 1, 0, 0],   # 14 RightShoulder
    [ 0, 1, 0],   # 15 Head
    [-1, 0, 0],   # 16 LeftArm       ← arms point SIDEWAYS (-X for left)
    [ 1, 0, 0],   # 17 RightArm      ← arms point SIDEWAYS (+X for right)
    [-1, 0, 0],   # 18 LeftForeArm
    [ 1, 0, 0],   # 19 RightForeArm
    [-1, 0, 0],   # 20 LeftHand
    [ 1, 0, 0],   # 21 RightHand
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers  (no torch dependency)
# ──────────────────────────────────────────────────────────────────────────────

def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """[..., 6] → [..., 3, 3] via Gram-Schmidt.  Columns = [b1, b2, b3]."""
    a1 = r6d[..., 0:3].astype(np.float64)
    a2 = r6d[..., 3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)   # [..., 3, 3] column-major


def qrot_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by quaternions q.  q=[w,x,y,z], shapes [...,4] and [...,3]."""
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return np.concatenate([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth recovery from HumanML3D [T, 263]
# ──────────────────────────────────────────────────────────────────────────────

def recover_root(data: np.ndarray):
    """
    Returns root_pos [T,3] and root_rot_quat [T,4] WXYZ from [T,263].
    Root rotation is Y-axis only (facing direction).
    """
    T = data.shape[0]
    theta = np.cumsum(data[:, 0].astype(np.float64))
    half  = theta * 0.5
    root_rot = np.zeros((T, 4), dtype=np.float64)
    root_rot[:, 0] = np.cos(half)   # w
    root_rot[:, 2] = np.sin(half)   # y  (Y-axis rotation)

    vel_local = np.stack([
        data[:, 2].astype(np.float64),
        np.zeros(T),
        data[:, 3].astype(np.float64),
    ], axis=-1)
    vel_world = qrot_np(root_rot, vel_local)

    root_pos = np.zeros((T, 3), dtype=np.float64)
    root_pos[:, 0] = np.cumsum(vel_world[:, 0])
    root_pos[:, 1] = data[:, 1].astype(np.float64)
    root_pos[:, 2] = np.cumsum(vel_world[:, 2])
    return root_pos, root_rot


def recover_world_positions(data: np.ndarray) -> np.ndarray:
    """
    Reconstruct world positions for all 22 joints from [T, 263].
    dims[4:67] = 21 joints (1-21) relative to root, in ROOT-LOCAL frame.
    Returns [T, 22, 3].
    """
    T = data.shape[0]
    root_pos, root_rot = recover_root(data)

    # dims[4:67]: 21 joints relative to root in local (root-facing) frame
    rel_pos = data[:, 4:67].astype(np.float64).reshape(T, 21, 3)

    # Rotate from root-local to world frame
    rr_rep = np.repeat(root_rot[:, np.newaxis], 21, axis=1)   # [T,21,4]
    world_j1_21 = qrot_np(rr_rep, rel_pos) + root_pos[:, np.newaxis]  # [T,21,3]

    world = np.zeros((T, 22, 3), dtype=np.float64)
    world[:, 0]  = root_pos
    world[:, 1:] = world_j1_21
    return world


# ──────────────────────────────────────────────────────────────────────────────
# FK engine
# ──────────────────────────────────────────────────────────────────────────────

def fk_from_cont6d(
    cont6d_root: np.ndarray,   # [T, 6]  root 6D rotation
    cont6d_joints: np.ndarray, # [T, 21, 6]  joints 1-21
    offsets: np.ndarray,       # [22, 3]  bone offset vectors (pre-scaled)
    root_pos: np.ndarray,      # [T, 3]
    kinematic_chain=T2M_KINEMATIC_CHAIN,
) -> np.ndarray:
    """
    Forward kinematics exactly matching skeleton.py's forward_kinematics_cont6d_np.
    Returns [T, 22, 3] world joint positions.
    """
    T = cont6d_root.shape[0]
    joints = np.zeros((T, 22, 3), dtype=np.float64)
    joints[:, 0] = root_pos

    # Pack all 22 joints' 6D into single array [T, 22, 6]
    cont6d = np.concatenate([cont6d_root[:, np.newaxis], cont6d_joints], axis=1)

    for chain in kinematic_chain:
        matR = rot6d_to_matrix(cont6d[:, chain[0]])  # [T, 3, 3]
        for i in range(1, len(chain)):
            ji = chain[i]
            local_mat = rot6d_to_matrix(cont6d[:, ji])   # [T, 3, 3]
            matR = matR @ local_mat                        # [T, 3, 3]
            off = offsets[ji][np.newaxis, :, np.newaxis]  # [1, 3, 1]
            # pos[ji] = pos[prev] + matR @ offset
            joints[:, ji] = (matR @ off).squeeze(-1) + joints[:, chain[i - 1]]

    return joints


# ──────────────────────────────────────────────────────────────────────────────
# Root rotation: Y-axis quaternion → 6D
# ──────────────────────────────────────────────────────────────────────────────

def root_quat_to_cont6d(q_wxyz: np.ndarray) -> np.ndarray:
    """[T,4] WXYZ quaternion → [T,6] 6D (first two columns of rotation matrix)."""
    T = q_wxyz.shape[0]
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    # Column 0 of rotation matrix (rotated X axis)
    col0 = np.stack([
        1 - 2*(y*y + z*z),
        2*(x*y + w*z),
        2*(x*z - w*y),
    ], axis=-1)
    # Column 1 of rotation matrix (rotated Y axis)
    col1 = np.stack([
        2*(x*y - w*z),
        1 - 2*(x*x + z*z),
        2*(y*z + w*x),
    ], axis=-1)
    return np.concatenate([col0, col1], axis=-1)  # [T, 6]


# ──────────────────────────────────────────────────────────────────────────────
# Bone length computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_bone_lengths(
    world_pos: np.ndarray,         # [T, 22, 3]
    parents=T2M_PARENTS,
) -> np.ndarray:
    """Average distance between each joint and its parent across all frames."""
    lengths = np.zeros(22, dtype=np.float64)
    for j in range(1, 22):
        p = parents[j]
        diff = world_pos[:, j] - world_pos[:, p]        # [T, 3]
        lengths[j] = np.mean(np.linalg.norm(diff, axis=-1))
    return lengths


def make_offsets(raw_offsets: np.ndarray, bone_lengths: np.ndarray) -> np.ndarray:
    """Scale unit direction vectors by bone lengths → actual bone offset vectors."""
    offsets = raw_offsets.copy()
    for j in range(1, 22):
        offsets[j] = raw_offsets[j] * bone_lengths[j]
    return offsets


# ──────────────────────────────────────────────────────────────────────────────
# Main test
# ──────────────────────────────────────────────────────────────────────────────

def run_test(data: np.ndarray, n_frames: int, verbose: bool):
    T = min(n_frames, data.shape[0])
    data = data[:T].astype(np.float64)

    print(f"\n{'='*60}")
    print(f"  Sample: {T} frames, {T/20:.1f}s @ 20 fps")
    print(f"{'='*60}")

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_world = recover_world_positions(data)          # [T, 22, 3]
    root_pos, root_rot = recover_root(data)

    # ── 6D rotations from features ────────────────────────────────────────────
    cont6d_joints = data[:, 67:193].reshape(T, 21, 6)  # joints 1-21
    cont6d_root   = root_quat_to_cont6d(root_rot)       # from integrated Y-axis vel

    # ── Bone lengths from ground truth ────────────────────────────────────────
    bone_lengths = compute_bone_lengths(gt_world)

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1: FK with t2m canonical offsets (arms DOWN)
    # Should reproduce ground truth with ~0 error
    # ──────────────────────────────────────────────────────────────────────────
    t2m_offsets = make_offsets(T2M_RAW_OFFSETS, bone_lengths)
    fk_t2m = fk_from_cont6d(cont6d_root, cont6d_joints, t2m_offsets, root_pos)

    err_t2m = np.linalg.norm(fk_t2m - gt_world, axis=-1)  # [T, 22]

    print("\n[TEST 1] FK with t2m canonical offsets (arms=[0,-1,0])")
    print(f"         Expected: near-zero error -> rotations ARE in t2m frame")
    print(f"  {'-'*58}")
    print(f"  {'Joint':<20} {'Mean err (mm)':>14}  {'Max err (mm)':>13}")
    print(f"  {'-'*58}")
    for j in range(22):
        mean_mm = err_t2m[:, j].mean() * 1000
        max_mm  = err_t2m[:, j].max()  * 1000
        flag = " <<" if mean_mm > 10 else ""
        print(f"  {SMPL_JOINT_NAMES[j]:<20} {mean_mm:>13.1f}  {max_mm:>13.1f}{flag}")
    print(f"{'─'*60}")
    arm_joints = [16, 17, 18, 19, 20, 21]
    arm_err_t2m = err_t2m[:, arm_joints].mean() * 1000
    total_err_t2m = err_t2m.mean() * 1000
    print(f"  {'ALL joints avg':<20} {total_err_t2m:>13.1f}  mm")
    print(f"  {'ARM joints avg':<20} {arm_err_t2m:>13.1f}  mm")

    if total_err_t2m < 5.0:
        print(f"\n  RESULT: PASS  — t2m FK matches ground truth (< 5mm avg error)")
    else:
        print(f"\n  RESULT: FAIL  — t2m FK does NOT match ground truth!")
        print(f"          (This may indicate a bug in the 6D extraction or FK logic)")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: FK with SMPL/UniRig offsets (arms SIDEWAYS ±X)
    # Should show large errors at arm joints IF test 1 passed
    # ──────────────────────────────────────────────────────────────────────────
    smpl_offsets = make_offsets(SMPL_RAW_OFFSETS, bone_lengths)
    fk_smpl = fk_from_cont6d(cont6d_root, cont6d_joints, smpl_offsets, root_pos)

    err_smpl = np.linalg.norm(fk_smpl - gt_world, axis=-1)  # [T, 22]

    print(f"\n[TEST 2] FK with SMPL/UniRig offsets (arms=[±1,0,0])")
    print(f"         Expected: large arm error → direct retargeting is WRONG")
    print(f"{'─'*60}")
    print(f"  {'Joint':<20} {'Mean err (mm)':>14}  {'Max err (mm)':>13}")
    print(f"{'─'*60}")
    for j in range(22):
        mean_mm = err_smpl[:, j].mean() * 1000
        max_mm  = err_smpl[:, j].max()  * 1000
        flag = " ← BAD" if j in arm_joints and mean_mm > 20 else ""
        print(f"  {SMPL_JOINT_NAMES[j]:<20} {mean_mm:>13.1f}  {max_mm:>13.1f}{flag}")
    print(f"{'─'*60}")
    arm_err_smpl = err_smpl[:, arm_joints].mean() * 1000
    total_err_smpl = err_smpl.mean() * 1000
    print(f"  {'ALL joints avg':<20} {total_err_smpl:>13.1f}  mm")
    print(f"  {'ARM joints avg':<20} {arm_err_smpl:>13.1f}  mm")

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnosis
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS")
    print(f"{'='*60}")

    t2m_pass  = total_err_t2m  < 5.0
    smpl_fail = arm_err_smpl   > 20.0

    if t2m_pass and smpl_fail:
        print(f"""
  [CONFIRMED] 6D rotations are in t2m CANONICAL frame.
  Arms canonical direction in t2m = [0,-1,0] (pointing DOWN).
  Arms canonical direction in UniRig = [+-1,0,0] (pointing SIDEWAYS).

  Current animate.py applies these rotations DIRECTLY to UniRig bones,
  which causes the arm/torso mangling seen in the output.

  FIX NEEDED: per-joint rest-pose correction before applying to GLB.
  For each joint j:
    C[j] = R_between(normalize(t2m_raw_offsets[j]),
                      normalize(ur_bone_direction[j]))
    R_local_ur[j] = C[parent[j]] @ R_local_t2m[j] @ inv(C[j])

  Then pass R_local_ur into gltf_io instead of the raw 6D quaternion.
""")
    elif t2m_pass and not smpl_fail:
        print(f"""
  [UNEXPECTED] t2m FK passes but SMPL FK also has low arm error.
  The 6D rotations may be approximately skeleton-agnostic (e.g. near-identity
  for a walking motion). Try a motion with strong arm movement to re-test.
""")
    elif not t2m_pass:
        print(f"""
  [BUG IN TEST] t2m FK does not reproduce ground truth.
  Possible causes:
    - dims[4:67] interpretation is wrong (not root-relative positions)
    - 6D Gram-Schmidt column order is wrong (rows vs columns)
    - FK accumulation order wrong (matR = matR @ local vs local @ matR)
    - Root rotation integration is wrong

  Check and fix recover_world_positions() and fk_from_cont6d().
""")

    # ──────────────────────────────────────────────────────────────────────────
    # Verbose: per-joint per-frame detail
    # ──────────────────────────────────────────────────────────────────────────
    if verbose:
        print("\n[VERBOSE] Per-frame t2m FK error for arm joints:")
        print(f"  {'Frame':>5}  " + "  ".join(f"{SMPL_JOINT_NAMES[j]:>12}" for j in arm_joints))
        for t in range(T):
            row = "  ".join(f"{err_t2m[t,j]*1000:>11.1f}" for j in arm_joints)
            print(f"  {t:>5}  {row}")

    return total_err_t2m, arm_err_smpl


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate HumanML3D 6D rotation frame")
    parser.add_argument("--frames",  type=int, default=60,    help="Number of frames to test")
    parser.add_argument("--verbose", action="store_true",     help="Print per-frame arm errors")
    parser.add_argument("--query",   default="a person walks forward",
                        help="Motion query to search in HumanML3D dataset")
    args = parser.parse_args()

    print("Loading motion from HumanML3D dataset...")
    from Retarget.search import search_motions
    results = search_motions(args.query, top_k=1, split="test", max_scan=4384)
    if not results:
        print(f"No results for '{args.query}'. Trying train split...")
        results = search_motions(args.query, top_k=1, split="train", max_scan=2000)
    if not results:
        print("ERROR: No motion found. Check your HuggingFace dataset access.")
        sys.exit(1)

    r = results[0]
    print(f"Found: '{r['caption']}'  ({r['frames']} frames)")
    data = r["motion"]

    run_test(data, n_frames=args.frames, verbose=args.verbose)


if __name__ == "__main__":
    main()
