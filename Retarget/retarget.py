"""
retarget.py
Pure-Python port of KeeMapBoneOperators.py core math.

Replaces bpy / mathutils with numpy.  No Blender dependency.
Public API mirrors the Blender operator flow:

    get_bone_position_ws(bone, arm)          → np.ndarray(3)
    get_bone_ws_quat(bone, arm)              → np.ndarray(4)  w,x,y,z
    set_bone_position_ws(bone, arm, pos)
    set_bone_rotation(...)
    set_bone_position(...)
    set_bone_position_pole(...)
    set_bone_scale(...)
    calc_rotation_offset(bone_item, src_arm, dst_arm, settings)
    calc_location_offset(bone_item, src_arm, dst_arm)
    transfer_frame(src_arm, dst_arm, bone_items, settings, do_keyframe)
      → Dict[bone_name → (pose_loc, pose_rot, pose_scale)]
    transfer_animation(src_anim, dst_arm, bone_items, settings)
      → List[Dict[bone_name → (pose_loc, pose_rot, pose_scale)]]
"""
from __future__ import annotations
import math
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

from .skeleton import Armature, PoseBone
from .math3d import (
    quat_identity, quat_normalize, quat_mul, quat_conjugate,
    quat_rotation_difference, quat_dot,
    quat_to_matrix4, matrix4_to_quat,
    euler_to_quat, quat_to_euler,
    translation_matrix, vec3, get_point_on_vector,
)
from .io.mapping import BoneMappingItem, KeeMapSettings


# ---------------------------------------------------------------------------
# Progress bar (console)
# ---------------------------------------------------------------------------

def _update_progress(job: str, progress: float) -> None:
    length = 40
    block = int(round(length * progress))
    msg = f"\r{job}: [{'#'*block}{'-'*(length-block)}] {round(progress*100, 1)}%"
    if progress >= 1:
        msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# World-space position / quaternion getters
# ---------------------------------------------------------------------------

def get_bone_position_ws(bone: PoseBone, arm: Armature) -> np.ndarray:
    """
    Return world-space position of bone head.
    Equivalent to Blender's GetBonePositionWS().
    """
    ws_matrix = arm.world_matrix @ bone.matrix_armature
    return ws_matrix[:3, 3].copy()


def get_bone_ws_quat(bone: PoseBone, arm: Armature) -> np.ndarray:
    """
    Return world-space rotation as quaternion (w,x,y,z).
    Equivalent to Blender's GetBoneWSQuat().
    """
    ws_matrix = arm.world_matrix @ bone.matrix_armature
    return matrix4_to_quat(ws_matrix)


# ---------------------------------------------------------------------------
# World-space position setter
# ---------------------------------------------------------------------------

def set_bone_position_ws(bone: PoseBone, arm: Armature, position: np.ndarray) -> None:
    """
    Move bone so its world-space head = position.
    Equivalent to Blender's SetBonePositionWS().

    Strategy:
      1. Build new armature-space matrix = old rotation + new translation
      2. Strip parent transform to get new local translation
      3. Update pose_location so FK matches
    """
    # Current armature-space matrix (rotation/scale part preserved)
    arm_mat = bone.matrix_armature.copy()

    # Target armature-space position
    arm_world_inv = np.linalg.inv(arm.world_matrix)
    target_arm_pos = (arm_world_inv @ np.append(position, 1.0))[:3]

    # New armature-space matrix with replaced translation
    new_arm_mat = arm_mat.copy()
    new_arm_mat[:3, 3] = target_arm_pos

    # Convert to local (parent-relative) space
    if bone.parent is not None:
        parent_arm_mat = bone.parent.matrix_armature
        new_local = np.linalg.inv(parent_arm_mat) @ new_arm_mat
    else:
        new_local = new_arm_mat

    # Extract translation from new_local = rest_local @ T(pose_loc) @ ...
    # Approximate: strip rest_local rotation contribution to isolate pose_location
    rest_inv = np.linalg.inv(bone.rest_matrix_local)
    pose_delta = rest_inv @ new_local
    bone.pose_location = pose_delta[:3, 3].copy()

    # Recompute FK for this bone and its subtree
    if bone.parent is not None:
        bone._fk(bone.parent.matrix_armature)
    else:
        bone._fk(np.eye(4))


# ---------------------------------------------------------------------------
# Rotation setter (core retargeting math)
# ---------------------------------------------------------------------------

def set_bone_rotation(
    src_arm: Armature, src_name: str,
    dst_arm: Armature, dst_name: str,
    dst_twist_name: str,
    correction_quat: np.ndarray,
    has_twist: bool,
    xfer_axis: str,
    transpose: str,
    mode: str,
) -> None:
    """
    Port of Blender's SetBoneRotation().
    Drives dst bone rotation to match src bone world-space rotation.

    mode:       "EULER" | "QUATERNION"
    xfer_axis:  "X" "Y" "Z" "XY" "XZ" "YZ" "XYZ"
    transpose:  "NONE" "ZYX" "ZXY" "XZY" "YZX" "YXZ"
    """
    src_bone = src_arm.get_bone(src_name)
    dst_bone = dst_arm.get_bone(dst_name)

    # ------------------------------------------------------------------
    # Get source and destination world-space quaternions (current pose)
    # ------------------------------------------------------------------
    src_ws_quat = get_bone_ws_quat(src_bone, src_arm)
    dst_ws_quat = get_bone_ws_quat(dst_bone, dst_arm)

    # Rotation difference: r such that  dst_ws @ r ≈ src_ws
    diff = quat_rotation_difference(dst_ws_quat, src_ws_quat)

    # FinalQuat = dst_local_pose_delta @ diff @ correction
    final_quat = quat_normalize(
        quat_mul(quat_mul(dst_bone.pose_rotation_quat, diff), correction_quat)
    )

    # ------------------------------------------------------------------
    # Apply axis masking / transpose (EULER mode)
    # ------------------------------------------------------------------
    if mode == "EULER":
        euler = quat_to_euler(final_quat, order="XYZ")

        # Transpose axes
        if transpose == "ZYX":
            euler = np.array([euler[2], euler[1], euler[0]])
        elif transpose == "ZXY":
            euler = np.array([euler[2], euler[0], euler[1]])
        elif transpose == "XZY":
            euler = np.array([euler[0], euler[2], euler[1]])
        elif transpose == "YZX":
            euler = np.array([euler[1], euler[2], euler[0]])
        elif transpose == "YXZ":
            euler = np.array([euler[1], euler[0], euler[2]])
        # else NONE — no change

        # Mask axes
        if xfer_axis == "X":
            euler[1] = 0.0; euler[2] = 0.0
        elif xfer_axis == "Y":
            euler[0] = 0.0; euler[2] = 0.0
        elif xfer_axis == "Z":
            euler[0] = 0.0; euler[1] = 0.0
        elif xfer_axis == "XY":
            euler[2] = 0.0
        elif xfer_axis == "XZ":
            euler[1] = 0.0
        elif xfer_axis == "YZ":
            euler[0] = 0.0
        # XYZ → no masking

        final_quat = euler_to_quat(euler[0], euler[1], euler[2], order="XYZ")

        # Twist bone: peel Y rotation off to twist bone
        if has_twist and dst_twist_name:
            twist_bone = dst_arm.get_bone(dst_twist_name)
            y_euler = quat_to_euler(final_quat, order="XYZ")[1]
            # Remove Y from main bone
            euler_no_y = quat_to_euler(final_quat, order="XYZ")
            euler_no_y[1] = 0.0
            final_quat = euler_to_quat(*euler_no_y, order="XYZ")
            # Apply Y to twist bone
            twist_euler = quat_to_euler(twist_bone.pose_rotation_quat, order="XYZ")
            twist_euler[1] = math.degrees(y_euler)
            twist_bone.pose_rotation_quat = euler_to_quat(*twist_euler, order="XYZ")

    else:  # QUATERNION
        if final_quat[0] < 0:
            final_quat = -final_quat
        final_quat = quat_normalize(final_quat)

    dst_bone.pose_rotation_quat = final_quat

    # Recompute FK
    parent = dst_bone.parent
    dst_bone._fk(parent.matrix_armature if parent else np.eye(4))


# ---------------------------------------------------------------------------
# Position setter
# ---------------------------------------------------------------------------

def set_bone_position(
    src_arm: Armature, src_name: str,
    dst_arm: Armature, dst_name: str,
    dst_twist_name: str,
    correction: np.ndarray,
    gain: float,
) -> None:
    """
    Port of Blender's SetBonePosition().
    Moves dst bone to match src bone world-space position, with offset/gain.
    """
    src_bone = src_arm.get_bone(src_name)
    dst_bone = dst_arm.get_bone(dst_name)

    target_ws = get_bone_position_ws(src_bone, src_arm)
    set_bone_position_ws(dst_bone, dst_arm, target_ws)

    # Apply correction and gain to pose_location
    dst_bone.pose_location[0] = (dst_bone.pose_location[0] + correction[0]) * gain
    dst_bone.pose_location[1] = (dst_bone.pose_location[1] + correction[1]) * gain
    dst_bone.pose_location[2] = (dst_bone.pose_location[2] + correction[2]) * gain

    parent = dst_bone.parent
    dst_bone._fk(parent.matrix_armature if parent else np.eye(4))


# ---------------------------------------------------------------------------
# Pole bone position setter
# ---------------------------------------------------------------------------

def set_bone_position_pole(
    src_arm: Armature, src_name: str,
    dst_arm: Armature, dst_name: str,
    dst_twist_name: str,
    pole_distance: float,
) -> None:
    """
    Port of Blender's SetBonePositionPole().
    Positions an IK pole target relative to source limb geometry.
    """
    src_bone = src_arm.get_bone(src_name)
    dst_bone = dst_arm.get_bone(dst_name)

    parent_src = src_bone.parent_recursive[0] if src_bone.parent_recursive else src_bone

    base_parent_ws = get_bone_position_ws(parent_src, src_arm)
    base_child_ws = get_bone_position_ws(src_bone, src_arm)

    # Tail = head + Y-axis direction of bone in world space
    src_ws_mat = src_arm.world_matrix @ src_bone.matrix_armature
    tail_ws = src_ws_mat[:3, 3] + src_ws_mat[:3, :3] @ np.array([0.0, 1.0, 0.0])

    length_parent = np.linalg.norm(base_child_ws - base_parent_ws)
    length_child = np.linalg.norm(tail_ws - base_child_ws)
    total = length_parent + length_child

    c_p_ratio = length_parent / total if total > 1e-12 else 0.5

    length_pp_to_tail = np.linalg.norm(base_parent_ws - tail_ws)
    average_location = get_point_on_vector(base_parent_ws, tail_ws, length_pp_to_tail * c_p_ratio)

    distance = np.linalg.norm(base_child_ws - average_location)

    if distance > 0.001:
        pole_pos = get_point_on_vector(base_child_ws, average_location, pole_distance)
        set_bone_position_ws(dst_bone, dst_arm, pole_pos)
        parent = dst_bone.parent
        dst_bone._fk(parent.matrix_armature if parent else np.eye(4))


# ---------------------------------------------------------------------------
# Scale setter
# ---------------------------------------------------------------------------

def set_bone_scale(
    src_arm: Armature, src_name: str,
    dst_arm: Armature, dst_name: str,
    src_scale_bone_name: str,
    gain: float,
    axis: str,
    max_scale: float,
    min_scale: float,
) -> None:
    """
    Port of Blender's SetBoneScale().
    Scales dst bone based on dot product between two source bone quaternions.
    """
    src_bone = src_arm.get_bone(src_name)
    dst_bone = dst_arm.get_bone(dst_name)
    secondary = src_arm.get_bone(src_scale_bone_name)

    q1 = get_bone_ws_quat(src_bone, src_arm)
    q2 = get_bone_ws_quat(secondary, src_arm)
    amount = quat_dot(q1, q2) * gain

    if amount < 0:
        amount = -amount
    amount = max(min_scale, min(max_scale, amount))

    s = dst_bone.pose_scale
    if axis == "X":
        s[0] = amount
    elif axis == "Y":
        s[1] = amount
    elif axis == "Z":
        s[2] = amount
    elif axis == "XY":
        s[0] = s[1] = amount
    elif axis == "XZ":
        s[0] = s[2] = amount
    elif axis == "YZ":
        s[1] = s[2] = amount
    else:  # XYZ
        s[:] = amount

    parent = dst_bone.parent
    dst_bone._fk(parent.matrix_armature if parent else np.eye(4))


# ---------------------------------------------------------------------------
# Correction calculators
# ---------------------------------------------------------------------------

def calc_rotation_offset(
    item: BoneMappingItem,
    src_arm: Armature,
    dst_arm: Armature,
    settings: KeeMapSettings,
) -> None:
    """
    Auto-compute the rotation correction factor for one bone mapping.
    Port of Blender's CalcRotationOffset().
    Modifies item.correction_factor and item.quat_correction_factor in-place.
    """
    if not item.source_bone_name or not item.destination_bone_name:
        return
    if not src_arm.has_bone(item.source_bone_name):
        return
    if not dst_arm.has_bone(item.destination_bone_name):
        return

    dst_bone = dst_arm.get_bone(item.destination_bone_name)

    # Snapshot destination bone state
    snap_r = dst_bone.pose_rotation_quat.copy()
    snap_t = dst_bone.pose_location.copy()

    starting_ws_quat = get_bone_ws_quat(dst_bone, dst_arm)

    # Apply with identity correction
    set_bone_rotation(
        src_arm, item.source_bone_name,
        dst_arm, item.destination_bone_name,
        item.twist_bone_name,
        quat_identity(),
        False,
        item.bone_rotation_application_axis,
        item.bone_transpose_axis,
        settings.bone_rotation_mode,
    )
    dst_arm.update_fk()

    modified_ws_quat = get_bone_ws_quat(dst_bone, dst_arm)

    # Correction = rotation that takes modified_ws back to starting_ws
    q_diff = quat_rotation_difference(modified_ws_quat, starting_ws_quat)
    euler = quat_to_euler(q_diff, order="XYZ")
    item.correction_factor = euler.copy()
    item.quat_correction_factor = q_diff.copy()

    # Restore
    dst_bone.pose_rotation_quat = snap_r
    dst_bone.pose_location = snap_t
    parent = dst_bone.parent
    dst_bone._fk(parent.matrix_armature if parent else np.eye(4))


def calc_location_offset(
    item: BoneMappingItem,
    src_arm: Armature,
    dst_arm: Armature,
) -> None:
    """
    Auto-compute position correction for one bone mapping.
    Port of Blender's CalcLocationOffset().
    """
    if not item.source_bone_name or not item.destination_bone_name:
        return
    if not src_arm.has_bone(item.source_bone_name):
        return
    if not dst_arm.has_bone(item.destination_bone_name):
        return

    src_bone = src_arm.get_bone(item.source_bone_name)
    dst_bone = dst_arm.get_bone(item.destination_bone_name)

    source_ws_pos = get_bone_position_ws(src_bone, src_arm)
    dest_ws_pos = get_bone_position_ws(dst_bone, dst_arm)

    # Snapshot
    snap_loc = dst_bone.pose_location.copy()

    # Move dest to source position
    set_bone_position_ws(dst_bone, dst_arm, source_ws_pos)
    dst_arm.update_fk()
    moved_pose_loc = dst_bone.pose_location.copy()

    # Restore
    set_bone_position_ws(dst_bone, dst_arm, dest_ws_pos)
    dst_arm.update_fk()

    delta = snap_loc - moved_pose_loc
    item.position_correction_factor = delta.copy()


def calc_all_corrections(
    bone_items: List[BoneMappingItem],
    src_arm: Armature,
    dst_arm: Armature,
    settings: KeeMapSettings,
) -> None:
    """Auto-calculate rotation and position corrections for all mapped bones."""
    for item in bone_items:
        calc_rotation_offset(item, src_arm, dst_arm, settings)
        if "pole" not in item.name.lower():
            calc_location_offset(item, src_arm, dst_arm)


# ---------------------------------------------------------------------------
# Single-frame transfer
# ---------------------------------------------------------------------------

def transfer_frame(
    src_arm: Armature,
    dst_arm: Armature,
    bone_items: List[BoneMappingItem],
    settings: KeeMapSettings,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply retargeting for all bone mappings at the current source frame.
    src_arm must already have FK updated for the current frame.

    Returns a dict of  bone_name → (pose_location, pose_rotation_quat, pose_scale)
    suitable for writing into a keyframe list.
    """
    for item in bone_items:
        if not item.source_bone_name or not item.destination_bone_name:
            continue
        if not src_arm.has_bone(item.source_bone_name):
            continue
        if not dst_arm.has_bone(item.destination_bone_name):
            continue

        # Build correction quaternion
        if settings.bone_rotation_mode == "EULER":
            cf = item.correction_factor
            correction_quat = euler_to_quat(cf[0], cf[1], cf[2], order="XYZ")
        else:
            correction_quat = quat_normalize(item.quat_correction_factor)

        # Rotation
        if item.set_bone_rotation:
            set_bone_rotation(
                src_arm, item.source_bone_name,
                dst_arm, item.destination_bone_name,
                item.twist_bone_name,
                correction_quat,
                item.has_twist_bone,
                item.bone_rotation_application_axis,
                item.bone_transpose_axis,
                settings.bone_rotation_mode,
            )
            dst_arm.update_fk()

        # Position
        if item.set_bone_position:
            if item.postion_type == "SINGLE_BONE_OFFSET":
                set_bone_position(
                    src_arm, item.source_bone_name,
                    dst_arm, item.destination_bone_name,
                    item.twist_bone_name,
                    item.position_correction_factor,
                    item.position_gain,
                )
            else:
                set_bone_position_pole(
                    src_arm, item.source_bone_name,
                    dst_arm, item.destination_bone_name,
                    item.twist_bone_name,
                    -item.position_pole_distance,
                )
            dst_arm.update_fk()

        # Scale
        if item.set_bone_scale and item.scale_secondary_bone_name:
            if src_arm.has_bone(item.scale_secondary_bone_name):
                set_bone_scale(
                    src_arm, item.source_bone_name,
                    dst_arm, item.destination_bone_name,
                    item.scale_secondary_bone_name,
                    item.scale_gain,
                    item.bone_scale_application_axis,
                    item.scale_max,
                    item.scale_min,
                )
                dst_arm.update_fk()

    # Snapshot destination bone state for this frame
    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for item in bone_items:
        if not item.destination_bone_name:
            continue
        if not dst_arm.has_bone(item.destination_bone_name):
            continue
        dst_bone = dst_arm.get_bone(item.destination_bone_name)
        result[item.destination_bone_name] = (
            dst_bone.pose_location.copy(),
            dst_bone.pose_rotation_quat.copy(),
            dst_bone.pose_scale.copy(),
        )
    return result


# ---------------------------------------------------------------------------
# Full animation transfer
# ---------------------------------------------------------------------------

def transfer_animation(
    src_anim,                        # BVHAnimation or any object with .armature + .apply_frame(i) + .num_frames
    dst_arm: Armature,
    bone_items: List[BoneMappingItem],
    settings: KeeMapSettings,
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Transfer all frames from src_anim to dst_arm.
    Returns list of keyframe dicts (one per frame sampled).

    Equivalent to Blender's PerformAnimationTransfer operator.
    """
    keyframes: List[Dict] = []
    step = max(1, settings.keyframe_every_n_frames)
    start = settings.start_frame_to_apply
    total = settings.number_of_frames_to_apply
    end = start + total

    src_arm = src_anim.armature

    i = start
    n_steps = len(range(start, end, step))
    step_i = 0
    while i < end and i < src_anim.num_frames:
        src_anim.apply_frame(i)    # updates src_arm FK
        dst_arm.update_fk()

        frame_data = transfer_frame(src_arm, dst_arm, bone_items, settings)
        keyframes.append(frame_data)

        step_i += 1
        _update_progress("Retargeting", step_i / n_steps)
        i += step

    _update_progress("Retargeting", 1.0)
    return keyframes
