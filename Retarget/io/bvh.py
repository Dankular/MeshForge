"""
io/bvh.py
BVH (Biovision Hierarchy) reader.

Returns an Armature in rest pose plus an iterator / list of frame states.
Each frame state sets the bone pose_rotation_quat / pose_location on the
source armature so that retarget.get_bone_ws_quat / get_bone_position_ws
return the correct world-space values.
"""
from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..skeleton import Armature, PoseBone
from ..math3d import translation_matrix, euler_to_quat, quat_identity, vec3


# ---------------------------------------------------------------------------
# Internal BVH data structures
# ---------------------------------------------------------------------------

class _BVHJoint:
    def __init__(self, name: str):
        self.name = name
        self.offset: np.ndarray = vec3()
        self.channels: List[str] = []
        self.children: List["_BVHJoint"] = []
        self.is_end_site: bool = False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.split(r"[\s]+", text.strip())


def _parse_hierarchy(tokens: List[str], idx: int) -> Tuple[_BVHJoint, int]:
    """Parse one joint block.  idx should point to joint name."""
    name = tokens[idx]; idx += 1
    joint = _BVHJoint(name)
    assert tokens[idx] == "{", f"Expected '{{' got '{tokens[idx]}'"
    idx += 1
    while tokens[idx] != "}":
        kw = tokens[idx].upper()
        if kw == "OFFSET":
            joint.offset = np.array([float(tokens[idx+1]), float(tokens[idx+2]), float(tokens[idx+3])])
            idx += 4
        elif kw == "CHANNELS":
            n = int(tokens[idx+1]); idx += 2
            joint.channels = [tokens[idx+i].upper() for i in range(n)]
            idx += n
        elif kw == "JOINT":
            idx += 1
            child, idx = _parse_hierarchy(tokens, idx)
            joint.children.append(child)
        elif kw == "END" and tokens[idx+1].upper() == "SITE":
            # End Site block — just parse and discard
            idx += 2
            assert tokens[idx] == "{"; idx += 1
            while tokens[idx] != "}":
                idx += 1
            idx += 1  # skip '}'
        else:
            idx += 1   # unknown token, skip
    idx += 1  # skip '}'
    return joint, idx


def _collect_joints(joint: _BVHJoint) -> List[_BVHJoint]:
    result = [joint]
    for c in joint.children:
        result.extend(_collect_joints(c))
    return result


# ---------------------------------------------------------------------------
# Build Armature from BVH hierarchy
# ---------------------------------------------------------------------------

def _build_armature(root_joint: _BVHJoint) -> Armature:
    arm = Armature("BVH_Source")

    def add_recursive(j: _BVHJoint, parent_name: Optional[str], parent_world: np.ndarray):
        # rest_matrix_local = T(offset) relative to parent
        rest_local = translation_matrix(j.offset)
        bone = PoseBone(j.name, rest_local)
        arm.add_bone(bone, parent_name)
        world = parent_world @ rest_local
        for child in j.children:
            add_recursive(child, j.name, world)

    add_recursive(root_joint, None, np.eye(4))
    arm.update_fk()
    return arm


# ---------------------------------------------------------------------------
# Frame application
# ---------------------------------------------------------------------------

_CHANNEL_MAP = {
    "XROTATION": ("rx",), "YROTATION": ("ry",), "ZROTATION": ("rz",),
    "XPOSITION": ("tx",), "YPOSITION": ("ty",), "ZPOSITION": ("tz",),
}


def _apply_frame(arm: Armature, all_joints: List[_BVHJoint], values: List[float]) -> None:
    """Set bone poses for one BVH frame."""
    vi = 0
    for j in all_joints:
        tx = ty = tz = 0.0
        rx = ry = rz = 0.0
        for ch in j.channels:
            key = _CHANNEL_MAP.get(ch, None)
            if key:
                val = values[vi]
                k = key[0]
                if k == "tx": tx = val
                elif k == "ty": ty = val
                elif k == "tz": tz = val
                elif k == "rx": rx = math.radians(val)
                elif k == "ry": ry = math.radians(val)
                elif k == "rz": rz = math.radians(val)
            vi += 1

        if j.name not in arm.pose_bones:
            continue
        bone = arm.pose_bones[j.name]

        # BVH rotation order is specified per channel list; rebuild from order
        rot_channels = [c for c in j.channels if "ROTATION" in c]
        order = "".join(c[0] for c in rot_channels)  # e.g. "ZXY"
        angles = {"X": rx, "Y": ry, "Z": rz}
        angle_seq = [angles[a] for a in order]
        bone.pose_rotation_quat = euler_to_quat(*angle_seq, order=order)

        # Translation — only root joints typically have it
        if tx or ty or tz:
            bone.pose_location = np.array([tx, ty, tz])

    arm.update_fk()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BVHAnimation:
    """Loaded BVH file.  Iterate frames by calling advance(frame_index)."""

    def __init__(
        self,
        armature: Armature,
        all_joints: List[_BVHJoint],
        frame_data: List[List[float]],
        frame_time: float,
    ):
        self.armature = armature
        self._all_joints = all_joints
        self._frame_data = frame_data
        self.frame_time = frame_time
        self.num_frames = len(frame_data)

    def apply_frame(self, frame_index: int) -> None:
        """Advance armature to frame_index and update FK."""
        if frame_index < 0 or frame_index >= self.num_frames:
            raise IndexError(f"Frame {frame_index} out of range [0, {self.num_frames})")
        _apply_frame(self.armature, self._all_joints, self._frame_data[frame_index])


def load_bvh(filepath: str) -> BVHAnimation:
    """
    Parse a BVH file.
    Returns BVHAnimation with an Armature ready for retargeting.
    """
    with open(filepath, "r") as f:
        text = f.read()

    tokens = _tokenize(text)
    idx = 0

    # Expect HIERARCHY keyword
    while tokens[idx].upper() != "HIERARCHY":
        idx += 1
    idx += 1

    root_kw = tokens[idx].upper()
    assert root_kw in ("ROOT", "JOINT"), f"Expected ROOT/JOINT, got '{tokens[idx]}'"
    idx += 1
    root_joint, idx = _parse_hierarchy(tokens, idx)

    # MOTION section
    while tokens[idx].upper() != "MOTION":
        idx += 1
    idx += 1

    assert tokens[idx].upper() == "FRAMES:"; idx += 1
    num_frames = int(tokens[idx]); idx += 1
    assert tokens[idx].upper() == "FRAME"; assert tokens[idx+1].upper() == "TIME:"; idx += 2
    frame_time = float(tokens[idx]); idx += 1

    all_joints = _collect_joints(root_joint)
    total_channels = sum(len(j.channels) for j in all_joints)

    frame_data: List[List[float]] = []
    for _ in range(num_frames):
        row = [float(tokens[idx + k]) for k in range(total_channels)]
        idx += total_channels
        frame_data.append(row)

    arm = _build_armature(root_joint)
    return BVHAnimation(arm, all_joints, frame_data, frame_time)
