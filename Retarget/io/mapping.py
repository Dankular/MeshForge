"""
io/mapping.py
Load / save bone mapping JSON in the exact same format as KeeMap.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from ..math3d import quat_identity, vec3


@dataclass
class BoneMappingItem:
    name: str = ""
    label: str = ""
    description: str = ""

    source_bone_name: str = ""
    destination_bone_name: str = ""

    keyframe_this_bone: bool = True

    # Rotation correction (Euler, radians)
    correction_factor: np.ndarray = field(default_factory=lambda: vec3())

    # Quaternion correction
    quat_correction_factor: np.ndarray = field(default_factory=quat_identity)

    has_twist_bone: bool = False
    twist_bone_name: str = ""

    set_bone_position: bool = False
    set_bone_rotation: bool = True
    set_bone_scale: bool = False

    # Rotation options
    bone_rotation_application_axis: str = "XYZ"   # X Y Z XY XZ YZ XYZ
    bone_transpose_axis: str = "NONE"             # NONE ZXY ZYX XZY YZX YXZ

    # Position options
    postion_type: str = "SINGLE_BONE_OFFSET"       # SINGLE_BONE_OFFSET | POLE
    position_correction_factor: np.ndarray = field(default_factory=lambda: vec3())
    position_gain: float = 1.0
    position_pole_distance: float = 0.3

    # Scale options
    scale_secondary_bone_name: str = ""
    bone_scale_application_axis: str = "Y"
    scale_gain: float = 1.0
    scale_max: float = 1.0
    scale_min: float = 0.5


@dataclass
class KeeMapSettings:
    source_rig_name: str = ""
    destination_rig_name: str = ""
    bone_mapping_file: str = ""
    bone_rotation_mode: str = "EULER"           # EULER | QUATERNION
    start_frame_to_apply: int = 0
    number_of_frames_to_apply: int = 100
    keyframe_every_n_frames: int = 1
    keyframe_test: bool = False


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_mapping(filepath: str):
    """
    Returns (KeeMapSettings, List[BoneMappingItem]).
    Reads the exact same JSON that KeeMap writes.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    settings = KeeMapSettings(
        source_rig_name=data.get("source_rig_name", ""),
        destination_rig_name=data.get("destination_rig_name", ""),
        bone_mapping_file=data.get("bone_mapping_file", ""),
        bone_rotation_mode=data.get("bone_rotation_mode", "EULER"),
        start_frame_to_apply=data.get("start_frame_to_apply", 0),
        number_of_frames_to_apply=data.get("number_of_frames_to_apply", 100),
        keyframe_every_n_frames=data.get("keyframe_every_n_frames", 1),
    )

    bones: List[BoneMappingItem] = []
    for p in data.get("bones", []):
        item = BoneMappingItem()
        item.name = p.get("name", "")
        item.label = p.get("label", "")
        item.description = p.get("description", "")
        item.source_bone_name = p.get("SourceBoneName", "")
        item.destination_bone_name = p.get("DestinationBoneName", "")
        item.keyframe_this_bone = p.get("keyframe_this_bone", True)

        item.correction_factor = np.array([
            p.get("CorrectionFactorX", 0.0),
            p.get("CorrectionFactorY", 0.0),
            p.get("CorrectionFactorZ", 0.0),
        ])

        item.quat_correction_factor = np.array([
            p.get("QuatCorrectionFactorw", 1.0),
            p.get("QuatCorrectionFactorx", 0.0),
            p.get("QuatCorrectionFactory", 0.0),
            p.get("QuatCorrectionFactorz", 0.0),
        ])

        item.has_twist_bone = p.get("has_twist_bone", False)
        item.twist_bone_name = p.get("TwistBoneName", "")
        item.set_bone_position = p.get("set_bone_position", False)
        item.set_bone_rotation = p.get("set_bone_rotation", True)
        item.set_bone_scale = p.get("set_bone_scale", False)
        item.bone_rotation_application_axis = p.get("bone_rotation_application_axis", "XYZ")
        item.bone_transpose_axis = p.get("bone_transpose_axis", "NONE")
        item.postion_type = p.get("postion_type", "SINGLE_BONE_OFFSET")

        item.position_correction_factor = np.array([
            p.get("position_correction_factorX", 0.0),
            p.get("position_correction_factorY", 0.0),
            p.get("position_correction_factorZ", 0.0),
        ])
        item.position_gain = p.get("position_gain", 1.0)
        item.position_pole_distance = p.get("position_pole_distance", 0.3)
        item.scale_secondary_bone_name = p.get("scale_secondary_bone_name", "")
        item.bone_scale_application_axis = p.get("bone_scale_application_axis", "Y")
        item.scale_gain = p.get("scale_gain", 1.0)
        item.scale_max = p.get("scale_max", 1.0)
        item.scale_min = p.get("scale_min", 0.5)
        bones.append(item)

    return settings, bones


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_mapping(filepath: str, settings: KeeMapSettings, bones: List[BoneMappingItem]) -> None:
    """Write mapping JSON readable by KeeMap."""
    root = {
        "source_rig_name": settings.source_rig_name,
        "destination_rig_name": settings.destination_rig_name,
        "bone_mapping_file": settings.bone_mapping_file,
        "bone_rotation_mode": settings.bone_rotation_mode,
        "start_frame_to_apply": settings.start_frame_to_apply,
        "number_of_frames_to_apply": settings.number_of_frames_to_apply,
        "keyframe_every_n_frames": settings.keyframe_every_n_frames,
        "bones": [],
    }
    for b in bones:
        root["bones"].append({
            "name": b.name,
            "label": b.label,
            "description": b.description,
            "SourceBoneName": b.source_bone_name,
            "DestinationBoneName": b.destination_bone_name,
            "keyframe_this_bone": b.keyframe_this_bone,
            "CorrectionFactorX": float(b.correction_factor[0]),
            "CorrectionFactorY": float(b.correction_factor[1]),
            "CorrectionFactorZ": float(b.correction_factor[2]),
            "QuatCorrectionFactorw": float(b.quat_correction_factor[0]),
            "QuatCorrectionFactorx": float(b.quat_correction_factor[1]),
            "QuatCorrectionFactory": float(b.quat_correction_factor[2]),
            "QuatCorrectionFactorz": float(b.quat_correction_factor[3]),
            "has_twist_bone": b.has_twist_bone,
            "TwistBoneName": b.twist_bone_name,
            "set_bone_position": b.set_bone_position,
            "set_bone_rotation": b.set_bone_rotation,
            "set_bone_scale": b.set_bone_scale,
            "bone_rotation_application_axis": b.bone_rotation_application_axis,
            "bone_transpose_axis": b.bone_transpose_axis,
            "postion_type": b.postion_type,
            "position_correction_factorX": float(b.position_correction_factor[0]),
            "position_correction_factorY": float(b.position_correction_factor[1]),
            "position_correction_factorZ": float(b.position_correction_factor[2]),
            "position_gain": b.position_gain,
            "position_pole_distance": b.position_pole_distance,
            "scale_secondary_bone_name": b.scale_secondary_bone_name,
            "bone_scale_application_axis": b.bone_scale_application_axis,
            "scale_gain": b.scale_gain,
            "scale_max": b.scale_max,
            "scale_min": b.scale_min,
        })
    with open(filepath, "w") as f:
        json.dump(root, f, indent=2)
