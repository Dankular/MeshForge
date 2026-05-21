"""
rig_retarget
============
Pure-Python rig retargeting library.
No Blender dependency. Targets TripoSG meshes auto-rigged by UniRig (SIGGRAPH 2025).

Quick start
-----------
    from rig_retarget.io.bvh import load_bvh
    from rig_retarget.io.gltf_io import load_gltf, write_gltf_animation
    from rig_retarget.io.mapping import load_mapping
    from rig_retarget.retarget import transfer_animation

    settings, bone_items = load_mapping("my_map.json")
    src_anim = load_bvh("motion.bvh")
    dst_arm  = load_gltf("unirig_char.glb")

    keyframes = transfer_animation(src_anim, dst_arm, bone_items, settings)
    write_gltf_animation("unirig_char.glb", dst_arm, keyframes, "output.glb")

CLI
---
    python -m rig_retarget.cli --source motion.bvh --dest char.glb \\
        --mapping map.json --output char_animated.glb
"""

from .skeleton import Armature, PoseBone
from .retarget import (
    get_bone_position_ws,
    get_bone_ws_quat,
    set_bone_position_ws,
    set_bone_rotation,
    set_bone_position,
    set_bone_position_pole,
    set_bone_scale,
    calc_rotation_offset,
    calc_location_offset,
    calc_all_corrections,
    transfer_frame,
    transfer_animation,
)

__all__ = [
    "Armature", "PoseBone",
    "get_bone_position_ws", "get_bone_ws_quat", "set_bone_position_ws",
    "set_bone_rotation", "set_bone_position", "set_bone_position_pole",
    "set_bone_scale", "calc_rotation_offset", "calc_location_offset",
    "calc_all_corrections", "transfer_frame", "transfer_animation",
]
