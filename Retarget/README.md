# rig_retarget

Pure-Python rig retargeting library. No Blender required.

Based on **[KeeMap Blender Rig Retargeting Addon](https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon)** by [Nick Keeline](https://github.com/nkeeline) (GPL v2).
All core retargeting math is a direct port of his work. Mapping JSON files are fully compatible with KeeMap.

---

## File layout

```
rig_retarget/
├── math3d.py        # Quaternion / matrix math (numpy + scipy), replaces mathutils
├── skeleton.py      # Armature + PoseBone with FK, replaces bpy armature objects
├── retarget.py      # Core retargeting logic — faithful port of KeeMapBoneOperators.py
├── cli.py           # CLI entry point
└── io/
    ├── bvh.py       # BVH mocap reader (source animation)
    ├── gltf_io.py   # glTF/GLB reader + animation writer (UniRig destination)
    └── mapping.py   # JSON bone mapping — same format as KeeMap
```

---

## Install

```bash
pip install numpy scipy pygltflib
```

---

## CLI

```bash
# Retarget BVH onto UniRig GLB
python -m rig_retarget.cli \
    --source  motion.bvh \
    --dest    unirig_character.glb \
    --mapping radical2unirig.json \
    --output  animated_character.glb \
    --fps 30 --start 0 --frames 200 --step 1

# Auto-calculate bone correction factors and save back to the mapping file
python -m rig_retarget.cli --calc-corrections \
    --source motion.bvh \
    --dest   unirig_character.glb \
    --mapping mymap.json
```

---

## Python API

```python
from rig_retarget.io.bvh     import load_bvh
from rig_retarget.io.gltf_io import load_gltf, write_gltf_animation
from rig_retarget.io.mapping  import load_mapping
from rig_retarget.retarget    import transfer_animation, calc_all_corrections

# Load
settings, bone_items = load_mapping("my_map.json")
src_anim = load_bvh("motion.bvh")
dst_arm  = load_gltf("unirig_char.glb")

# Optional: auto-calc corrections at first frame
src_anim.apply_frame(0)
calc_all_corrections(bone_items, src_anim.armature, dst_arm, settings)

# Transfer
settings.number_of_frames_to_apply = src_anim.num_frames
keyframes = transfer_animation(src_anim, dst_arm, bone_items, settings)

# Write output GLB
write_gltf_animation("unirig_char.glb", dst_arm, keyframes, "output.glb")
```

---

## Mapping JSON format

100% compatible with KeeMap's `.json` files. Use KeeMap in Blender to create
and tune mappings, then use this library offline for batch processing.

Key fields per bone:

| Field | Description |
|---|---|
| `SourceBoneName` | Bone name in the source rig (BVH joint name) |
| `DestinationBoneName` | Bone name in the UniRig skeleton (glTF node name) |
| `set_bone_rotation` | Drive rotation from source |
| `set_bone_position` | Drive position from source |
| `bone_rotation_application_axis` | Mask axes: `X` `Y` `Z` `XY` `XZ` `YZ` `XYZ` |
| `bone_transpose_axis` | Swap axes: `NONE` `ZYX` `ZXY` `XZY` `YZX` `YXZ` |
| `CorrectionFactorX/Y/Z` | Euler correction (radians) |
| `postion_type` | `SINGLE_BONE_OFFSET` or `POLE` |
| `position_pole_distance` | IK pole distance |

---

## Blender → pure-Python mapping

| Blender | rig_retarget |
|---|---|
| `bpy.data.objects[name]` | `Armature` + `load_gltf()` / `load_bvh()` |
| `arm.pose.bones[name]` | `arm.get_bone(name)` → `PoseBone` |
| `bone.matrix` (pose space) | `bone.matrix_armature` |
| `arm.matrix_world` | `arm.world_matrix` |
| `arm.convert_space(...)` | `arm.world_matrix @ bone.matrix_armature` |
| `bone.rotation_quaternion` | `bone.pose_rotation_quat` |
| `bone.location` | `bone.pose_location` |
| `bone.keyframe_insert(...)` | returned in `keyframes` list from `transfer_frame()` |
| `bpy.context.scene.frame_set(i)` | `src_anim.apply_frame(i)` |
| `mathutils.Quaternion` | `np.ndarray [w,x,y,z]` + `math3d.*` |
| `mathutils.Matrix` | `np.ndarray (4,4)` |

---

## Limitations / TODO

- **glTF source animation** reading not yet implemented (BVH only for now).
  Add `io/gltf_anim_reader.py` reading `gltf.animations[0]` sampler data.
- FBX source support: use `pyassimp` or `bpy` offline with `--background`.
- IK solving: pole bone positioning is FK-only; a full IK solver (FABRIK/CCD)
  would improve accuracy for limb targets.
- Quaternion mode twist bones: parity with Blender not guaranteed for complex twist rigs.
