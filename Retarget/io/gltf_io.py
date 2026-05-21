"""
io/gltf_io.py
Load a glTF/GLB skeleton (e.g. UniRig output) into an Armature.
Write retargeted animation back into a glTF/GLB file.

Requires:  pip install pygltflib
"""
from __future__ import annotations
import base64
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pygltflib
except ImportError:
    raise ImportError("pip install pygltflib")

from ..skeleton import Armature, PoseBone
from ..math3d import (
    quat_identity, quat_normalize, matrix4_to_quat, matrix4_to_trs,
    trs_to_matrix4, vec3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_local_trs(node: "pygltflib.Node"):
    """Extract TRS from a glTF node.  Returns (t[3], r_wxyz[4], s[3])."""
    t = np.array(node.translation or [0.0, 0.0, 0.0])
    r_xyzw = np.array(node.rotation or [0.0, 0.0, 0.0, 1.0])
    s = np.array(node.scale or [1.0, 1.0, 1.0])
    # Convert glTF (x,y,z,w) → our (w,x,y,z)
    r_wxyz = np.array([r_xyzw[3], r_xyzw[0], r_xyzw[1], r_xyzw[2]])
    return t, r_wxyz, s


def _node_local_matrix(node: "pygltflib.Node") -> np.ndarray:
    if node.matrix:
        # glTF stores column-major; convert to row-major
        m = np.array(node.matrix, dtype=float).reshape(4, 4).T
        return m
    t, r, s = _node_local_trs(node)
    return trs_to_matrix4(t, r, s)


def _read_accessor(gltf: "pygltflib.GLTF2", accessor_idx: int) -> np.ndarray:
    """Read a glTF accessor into a numpy array."""
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[bv.buffer]

    # Inline base64 data URI
    if buf.uri and buf.uri.startswith("data:"):
        _, b64 = buf.uri.split(",", 1)
        raw = base64.b64decode(b64)
    elif buf.uri:
        base_dir = Path(gltf._path).parent if hasattr(gltf, "_path") and gltf._path else Path(".")
        raw = (base_dir / buf.uri).read_bytes()
    else:
        # Binary GLB — data stored in gltf.binary_blob
        raw = bytes(gltf.binary_blob())

    start = bv.byteOffset + (acc.byteOffset or 0)
    count = acc.count

    type_to_components = {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
        "MAT2": 4, "MAT3": 9, "MAT4": 16,
    }
    component_type_to_fmt = {
        5120: "b", 5121: "B", 5122: "h", 5123: "H",
        5125: "I", 5126: "f",
    }
    n_comp = type_to_components[acc.type]
    fmt = component_type_to_fmt[acc.componentType]
    item_size = struct.calcsize(fmt) * n_comp
    stride = bv.byteStride or item_size

    items = []
    for i in range(count):
        offset = start + i * stride
        vals = struct.unpack_from(f"{n_comp}{fmt}", raw, offset)
        items.append(vals)

    return np.array(items, dtype=float).squeeze()


# ---------------------------------------------------------------------------
# Load skeleton from glTF
# ---------------------------------------------------------------------------

def load_gltf(filepath: str, skin_index: int = 0) -> Armature:
    """
    Load the first (or specified) skin from a glTF/GLB file into an Armature.

    The armature world_matrix is set to identity (typical for UniRig output).
    """
    gltf = pygltflib.GLTF2().load(filepath)
    gltf._path = filepath

    if not gltf.skins:
        raise ValueError(f"No skins found in '{filepath}'")
    skin = gltf.skins[skin_index]

    # Read inverse bind matrices
    n_joints = len(skin.joints)
    ibm_array: Optional[np.ndarray] = None
    if skin.inverseBindMatrices is not None:
        raw = _read_accessor(gltf, skin.inverseBindMatrices)
        ibm_array = raw.reshape(n_joints, 4, 4)

    # Compute bind-pose world matrices: world_bind = inv(ibm)
    joint_world_bind: Dict[int, np.ndarray] = {}
    for i, j_idx in enumerate(skin.joints):
        if ibm_array is not None:
            ibm = ibm_array[i].T   # glTF column-major → numpy row-major
            joint_world_bind[j_idx] = np.linalg.inv(ibm)
        else:
            # Fallback: compute from FK over node local matrices
            joint_world_bind[j_idx] = np.eye(4)

    # Build parent map for nodes
    parent_of: Dict[int, Optional[int]] = {}
    for ni, node in enumerate(gltf.nodes):
        for child_idx in (node.children or []):
            parent_of[child_idx] = ni

    arm = Armature(skin.name or f"Skin_{skin_index}")

    # Process joints in order (parent always before child in glTF spec)
    joint_set = set(skin.joints)
    processed: Dict[int, str] = {}

    for i, j_idx in enumerate(skin.joints):
        node = gltf.nodes[j_idx]
        bone_name = node.name or f"joint_{i}"

        # Find parent joint node
        parent_node_idx = parent_of.get(j_idx)
        parent_bone_name: Optional[str] = None
        while parent_node_idx is not None:
            if parent_node_idx in joint_set:
                parent_bone_name = processed.get(parent_node_idx)
                break
            parent_node_idx = parent_of.get(parent_node_idx)

        # rest_matrix_local in parent space
        if parent_bone_name and parent_bone_name in [b for b in processed.values()]:
            parent_world = joint_world_bind.get(
                next(k for k, v in processed.items() if v == parent_bone_name),
                np.eye(4)
            )
            rest_local = np.linalg.inv(parent_world) @ joint_world_bind[j_idx]
        else:
            rest_local = joint_world_bind[j_idx]

        bone = PoseBone(bone_name, rest_local)
        arm.add_bone(bone, parent_bone_name)
        processed[j_idx] = bone_name

    arm.update_fk()
    return arm


# ---------------------------------------------------------------------------
# Write animation to glTF
# ---------------------------------------------------------------------------

def write_gltf_animation(
    source_filepath: str,
    dest_armature: Armature,
    keyframes: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    output_filepath: str,
    fps: float = 30.0,
    skin_index: int = 0,
) -> None:
    """
    Embed animation keyframes into a copy of source_filepath (the UniRig GLB).

    keyframes:  list of dicts, one per frame.
                Each dict maps bone_name → (pose_location, pose_rotation_quat, pose_scale)
                These are LOCAL values (relative to rest pose local matrix).

    The function adds one glTF Animation with channels for each bone that has data.
    """
    gltf = pygltflib.GLTF2().load(source_filepath)
    gltf._path = source_filepath

    if not gltf.skins:
        raise ValueError("No skins in source file")
    skin = gltf.skins[skin_index]

    # Build node_name → node_index map for skin joints
    joint_name_to_node: Dict[str, int] = {}
    for j_idx in skin.joints:
        node = gltf.nodes[j_idx]
        name = node.name or f"joint_{j_idx}"
        joint_name_to_node[name] = j_idx

    n_frames = len(keyframes)
    times = np.array([i / fps for i in range(n_frames)], dtype=np.float32)

    # Gather binary data
    binary_chunks: List[bytes] = []
    accessors: List[dict] = []
    buffer_views: List[dict] = []

    def _add_data(data: np.ndarray, acc_type: str) -> int:
        """Append numpy array to binary, return accessor index."""
        raw = data.astype(np.float32).tobytes()
        bv_offset = sum(len(c) for c in binary_chunks)
        binary_chunks.append(raw)
        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=bv_offset,
            byteLength=len(raw),
        ))
        acc_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=bv_idx,
            componentType=pygltflib.FLOAT,
            count=len(data),
            type=acc_type,
            max=data.max(axis=0).tolist() if data.ndim > 1 else [float(data.max())],
            min=data.min(axis=0).tolist() if data.ndim > 1 else [float(data.min())],
        ))
        return acc_idx

    time_acc_idx = _add_data(times, "SCALAR")

    channels: List[pygltflib.AnimationChannel] = []
    samplers: List[pygltflib.AnimationSampler] = []

    bone_names = set()
    for frame in keyframes:
        bone_names |= frame.keys()

    for bone_name in sorted(bone_names):
        if bone_name not in joint_name_to_node:
            continue
        node_idx = joint_name_to_node[bone_name]
        node = gltf.nodes[node_idx]

        # Collect TRS arrays across frames
        rot_data = np.zeros((n_frames, 4), dtype=np.float32)  # (x,y,z,w)
        trans_data = np.zeros((n_frames, 3), dtype=np.float32)
        scale_data = np.ones((n_frames, 3), dtype=np.float32)

        rest_t, rest_r, rest_s = _node_local_trs(node)

        for fi, frame in enumerate(keyframes):
            if bone_name in frame:
                pose_loc, pose_rot, pose_scale = frame[bone_name]
            else:
                pose_loc = vec3()
                pose_rot = quat_identity()
                pose_scale = np.ones(3)

            # Final local = rest + delta (simple addition for translation, multiply for rotation)
            from ..math3d import quat_mul, trs_to_matrix4
            final_t = rest_t + pose_loc
            final_r = quat_mul(rest_r, pose_rot)      # (w,x,y,z)
            final_s = rest_s * pose_scale

            # Convert rotation to glTF (x,y,z,w)
            w, x, y, z = final_r
            rot_data[fi] = [x, y, z, w]
            trans_data[fi] = final_t
            scale_data[fi] = final_s

        s_idx = len(samplers)
        rot_acc = _add_data(rot_data, "VEC4")
        samplers.append(pygltflib.AnimationSampler(input=time_acc_idx, output=rot_acc, interpolation="LINEAR"))
        channels.append(pygltflib.AnimationChannel(
            sampler=s_idx,
            target=pygltflib.AnimationChannelTarget(node=node_idx, path="rotation"),
        ))

        s_idx = len(samplers)
        trans_acc = _add_data(trans_data, "VEC3")
        samplers.append(pygltflib.AnimationSampler(input=time_acc_idx, output=trans_acc, interpolation="LINEAR"))
        channels.append(pygltflib.AnimationChannel(
            sampler=s_idx,
            target=pygltflib.AnimationChannelTarget(node=node_idx, path="translation"),
        ))

    if not channels:
        print("[gltf_io] Warning: no channels written — check bone name mapping.")
        return

    gltf.animations.append(pygltflib.Animation(
        name="RetargetedAnimation",
        samplers=samplers,
        channels=channels,
    ))

    # Patch buffer 0 size with our new data
    new_blob = b"".join(binary_chunks)
    existing_blob = bytes(gltf.binary_blob()) if gltf.binary_blob() else b""
    full_blob = existing_blob + new_blob

    # Update buffer 0 byteOffset of new views
    for bv in gltf.bufferViews[-len(binary_chunks):]:
        bv.byteOffset += len(existing_blob)

    gltf.set_binary_blob(full_blob)
    gltf.buffers[0].byteLength = len(full_blob)

    gltf.save(output_filepath)
    print(f"[gltf_io] Saved animated GLB -> {output_filepath}")
