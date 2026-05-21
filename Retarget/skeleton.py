"""
skeleton.py
Pure-Python armature / pose-bone system.

Design matches Blender's pose-mode semantics:
  - bone.rest_matrix_local   = 4×4 rest pose in parent space  (edit-mode)
  - bone.pose_rotation_quat  = local rotation DELTA from rest  (≡ bone.rotation_quaternion)
  - bone.pose_location       = local translation DELTA from rest (≡ bone.location)
  - bone.pose_scale          = local scale                       (≡ bone.scale)
  - bone.matrix_armature     = FK-computed 4×4 in armature space (≡ bone.matrix in pose mode)

Armature.world_matrix corresponds to arm.matrix_world.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from .math3d import (
    quat_identity, quat_normalize, quat_mul,
    quat_to_matrix4, matrix4_to_quat,
    translation_matrix, scale_matrix, trs_to_matrix4, matrix4_to_trs,
    vec3,
)


class PoseBone:
    def __init__(
        self,
        name: str,
        rest_matrix_local: np.ndarray,   # 4×4, in parent local space
        parent: Optional["PoseBone"] = None,
    ):
        self.name = name
        self.parent: Optional[PoseBone] = parent
        self.children: List[PoseBone] = []
        self.rest_matrix_local: np.ndarray = rest_matrix_local.copy()

        # Pose state — start at rest (delta = identity)
        self.pose_rotation_quat: np.ndarray = quat_identity()
        self.pose_location: np.ndarray = vec3()
        self.pose_scale: np.ndarray = np.ones(3)

        # Cached FK result — call armature.update_fk() to refresh
        self._matrix_armature: np.ndarray = np.eye(4)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def matrix_armature(self) -> np.ndarray:
        """4×4 FK result in armature space.  Refresh with armature.update_fk()."""
        return self._matrix_armature

    @property
    def head(self) -> np.ndarray:
        """Bone head position in armature space."""
        return self._matrix_armature[:3, 3].copy()

    @property
    def tail(self) -> np.ndarray:
        """
        Approximate tail position (Y-axis in bone space, length 1).
        Works for Y-along-bone convention (Blender / BVH default).
        """
        y_axis = self._matrix_armature[:3, :3] @ np.array([0.0, 1.0, 0.0])
        return self._matrix_armature[:3, 3] + y_axis

    # -----------------------------------------------------------------------
    # FK
    # -----------------------------------------------------------------------

    def _compute_local_matrix(self) -> np.ndarray:
        """rest_local @ T(pose_loc) @ R(pose_rot) @ S(pose_scale)."""
        T = translation_matrix(self.pose_location)
        R = quat_to_matrix4(self.pose_rotation_quat)
        S = scale_matrix(self.pose_scale)
        return self.rest_matrix_local @ T @ R @ S

    def _fk(self, parent_matrix: np.ndarray) -> None:
        self._matrix_armature = parent_matrix @ self._compute_local_matrix()
        for child in self.children:
            child._fk(self._matrix_armature)

    # -----------------------------------------------------------------------
    # Parent-chain helpers (Blender: bone.parent_recursive)
    # -----------------------------------------------------------------------

    @property
    def parent_recursive(self) -> List["PoseBone"]:
        chain: List[PoseBone] = []
        cur = self.parent
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        return chain


class Armature:
    """
    Collection of PoseBones with a world transform.
    Corresponds to a Blender armature object.
    """

    def __init__(self, name: str = "Armature"):
        self.name = name
        self.world_matrix: np.ndarray = np.eye(4)    # arm.matrix_world
        self._bones: Dict[str, PoseBone] = {}
        self._roots: List[PoseBone] = []

    # -----------------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------------

    def add_bone(self, bone: PoseBone, parent_name: Optional[str] = None) -> PoseBone:
        self._bones[bone.name] = bone
        if parent_name and parent_name in self._bones:
            parent = self._bones[parent_name]
            bone.parent = parent
            parent.children.append(bone)
        elif bone.parent is None:
            self._roots.append(bone)
        return bone

    @property
    def pose_bones(self) -> Dict[str, PoseBone]:
        return self._bones

    def get_bone(self, name: str) -> PoseBone:
        if name not in self._bones:
            raise KeyError(f"Bone '{name}' not found in armature '{self.name}'")
        return self._bones[name]

    def has_bone(self, name: str) -> bool:
        return name in self._bones

    # -----------------------------------------------------------------------
    # FK update
    # -----------------------------------------------------------------------

    def update_fk(self) -> None:
        """Recompute all bone armature-space matrices via FK."""
        for root in self._roots:
            root._fk(np.eye(4))

    # -----------------------------------------------------------------------
    # Snapshot / restore (for calc-correction passes)
    # -----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return {
            name: (
                bone.pose_rotation_quat.copy(),
                bone.pose_location.copy(),
                bone.pose_scale.copy(),
            )
            for name, bone in self._bones.items()
        }

    def restore(self, snap: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        for name, (r, t, s) in snap.items():
            if name in self._bones:
                self._bones[name].pose_rotation_quat = r.copy()
                self._bones[name].pose_location = t.copy()
                self._bones[name].pose_scale = s.copy()
        self.update_fk()
