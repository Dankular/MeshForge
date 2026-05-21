"""
smpl.py
───────────────────────────────────────────────────────────────────────────────
Parse HumanML3D [T, 263] feature vectors into structured SMPL motion data.

HumanML3D 263-dim layout per frame
  [0]       root angular-velocity   (Y-axis, rad/frame)
  [1]       root height Y           (metres)
  [2:4]     root XZ velocity        (local-frame, metres/frame)
  [4:67]    joint local positions   joints 1-21 relative to root, 21×3 (unused here)
  [67:193]  6D joint rotations      joints 1-21, 21×6
  [193:259] joint velocities        joints 0-21, 22×3 (unused here)
  [259:263] foot contact flags      (unused here)

Root rotation  = cumulative integral of dim[0] → Y-axis quaternion.
Root position  = dim[1] (height) + integrated XZ velocity.
Joint 1-21 rot = dims 67:193 as 6D continuous rotation representation
                 [Zhou et al. 2019] → Gram-Schmidt → 3×3 rotation matrix → quaternion.
                 These are LOCAL rotations relative to the SMPL parent joint's rest
                 frame, where the canonical T-pose is the zero (identity) rotation.
"""
from __future__ import annotations
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 6D rotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """
    [..., 6] → [..., 3, 3]
    Reconstructs a rotation matrix from two columns using Gram-Schmidt.
    The two columns are [a1 = r6d[..., 0:3], a2 = r6d[..., 3:6]].
    """
    a1 = r6d[..., 0:3].astype(np.float64)
    a2 = r6d[..., 3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)   # columns → [..., 3, 3]


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    [..., 3, 3] → [..., 4]  WXYZ quaternion, positive-W convention.
    Uses scipy for numerical stability.
    """
    from scipy.spatial.transform import Rotation
    shape = mat.shape[:-2]
    flat  = mat.reshape(-1, 3, 3).astype(np.float64)
    xyzw  = Rotation.from_matrix(flat).as_quat()   # scipy → XYZW
    wxyz  = xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    wxyz[wxyz[:, 0] < 0] *= -1                     # positive-W
    return wxyz.reshape(*shape, 4)


def rot6d_to_quat(r6d: np.ndarray) -> np.ndarray:
    """[..., 6] → [..., 4] WXYZ.  Convenience: 6D → matrix → quaternion."""
    return matrix_to_quat(rot6d_to_matrix(r6d))


# ──────────────────────────────────────────────────────────────────────────────
# Root motion recovery
# ──────────────────────────────────────────────────────────────────────────────

def _qrot_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate [N, 3] vectors by [N, 4] WXYZ quaternions (batch)."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    vx, vy, vz  = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    # Rodrigues-style: v + 2w*(q.xyz × v) + 2*(q.xyz × (q.xyz × v))
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return np.concatenate([
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ], axis=-1)


def recover_root_motion(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover root world-space position and rotation from [T, 263] features.

    Returns
    -------
    root_pos : [T, 3]  world-space root position  (Y = height above ground)
    root_rot : [T, 4]  WXYZ quaternion — Y-axis only (global facing direction)
    """
    T = data.shape[0]

    # Facing direction: integrate Y-axis angular velocity
    theta    = np.cumsum(data[:, 0].astype(np.float32))
    half     = theta * 0.5
    root_rot = np.zeros((T, 4), dtype=np.float32)
    root_rot[:, 0] = np.cos(half)
    root_rot[:, 2] = np.sin(half)

    # XZ velocity encoded in root-local frame → world frame
    vel_local = np.stack([
        data[:, 2].astype(np.float32),
        np.zeros(T, dtype=np.float32),
        data[:, 3].astype(np.float32),
    ], axis=-1)
    vel_world = _qrot_vec(root_rot, vel_local)

    root_pos = np.zeros((T, 3), dtype=np.float32)
    root_pos[:, 0] = np.cumsum(vel_world[:, 0])
    root_pos[:, 1] = data[:, 1]
    root_pos[:, 2] = np.cumsum(vel_world[:, 2])

    return root_pos, root_rot


# ──────────────────────────────────────────────────────────────────────────────
# SMPLMotion container
# ──────────────────────────────────────────────────────────────────────────────

class SMPLMotion:
    """
    Structured SMPL motion data parsed from a single HumanML3D clip.

    Attributes
    ----------
    root_pos  : [T, 3]     world-space root position  (metres)
    root_rot  : [T, 4]     WXYZ root Y-axis rotation  (global facing)
    local_rot : [T, 21, 4] WXYZ local quaternions for joints 1-21
                            T-pose = identity; relative to SMPL parent frame
    fps       : float      capture frame rate (20 for HumanML3D)
    """

    def __init__(
        self,
        root_pos:  np.ndarray,
        root_rot:  np.ndarray,
        local_rot: np.ndarray,
        fps: float = 20.0,
    ):
        self.root_pos  = np.asarray(root_pos,  dtype=np.float32)
        self.root_rot  = np.asarray(root_rot,  dtype=np.float32)
        self.local_rot = np.asarray(local_rot, dtype=np.float32)
        self.fps       = float(fps)

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]

    def slice(self, start: int = 0, end: int = -1) -> "SMPLMotion":
        e = end if end > 0 else self.num_frames
        return SMPLMotion(
            self.root_pos[start:e],
            self.root_rot[start:e],
            self.local_rot[start:e],
            self.fps,
        )


def hml3d_to_smpl_motion(data: np.ndarray, fps: float = 20.0) -> SMPLMotion:
    """
    Convert HumanML3D [T, 263] feature array to a SMPLMotion.

    Uses the actual 6D rotation data (dims 67:193) — NOT position-derived
    rotations.  This preserves twist and gives physically correct limb poses.

    Parameters
    ----------
    data : [T, 263]  raw HumanML3D features (e.g. from MoMask or dataset row)
    fps  : float     frame rate (default 20 = HumanML3D native)
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 193:
        raise ValueError(f"Expected [T, >=193] but got {data.shape}")

    T = data.shape[0]

    root_pos, root_rot = recover_root_motion(data)

    # 6D rotations for joints 1-21:  dims [67:193]  →  [T, 21, 6]
    r6d       = data[:, 67:193].reshape(T, 21, 6)
    local_rot = rot6d_to_quat(r6d)              # [T, 21, 4]  WXYZ

    return SMPLMotion(root_pos, root_rot, local_rot, fps)
