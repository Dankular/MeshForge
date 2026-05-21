"""
math3d.py
Pure numpy / scipy replacement for Blender's mathutils.
Quaternion convention throughout: (w, x, y, z)
Matrix convention: row-major, right-multiplied  (Numpy default)
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Quaternion helpers  (w, x, y, z)
# ---------------------------------------------------------------------------

def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else quat_identity()


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate == inverse for unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (Blender @ operator)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotation_difference(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Rotation that takes q1 to q2.
    r  such that  q1 @ r == q2
    r = conj(q1) @ q2
    Matches Blender's Quaternion.rotation_difference()
    """
    return quat_normalize(quat_mul(quat_conjugate(q1), q2))


def quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    """Dot product of two quaternions (used for scale retargeting)."""
    return float(np.dot(q1, q2))


def quat_to_matrix4(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) → 4×4 rotation matrix."""
    w, x, y, z = q
    m = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w), 0],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w), 0],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y), 0],
        [                0,                 0,                 0,  1],
    ], dtype=float)
    return m


def matrix4_to_quat(m: np.ndarray) -> np.ndarray:
    """4×4 matrix → unit quaternion (w,x,y,z)."""
    r = Rotation.from_matrix(m[:3, :3])
    x, y, z, w = r.as_quat()       # scipy uses (x,y,z,w)
    q = np.array([w, x, y, z])
    # Ensure positive w to match Blender convention
    if q[0] < 0:
        q = -q
    return quat_normalize(q)


# ---------------------------------------------------------------------------
# Euler ↔ Quaternion
# ---------------------------------------------------------------------------

def euler_to_quat(rx: float, ry: float, rz: float, order: str = "XYZ") -> np.ndarray:
    """Euler angles (radians) to quaternion (w,x,y,z)."""
    r = Rotation.from_euler(order, [rx, ry, rz])
    x, y, z, w = r.as_quat()
    return quat_normalize(np.array([w, x, y, z]))


def quat_to_euler(q: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Quaternion (w,x,y,z) to Euler angles (radians)."""
    w, x, y, z = q
    r = Rotation.from_quat([x, y, z, w])
    return r.as_euler(order)


# ---------------------------------------------------------------------------
# Matrix constructors
# ---------------------------------------------------------------------------

def translation_matrix(v) -> np.ndarray:
    m = np.eye(4)
    m[0, 3] = v[0]
    m[1, 3] = v[1]
    m[2, 3] = v[2]
    return m


def scale_matrix(s) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = s[0]
    m[1, 1] = s[1]
    m[2, 2] = s[2]
    return m


def trs_to_matrix4(t, r_quat, s) -> np.ndarray:
    """Combine translation, rotation (w,x,y,z quat), scale into 4×4."""
    T = translation_matrix(t)
    R = quat_to_matrix4(r_quat)
    S = scale_matrix(s)
    return T @ R @ S


def matrix4_to_trs(m: np.ndarray):
    """Decompose 4×4 into (translation[3], rotation_quat[4], scale[3])."""
    t = m[:3, 3].copy()
    sx = np.linalg.norm(m[:3, 0])
    sy = np.linalg.norm(m[:3, 1])
    sz = np.linalg.norm(m[:3, 2])
    s = np.array([sx, sy, sz])
    rot_m = m[:3, :3].copy()
    if sx > 1e-12: rot_m[:, 0] /= sx
    if sy > 1e-12: rot_m[:, 1] /= sy
    if sz > 1e-12: rot_m[:, 2] /= sz
    r = Rotation.from_matrix(rot_m)
    x, y, z, w = r.as_quat()
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return t, quat_normalize(q), s


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def vec3(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


def get_point_on_vector(initial_pt: np.ndarray, terminal_pt: np.ndarray, distance: float) -> np.ndarray:
    """
    Point at 'distance' from initial_pt along (initial_pt → terminal_pt).
    Matches Blender's get_point_on_vector helper in KeeMapBoneOperators.
    """
    n = initial_pt - terminal_pt
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return initial_pt.copy()
    n = n / norm
    return initial_pt - distance * n


def apply_rotation_matrix4(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply only the rotation part of a 4×4 matrix to a 3-vector."""
    return m[:3, :3] @ v
