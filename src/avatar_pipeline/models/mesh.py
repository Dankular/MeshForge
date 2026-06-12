from dataclasses import dataclass, field
import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None
    semantic_regions: list[str] = field(default_factory=list)


@dataclass
class RiggedMesh:
    mesh: Mesh
    joint_names: list[str]
    joint_positions: np.ndarray
    skin_weights: np.ndarray
    joint_parents: np.ndarray | None = None


@dataclass
class BakedTextures:
    albedo: np.ndarray
    normal: np.ndarray
    ambient_occlusion: np.ndarray
