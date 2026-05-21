import numpy as np

from avatar_pipeline.models.mesh import BakedTextures, RiggedMesh


def validate_runtime_contracts(rigged: RiggedMesh, textures: BakedTextures) -> None:
    mesh = rigged.mesh
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError("Invalid vertex layout")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Invalid face layout")
    if np.isnan(mesh.vertices).any() or np.isnan(mesh.faces).any():
        raise ValueError("Mesh contains NaN values")

    names = rigged.joint_names
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("Joint names must be non-empty and unique")
    if rigged.joint_positions.shape != (len(names), 3):
        raise ValueError("Joint positions shape mismatch")
    if rigged.skin_weights.shape != (mesh.vertices.shape[0], len(names)):
        raise ValueError("Skin weight shape mismatch")

    skin = rigged.skin_weights
    if np.isnan(skin).any() or np.any(skin < -1e-6):
        raise ValueError("Invalid skin weights")
    s = np.sum(skin, axis=1)
    if np.max(np.abs(s - 1.0)) > 1e-3:
        raise ValueError("Skin weights must sum to 1 per vertex")

    if textures.albedo.ndim != 3 or textures.albedo.shape[2] != 3:
        raise ValueError("Albedo must be HxWx3")
    if textures.normal.ndim != 3 or textures.normal.shape[2] != 3:
        raise ValueError("Normal must be HxWx3")
    if textures.ambient_occlusion.ndim != 3 or textures.ambient_occlusion.shape[2] != 1:
        raise ValueError("AO must be HxWx1")

    if (
        textures.albedo.shape[:2] != textures.normal.shape[:2]
        or textures.albedo.shape[:2] != textures.ambient_occlusion.shape[:2]
    ):
        raise ValueError("Texture resolutions must match")
