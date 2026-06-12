import numpy as np

from avatar_pipeline.models.mesh import BakedTextures, RiggedMesh


# v2: bake fills view-uncovered UV texels with big-lama inpainting; v1 atlases
# left them black, so any cached v1 textures must be rebaked.
# v3: bake renders in the proper z-up rig frame (load_mesh-equivalent
# rotation) so the reference image conditions upright views, and the normal
# atlas is a real tangent-space bake instead of a photo resize.
# v4: 50-step views, exposure anchoring, RealESRGAN upscale, SmartPainter
# view inpaint, poisson reprojection, 3D-nearest fill, PSHuman head colors.
# v5: texturing via the reference TexturePipeline (UVAtlas unwrap inside),
# Space-exact; the textured mesh geometry comes from the texture stage.
TEXTURE_SCHEMA_VERSION = 5
MIN_ALBEDO_ACTIVE_RATIO = 0.02
ALBEDO_ACTIVE_THRESHOLD = 1.0 / 255.0


def albedo_active_ratio(albedo: np.ndarray) -> float:
    if albedo.ndim != 3 or albedo.shape[2] != 3 or albedo.size == 0:
        return 0.0
    active = np.max(albedo, axis=2) > ALBEDO_ACTIVE_THRESHOLD
    return float(np.mean(active))


def has_sparse_albedo(textures: BakedTextures) -> bool:
    return albedo_active_ratio(textures.albedo) < MIN_ALBEDO_ACTIVE_RATIO


def validate_texture_contracts(textures: BakedTextures) -> None:
    arrays = {
        "Albedo": textures.albedo,
        "Normal": textures.normal,
        "AO": textures.ambient_occlusion,
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

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

    if np.any(textures.albedo < 0.0) or np.any(textures.albedo > 1.0):
        raise ValueError("Albedo values must be in [0, 1]")
    if np.any(textures.normal < -1.0) or np.any(textures.normal > 1.0):
        raise ValueError("Normal values must be in [-1, 1]")
    if np.any(textures.ambient_occlusion < 0.0) or np.any(
        textures.ambient_occlusion > 1.0
    ):
        raise ValueError("AO values must be in [0, 1]")


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
    if rigged.joint_parents is not None:
        if rigged.joint_parents.shape != (len(names),):
            raise ValueError("Joint parent shape mismatch")
        if np.any(rigged.joint_parents >= len(names)):
            raise ValueError("Joint parent index is out of range")

    skin = rigged.skin_weights
    if np.isnan(skin).any() or np.any(skin < -1e-6):
        raise ValueError("Invalid skin weights")
    s = np.sum(skin, axis=1)
    if np.max(np.abs(s - 1.0)) > 1e-3:
        raise ValueError("Skin weights must sum to 1 per vertex")

    validate_texture_contracts(textures)
