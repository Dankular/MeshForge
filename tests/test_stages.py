import numpy as np

from avatar_pipeline.baking.ao_bake import AmbientOcclusionBaker
from avatar_pipeline.models.mesh import Mesh
from avatar_pipeline.rigging.skintokens import SkinTokensRigger


def _simple_mesh() -> Mesh:
    vertices = np.array(
        [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
            [0.0, 1.5, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 3, 2],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )
    return Mesh(vertices=vertices, faces=faces)


def test_ao_bake_produces_bounded_per_vertex_occlusion():
    mesh = _simple_mesh()
    ao = AmbientOcclusionBaker().bake(mesh)

    assert ao.shape == (len(mesh.vertices),)
    assert ao.dtype == np.float32
    assert np.all(ao >= 0.0) and np.all(ao <= 1.0)


def test_skintokens_normalization_and_weight_sparsification():
    mesh = _simple_mesh()
    rigger = SkinTokensRigger()

    normalized, center, scale = rigger._normalization(mesh.vertices)
    dense_weights = np.arange(6 * 8, dtype=np.float32).reshape(6, 8) + 1.0
    sparse_weights = rigger._top_four_weights(dense_weights)

    np.testing.assert_allclose(normalized * scale + center, mesh.vertices, atol=1e-6)
    assert np.all(np.count_nonzero(sparse_weights, axis=1) == 4)
    np.testing.assert_allclose(np.sum(sparse_weights, axis=1), 1.0, atol=1e-6)
