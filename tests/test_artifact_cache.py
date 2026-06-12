import numpy as np

from avatar_pipeline.models.mesh import Mesh
from avatar_pipeline.pipeline import AvatarPipeline, PipelineConfig
from avatar_pipeline.runtime.cache import artifact_get, artifact_put, content_key


def test_content_key_changes_with_array_content_and_params():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)

    assert content_key(a, "p=1") == content_key(a.copy(), "p=1")
    assert content_key(a, "p=1") != content_key(b, "p=1")
    assert content_key(a, "p=1") != content_key(a, "p=2")
    assert content_key(a, "v1") != content_key(a.astype(np.float64), "v1")


def test_artifact_roundtrip_and_key_mismatch():
    state: dict = {}
    artifact_put(state, "thing", "k1", 42)

    assert artifact_get(state, "thing", "k1") == 42
    assert artifact_get(state, "thing", "other") is None
    assert artifact_get(state, "missing", "k1") is None


class _CountingBaker:
    num_inference_steps = 15

    def __init__(self):
        self.view_calls = 0
        self.texture_calls = 0
        self.normal_calls = 0

    def generate_views(self, vertices, faces, processed, seed=0):
        self.view_calls += 1
        return np.full((6, 8, 8, 3), 128, dtype=np.uint8)

    def texture_mesh(self, verts, faces, views, uv_size, work_dir):
        self.texture_calls += 1
        uvs = np.zeros((len(verts), 2), dtype=np.float32)
        albedo = np.full((uv_size, uv_size, 3), 0.5, dtype=np.float32)
        return verts, faces, uvs, albedo

    def bake_normal_map(self, vertices, faces, uvs, photo_normals, alpha, uv_size):
        self.normal_calls += 1
        normal = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0
        return normal

    def bake_vertex_attribute(self, faces, uvs, attribute, uv_size, background):
        return (
            np.full((uv_size, uv_size, 1), 0.6, dtype=np.float32),
            np.ones((uv_size, uv_size), dtype=bool),
        )


class _FakeAO:
    num_samples = 64
    seed = 0

    def bake(self, body):
        return np.full(len(body.vertices), 0.5, dtype=np.float32)


def _mesh() -> Mesh:
    return Mesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_bake_textures_reuses_cached_artifacts_per_stage():
    pipeline = AvatarPipeline.__new__(AvatarPipeline)
    pipeline.config = PipelineConfig(uv_size=8)
    pipeline.tex_baker = _CountingBaker()
    pipeline.ao_baker = _FakeAO()

    body = _mesh()
    processed = np.ones((4, 4, 4), dtype=np.float32)
    hi_nrm = np.dstack(
        (
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        )
    )

    state: dict = {}
    pipeline._bake_textures(body, processed, hi_nrm, state, work_dir="ignored")
    pipeline._bake_textures(body, processed, hi_nrm, state, work_dir="ignored")

    assert pipeline.tex_baker.view_calls == 1
    assert pipeline.tex_baker.texture_calls == 1
    assert pipeline.tex_baker.normal_calls == 1

    # A normal-input change rebakes ONLY the normal atlas.
    hi_nrm2 = hi_nrm.copy()
    hi_nrm2[0, 0, 0] = 0.5
    pipeline._bake_textures(body, processed, hi_nrm2, state, work_dir="ignored")
    assert pipeline.tex_baker.view_calls == 1
    assert pipeline.tex_baker.texture_calls == 1
    assert pipeline.tex_baker.normal_calls == 2
