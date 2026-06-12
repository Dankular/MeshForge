import pickle
from pathlib import Path

import numpy as np
import pytest

from avatar_pipeline.models.mesh import BakedTextures, Mesh, RiggedMesh
from avatar_pipeline.pipeline import (
    AvatarPipeline,
    PipelineConfig,
    _snapshot_texture_rebuild_reason,
)
from avatar_pipeline.runtime.contracts import TEXTURE_SCHEMA_VERSION


def _textures(size: int = 8, albedo_value: float = 0.5) -> BakedTextures:
    normal = np.zeros((size, size, 3), dtype=np.float32)
    normal[:, :, 2] = 1.0
    return BakedTextures(
        albedo=np.full(
            (size, size, 3),
            albedo_value,
            dtype=np.float32,
        ),
        normal=normal,
        ambient_occlusion=np.ones((size, size, 1), dtype=np.float32),
    )


def _mesh() -> Mesh:
    return Mesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def _snapshot_state(textures: BakedTextures) -> dict[str, object]:
    return {
        "processed": np.ones((4, 4, 4), dtype=np.float32),
        "seg": object(),
        "pose": object(),
        "depth": np.ones((4, 4), dtype=np.float32),
        "hi_nrm": np.dstack(
            (
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            )
        ),
        "pointmap": np.zeros((4, 4, 3), dtype=np.float32),
        "body": _mesh(),
        "textures": textures,
        # These fixtures exercise texture-cache semantics; the PSHuman
        # head-upgrade path is integration-level behavior.
        "head_detail": True,
    }


def test_legacy_snapshot_rebuilds_even_when_texture_looks_valid():
    state = _snapshot_state(_textures())

    reason = _snapshot_texture_rebuild_reason(state, expected_uv_size=8)

    assert reason == f"texture schema None does not match {TEXTURE_SCHEMA_VERSION}"


def test_current_snapshot_uses_sparse_atlas_as_corruption_backstop():
    state = _snapshot_state(_textures(albedo_value=0.0))
    state["texture_schema_version"] = TEXTURE_SCHEMA_VERSION

    reason = _snapshot_texture_rebuild_reason(state, expected_uv_size=8)

    assert reason == "albedo atlas is black or sparse"


def test_current_snapshot_rebuilds_when_uv_size_changes():
    state = _snapshot_state(_textures(size=4))
    state["texture_schema_version"] = TEXTURE_SCHEMA_VERSION

    reason = _snapshot_texture_rebuild_reason(state, expected_uv_size=8)

    assert reason == "texture resolution (4, 4) does not match (8, 8)"


def test_current_valid_snapshot_does_not_rebuild():
    state = _snapshot_state(_textures())
    state["texture_schema_version"] = TEXTURE_SCHEMA_VERSION

    assert _snapshot_texture_rebuild_reason(state, expected_uv_size=8) is None


def test_bake_textures_rasterizes_ao_into_the_uv_atlas():
    class FakeTextureBaker:
        def __init__(self):
            self.attribute_call = None

        num_inference_steps = 15

        def generate_views(self, vertices, faces, processed, seed=0):
            return np.full((6, 8, 8, 3), 128, dtype=np.uint8)

        def texture_mesh(self, verts, faces, views, uv_size, work_dir):
            uvs = np.zeros((len(verts), 2), dtype=np.float32)
            albedo = np.full((uv_size, uv_size, 3), 0.5, dtype=np.float32)
            return verts, faces, uvs, albedo

        def bake_normal_map(self, vertices, faces, uvs, photo_normals, alpha, uv_size):
            normal = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
            normal[:, :, 2] = 1.0
            return normal

        def bake_vertex_attribute(
            self,
            faces,
            uvs,
            attribute,
            uv_size,
            background,
        ):
            self.attribute_call = (faces, uvs, attribute, uv_size, background)
            return (
                np.full((uv_size, uv_size, 1), 0.6, dtype=np.float32),
                np.ones((uv_size, uv_size), dtype=bool),
            )

    class FakeAOBaker:
        num_samples = 64
        seed = 0

        def bake(self, body):
            return np.array([0.2, 0.4, 0.6], dtype=np.float32)

    pipeline = AvatarPipeline.__new__(AvatarPipeline)
    pipeline.config = PipelineConfig(uv_size=8)
    pipeline.tex_baker = FakeTextureBaker()
    pipeline.ao_baker = FakeAOBaker()

    textures, _body = pipeline._bake_textures(
        _mesh(),
        np.ones((4, 4, 4), dtype=np.float32),
        _snapshot_state(_textures())["hi_nrm"],
        work_dir="ignored",
    )

    call = pipeline.tex_baker.attribute_call
    np.testing.assert_allclose(call[2][:, 0], [0.2, 0.4, 0.6])
    assert call[3:] == (8, 1.0)
    assert textures.ambient_occlusion.shape == (8, 8, 1)


def test_legacy_snapshot_is_rebaked_and_rewritten_with_current_schema(tmp_path):
    snapshot_path = tmp_path / "pre-rig.pkl"
    state = _snapshot_state(_textures(albedo_value=0.0))
    snapshot_path.write_bytes(pickle.dumps(state))

    replacement = _textures()
    bake_calls = []

    class FakeRigger:
        def rig(self, body):
            return RiggedMesh(
                mesh=body,
                joint_names=["root"],
                joint_positions=np.zeros((1, 3), dtype=np.float32),
                skin_weights=np.ones((len(body.vertices), 1), dtype=np.float32),
                joint_parents=np.array([-1], dtype=np.int32),
            )

    class FakeExporter:
        def export(self, rigged, textures, output_path):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"glb")
            return path

    pipeline = AvatarPipeline.__new__(AvatarPipeline)
    pipeline.config = PipelineConfig(uv_size=8)
    pipeline.validate_memory_profile = lambda: None
    # Texture-cache tests: head staleness is covered at integration level.
    pipeline._head_detail_is_stale = lambda state: False
    pipeline._bake_textures = (
        lambda body, processed, hi_nrm, state=None, persist=None, work_dir=None: bake_calls.append(body) or (replacement, body)
    )
    pipeline.rigger = FakeRigger()
    pipeline.exporter = FakeExporter()

    output = pipeline.run(
        "unused.png",
        str(tmp_path / "out"),
        snapshot=str(snapshot_path),
    )

    rewritten = pickle.loads(snapshot_path.read_bytes())
    assert output.exists()
    assert len(bake_calls) == 1
    np.testing.assert_array_equal(
        bake_calls[0].vertices,
        state["body"].vertices,
    )
    assert rewritten["texture_schema_version"] == TEXTURE_SCHEMA_VERSION
    np.testing.assert_array_equal(rewritten["textures"].albedo, replacement.albedo)


def test_failed_rebake_does_not_mark_or_replace_legacy_snapshot(tmp_path):
    snapshot_path = tmp_path / "pre-rig.pkl"
    snapshot_path.write_bytes(pickle.dumps(_snapshot_state(_textures(8, 0.0))))
    original_bytes = snapshot_path.read_bytes()

    pipeline = AvatarPipeline.__new__(AvatarPipeline)
    pipeline.config = PipelineConfig(uv_size=8)
    pipeline.validate_memory_profile = lambda: None
    # Texture-cache tests: head staleness is covered at integration level.
    pipeline._head_detail_is_stale = lambda state: False
    pipeline._bake_textures = lambda body, processed, hi_nrm, state=None, persist=None, work_dir=None: (
        _ for _ in ()
    ).throw(RuntimeError("bake failed"))

    with pytest.raises(RuntimeError, match="bake failed"):
        pipeline.run(
            "unused.png",
            str(tmp_path / "out"),
            snapshot=str(snapshot_path),
        )

    assert snapshot_path.read_bytes() == original_bytes
