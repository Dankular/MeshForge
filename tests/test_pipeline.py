import json
from pathlib import Path

import numpy as np
from PIL import Image

from avatar_pipeline.pipeline import AvatarPipeline


def test_pipeline_runs_end_to_end(tmp_path):
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "out"

    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[8:56, 16:48, :] = 40
    Image.fromarray(image).save(input_path)

    pipeline = AvatarPipeline()
    output_path = pipeline.run(str(input_path), str(output_dir))

    assert output_path.exists()
    payload = json.loads(
        output_path.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert payload["asset"]["version"] == "2.0"
    assert payload["mesh"]["vertex_count"] > 0
    assert payload["rig"]["joint_count"] >= 4


def test_pipeline_runs_with_workspace_image(tmp_path):
    workspace_image = Path(__file__).resolve().parents[1] / "test.png"
    assert workspace_image.exists(), f"Missing expected image: {workspace_image}"

    pipeline = AvatarPipeline()
    output_path = pipeline.run(str(workspace_image), str(tmp_path / "workspace-out"))

    assert output_path.exists()
    payload = json.loads(
        output_path.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert payload["mesh"]["vertex_count"] > 0
    assert payload["textures"]["albedo_shape"][0] == 256


def test_export_payload_has_required_fields_and_ranges(tmp_path):
    workspace_image = Path(__file__).resolve().parents[1] / "test.png"
    pipeline = AvatarPipeline()
    output_path = pipeline.run(str(workspace_image), str(tmp_path / "strict-out"))
    payload = json.loads(
        output_path.with_suffix(".meta.json").read_text(encoding="utf-8")
    )

    assert payload["asset"]["version"] == "2.0"
    assert payload["asset"]["generator"] == "avatar-pipeline"

    mesh = payload["mesh"]
    assert mesh["vertex_count"] > 100
    assert mesh["face_count"] > 100
    assert len(mesh["bounds_min"]) == 3
    assert len(mesh["bounds_max"]) == 3
    assert all(isinstance(v, (int, float)) for v in mesh["bounds_min"])
    assert all(isinstance(v, (int, float)) for v in mesh["bounds_max"])
    assert all(lo <= hi for lo, hi in zip(mesh["bounds_min"], mesh["bounds_max"]))

    rig = payload["rig"]
    assert rig["joint_count"] == len(rig["joint_names"])
    assert rig["joint_count"] == len(rig["joint_positions"])
    assert rig["joint_count"] >= 4
    assert all(len(p) == 3 for p in rig["joint_positions"])

    tex = payload["textures"]
    assert tex["albedo_shape"] == [256, 256, 3]
    assert tex["normal_shape"] == [256, 256, 3]
    assert tex["ao_shape"] == [256, 256, 1]
    assert 0.0 <= tex["albedo_mean"] <= 1.0
    assert -1.0 <= tex["normal_mean_z"] <= 1.0
    assert 0.0 <= tex["ao_mean"] <= 1.0
