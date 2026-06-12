import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from avatar_pipeline.main import app


@pytest.mark.slow
def test_cli_run_command_with_workspace_image(tmp_path):
    workspace_image = Path(__file__).resolve().parents[1] / "test.png"
    assert workspace_image.exists()

    out_dir = tmp_path / "cli-out"
    runner = CliRunner()
    result = runner.invoke(app, [str(workspace_image), str(out_dir)])

    assert result.exit_code == 0
    output_file = out_dir / "avatar.glb"
    assert output_file.exists()

    payload = json.loads(
        output_file.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert payload["mesh"]["vertex_count"] > 0
    assert payload["rig"]["joint_count"] >= 4
    assert "Avatar exported to:" in result.stdout
