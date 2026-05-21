from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import WorkflowConfig, ensure_job_dirs
from .stages import (
    materialize_artifacts,
    run_rig_facefusion_stage,
    run_shape_stage,
    run_texture_stage,
)


class UnifiedWorkflow:
    def __init__(
        self, config: WorkflowConfig, output_dir: Path, seed: int, tex_seed: int
    ):
        self.config = config
        self.output_dir = output_dir
        self.seed = int(seed)
        self.tex_seed = int(tex_seed)
        self.artifacts_dir, self.logs_dir = ensure_job_dirs(output_dir)

    def run(self, image_path: Path) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        manifest: dict[str, Any] = {
            "workflow": "meshforge-v2-unified",
            "started_at": started.isoformat(),
            "input": {"image": str(image_path)},
            "config": asdict(self.config),
            "seed": self.seed,
            "tex_seed": self.tex_seed,
            "stages": [],
            "artifacts": {},
            "status": "running",
        }

        all_outputs: dict[str, Any] = {}

        try:
            image_array = np.array(Image.open(image_path).convert("RGB"))

            shape = run_shape_stage(image_array, self.config, self.seed)
            all_outputs.update(shape)
            manifest["stages"].append(
                {"name": "shape", "status": "ok", "detail": shape["status"]}
            )

            texture = run_texture_stage(
                shape["shape_glb"], image_array, self.config, self.tex_seed
            )
            all_outputs.update(texture)
            manifest["stages"].append(
                {"name": "texture", "status": "ok", "detail": texture["status"]}
            )

            fused = run_rig_facefusion_stage(
                texture["textured_glb"], image_array, self.config
            )
            all_outputs.update(fused)
            manifest["stages"].append(
                {
                    "name": "rig_pshuman_fusion",
                    "status": "ok",
                    "detail": fused["status"],
                }
            )

            manifest["artifacts"] = materialize_artifacts(
                all_outputs, self.artifacts_dir
            )
            manifest["status"] = "completed"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest(manifest)
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        out = self.output_dir / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
