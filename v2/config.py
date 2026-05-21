from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowConfig:
    preset: str = "quality"
    remove_background: bool = True
    shape_steps: int = 30
    shape_guidance: float = 7.5
    shape_face_count: int = 120000
    texture_variant: str = "sdxl"
    enhance_face_texture: bool = True
    rembg_threshold: float = 0.5
    rembg_erode: int = 2
    export_fbx: bool = False
    pshuman_weight_threshold: float = 0.5
    pshuman_retract_mm: float = 2.0


PRESETS: dict[str, WorkflowConfig] = {
    "preview": WorkflowConfig(
        preset="preview",
        shape_steps=18,
        shape_guidance=6.0,
        shape_face_count=60000,
        enhance_face_texture=False,
    ),
    "quality": WorkflowConfig(
        preset="quality",
        shape_steps=30,
        shape_guidance=7.5,
        shape_face_count=120000,
        enhance_face_texture=True,
    ),
    "cinematic": WorkflowConfig(
        preset="cinematic",
        shape_steps=45,
        shape_guidance=8.5,
        shape_face_count=200000,
        enhance_face_texture=True,
        pshuman_weight_threshold=0.45,
        pshuman_retract_mm=1.0,
    ),
}


def get_preset(name: str) -> WorkflowConfig:
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
    return PRESETS[name]


def ensure_job_dirs(output_dir: Path) -> tuple[Path, Path]:
    artifacts = output_dir / "artifacts"
    logs = output_dir / "logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return artifacts, logs
