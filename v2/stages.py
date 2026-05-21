from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any


def _import_app_module():
    repo_root = Path(__file__).resolve().parent.parent
    pipeline_dir = repo_root / "pipeline"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    import app as meshforge_app

    return meshforge_app


def run_shape_stage(image_array, cfg, seed: int) -> dict[str, Any]:
    app = _import_app_module()
    glb_path, status = app.generate_shape(
        input_image=image_array,
        remove_background=cfg.remove_background,
        num_steps=cfg.shape_steps,
        guidance_scale=cfg.shape_guidance,
        seed=seed,
        face_count=cfg.shape_face_count,
    )
    if not glb_path:
        raise RuntimeError(status)
    return {"shape_glb": glb_path, "status": status}


def run_texture_stage(
    shape_glb: str, image_array, cfg, tex_seed: int
) -> dict[str, Any]:
    app = _import_app_module()
    textured_glb, mv_img, status = app.apply_texture(
        glb_path=shape_glb,
        input_image=image_array,
        remove_background=cfg.remove_background,
        variant=cfg.texture_variant,
        tex_seed=tex_seed,
        enhance_face=cfg.enhance_face_texture,
        rembg_threshold=cfg.rembg_threshold,
        rembg_erode=cfg.rembg_erode,
    )
    if not textured_glb:
        raise RuntimeError(status)
    return {"textured_glb": textured_glb, "mv_preview": mv_img, "status": status}


def run_rig_facefusion_stage(textured_glb: str, image_array, cfg) -> dict[str, Any]:
    app = _import_app_module()
    final_glb, _animated, fbx, status, *_ = app.gradio_rig(
        input_image=image_array,
        glb_state_path=textured_glb,
        export_fbx_flag=cfg.export_fbx,
        pshuman_weight_threshold=cfg.pshuman_weight_threshold,
        pshuman_retract_mm=cfg.pshuman_retract_mm,
    )
    if not final_glb:
        raise RuntimeError(status)
    return {"final_glb": final_glb, "fbx": fbx, "status": status}


def materialize_artifacts(
    stage_outputs: dict[str, Any], artifacts_dir: Path
) -> dict[str, str]:
    copied: dict[str, str] = {}
    mapping = {
        "shape_glb": artifacts_dir / "01_shape.glb",
        "textured_glb": artifacts_dir / "02_textured.glb",
        "final_glb": artifacts_dir / "03_final_rigged_hdface.glb",
    }
    if stage_outputs.get("fbx"):
        mapping["fbx"] = artifacts_dir / "04_final_rigged_hdface.fbx"

    for key, dst in mapping.items():
        src = stage_outputs.get(key)
        if src:
            shutil.copy2(src, dst)
            copied[key] = str(dst)
    return copied
