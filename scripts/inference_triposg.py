from __future__ import annotations

from typing import Any

import torch

from scripts.image_process import prepare_image


def _call_pipeline(
    pipe: Any, image, seed: int, num_inference_steps: int, guidance_scale: float
):
    # TripoSG uses torch.Generator instead of seed parameter
    generator = torch.Generator().manual_seed(int(seed))

    if callable(pipe):
        try:
            return pipe(
                image=image,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            )
        except TypeError as e:
            pass

    if hasattr(pipe, "run"):
        try:
            return pipe.run(
                image=image,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            )
        except TypeError:
            pass

    for method_name in ("generate", "infer", "sample"):
        method = getattr(pipe, method_name, None)
        if method is None:
            continue
        try:
            return method(
                image=image,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            )
        except TypeError:
            continue

    raise RuntimeError(
        "Unsupported TripoSG pipeline interface in scripts/inference_triposg.py"
    )


def run_triposg(
    pipe: Any,
    image_input: str,
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    faces: int = -1,
):
    bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).numpy()
    img = prepare_image(image_input, bg_color=bg_color, rmbg_net=rmbg_net)
    result = _call_pipeline(
        pipe, img, int(seed), int(num_inference_steps), float(guidance_scale)
    )

    # Extract mesh from TripoSGPipelineOutput
    if hasattr(result, "mesh"):
        mesh = result.mesh
    elif hasattr(result, "meshes"):
        mesh = result.meshes[0] if isinstance(result.meshes, (list, tuple)) else result.meshes
    else:
        # Assume result is already a mesh
        mesh = result

    if (
        hasattr(mesh, "simplify_quadric_decimation")
        and isinstance(faces, int)
        and faces > 0
    ):
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=faces)
        except Exception:
            pass
    return mesh
