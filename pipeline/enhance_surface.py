"""
Surface enhancement for TripoSG GLB outputs.

StableNormal  — high-quality normal map from portrait reference
Depth-Anything V2 — metric depth map → displacement intensity

Both run on the reference portrait, produce calibrated maps that
are baked as PBR textures (normalTexture + occlusion/displacement)
into the output GLB.
"""

import os
import numpy as np
import torch
from PIL import Image


STABLE_NORMAL_PATH  = "/root/models/stable-normal"
DEPTH_ANYTHING_PATH = "/root/models/depth-anything-v2"

_normal_pipe  = None
_depth_pipe   = None


# ── model loading ──────────────────────────────────────────────────────────────

def load_normal_model():
    global _normal_pipe
    if _normal_pipe is not None:
        return _normal_pipe
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
    import torch
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        STABLE_NORMAL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        t_start=int(0.3 * 1000),
    ).to("cuda")
    _normal_pipe = YOSONormalsPipeline.from_pretrained(
        STABLE_NORMAL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        scheduler=HEURI_DDIMScheduler.from_pretrained(
            STABLE_NORMAL_PATH, subfolder="scheduler",
            ddim_timestep_respacing="ddim10", x_start_pipeline=x_start_pipeline,
        ),
    ).to("cuda")
    _normal_pipe.set_progress_bar_config(disable=True)
    return _normal_pipe


def load_depth_model():
    global _depth_pipe
    if _depth_pipe is not None:
        return _depth_pipe
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(DEPTH_ANYTHING_PATH)
    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_ANYTHING_PATH, torch_dtype=torch.float16
    ).to("cuda")
    _depth_pipe = (processor, model)
    return _depth_pipe


def unload_models():
    global _normal_pipe, _depth_pipe
    if _normal_pipe is not None:
        del _normal_pipe; _normal_pipe = None
    if _depth_pipe is not None:
        del _depth_pipe; _depth_pipe = None
    torch.cuda.empty_cache()


# ── inference ──────────────────────────────────────────────────────────────────

def run_stable_normal(image: Image.Image, resolution: int = 768) -> Image.Image:
    """Returns normal map as RGB PIL image ([-1,1] encoded as [0,255])."""
    pipe = load_normal_model()
    img = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    with torch.inference_mode(), torch.autocast("cuda"):
        result = pipe(img)
    normal_img = result.prediction  # numpy [H,W,3] in [-1,1]
    normal_rgb = ((normal_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(normal_rgb)


def run_depth_anything(image: Image.Image, resolution: int = 768) -> Image.Image:
    """Returns depth map as 16-bit grayscale PIL image (normalized 0–65535)."""
    processor, model = load_depth_model()
    img = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.float16) for k, v in inputs.items()}
    with torch.inference_mode():
        depth = model(**inputs).predicted_depth[0].float().cpu().numpy()
    # Normalize to 0–1
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_16 = (depth * 65535).astype(np.uint16)
    return Image.fromarray(depth_16, mode="I;16")


# ── GLB baking ─────────────────────────────────────────────────────────────────

def bake_normal_into_glb(
    glb_path: str,
    normal_img: Image.Image,
    out_path: str,
    normal_strength: float = 1.0,
) -> str:
    """
    Adds normalTexture to the first material of the GLB.
    Normal map is resized to match the existing base color texture resolution.
    """
    import pygltflib, struct, io

    gltf = pygltflib.GLTF2().load(glb_path)

    # Find existing base color texture size for matching resolution
    target_size = 1024
    if gltf.materials and gltf.materials[0].pbrMetallicRoughness:
        pbr = gltf.materials[0].pbrMetallicRoughness
        if pbr.baseColorTexture is not None:
            tex_idx = pbr.baseColorTexture.index
            img_idx = gltf.textures[tex_idx].source
            blob = gltf.binary_blob()
            bv = gltf.bufferViews[gltf.images[img_idx].bufferView]
            img_bytes = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
            existing = Image.open(io.BytesIO(img_bytes))
            target_size = existing.width

    normal_resized = normal_img.resize((target_size, target_size), Image.LANCZOS)

    # Encode normal map as PNG and append to binary blob
    buf = io.BytesIO()
    normal_resized.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    blob = bytearray(gltf.binary_blob() or b"")
    byte_offset = len(blob)
    blob.extend(png_bytes)

    # Pad to 4-byte alignment
    while len(blob) % 4:
        blob.append(0)

    # Add bufferView, image, texture
    bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=byte_offset, byteLength=len(png_bytes),
    ))
    img_idx = len(gltf.images)
    gltf.images.append(pygltflib.Image(
        bufferView=bv_idx, mimeType="image/png",
    ))
    tex_idx = len(gltf.textures)
    gltf.textures.append(pygltflib.Texture(source=img_idx))

    # Update material
    if gltf.materials:
        gltf.materials[0].normalTexture = pygltflib.NormalMaterialTexture(
            index=tex_idx, scale=normal_strength,
        )

    # Update buffer length
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save(out_path)
    return out_path


def bake_depth_as_occlusion(
    glb_path: str,
    depth_img: Image.Image,
    out_path: str,
    displacement_scale: float = 1.0,
) -> str:
    """
    Bakes depth map as occlusionTexture (R channel) — approximates displacement
    in PBR renderers. Depth is inverted and normalized for AO-style use.
    """
    import pygltflib, io

    gltf = pygltflib.GLTF2().load(glb_path)

    target_size = 1024
    if gltf.materials and gltf.materials[0].pbrMetallicRoughness:
        pbr = gltf.materials[0].pbrMetallicRoughness
        if pbr.baseColorTexture is not None:
            tex_idx = pbr.baseColorTexture.index
            img_idx = gltf.textures[tex_idx].source
            blob = gltf.binary_blob()
            bv = gltf.bufferViews[gltf.images[img_idx].bufferView]
            img_bytes = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
            existing = Image.open(io.BytesIO(img_bytes))
            target_size = existing.width

    # Convert 16-bit depth to 8-bit RGB occlusion (inverted, scaled)
    depth_arr = np.array(depth_img).astype(np.float32) / 65535.0
    depth_arr = 1.0 - depth_arr  # invert: close = bright
    depth_arr = np.clip(depth_arr * displacement_scale, 0, 1)
    occ_8 = (depth_arr * 255).astype(np.uint8)
    occ_rgb = Image.fromarray(np.stack([occ_8, occ_8, occ_8], axis=-1))
    occ_rgb = occ_rgb.resize((target_size, target_size), Image.LANCZOS)

    buf = io.BytesIO()
    occ_rgb.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    blob = bytearray(gltf.binary_blob() or b"")
    byte_offset = len(blob)
    blob.extend(png_bytes)
    while len(blob) % 4:
        blob.append(0)

    bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=byte_offset, byteLength=len(png_bytes),
    ))
    img_idx = len(gltf.images)
    gltf.images.append(pygltflib.Image(
        bufferView=bv_idx, mimeType="image/png",
    ))
    tex_idx = len(gltf.textures)
    gltf.textures.append(pygltflib.Texture(source=img_idx))

    if gltf.materials:
        gltf.materials[0].occlusionTexture = pygltflib.OcclusionTextureInfo(
            index=tex_idx, strength=displacement_scale,
        )

    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save(out_path)
    return out_path
