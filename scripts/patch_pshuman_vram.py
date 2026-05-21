"""
patch_pshuman_vram.py
=====================
Apply VRAM-reduction optimisations to /root/PSHuman/inference.py.

Patches applied:
  1. load_pshuman_pipeline — adds VAE slicing + CPU offload on top of
     the existing fp16 + xformers that are already in the file.
  2. run_inference — adds torch.cuda.empty_cache() after the pipeline
     call so fragmented VRAM is reclaimed between multi-view denoising.

fp32 @ 768 res ≈ 40 GB.  fp16 ≈ 20 GB.  fp16 + xformers ≈ 16-18 GB.
fp16 + xformers + VAE slicing + CPU offload ≈ 14-16 GB peak → fits 24 GB.

Run:
    /root/miniconda/envs/pshuman/bin/python /root/MeshForge/scripts/patch_pshuman_vram.py
"""
import pathlib, sys

TARGET = pathlib.Path("/root/PSHuman/inference.py")
if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found — run after PSHuman is cloned")

src = TARGET.read_text()
original = src  # keep a backup reference

# ─────────────────────────────────────────────────────────────────
# Patch 1: load_pshuman_pipeline
# ─────────────────────────────────────────────────────────────────
OLD_LOAD = """\
def load_pshuman_pipeline(cfg):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda')
    return pipeline"""

NEW_LOAD = """\
def load_pshuman_pipeline(cfg):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,          # float16 — halves VRAM vs fp32
    )

    # xformers: reduces peak VRAM during multi-head denoising attention
    try:
        pipeline.unet.enable_xformers_memory_efficient_attention()
        print("[PSHuman] xformers memory-efficient attention enabled")
    except Exception as _xe:
        print(f"[PSHuman] xformers unavailable ({_xe}) — falling back to attention slicing")
        pipeline.unet.enable_attention_slicing(1)

    # VAE slicing: prevents OOM when decoding a 7-view 768-res batch at once
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print("[PSHuman] VAE slicing enabled")

    # CPU offload: idle pipeline components (text encoder, VAE, safety checker)
    # move to RAM when not actively used, freeing ~3-4 GB of static VRAM.
    # pipeline() is called via standard diffusers __call__, so hooks work.
    if torch.cuda.is_available():
        try:
            pipeline.enable_model_cpu_offload()
            print("[PSHuman] model CPU offload enabled")
        except Exception as _oe:
            print(f"[PSHuman] CPU offload unavailable ({_oe}) — loading to CUDA directly")
            pipeline.to("cuda")

    return pipeline"""

if OLD_LOAD in src:
    src = src.replace(OLD_LOAD, NEW_LOAD)
    print("[patch 1] load_pshuman_pipeline — VRAM optimisations applied")
elif "enable_vae_slicing" in src:
    print("[patch 1] load_pshuman_pipeline — already patched, skipping")
else:
    # Looser match for minor whitespace/version differences
    import re
    m = re.search(
        r'def load_pshuman_pipeline\(cfg\):.*?return pipeline',
        src, re.DOTALL
    )
    if m:
        src = src[:m.start()] + NEW_LOAD + src[m.end():]
        print("[patch 1] load_pshuman_pipeline — applied via regex fallback")
    else:
        print("[patch 1] WARNING: could not locate load_pshuman_pipeline — skipping")

# ─────────────────────────────────────────────────────────────────
# Patch 2: empty CUDA cache after pipeline call in run_inference
# ─────────────────────────────────────────────────────────────────
# Insert torch.cuda.empty_cache() right after the pipeline __call__ block.
# The existing code already has `torch.cuda.empty_cache()` at the bottom of
# the batch loop — so only add if it's missing near the unet_out line.

OLD_CACHE_ANCHOR = """\
            with torch.autocast("cuda"):
                # B*Nv images
                guidance_scale = cfg.validation_guidance_scales
                unet_out = pipeline("""

NEW_CACHE_ANCHOR = """\
            torch.cuda.empty_cache()  # free fragmented VRAM before denoising
            with torch.autocast("cuda"):
                # B*Nv images
                guidance_scale = cfg.validation_guidance_scales
                unet_out = pipeline("""

if OLD_CACHE_ANCHOR in src and "empty_cache()  # free fragmented" not in src:
    src = src.replace(OLD_CACHE_ANCHOR, NEW_CACHE_ANCHOR)
    print("[patch 2] run_inference — added pre-denoising cache flush")
else:
    print("[patch 2] run_inference — cache flush already present or anchor not found, skipping")

# ─────────────────────────────────────────────────────────────────
# Write back only if changed
# ─────────────────────────────────────────────────────────────────
if src != original:
    backup = TARGET.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(original)
        print(f"[patch] Backup saved → {backup}")
    TARGET.write_text(src)
    print(f"[patch] Written → {TARGET}")
else:
    print("[patch] No changes made.")
