# Avatar Pipeline Engineering Scope

## Core Philosophy

This project is a semantic human reconstruction engine.

Reference systems — vendored under `external/` and run **in-process** with
their real published weights:

- TripoSG   — body geometry (SDF diffusion)
- PSHuman   — head/face detail (7-view unclip diffusion + SMPL-X carve)
- Sapiens   — semantic backbone (seg / pose / depth / normals)
- MV-Adapter — multiview texture generation + UV projection
- SkinTokens — joint skeleton + skinning-weight generation
- rembg     — background removal

## Hard Constraints

- No subprocess orchestration, no CLI wrappers, no Blender
- No hash-derived outputs, no stub stages, no silent fallbacks:
  every stage runs its real model or fails loudly
- All heavy torch models live in **one shared mmgp offload domain**
  (`runtime/memory.py`), built on CPU and streamed to the GPU on demand;
  `AvatarPipeline.validate_memory_profile()` enforces that nothing escapes.
  Non-torch components (onnxruntime sessions, PIXIE detector, optimizers)
  take explicit short-lived GPU residency instead.
  mmgp setup is always the LAST construction step: its global cuda
  default-device mode breaks numpy/tensor interop inside model constructors.

## Pipeline (implemented)

INPUT IMAGE
    ↓
preprocessing            rembg u2net_human_seg → RGBA
    ↓
semantic parsing         Sapiens-1b seg (verified Goliath class order),
                         pose, depth, normals + geometric pointmap
    ↓
body generation          TripoSG (+ outward-winding normalization)
    ↓
head generation          PSHuman head detail from the seg-derived head crop
    ↓
semantic mesh fusion     pre-rig transplant (fusion/head_fusion.py)
    ↓
UV generation            xatlas unwrap of the fused mesh
    ↓
texture baking           MV-Adapter 6-view diffusion bake (z-up rig frame)
                         + big-lama fill for view-uncovered texels
                         + hemispheric raycast AO (Embree)
                         + tangent-space normal bake (front-view projection)
    ↓
rigging                  SkinTokens (skeleton + top-4 skinning weights)
    ↓
GLB export               trimesh PBR + sidecar maps + meta.json

## Still TODO

- Canonical transfer: runtime assets sharing topology / UVs / skeletons /
  material slots across avatars (the canonical template does not exist yet)
- PSHuman normal-map reprojection for the head region of the atlas
- CV-CUDA-gated MV-Adapter quality passes (poisson_reprojection,
  SmartPainter view inpaint) — no Windows wheels exist

## Snapshots

`--snapshot` caches the pre-rig state (stages 1–5). `TEXTURE_SCHEMA_VERSION`
in `runtime/contracts.py` gates cached textures: bump it whenever bake
semantics change and stale snapshots rebake automatically without rerunning
TripoSG.

## Major Systems

preprocess/   rembg
sapiens/      seg, pose, depth, normals, pointmap (+ shared TorchScript loader)
generators/   triposg_body
fusion/       head_fusion (PSHuman pre-rig transplant)
baking/       mv_adapter_texture (color + normal + vertex attrs), ao_bake
rigging/      skintokens
export/       glb
runtime/      contracts, memory (mmgp domain)
