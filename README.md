# MeshForge — Avatar Pipeline

Single-image -> rigged, textured GLB avatar. Every stage runs its real
reference model in-process; see `CODEX.md` for the architecture and the
hard rules (no stubs, no hash-derived outputs, no silent fallbacks).

## Stages

1. **Preprocess** — rembg (u2net_human_seg) background removal
2. **Sapiens backbone** — segmentation (verified Goliath classes), pose,
   depth, surface normals
3. **TripoSG** — body mesh (SDF diffusion, outward-winding normalized)
4. **PSHuman** — head detail (7-view unclip diffusion + SMPL-X carve),
   fused pre-rig onto the body
5. **xatlas** — UV unwrap of the fused mesh
6. **Bake** — MV-Adapter 6-view diffusion albedo + big-lama fill,
   Embree hemispheric raycast AO, tangent-space normal bake
7. **SkinTokens** — skeleton + skinning weights
8. **GLB export** — PBR material, sidecar PNGs, meta.json

The stage list above is what runs today. `CODEX.md` carries the revised
target architecture (PSHuman/SMPL-X-scaffolded body base, TripoSG demoted
to supplemental shells) and the migration order.

## Run

```powershell
$env:PYTHONPATH = "src"
python -m avatar_pipeline.main <input.png> <output_dir> --snapshot <cache.pkl>
```

`--snapshot` caches the pre-rig state; stale caches (old texture schema,
missing head detail, inward winding) upgrade themselves on load without
rerunning TripoSG.

PSHuman's native full-figure SMPL-X-guided reconstruction is the default
body+head source (A/B-verified: silhouette IoU 0.97 vs TripoSG's 0.72);
`--triposg-body` restores the legacy TripoSG + head-transplant arm.

`--from-candid` accepts a candid photo (face or partial face+body) instead
of a T-pose reference: IP-Adapter FaceID PlusV2 (external/IP-Adapter, weights
from h94/IP-Adapter-FaceID) + an OpenPose ControlNet conditioned on a
synthetic T-pose skeleton synthesize a gated best-of-N
`<output_dir>/tpose_reference.png` (strict-pose + identity scoring,
HyperSwap + GFPGAN identity enhancement), plus a perspective-neutral
`face_portrait.png` that PSHuman carves into a high-fidelity head
(multiview diffusion supplies the depth a single warped candid view
cannot) transplanted onto the full-figure body. The SD15 stack loads,
runs, and unloads before the main pipeline's mmgp domain comes up.

## Memory

All heavy torch models share one mmgp offload domain (8 GB-class GPUs work;
weights idle in host RAM and stream on demand). Non-torch components take
short-lived explicit GPU residency. `nvdiffrast` JIT-compiles its CUDA
plugin on first use — ninja and MSVC must be reachable (see
`requirements.txt` for the torch-version-coupled wheels: pytorch3d, kaolin,
torch_scatter, xformers).

## Tests

```
python -m pytest            # unit tests (fast; slow e2e excluded)
python -m pytest -m slow    # full end-to-end pipeline runs (~1h each)
```
