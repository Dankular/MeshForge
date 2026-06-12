# MeshForge — Avatar Pipeline

Single image -> rigged, textured GLB avatar. Every stage runs its real
reference model in-process; see `CODEX.md` for the architecture and the
hard rules (no stubs, no hash-derived outputs, no silent fallbacks).

## Stages (default path)

1. **Preprocess** — rembg (u2net_human_seg) background removal
2. **Sapiens backbone** — segmentation (verified Goliath classes), pose
   (Goliath-308 keypoint subset), depth, surface normals, pointmap
3. **PSHuman full figure** — SMPL-X-guided 7-view unclip diffusion +
   carve: body AND head in one piece (A/B-verified over the legacy
   TripoSG arm: silhouette IoU 0.97 vs 0.72)
4. **Portrait head** (candid flow) — PSHuman carves a second,
   high-fidelity head from the FaceID portrait (the full 768px view
   budget spent on the face alone) and transplants it onto the body at
   the width-profile-detected neck line
5. **UV unwrap** — open3d UVAtlas inside the reference TexturePipeline
   (fusion output is repaired to full UVAtlas cleanliness first)
6. **Bake** — MV-Adapter 6-view diffusion albedo + big-lama fill,
   PSHuman head-color composite, Embree hemispheric raycast AO,
   tangent-space normal bake (all atlases share one row convention)
7. **SkinTokens** — skeleton + skinning weights (+ rig hygiene report)
8. **GLB export** — PBR material, sidecar PNGs, meta.json

`--triposg-body` restores the legacy TripoSG + head-transplant arm.
Proportion validators print `[validate]` reports after reconstruction;
warnings (vestigial head, broken wingspan, asymmetry) surface in seconds
instead of after a full bake.

## Run

```powershell
$env:PYTHONPATH = "src"

# single run (T-pose reference input)
python -m avatar_pipeline.main run <input.png> <output_dir> --snapshot <out>\state.pkl

# from a candid photo (face or partial face+body)
python -m avatar_pipeline.main run <candid.png> <output_dir> --from-candid --snapshot <out>\state.pkl

# multiple runs: ONE resident pipeline serves the whole queue
python -m avatar_pipeline.main batch <jobs.txt>
```

Jobs file: one `name | image_path [| no-clothes]` per line (`#` comments).

`--from-candid` synthesizes the missing T-pose reference: IP-Adapter
FaceID PlusV2 (external/IP-Adapter, weights from h94/IP-Adapter-FaceID)
+ an OpenPose ControlNet conditioned on an anatomically-proportioned
synthetic T-pose skeleton. Best-of-N candidates are gated on strict-pose
scores (Sapiens keypoints: arm line, wingspan ratio, uprightness) and
FaceID identity, then the winner gets the true candid face transplanted
(HyperSwap) and restored (GFPGAN). A perspective-neutral
`face_portrait.png` (85mm framing — a candid's selfie-distance warp never
feeds geometry) is produced the same way for the portrait-head carve.
The identity phase also reads sex/age from the aligned face crop to set
the prompt subject. The SD15 stack loads, runs, and unloads before the
main pipeline's mmgp domain comes up.

`--no-clothes` drops the clothing fragment from the prompt and negates
all garment terms — an unobstructed body silhouette for anatomy
validation and reconstruction.

`--snapshot` caches the pre-rig state; stale caches (old texture schema,
missing pre-unwrap mesh, inward winding) upgrade themselves on load
without rerunning the generators.

## Feedback loop

```
scripts\compare_runs.py runA runB --candid <photo>   # A/B sheet + metrics
scripts\pose_smoke.py <run_dir>                      # LBS elbow bend render
scripts\render_glb.py <model.glb>                    # turntable views
scripts\download_checkpoints.py                      # fetch model weights
```

`compare_runs` scores mesh stats, anatomical proportions, silhouette IoU
against the run's own rembg alpha, FaceID identity vs the candid (faces
under 160px are zoom-re-embedded), and rig hygiene into `report.md`.

## Memory

All heavy torch models share one mmgp offload domain (8 GB-class GPUs work;
weights idle in host RAM and stream on demand). TripoSG's weights are not
loaded at all on the default path. Non-torch components take short-lived
explicit GPU residency. `nvdiffrast` JIT-compiles its CUDA plugin on first
use — ninja and MSVC must be reachable (see `requirements.txt` for the
torch-version-coupled wheels: pytorch3d, kaolin, torch_scatter, xformers).

## Tests

```
python -m pytest            # unit tests (fast; slow e2e excluded)
python -m pytest -m slow    # full end-to-end pipeline runs (~1h each)
```
