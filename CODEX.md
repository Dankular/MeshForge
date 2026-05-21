# MeshForge — Project Scope & Requirements for Codex

## What This Project Is

Single-image-to-rigged-avatar pipeline. User uploads a portrait photo; the app outputs a fully textured, rigged 3D GLB avatar with a PSHuman-quality face mesh.

**Pipeline stages (in order, single pass):**
1. Background removal (RMBG-1.4)
2. Shape generation (TripoSG)
3. Texture generation (MV-Adapter SDXL)
4. Rigging (UniRig)
5. HD face replacement (PSHuman → face_transplant)

---

## Repo Structure

```
MeshForge/
├── app.py                        # v1 working Gradio app (IONOS VPS, Linux)
├── requirements.txt              # VPS Python deps (PyTorch cu128, Python 3.10)
├── pipeline/
│   ├── face_transplant.py        # Stitches PSHuman face into rigged body mesh
│   ├── pshuman_client.py         # Runs PSHuman inference (direct subprocess)
│   ├── face_enhance.py
│   └── ...
├── patches/
│   └── TripoSG_image_process.py  # Patched prepare_image with RMBG-1.4 support
├── scripts/
│   ├── setup_unirig.sh           # UniRig conda env setup
│   ├── inference_triposg.py      # TripoSG pipeline wrapper
│   └── texture_i2tex.py          # MV-Adapter texture subprocess entry point
├── external/
│   ├── TripoSG/                  # Cloned TripoSG repo
│   └── MV-Adapter/               # Cloned MV-Adapter repo
├── v2/                           # CLI wrapper (shape stage works, texture broken locally)
│   ├── cli.py
│   ├── config.py
│   ├── workflow.py
│   └── stages.py
└── ZeroGPU/
    ├── app.py                    # HuggingFace Space app (BROKEN — needs fixing)
    └── requirements_space.txt    # Space requirements (BROKEN — needs fixing)
```

---

## Working Reference: VAST-AI/TripoSG Space

**This is the proven base. Start here.**

- Space: https://huggingface.co/spaces/VAST-AI/TripoSG
- Their `app.py`: https://huggingface.co/spaces/VAST-AI/TripoSG/resolve/main/app.py
- Their `requirements.txt`: (exact content below)

```
torchvision==0.20.1
diffusers
transformers==4.49.0
einops
huggingface_hub
opencv-python
trimesh==4.5.3
omegaconf
scikit-image
numpy
peft
scipy==1.11.4
jaxtyping
typeguard
pymeshlab==2022.2.post4
open3d
timm
kornia
ninja
https://huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl?download=true
cvcuda_cu12
gltflib
https://huggingface.co/spaces/VAST-AI/TripoSG/resolve/main/diso-0.1.4-cp310-cp310-linux_x86_64.whl?download=true
```

**Critical notes from their setup:**
- `scipy==1.11.4` pins numpy to <1.28 — this is what keeps trimesh working on ZeroGPU
- `cvcuda_cu12` is Linux-only but works on ZeroGPU (don't replace with OpenCV)
- `nvdiffrast` and `diso` are pre-built Linux wheels from their space — use exactly these URLs
- `spandrel==0.4.1` must be installed via `subprocess.run("pip install spandrel==0.4.1 --no-deps", ...)` at startup
- They clone TripoSG and MV-Adapter at runtime (not bundled)
- The `texture.cpython-310-x86_64-linux-gnu.so` file in their space root is a compiled C extension for the texture pipeline — **download it from their space and include it**

---

## HuggingFace Space Target

- **Space**: `Daankular/MeshForge`
- **Token**: `[set via HF_TOKEN env var]`
- **SDK**: gradio (ZeroGPU)
- **Hardware**: zero-a10g

### README.md header (must be exactly this):
```yaml
---
title: MeshForge
emoji: 🧊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: ZeroGPU/app.py
pinned: false
license: mit
---
```

---

## What Needs to Be Built

### ZeroGPU/app.py

Single-pass pipeline — one image in, one rigged GLB out. No tabs, no separate steps.

**Base**: Start from VAST-AI/TripoSG `app.py` verbatim (it handles shape + texture correctly).

**Add after texture step (inside `@spaces.GPU`):**

1. **UniRig rigging**
   - Clone: `https://github.com/VAST-AI-Research/UniRig.git`
   - Run three bash scripts sequentially:
     - `launch/inference/generate_skeleton.sh --input <textured.glb> --output skeleton.fbx`
     - `launch/inference/generate_skin.sh --input skeleton.fbx --output skin.fbx`
     - `launch/inference/merge.sh --source skin.fbx --target <textured.glb> --output rigged.glb`
   - UniRig ignores `--output` and writes to `/tmp/rig_out/rigged.glb` — copy from there if needed
   - If UniRig fails, fall through to next step with the textured mesh

2. **PSHuman face replacement**
   - Clone: `https://github.com/pengHTYX/PSHuman.git`
   - Model: `pengHTYX/PSHuman_Unclip_768_6views` (snapshot_download)
   - Run RMBG mask on portrait → save as RGBA PNG
   - Run `inference.py --config configs/inference-768-6view.yaml --data_dir <work> --case_name face --output_dir <out> --pretrained_model_name_or_path <ckpt>`
   - Find the `.obj` output file
   - Call `pipeline/face_transplant.transplant_face(body_glb_path, pshuman_mesh_path, output_path, weight_threshold=0.5, retract_mm=2.0)`
   - The `pipeline/` directory is in the repo root — add it to sys.path
   - If PSHuman fails, fall through with rigged mesh

### UI

```
[ Portrait Photo input ]
[ Seed slider + Randomize checkbox ]
[ Generate Avatar button ]
─────────────────────────
[ 3D Model viewer (output) ]
[ Download GLB button ]
```

Single button, single output. No intermediate previews needed.

### demo.launch()

Must use `spaces.zero.gradio` launcher (the `spaces` package wraps this). Use:
```python
demo.launch()
```
Do NOT add `share=True` — the `spaces` package handles this automatically on ZeroGPU.
If `ValueError: When localhost is not accessible` appears, the `spaces` package version is wrong.

---

## Known Issues to Avoid

1. **Do not replace `cvcuda_cu12` with OpenCV** — cvcuda works on ZeroGPU Linux, OpenCV doesn't have the same GPU-accelerated API
2. **Do not patch numpy** — `scipy==1.11.4` in requirements.txt is what keeps numpy at 1.x during Docker build
3. **Do not add `--no-build-isolation` to requirements.txt** — newer pip rejects it as invalid
4. **UniRig requires `bpy==4.2`** in its requirements.txt which is not on PyPI — skip it when installing UniRig deps: `pip install -r requirements.txt --ignore-requires-python` or filter out `bpy` lines
5. **`texture.cpython-310-x86_64-linux-gnu.so`** — this compiled file from the VAST-AI space is needed for `from texture import TexturePipeline`. Download it from `https://huggingface.co/spaces/VAST-AI/TripoSG/resolve/main/texture.cpython-310-x86_64-linux-gnu.so` and place it in `ZeroGPU/`
6. **lora_scale in MV-Adapter** — `inference_ig2mv_sdxl.py` has `lora_scale=1.0` as default which triggers LoRA code on the text encoder even when no LoRA is loaded, causing `AttributeError: 'CLIPTextModel' has no attribute 'text_model'` with newer transformers. Set `lora_scale=None` as default and guard: `cross_attention_kwargs={"scale": lora_scale} if lora_scale is not None else None`

---

## v1 Working App (Reference)

`app.py` in the repo root is the working v1 app running on an IONOS VPS (Linux, CUDA 12.8, Python 3.10 miniconda env `triposg`). It has the full pipeline working including UniRig and PSHuman. Read it to understand how each stage works before implementing anything.

Key functions:
- `generate_shape()` — TripoSG shape generation with RMBG
- `apply_texture()` — MV-Adapter texture subprocess
- `gradio_rig()` — UniRig + PSHuman face transplant (lines 1266–1400)
- `_run_unirig()` — UniRig subprocess wrapper (lines 1210–1263)
- `_portrait_to_rgba()` — RMBG mask for PSHuman input

---

## HuggingFace Token

Set as environment variable before running any HF commands:

```bash
export HF_TOKEN=<token>
```

Use `HF_TOKEN` in code:
```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
```

---

## Space Management

**Space**: `Daankular/MeshForge`

### Watch live logs (run in two terminals)

```bash
# Build logs
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Daankular/MeshForge/logs/build"

# Runtime logs
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Daankular/MeshForge/logs/run"
```

### Check Space status

```bash
curl -s \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Daankular/MeshForge" \
  | python -c "import sys,json; r=json.load(sys.stdin).get('runtime',{}); print(r.get('stage')); print(r.get('errorMessage','')[:500])"
```

### Restart Space

```bash
curl -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Daankular/MeshForge/restart"
```

### Upload a file

```bash
python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='local/path/to/file.py',
    path_in_repo='remote/path/in/space.py',
    repo_id='Daankular/MeshForge',
    repo_type='space',
)
"
```

### Upload a whole folder

```bash
python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path='ZeroGPU',
    repo_id='Daankular/MeshForge',
    repo_type='space',
    path_in_repo='ZeroGPU',
    ignore_patterns=['__pycache__', '*.pyc'],
)
"
```

### Stages meaning

| Stage | Meaning |
|-------|---------|
| `BUILDING` | Docker build running |
| `APP_STARTING` | Build done, app starting up |
| `RUNNING` | App is up — **verify logs too, it may still be crashing** |
| `RUNTIME_ERROR` | App crashed |
| `BUILD_ERROR` | Docker build failed |
| `CONFIG_ERROR` | README.md YAML header invalid |

> **Important**: `RUNNING` does not mean the app works. Always stream the run logs and look for tracebacks.
