# MeshForge

Portrait-to-mesh pipeline. Upload a photo, get a rigged, textured, animation-ready GLB.

---

## What it does

1. **Background removal** — RMBG-2.0 strips the background from the input portrait
2. **3D shape generation** — TripoSG reconstructs a high-quality mesh from the cleaned image
3. **Multiview texturing** — MV-Adapter generates 6 consistent views (SDXL or SD2.1), baked into a UV texture
4. **Face enhancement** — HyperSwap 1A 256 transfers reference identity; RealESRGAN x4plus sharpens with unsharp mask
5. **Rigging** — YOLO-pose detects 17 body keypoints → unprojected to 3D → proximity LBS weights → skinned GLB
6. **SKEL anatomy layer** — SKEL BSM bone mesh optionally exported alongside rigged surface
7. **MDM animation** — text-to-motion via MDM (HumanML3D) baked into animated GLB
8. **Surface enhancement** — StableNormal (yoso-normal-v1-5) bakes normal maps; Depth-Anything V2 bakes displacement

---

## Technology Stack

| Stage | Model / Library | Notes |
|---|---|---|
| Background removal | [RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0) | BiRefNet-based, ~4GB VRAM |
| 3D reconstruction | [TripoSG](https://github.com/VAST-AI-Research/TripoSG) | Diffusion-based SDF, ~7.5GB weights |
| Multiview generation | [MV-Adapter](https://github.com/huanngzh/MV-Adapter) + SDXL / SD2.1 | 6-view consistent texturing |
| Face swap | [HyperSwap 1A 256](https://github.com/...) | ONNX, 256×256 aligned crop, identity L2-normalised |
| Face enhancement | [RealESRGAN x4plus](https://github.com/xinntao/Real-ESRGAN) | Full float32, tile=0, unsharp mask post-pass |
| Pose estimation | [YOLOv8x-pose](https://github.com/ultralytics/ultralytics) | 17-keypoint COCO body |
| Skinning | Custom LBS | Proximity-weighted, SMPL-24 joint layout |
| Bone mesh | [SKEL BSM](https://github.com/MarilynKeller/SKEL) | Anatomical inner skeleton, optional export |
| Animation | [MDM](https://github.com/GuyTevet/motion-diffusion-model) | HumanML3D, text-to-motion, patched for synthetic SMPL |
| Normal maps | [StableNormal yoso-normal-v1-5](https://huggingface.co/Stable-X/yoso-normal-v1-5) | 2.45GB, Apache 2.0 |
| Depth / displacement | [Depth-Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | 1.25GB, baked as occlusionTexture |
| UI | [Gradio 6](https://github.com/gradio-app/gradio) | Multi-tab, GLB preview, streaming status |
| GLB I/O | [pygltflib](https://github.com/KhronosGroup/glTF) | Direct binary accessor read/write |

---

## Pipeline Files

```
app.py                        # Gradio UI — all tabs wired here
pipeline/
  rig_yolo.py                 # YOLO pose → 3D joints → LBS skinning → GLB export
  rig_stage.py                # 4D-Humans HMR2.0 → SMPL shape → weight transfer
  face_enhance.py             # HyperSwap + RealESRGAN multiview enhancement
  enhance_surface.py          # StableNormal + Depth-Anything → PBR map baking
  tpose_smpl.py               # SMPL T-pose proxy skeleton generation
scripts/
  deploy.sh                   # Full instance bootstrap (conda env, weights, deps)
```

---

## Instance Requirements

- **GPU**: RTX 3090 24GB (minimum ~20GB VRAM for full pipeline)
- **RAM**: 64GB recommended
- **Disk**: 80GB+ (model weights alone ~25GB)
- **CUDA**: 12.8+
- **Python**: 3.10 (conda env `triposg`)

### Key model paths (on instance)
```
/root/.cache/huggingface/hub/   # TripoSG, MV-Adapter, RMBG-2.0, SDXL
/root/models/stable-normal/     # StableNormal yoso-normal-v1-5
/root/models/depth-anything-v2/ # Depth-Anything V2 Large
/root/body_models/skel/         # SKEL BSM weights
/root/MDM/                      # MDM checkpoint + HumanML3D
/root/MV-Adapter/checkpoints/   # HyperSwap, RealESRGAN, InsightFace
```

> TripoSG and MV-Adapter caches are symlinked to `/dev/shm` for RAM-speed loading.

---

## Tabs

| Tab | Function |
|---|---|
| **Generate** | Shape → Texture → optional face enhance → GLB download |
| **Rig & Export** | Step 1: SKEL anatomy layer · Step 2: Rig + MDM animation + FBX |
| **Enhancement** | StableNormal normal map · Depth-Anything displacement · sliders + preview |
| **Settings** | VRAM management — preload / unload individual models, live GPU status |

---

## Gradio API Endpoints

| Endpoint | Description |
|---|---|
| `/generate_shape` | TripoSG shape from portrait |
| `/apply_texture` | MV-Adapter texture bake |
| `/rig_mesh` | YOLO rig + optional MDM animation |
| `/run_full_pipeline` | Shape + texture + rig in one call |
| `/preview_rembg` | Background removal preview |
| `/render_last` | Render cached GLB to 5 views |

---

## Notes

- `GRADIO_CDN_BACKEND_ENABLED=False` — assets served locally, not from S3 CDN
- `image_process.py` patched: resize before squeeze to fix `ValueError: spatial dimensions of [1024]`
- MDM patched for synthetic SMPL proxy (gated SMPL weights unavailable)
- MV-Adapter `fish_speech` UnboundLocalError scoping bug patched in `reference_loader.py`
