---
title: Image2Model
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.11.0"
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# Image2Model

Portrait-to-mesh pipeline on HuggingFace ZeroGPU.

Upload a photo → rigged, textured, animation-ready GLB in minutes.

**Pipeline stages**
1. Background removal — RMBG-2.0
2. 3D shape generation — TripoSG (diffusion SDF)
3. Multiview texturing — MV-Adapter + SDXL
4. Face enhancement — HyperSwap 1A 256 + RealESRGAN x4plus
5. Rigging — YOLO-pose → 3D joints → LBS weights
6. SKEL anatomy layer — anatomical bone mesh
7. MDM animation — text-to-motion
8. Surface enhancement — StableNormal normal maps + Depth-Anything displacement
