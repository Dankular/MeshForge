# Avatar Pipeline Engineering Scope

## Core Philosophy

This project is a semantic human reconstruction engine.

Reference systems — vendored under `external/` and run **in-process** with
their real published weights:

- PSHuman   — human-specific reconstruction (7-view unclip diffusion +
              SMPL-X carve via PIXIE) — body + head base
- Sapiens   — semantic backbone (seg / pose / depth / normals) — drives
              masks and validation, not just conditioning
- TripoSG   — supplemental geometry only (hair mass, shoes, accessories,
              loose clothing shells) — NOT the body base
- MV-Adapter — multiview texture generation + UV projection
- SkinTokens — joint skeleton + skinning-weight generation
- IP-Adapter (FaceID PlusV2) — identity-preserving T-pose reference
              synthesis from candid photos (`--from-candid`)
- rembg     — background removal

**Why TripoSG was demoted:** it is a general SDF-diffusion object
reconstructor. It produces visually good shapes but does not respect human
proportions, joint placement, or deformation topology. A pretty mesh is not
a riggable mesh. PSHuman is human-specific and already carries an SMPL-X
prior internally; the body must come from a human-aware source.

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
  (`--from-candid` runs and unloads BEFORE the domain is built.)

## Target Architecture

INPUTS
  body_source_image: T-pose full body (or candid photo via --from-candid →
                     IP-Adapter FaceID PlusV2 + OpenPose ControlNet)
  face_reference_images: candid / close-up, optional

STAGE A — image understanding                              [implemented]
  rembg → Sapiens seg / pose / depth / normals / pointmap
  Body-part masks (face, hair, hands, feet) feed every later stage.

STAGE B — human scaffold                                   [partially: buried]
  SMPL-X / PIXIE fit → canonical skeleton landmarks, body scale,
  joint map, head-neck alignment, limb proportions.
  The scaffold is alignment + validation truth, not the visible mesh.
  (PIXIE already runs inside PSHuman's carve — it must be promoted to a
  first-class stage whose joints/scale outputs persist in the snapshot.)

STAGE C — reconstruction                                   [inverted today]
  PSHuman body + head base (full figure, not just the head crop).
  TripoSG only for supplemental shells (hair mass, shoes, accessories),
  carved by Sapiens masks.
  Face detail: FaceID reference + PIXIE/DECA geometry prior, refined onto
  the SMPL-X-aligned head.
  Validation: reconstruction depth vs Sapiens depth/pointmap consistency.

STAGE D — fusion                                           [head-only today]
  Align all meshes to the scaffold → fuse head/body → repair neck seam
  (weld / proximity-bind) → separate hair/accessory shells →
  remove self-intersections → outward winding → normalize units/orientation.

STAGE E — production mesh                                  [minimal today]
  Cleanup: floaters, holes, noise smoothing with detail masks preserved.
  Keep deformation-critical loops: shoulders, elbows, wrists, hips, knees,
  ankles, neck (jaw/eyes/mouth when face rigging lands).
  Optional quad remesh for deformation zones (deferred — heavy).
  UV unwrap (open3d UVAtlas / xatlas) with material regions seeded by
  Sapiens masks: skin / face / hair / eyes; high texel density on
  face and hands.

STAGE F — baking                                           [implemented]
  MV-Adapter 6-view diffusion albedo + big-lama fill for uncovered texels,
  Embree hemispheric raycast AO, tangent-space normal bake.
  Optional later: cavity/curvature, roughness/specular estimation.

STAGE G — rig                                              [implemented, unvalidated]
  SkinTokens skeleton + top-4 skinning weights on the FINAL cleaned mesh.
  Validation pass: SkinTokens joints vs STAGE B scaffold joints (loud
  failure on gross mismatch), weight sanity (normalization, max-influence,
  detached-island coverage), corrective smoothing.

STAGE H — export                                           [implemented]
  GLB (trimesh PBR) + sidecar PNGs + meta.json.

## Migration Order (each step ships independently, snapshot-compatible)

1. **Expose the scaffold.** Lift the PIXIE/SMPL-X fit out of PSHuman's
   carve into a STAGE B output (joints, scale, landmarks) persisted in the
   snapshot. No new models — re-plumbing only.
2. **Rig validation.** Cheap, immediate win: compare SkinTokens joints
   against the scaffold joints; validate weights. Catches bad rigs today,
   before any reconstruction change.
3. **PSHuman full-figure.** DONE — default path. A/B-verified against the
   TripoSG arm: silhouette IoU 0.973 vs 0.716, anatomical head fraction,
   5/27 vs 32/52 asymmetric joints. The candid flow also carves the head
   from a perspective-neutral FaceID portrait (HyperSwap + GFPGAN
   identity, PSHuman multiview depth) transplanted onto the body.
   Depth/pointmap validation still TODO.
4. **TripoSG demotion.** Sapiens-mask-carved supplemental shells (hair,
   shoes, accessories) aligned to the scaffold; drop it from the body path.
5. **Fusion + cleanup hardening.** Neck-seam weld, self-intersection
   removal, floater/hole cleanup (pymeshlab), deformation-loop checks —
   all BEFORE the unwrap, so UVs and bakes see final geometry.
6. **Material-region UVs.** Seed charts from Sapiens masks; boost
   face/hand texel density.

Ordering rationale: the bake→rig order is already correct (fusion →
cleanup → UV → bake → SkinTokens); nothing downstream of STAGE E moves.
Steps 1–2 de-risk steps 3–4 by giving every reconstruction change a
quantitative joint/depth check.

## Still TODO (unchanged)

- Canonical transfer: runtime assets sharing topology / UVs / skeletons /
  material slots across avatars (the canonical template does not exist yet)
- PSHuman normal-map reprojection for the head region of the atlas
- CV-CUDA-gated MV-Adapter quality passes (poisson_reprojection,
  SmartPainter view inpaint) — no Windows wheels exist

## Snapshots

`--snapshot` caches the pre-rig state (stages 1–5). `TEXTURE_SCHEMA_VERSION`
in `runtime/contracts.py` gates cached textures: bump it whenever bake
semantics change and stale snapshots rebake automatically without rerunning
the generators.

## Major Systems

preprocess/   rembg, tpose_reference (IP-Adapter FaceID --from-candid)
sapiens/      seg, pose, depth, normals, pointmap (+ shared TorchScript loader)
generators/   triposg_body, pshuman_head
fusion/       head_fusion (PSHuman pre-rig transplant)
baking/       mv_adapter_texture (color + normal + vertex attrs), ao_bake
rigging/      skintokens
export/       glb
runtime/      contracts, memory (mmgp domain), cache (artifact keys)
