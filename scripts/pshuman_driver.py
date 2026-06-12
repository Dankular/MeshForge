"""Run PSHuman's real multiview diffusion + SMPL-X carve on one RGBA image.

This is inference.py's main() orchestrated for a single case. The unet runs
with its real xformers multiview attention processors (mandatory: generic
processor swaps like enable_attention_slicing silently disable the
cross-view attention — the kwargs get ignored — and the carve NaNs out).
Memory is managed by the same shared-mmgp pattern the pipeline uses: the
diffusion components are built on CPU and streamed to the GPU on demand.

Outputs PSHuman's colored mesh: <save_dir>/recon/<scene>/result_clr_scale4_<scene>.obj
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
PSHUMAN_DIR = REPO_ROOT / "external" / "PSHuman"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "pshuman"

sys.path.insert(0, str(PSHUMAN_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

# PSHuman's utils/ has no __init__.py (namespace package) and the vastai CLI
# installs a regular top-level `utils` package, which wins resolution. Bind
# PSHuman's directory explicitly so `from utils.smpl_util import ...` inside
# the repo resolves correctly.
import types

_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = [str(PSHUMAN_DIR / "utils")]
sys.modules["utils"] = _utils_pkg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGBA image (alpha = mask)")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    import os
    os.chdir(PSHUMAN_DIR)  # smpl_related/, mvdiffusion/data paths are CWD-relative

    save_dir = Path(args.save_dir).resolve()
    img_dir = save_dir / "input"
    img_dir.mkdir(parents=True, exist_ok=True)
    scene = Path(args.image).stem.replace(" ", "_")
    shutil.copy(args.image, img_dir / f"{scene}.png")

    from utils.misc import load_config

    cfg = load_config(
        str(PSHUMAN_DIR / "configs" / "inference-768-6view.yaml"),
        cli_args=[
            f"pretrained_model_name_or_path={CHECKPOINT_DIR.as_posix()}",
            f"validation_dataset.root_dir={img_dir.as_posix()}",
            f"save_dir={save_dir.as_posix()}",
            "validation_dataset.crop_size=740",
            "with_smpl=false",
            "num_views=7",
            "save_mode=rgb",
            f"seed={args.seed}",
            f"pipe_validation_kwargs.num_inference_steps={args.steps}",
            f"recon_opt.res_path={(save_dir / 'recon').as_posix()}",
        ],
    )

    from inference import TestConfig, run_inference
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    from accelerate.utils import set_seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    print("[driver] loading PSHuman unclip pipeline (fp16)...")
    t0 = time.time()
    from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import (
        StableUnCLIPImg2ImgPipeline,
    )
    # Built on CPU; mmgp streams weights to the GPU on demand — the same
    # memory-management domain pattern the pipeline uses for every heavy model.
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path, torch_dtype=torch.float16
    )
    # The model's intended attention path: custom xformers multiview
    # processors (same call inference.py's load_pshuman_pipeline makes).
    pipeline.unet.enable_xformers_memory_efficient_attention()
    print(f"[driver] pipeline ready in {time.time()-t0:.0f}s")

    from mvdiffusion.data.single_image_dataset import SingleImageDataset
    dataset = SingleImageDataset(**cfg.validation_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.validation_batch_size, shuffle=False,
        num_workers=0,
    )

    print("[driver] building SMPLDataset (PIXIE)...")
    from econdataset import SMPLDataset
    econdata = SMPLDataset(
        {
            "image_dir": str(img_dir),
            "seg_dir": None,
            "colab": False,
            "has_det": True,
            "hps_type": "pixie",
        },
        device="cuda",
    )

    print("[driver] building ReMesh carving...")
    from reconstruct import ReMesh
    carving = ReMesh(cfg.recon_opt, econ_dataset=econdata)

    # Precompute the per-case SMPL-X pose data NOW, before mmgp installs its
    # global cuda default-device mode: SMPLDataset.__getitem__ lazily builds
    # a torchvision Mask-RCNN detector whose weights would land on cuda while
    # its from_numpy inputs stay CPU. The pose data is independent of the
    # diffusion, so caching it up front is exact, not an approximation.
    print("[driver] precomputing SMPL-X poses (PIXIE + detection)...")

    class _CachedPoses:
        def __init__(self, econ, count):
            self._econ = econ
            self._cache = {i: econ[i] for i in range(count)}

        def __getitem__(self, i):
            return self._cache[i]

        def __getattr__(self, name):
            return getattr(self._econ, name)

    econdata = _CachedPoses(econdata, len(dataset))

    # mmgp LAST, after all construction — its profile installs a global
    # cuda default-device mode that breaks numpy/tensor interop inside
    # constructors like smplx (np.concatenate over fresh tensors lands on
    # cuda). Same ordering rule as AvatarPipeline._load_checkpoints, where
    # _setup_offload() is the final step.
    from avatar_pipeline.runtime.memory import (
        create_shared_profile,
        normalize_mmgp_module,
    )

    _, report = create_shared_profile(
        {
            name: normalize_mmgp_module(module)
            for name, module in {
                "pshuman_unet": pipeline.unet,
                "pshuman_vae": pipeline.vae,
                "pshuman_image_encoder": pipeline.image_encoder,
                "pshuman_text_encoder": pipeline.text_encoder,
            }.items()
        }
    )
    print(f"[driver] mmgp domain over {report.module_names}")
    # Tiny buffer-only module used during image-embedding noising; not worth
    # streaming — pin it on the GPU so it matches the embedding device.
    if torch.cuda.is_available():
        pipeline.image_normalizer.to("cuda")

    # The carve was written with NO global default device: it mixes
    # numpy-derived CPU tensors, default-device creations (kornia kernels),
    # and pytorch3d's legacy sparse constructors — the latter misallocate
    # under ANY active device mode (cuda or cpu). Disable the mode entirely
    # around the carve to restore PSHuman's exact native semantics, then
    # re-enable mmgp's cuda default for anything after.
    class _NativeDeviceCarving:
        def __init__(self, inner):
            self._inner = inner

        def optimize_case(self, *args, **kwargs):
            torch.set_default_device(None)
            try:
                return self._inner.optimize_case(*args, **kwargs)
            finally:
                torch.set_default_device("cuda")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    print("[driver] running inference + reconstruction...")
    run_inference(
        dataloader, econdata, pipeline, _NativeDeviceCarving(carving), cfg, str(save_dir)
    )

    hits = sorted((save_dir / "recon").rglob("result_clr_scale*.obj"))
    if not hits:
        raise FileNotFoundError(f"No colored OBJ produced under {save_dir / 'recon'}")
    print(f"[driver] DONE: {hits[-1]}")


if __name__ == "__main__":
    main()
