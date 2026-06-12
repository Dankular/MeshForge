"""PSHuman head-detail generator — real 7-view unclip diffusion + SMPL-X carve.

Runs PSHuman (external/PSHuman) in-process on the head crop derived from the
Sapiens segmentation. The diffusion components are built on CPU at load time
and registered into the pipeline's shared mmgp offload domain; PIXIE, the
person detector, and the carve optimizer take short-lived explicit GPU
residency inside generate() instead (they are not plain torch modules).

Two hard rules learned the expensive way:
  - never swap the unet's attention processors (enable_attention_slicing
    silently disables the custom multiview attention and the carve NaNs out);
    xformers is the model's intended path.
  - construct smplx/PIXIE under a CPU default-device context: mmgp's global
    cuda default-device mode breaks the numpy/tensor interop inside their
    constructors and lazy detector builds.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
PSHUMAN_DIR = _REPO_ROOT / "external" / "PSHuman"
PSHUMAN_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "pshuman"

_HEAD_LABELS = (1, 5)  # internal palette: head, hair


def _bootstrap_pshuman_imports() -> None:
    if str(PSHUMAN_DIR) not in sys.path:
        sys.path.insert(0, str(PSHUMAN_DIR))
    # PSHuman's utils/ is a namespace package and the vastai CLI installs a
    # regular top-level `utils` package that would win resolution.
    existing = sys.modules.get("utils")
    if existing is None or not getattr(existing, "__path__", [None])[0] == str(
        PSHUMAN_DIR / "utils"
    ):
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(PSHUMAN_DIR / "utils")]
        sys.modules["utils"] = utils_pkg


def extract_head_crop_rgba(
    processed: np.ndarray, seg_labels: np.ndarray, margin: float = 0.35
) -> np.ndarray:
    """Square RGBA crop of the head+hair region; alpha limited to the head."""
    head_mask = np.isin(seg_labels, _HEAD_LABELS)
    if head_mask.sum() < 100:
        raise RuntimeError(
            f"Segmentation found only {int(head_mask.sum())} head pixels; "
            "cannot build a head crop"
        )
    ys, xs = np.where(head_mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = head_mask.shape
    side = int(max(y1 - y0, x1 - x0) * (1.0 + margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2
    top, left = max(0, cy - half), max(0, cx - half)
    bottom, right = min(h, top + side), min(w, left + side)

    crop = processed[top:bottom, left:right]
    alpha = head_mask[top:bottom, left:right].astype(np.float32)
    rgba = np.dstack([crop[:, :, :3], np.minimum(crop[:, :, 3], alpha)])
    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8)


class PSHumanHeadGenerator:
    def __init__(self, num_inference_steps: int = 40, seed: int = 42) -> None:
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self._pipe = None

    def load_pretrained(self, checkpoint_dir: Path | None = None) -> None:
        _bootstrap_pshuman_imports()
        from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import (
            StableUnCLIPImg2ImgPipeline,
        )

        ckpt = checkpoint_dir or PSHUMAN_CHECKPOINT_DIR
        # CPU + fp16: the shared mmgp domain streams these to the GPU.
        self._pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            str(ckpt), torch_dtype=torch.float16
        )
        self._pipe.unet.enable_xformers_memory_efficient_attention()

    def heavy_components(self) -> dict[str, torch.nn.Module]:
        """Diffusion components for the shared mmgp domain."""
        if self._pipe is None:
            raise RuntimeError("PSHumanHeadGenerator is not loaded")
        return {
            "pshuman_unet": self._pipe.unet,
            "pshuman_vae": self._pipe.vae,
            "pshuman_image_encoder": self._pipe.image_encoder,
            "pshuman_text_encoder": self._pipe.text_encoder,
        }

    def generate(
        self,
        processed: np.ndarray,    # float32 (H, W, 4) RGBA in [0, 1]
        seg_labels: np.ndarray,   # (H, W) int32 internal palette
        work_dir: str,
    ) -> str:
        """Run PSHuman on the head crop. Returns the colored OBJ path."""
        crop = extract_head_crop_rgba(processed, seg_labels)
        return self._run_case(crop, "head", work_dir)

    def generate_full(
        self,
        processed: np.ndarray,    # float32 (H, W, 4) RGBA in [0, 1]
        work_dir: str,
    ) -> str:
        """Run PSHuman on the full figure (its native use: SMPL-X-guided
        single-image human reconstruction). Returns the colored OBJ path."""
        rgba = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
        return self._run_case(rgba, "body", work_dir)

    def generate_from_rgba(
        self,
        rgba: np.ndarray,         # uint8 (H, W, 4) RGBA
        work_dir: str,
        scene: str = "portrait",
    ) -> str:
        """Run PSHuman on an arbitrary RGBA crop (e.g. the FaceID-synthesized
        perspective-neutral portrait). Returns the colored OBJ path."""
        return self._run_case(rgba, scene, work_dir)

    def _run_case(
        self,
        rgba: np.ndarray,         # uint8 (H, W, 4) RGBA
        scene: str,
        work_dir: str,
    ) -> str:
        """7-view unclip diffusion + SMPL-X carve on one RGBA image."""
        if self._pipe is None:
            raise RuntimeError("PSHumanHeadGenerator is not loaded")
        _bootstrap_pshuman_imports()
        if torch.cuda.is_available():
            # Tiny buffer-only module used during image-embedding noising;
            # not in the mmgp domain, so pin it to the GPU explicitly.
            self._pipe.image_normalizer.to("cuda")
        import os

        from omegaconf import OmegaConf
        from PIL import Image

        # Absolute: everything below runs with CWD switched to the PSHuman
        # repo (its data paths are CWD-relative), so a relative work_dir
        # would silently re-root.
        work = Path(work_dir).resolve()
        img_dir = work / "input"
        img_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, "RGBA").save(img_dir / f"{scene}.png")

        # PSHuman's utils.misc registers OmegaConf resolvers at import; in
        # the integrated process SkinTokens' loader already registered some
        # of the same names (identical lambdas). Clear them so PSHuman's
        # import can re-register without colliding.
        for _name in ("calc_exp_lr_decay_rate", "add", "sub", "mul",
                      "div", "idiv", "basename"):
            OmegaConf.clear_resolver(_name)

        # PSHuman resolves smpl_related/ and mvdiffusion/data/ from CWD.
        prev_cwd = os.getcwd()
        os.chdir(PSHUMAN_DIR)
        try:
            from utils.misc import load_config

            cfg = load_config(
                str(PSHUMAN_DIR / "configs" / "inference-768-6view.yaml"),
                cli_args=[
                    f"pretrained_model_name_or_path={PSHUMAN_CHECKPOINT_DIR.as_posix()}",
                    f"validation_dataset.root_dir={img_dir.as_posix()}",
                    f"save_dir={work.as_posix()}",
                    "validation_dataset.crop_size=740",
                    "with_smpl=false",
                    "num_views=7",
                    "save_mode=rgb",
                    f"seed={self.seed}",
                    f"pipe_validation_kwargs.num_inference_steps={self.num_inference_steps}",
                    f"recon_opt.res_path={(work / 'recon').as_posix()}",
                ],
            )
            from inference import TestConfig, run_inference

            cfg = OmegaConf.merge(OmegaConf.structured(TestConfig), cfg)

            from mvdiffusion.data.single_image_dataset import SingleImageDataset

            dataset = SingleImageDataset(**cfg.validation_dataset)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.validation_batch_size,
                shuffle=False, num_workers=0,
            )

            # CPU default-device context: smplx/PIXIE constructors and the
            # lazily-built detector do numpy/tensor interop that breaks under
            # mmgp's global cuda default-device mode. Explicit .to(device)
            # calls inside still place the models on the GPU.
            with torch.device("cpu"):
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
                from reconstruct import ReMesh

                carving = ReMesh(cfg.recon_opt, econ_dataset=econdata)
                poses = {i: econdata[i] for i in range(len(dataset))}

            class _CachedPoses:
                def __init__(self, econ, cache):
                    self._econ, self._cache = econ, cache

                def __getitem__(self, i):
                    return self._cache[i]

                def __getattr__(self, name):
                    return getattr(self._econ, name)

            class _NativeDeviceCarving:
                """Run the carve with NO default-device mode active: it
                mixes numpy-derived CPU tensors, default-device creations
                (kornia kernels) and pytorch3d legacy sparse constructors,
                which misallocate under any active device mode. Restore
                mmgp's cuda default afterwards."""

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

            run_inference(
                dataloader,
                _CachedPoses(econdata, poses),
                self._pipe,
                _NativeDeviceCarving(carving),
                cfg,
                str(work),
            )
        finally:
            os.chdir(prev_cwd)

        hits = sorted((work / "recon").rglob("result_clr_scale*.obj"))
        if not hits:
            raise RuntimeError(
                f"PSHuman produced no colored mesh under {work / 'recon'}"
            )
        del econdata, carving
        torch.cuda.empty_cache()
        print(f"  PSHuman {scene} mesh: {hits[-1]}")
        return str(hits[-1])
