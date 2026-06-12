"""OpenCV-backed drop-in for mvadapter.utils.mesh_utils.cv_ops.

CV-CUDA ships no Windows wheels, but every operation the mvadapter UV code
needs from it is a direct GPU port of classic OpenCV: cvcuda.inpaint is
Telea inpainting (cv2.inpaint), cvcuda.morphology is cv2.erode/cv2.dilate.
This module mirrors the exact cv_ops contract on CPU and is registered in
sys.modules under the mvadapter name BEFORE their package imports it, so
uv_padding / poisson blending / SmartPainter run unmodified on Windows.

Cost: one GPU->CPU->GPU round trip per call on uv-sized images (~10ms class),
negligible next to the diffusion.
"""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np
import torch


def inpaint_cvc(
    image: torch.Tensor,
    mask: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    input_dtype = image.dtype
    input_device = image.device

    image = image.detach()
    mask = mask.detach()

    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)
    if mask.dtype != torch.uint8:
        mask = (mask * 255).to(torch.uint8)

    img_np = np.ascontiguousarray(image.cpu().numpy())
    mask_np = np.ascontiguousarray(mask.cpu().numpy())
    out_np = cv2.inpaint(img_np, mask_np, float(padding_size), cv2.INPAINT_TELEA)
    output = torch.from_numpy(out_np).to(input_device)

    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return output
    return output.to(dtype=input_dtype) / 255.0


def batch_inpaint_cvc(
    images: torch.Tensor,
    masks: torch.Tensor,
    padding_size: int,
    return_dtype: Optional[torch.dtype] = None,
):
    return torch.stack(
        [
            inpaint_cvc(image, mask, padding_size, return_dtype)
            for (image, mask) in zip(images, masks)
        ],
        axis=0,
    )


def _batch_morphology(masks: torch.Tensor, kernel_size: int, op, return_dtype):
    input_dtype = masks.dtype
    input_device = masks.device
    masks = masks.detach()
    if masks.dtype != torch.uint8:
        masks = (masks.float() * 255).to(torch.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = torch.stack(
        [
            torch.from_numpy(op(np.ascontiguousarray(m.cpu().numpy()), kernel))
            for m in masks
        ],
        axis=0,
    ).to(input_device)
    if return_dtype == torch.uint8 or input_dtype == torch.uint8:
        return out
    return (out > 0).to(dtype=input_dtype)


def batch_erode(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    return _batch_morphology(masks, kernel_size, cv2.erode, return_dtype)


def batch_dilate(
    masks: torch.Tensor, kernel_size: int, return_dtype: Optional[torch.dtype] = None
):
    return _batch_morphology(masks, kernel_size, cv2.dilate, return_dtype)


def register() -> None:
    """Install this module as mvadapter.utils.mesh_utils.cv_ops.

    Must run before anything imports the real module (which would fail on
    its `import cvcuda` line anyway on Windows).
    """
    sys.modules.setdefault(
        "mvadapter.utils.mesh_utils.cv_ops", sys.modules[__name__]
    )
