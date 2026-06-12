from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class SharedProfileReport:
    module_names: tuple[str, ...]
    cuda_bytes_before: int
    cuda_bytes_after: int


def _module_devices(module: torch.nn.Module) -> set[str]:
    devices = {tensor.device.type for tensor in module.parameters()}
    devices.update(tensor.device.type for tensor in module.buffers())
    return devices


def normalize_mmgp_module(module: torch.nn.Module) -> torch.nn.Module:
    """Match MMGP's one-child wrapper normalization before registration."""
    children = module._modules
    if children is None or len(children) != 1:
        return module
    child = children[next(iter(children))]
    if hasattr(child, "config") or hasattr(child, "base_model"):
        return child
    return module


def validate_cpu_constructed(modules: Mapping[str, torch.nn.Module]) -> None:
    if not modules:
        raise RuntimeError("No heavy modules were registered for MMGP")

    identities: dict[int, str] = {}
    for name, module in modules.items():
        prior = identities.setdefault(id(module), name)
        if prior != name:
            raise RuntimeError(
                f"Heavy module is registered twice: {prior!r} and {name!r}"
            )
    for name, module in modules.items():
        if "cuda" in _module_devices(module):
            raise RuntimeError(
                f"Heavy module {name!r} reached CUDA before MMGP profiling"
            )


def validate_shared_profile(
    manager,
    modules: Mapping[str, torch.nn.Module],
) -> None:
    if manager is None:
        raise RuntimeError("MMGP shared profile has not been created")
    if set(manager.models) != set(modules):
        raise RuntimeError("MMGP model registry does not match the pipeline registry")

    for name, module in modules.items():
        if manager.models[name] is not module:
            raise RuntimeError(f"MMGP owns the wrong module object for {name!r}")
        if not hasattr(module, "_hf_hook"):
            raise RuntimeError(f"MMGP hook is missing from {name!r}")
        if getattr(module, "_force_device", None) != "cuda":
            raise RuntimeError(f"MMGP force-device marker is missing from {name!r}")
        if "cuda" in _module_devices(module):
            raise RuntimeError(f"Heavy module {name!r} is not idle on CPU")


def create_shared_profile(
    modules: Mapping[str, torch.nn.Module],
) -> tuple[object, SharedProfileReport]:
    from mmgp import offload, profile_type

    validate_cpu_constructed(modules)
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    manager = offload.profile(
        dict(modules),
        profile_no=profile_type.LowRAM_LowVRAM,
        convertWeightsFloatTo=torch.float16,
    )
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    validate_shared_profile(manager, modules)
    return manager, SharedProfileReport(
        module_names=tuple(sorted(modules)),
        cuda_bytes_before=before,
        cuda_bytes_after=after,
    )
