from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import urllib.request


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    url: str
    relative_path: str
    sha256: str | None = None


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        name="triposr_model",
        url="https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
        relative_path="triposr/model.ckpt",
    ),
    CheckpointSpec(
        name="pshuman_unet",
        url="https://huggingface.co/pengHTYX/PSHuman_Unclip_768_6views/resolve/main/unet/diffusion_pytorch_model.safetensors",
        relative_path="pshuman/unet/diffusion_pytorch_model.safetensors",
    ),
    CheckpointSpec(
        name="pshuman_vae",
        url="https://huggingface.co/pengHTYX/PSHuman_Unclip_768_6views/resolve/main/vae/diffusion_pytorch_model.safetensors",
        relative_path="pshuman/vae/diffusion_pytorch_model.safetensors",
    ),
    CheckpointSpec(
        name="skintokens_tokenrig",
        url="https://huggingface.co/VAST-AI/SkinTokens/resolve/main/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        relative_path="skintokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
    ),
    CheckpointSpec(
        name="skintokens_vae",
        url="https://huggingface.co/VAST-AI/SkinTokens/resolve/main/experiments/skin_vae_2_10_32768/last.ckpt",
        relative_path="skintokens/experiments/skin_vae_2_10_32768/last.ckpt",
    ),
    CheckpointSpec(
        name="skintokens_qwen_config",
        url="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
        relative_path="skintokens/models/Qwen3-0.6B/config.json",
    ),
    CheckpointSpec(
        name="ip_adapter_faceid_plusv2",
        url="https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
        relative_path="ip_adapter/ip-adapter-faceid-plusv2_sd15.bin",
    ),
    CheckpointSpec(
        name="sapiens2_seg_1b",
        url="https://huggingface.co/facebook/sapiens2-seg-1b/resolve/main/model.safetensors",
        relative_path="sapiens2/seg/model.safetensors",
    ),
    CheckpointSpec(
        name="sapiens2_normal_1b",
        url="https://huggingface.co/facebook/sapiens2-normal-1b/resolve/main/model.safetensors",
        relative_path="sapiens2/normal/model.safetensors",
    ),
    CheckpointSpec(
        name="sapiens2_pointmap_1b",
        url="https://huggingface.co/facebook/sapiens2-pointmap-1b/resolve/main/model.safetensors",
        relative_path="sapiens2/pointmap/model.safetensors",
    ),
    CheckpointSpec(
        name="sapiens2_matting_1b",
        url="https://huggingface.co/facebook/sapiens2-matting-1b/resolve/main/model.safetensors",
        relative_path="sapiens2/matting/model.safetensors",
    ),
)


def resolve_dynamic_checkpoint_urls() -> dict[str, str]:
    mapping: dict[str, tuple[str, str]] = {
        "triposr_model": ("stabilityai/TripoSR", "model.ckpt"),
        "pshuman_unet": (
            "pengHTYX/PSHuman_Unclip_768_6views",
            "unet/diffusion_pytorch_model.safetensors",
        ),
        "pshuman_vae": (
            "pengHTYX/PSHuman_Unclip_768_6views",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        "skintokens_tokenrig": (
            "VAST-AI/SkinTokens",
            "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        ),
        "skintokens_vae": (
            "VAST-AI/SkinTokens",
            "experiments/skin_vae_2_10_32768/last.ckpt",
        ),
        "skintokens_qwen_config": (
            "Qwen/Qwen3-0.6B",
            "config.json",
        ),
        "ip_adapter_faceid_plusv2": (
            "h94/IP-Adapter-FaceID",
            "ip-adapter-faceid-plusv2_sd15.bin",
        ),
        "sapiens2_seg_1b": ("facebook/sapiens2-seg-1b", "model.safetensors"),
        "sapiens2_normal_1b": ("facebook/sapiens2-normal-1b", "model.safetensors"),
        "sapiens2_pointmap_1b": (
            "facebook/sapiens2-pointmap-1b",
            "model.safetensors",
        ),
        "sapiens2_matting_1b": (
            "facebook/sapiens2-matting-1b",
            "model.safetensors",
        ),
    }
    out: dict[str, str] = {}
    for name, (repo, hint) in mapping.items():
        url = resolve_hf_file_url(repo, hint)
        if url is not None:
            out[name] = url
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(spec: CheckpointSpec, root: Path, force: bool = False) -> Path:
    dst = root / spec.relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    cleanup_incomplete_files(dst.parent)
    if dst.exists() and not force:
        if spec.sha256 is None or _sha256(dst) == spec.sha256:
            return dst
    tmp = dst.with_suffix(dst.suffix + ".incomplete")
    tmp.unlink(missing_ok=True)
    with urllib.request.urlopen(spec.url) as response, tmp.open("wb") as out:
        while True:
            data = response.read(1024 * 1024)
            if not data:
                break
            out.write(data)
    if spec.sha256 is not None and _sha256(tmp) != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {spec.name}")
    tmp.replace(dst)
    return dst


def download_all(
    root: Path, names: Iterable[str] | None = None, force: bool = False
) -> dict[str, Path]:
    selected = set(names) if names is not None else None
    out: dict[str, Path] = {}
    dynamic_urls = resolve_dynamic_checkpoint_urls()
    for spec in CHECKPOINTS:
        if selected is not None and spec.name not in selected:
            continue
        effective = spec
        if spec.name in dynamic_urls:
            effective = CheckpointSpec(
                name=spec.name,
                url=dynamic_urls[spec.name],
                relative_path=spec.relative_path,
                sha256=spec.sha256,
            )
        out[spec.name] = download_checkpoint(effective, root=root, force=force)
    return out


def cleanup_incomplete_files(root: Path) -> list[Path]:
    removed: list[Path] = []
    if not root.exists():
        return removed
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".incomplete", ".part"}:
            p.unlink(missing_ok=True)
            removed.append(p)
    return removed


def resolve_hf_file_url(repo_id: str, filename_hint: str) -> str | None:
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    with urllib.request.urlopen(api_url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    siblings = payload.get("siblings", [])
    files = [item.get("rfilename", "") for item in siblings]
    if filename_hint in files:
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename_hint}"
    for f in files:
        if f.endswith(filename_hint):
            return f"https://huggingface.co/{repo_id}/resolve/main/{f}"
    for f in files:
        low = f.lower()
        if low.endswith("model.safetensors") or low.endswith("model.ckpt"):
            return f"https://huggingface.co/{repo_id}/resolve/main/{f}"
    for f in files:
        low = f.lower()
        if (
            low.endswith(".safetensors")
            or low.endswith(".ckpt")
            or low.endswith(".pt")
            or low.endswith(".pth")
        ):
            return f"https://huggingface.co/{repo_id}/resolve/main/{f}"
    return None


def resolve_checkpoint_paths(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in CHECKPOINTS:
        path = root / spec.relative_path
        if path.exists():
            out[spec.name] = path
    return out
