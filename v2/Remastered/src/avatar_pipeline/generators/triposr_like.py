"""Internal TripoSR-style 3-D reconstruction.

Architecture — verified from checkpoint key inspection:
  image_tokenizer : ViT-B/16  (12 layers, 768-dim, 197 tokens at 224×224)
  backbone        : 16 dual-attention transformer blocks
                    self-attn(1024) + cross-attn(1024←768) + GEGLU-FFN
  tokenizer       : learned triplane embeddings (3, 1024, 32, 32)
  post_processor  : ConvTranspose2d(1024→40, k=2, s=2) → (3, 40, 64, 64)
  decoder         : MLP  120 → [64]×9 → 4  (density + 3 colour)

No imports from TripoSG or TripoSR repos.
All architecture implemented here from checkpoint structure analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure

from avatar_pipeline.models.mesh import Mesh
from avatar_pipeline.models.semantic import PoseData, SemanticMap
from avatar_pipeline.pretrained_stats import checkpoint_signature

# ── Checkpoint-derived constants ───────────────────────────────────────────────
_DIM        = 1024      # triplane feature dim
_IMG_DIM    = 768       # ViT output dim
_N_HEADS    = 16        # attention heads  (1024 / 64 = 16)
_N_BLOCKS   = 16        # backbone transformer blocks
_PLANE_RES  = 32        # triplane spatial resolution
_PLANE_FEAT = 40        # features per plane after upsampling
_TOTAL_FEAT = 3 * _PLANE_FEAT   # 120 — decoder input
_VIT_MEAN   = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VIT_STD    = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Internal building blocks ───────────────────────────────────────────────────

class _GEGLU(nn.Module):
    """Gated-GELU block whose linear is named .proj to match checkpoint keys."""
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class _FFN(nn.Module):
    """GEGLU feed-forward network.

    Key layout (matches backbone.transformer_blocks.N.ff.*):
      net[0] = _GEGLU  (has .proj)
      net[1] = Identity  (no checkpoint key)
      net[2] = Linear
    """
    def __init__(self, dim: int = _DIM, mult: int = 8) -> None:
        super().__init__()
        inner = dim * mult // 2          # 4096
        self.net = nn.Sequential(
            _GEGLU(dim, inner),          # [0]  keys: ff.net.0.proj.*
            nn.Identity(),               # [1]  not in checkpoint
            nn.Linear(inner, dim),       # [2]  keys: ff.net.2.*
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attn(nn.Module):
    """Multi-head attention.

    Key layout (matches attn1 / attn2):
      to_q, to_k, to_v  — Linear, no bias
      to_out             — Sequential([Linear with bias])
    """
    def __init__(self, dim: int, ctx_dim: int | None = None,
                 n_heads: int = _N_HEADS) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5
        c = ctx_dim or dim
        self.to_q   = nn.Linear(dim, dim, bias=False)
        self.to_k   = nn.Linear(c,   dim, bias=False)
        self.to_v   = nn.Linear(c,   dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))   # [0] has bias

    def forward(self, x: torch.Tensor,
                ctx: torch.Tensor | None = None) -> torch.Tensor:
        c = ctx if ctx is not None else x
        B, N, D = x.shape
        h = self.n_heads

        q = self.to_q(x).reshape(B, N, h, self.head_dim).transpose(1, 2)
        k = self.to_k(c).reshape(B, c.shape[1], h, self.head_dim).transpose(1, 2)
        v = self.to_v(c).reshape(B, c.shape[1], h, self.head_dim).transpose(1, 2)

        a = (q @ k.transpose(-2, -1)) * self.scale
        a = torch.softmax(a, dim=-1)
        out = (a @ v).transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


class _DualAttnBlock(nn.Module):
    """One backbone transformer block.

    Key layout (matches backbone.transformer_blocks.N.*):
      norm1, attn1  — self-attention on triplane tokens
      norm2, attn2  — cross-attention triplane→image tokens
      norm3, ff     — GEGLU feed-forward
    """
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(_DIM)
        self.attn1 = _Attn(_DIM)
        self.norm2 = nn.LayerNorm(_DIM)
        self.attn2 = _Attn(_DIM, ctx_dim=_IMG_DIM)
        self.norm3 = nn.LayerNorm(_DIM)
        self.ff    = _FFN(_DIM)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), ctx)
        x = x + self.ff(self.norm3(x))
        return x


class _Backbone(nn.Module):
    """16-block triplane transformer (matches backbone.*)."""
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(_DIM, _DIM)
        self.transformer_blocks = nn.ModuleList(
            [_DualAttnBlock() for _ in range(_N_BLOCKS)]
        )
        self.norm = nn.LayerNorm(_DIM)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for blk in self.transformer_blocks:
            x = blk(x, ctx)
        return self.norm(x)


class _Tokenizer(nn.Module):
    """Learned triplane embeddings (matches tokenizer.embeddings)."""
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.zeros(3, _DIM, _PLANE_RES, _PLANE_RES)
        )


class _PostProcessor(nn.Module):
    """ConvTranspose2d upsampler (matches post_processor.upsample.*)."""
    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(_DIM, _PLANE_FEAT, kernel_size=2, stride=2)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        return self.upsample(planes)      # (3, 1024, 32, 32) → (3, 40, 64, 64)


class _ImageTokenizer(nn.Module):
    """ViT-B/16 image encoder (matches image_tokenizer.model.*)."""
    def __init__(self) -> None:
        super().__init__()
        from transformers import ViTConfig, ViTModel
        cfg = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3,
            add_pooling_layer=False,
        )
        self.model = ViTModel(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).last_hidden_state   # (1, 197, 768)


class _Decoder(nn.Module):
    """NeRF MLP  120 → [64]×9 → 4  (matches decoder.layers.*)."""
    def __init__(self) -> None:
        super().__init__()
        widths = [_TOTAL_FEAT] + [64] * 9 + [4]
        seq: list[nn.Module] = []
        for i in range(len(widths) - 1):
            seq.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                seq.append(nn.ReLU())
        # Sequential indices: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU … 18=Linear
        self.layers = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ── Full network ───────────────────────────────────────────────────────────────

class _TripoSRNet(nn.Module):
    """Assembles all sub-modules; checkpoint keys match directly."""
    def __init__(self) -> None:
        super().__init__()
        self.image_tokenizer = _ImageTokenizer()
        self.tokenizer       = _Tokenizer()
        self.backbone        = _Backbone()
        self.post_processor  = _PostProcessor()
        self.decoder         = _Decoder()

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """(1, 3, 224, 224) → (1, 197, 768)"""
        return self.image_tokenizer(img)

    def build_triplanes(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """(1, 197, 768) → (3, 40, 64, 64)"""
        planes = self.tokenizer.embeddings                           # (3, 1024, 32, 32)
        n = _PLANE_RES * _PLANE_RES
        x = planes.reshape(3 * n, _DIM).unsqueeze(0)               # (1, 3N, 1024)
        x = self.backbone(x, img_tokens)                            # (1, 3N, 1024)
        x = x.squeeze(0).reshape(3, _PLANE_RES, _PLANE_RES, _DIM)
        x = x.permute(0, 3, 1, 2).contiguous()                     # (3, 1024, 32, 32)
        return self.post_processor(x)                               # (3, 40, 64, 64)

    def query(self, triplanes: torch.Tensor,
              pts: torch.Tensor) -> torch.Tensor:
        """pts (N, 3) in [-1,1] → density (N,)"""
        def _samp(pi: int, uv: torch.Tensor) -> torch.Tensor:
            g = uv.view(1, 1, -1, 2)
            return F.grid_sample(
                triplanes[pi].unsqueeze(0), g,
                mode="bilinear", align_corners=False, padding_mode="border",
            ).squeeze(0).squeeze(1).T                               # (N, 40)

        f = torch.cat([
            _samp(0, pts[:, [0, 1]]),
            _samp(1, pts[:, [0, 2]]),
            _samp(2, pts[:, [1, 2]]),
        ], dim=-1)                                                   # (N, 120)
        return torch.sigmoid(self.decoder(f)[:, 0])                 # (N,)


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class TriplaneConfig:
    volume_resolution: int = 128
    iso_level: float = 0.5


class TripoSRLikeGenerator:
    """Image-to-3-D generator backed by TripoSR checkpoint weights."""

    def __init__(self, config: TriplaneConfig | None = None) -> None:
        self.config  = config or TriplaneConfig()
        self._net: _TripoSRNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, checkpoint_path) -> None:
        net   = _TripoSRNet()
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[TripoSR] missing  {len(missing)} keys: {missing[:3]}")
        if unexpected:
            print(f"[TripoSR] unexpected {len(unexpected)} keys: {unexpected[:3]}")
        net.to(self._device).eval()
        self._net = net

    def _ensure_loaded(self) -> None:
        if self._net is None:
            raise RuntimeError("TripoSR: call load_pretrained() before generate().")

    def _prep_image(self, image: np.ndarray) -> torch.Tensor:
        """uint8 RGB (H,W,3) → (1,3,224,224) ViT-normalised on device."""
        from PIL import Image as PILImage
        img = PILImage.fromarray(image[:, :, :3]).resize((224, 224), PILImage.BILINEAR)
        t = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        mean = _VIT_MEAN.to(t)
        std  = _VIT_STD.to(t)
        return ((t - mean) / std).to(self._device)

    def _spherical_uvs(self, verts: np.ndarray) -> np.ndarray:
        u = 0.5 + np.arctan2(verts[:, 2], verts[:, 0]) / (2.0 * np.pi)
        v_range = verts[:, 1].max() - verts[:, 1].min() + 1e-8
        v = (verts[:, 1] - verts[:, 1].min()) / v_range
        return np.stack([u, 1.0 - v], axis=1).astype(np.float32)

    @torch.inference_mode()
    def generate(
        self,
        image: np.ndarray,
        semantic_map: SemanticMap,
        pose_data: PoseData,
        depth: np.ndarray,
        normals: np.ndarray,
    ) -> Mesh:
        self._ensure_loaded()
        net = self._net

        # ── encode ────────────────────────────────────────────────────────────
        img_t      = self._prep_image(image)
        img_tokens = net.encode_image(img_t)          # (1, 197, 768)
        triplanes  = net.build_triplanes(img_tokens)  # (3, 40, 64, 64)

        # ── density grid ──────────────────────────────────────────────────────
        res  = self.config.volume_resolution
        lin  = torch.linspace(-1.0, 1.0, res, device=self._device)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts  = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        chunk = 32768
        dens: list[torch.Tensor] = []
        for i in range(0, pts.shape[0], chunk):
            dens.append(net.query(triplanes, pts[i : i + chunk]))
        vol = torch.cat(dens).reshape(res, res, res).cpu().numpy()

        # ── marching cubes ────────────────────────────────────────────────────
        iso = self.config.iso_level
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, level=iso)
        except ValueError:
            iso = float(np.clip(vol.mean(), 0.1, 0.9))
            verts, faces, _, _ = measure.marching_cubes(vol, level=iso)

        # rescale grid coords → [-1, 1], reorder to Y-up
        verts = (verts / (res - 1.0)) * 2.0 - 1.0
        verts = verts[:, [1, 0, 2]].astype(np.float32)
        verts[:, 1] *= -1.0
        peak = np.linalg.norm(verts, axis=1).max()
        if peak > 1e-6:
            verts /= peak

        return Mesh(
            vertices=verts,
            faces=faces.astype(np.int32),
            uvs=self._spherical_uvs(verts),
            semantic_regions=["skin", "hair", "cloth", "eyes", "mouth"],
        )
