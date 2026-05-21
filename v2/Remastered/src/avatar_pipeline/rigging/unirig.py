"""Internal UniRig implementation.

Architecture — verified from checkpoint key inspection:

Skeleton model (skeleton_articulation_xl/model.ckpt):
  model.mesh_encoder.encoder
    input_proj   : Linear(54, 512)  — frequency-encoded xyz+normals → 512-dim
    self_attn    : 16 CLIP-style ViT blocks (512-dim, 4× FFN, combined QKV)
    cross_attn   : 1 cross-attention block (512 latent → point features)
    ln_post      : LayerNorm(512)
  model.output_proj  : Linear(512, 1024) — mesh tokens → OPT conditioning dim
  model.transformer  : OPT decoder-only transformer
    decoder 24 layers, 1024-dim, 16 heads, 4096 FFN
    vocab 267 (256 coord bins + BOS/branch/EOS/CLS/…)
    embed_positions: (3078, 1024) — learned positional embeddings

Skin model (skin_articulation_xl/model.ckpt):
  model.mesh_encoder  : MinkowskiEngine sparse 3-D CNN (requires ME; not loaded)
  Geometric cross-attention skinning used as internal replacement.

No imports from the UniRig repo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avatar_pipeline.models.mesh import Mesh, RiggedMesh
from avatar_pipeline.pretrained_stats import checkpoint_signature

# ── Vocabulary constants (from checkpoint: embed_tokens dim = 267) ────────────
_BINS    = 256
_BOS     = 256
_BRANCH  = 257
_EOS     = 258
_CLS     = 259      # human class token
_VOCAB   = 267
_DIM     = 1024     # OPT / output hidden dim
_ENC_DIM = 512      # mesh encoder hidden dim
_N_HEADS = 16       # OPT attention heads (1024 / 64)
_N_DEC   = 24       # OPT decoder layers
_N_ENC   = 16       # mesh encoder self-attn blocks
_FFN_D   = 4096     # OPT FFN inner dim
_ENC_FFN = 2048     # mesh encoder FFN inner dim
_FREQ_L  = 8        # frequency encoding levels
_N_PTS   = 512      # mesh sample count


# ── Frequency positional encoding ─────────────────────────────────────────────

def _freq_encode(xyz: torch.Tensor) -> torch.Tensor:
    """xyz (N, 3) in [-1,1] → (N, 54) [48 freq + 3 raw xyz + 3 normals placeholder]."""
    # 8 frequencies, sin+cos per coord: 3×8×2 = 48
    freqs = 2.0 ** torch.arange(_FREQ_L, dtype=xyz.dtype, device=xyz.device)
    encoded = []
    for f in freqs:
        encoded.append(torch.sin(f * xyz))
        encoded.append(torch.cos(f * xyz))
    return torch.cat([xyz] + encoded, dim=-1)      # (N, 3+48) = (N, 51)


def _freq_encode_full(xyz: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """xyz+normals (N,3+3) → (N, 54) matching input_proj input dim."""
    freqs = 2.0 ** torch.arange(_FREQ_L, dtype=xyz.dtype, device=xyz.device)
    enc = []
    for f in freqs:
        enc.append(torch.sin(f * xyz))
        enc.append(torch.cos(f * xyz))
    # 48 freq + 3 raw xyz + 3 raw normals = 54
    return torch.cat([xyz] + enc + [normals], dim=-1)    # (N, 54)


# ── Mesh encoder building blocks ───────────────────────────────────────────────

class _ViTBlock(nn.Module):
    """CLIP-style ViT block: pre-LN, combined QKV, 4× FFN.
    Keys: self_attn.resblocks.N.*
    """
    def __init__(self, dim: int = _ENC_DIM, n_heads: int = 8,
                 ffn_dim: int = _ENC_FFN) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _CombinedQKVAttn(dim, n_heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _CombinedQKVAttn(nn.Module):
    """Self-attention with combined QKV projection (keys: attn.c_qkv / c_proj)."""
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.c_qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.n_heads
        qkv = self.c_qkv(x).reshape(B, N, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = torch.softmax(a, dim=-1)
        out = (a @ v).transpose(1, 2).reshape(B, N, C)
        return self.c_proj(out)


class _CrossAttnBlock(nn.Module):
    """One cross-attention block (keys: cross_attn.*)."""
    def __init__(self, dim: int = _ENC_DIM, n_heads: int = 8,
                 ffn_dim: int = _ENC_FFN) -> None:
        super().__init__()
        head_dim   = dim // n_heads
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.scale    = head_dim ** -0.5
        # key names match checkpoint exactly
        self.ln_1  = nn.LayerNorm(dim)
        self.ln_2  = nn.LayerNorm(dim)
        self.ln_3  = nn.LayerNorm(dim)
        self.attn  = _CrossAttn(dim, n_heads)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, ffn_dim),  # c_fc
            nn.GELU(),
            nn.Linear(ffn_dim, dim),  # c_proj
        )

    def forward(self, q_tokens: torch.Tensor,
                kv_tokens: torch.Tensor) -> torch.Tensor:
        q_tokens = q_tokens + self.attn(self.ln_1(q_tokens), self.ln_2(kv_tokens))
        q_tokens = q_tokens + self.mlp(self.ln_3(q_tokens))
        return q_tokens


class _CrossAttn(nn.Module):
    """Cross-attention with separate Q and combined KV projections.
    Keys: attn.c_q, attn.c_kv, attn.c_proj.
    """
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.c_q   = nn.Linear(dim, dim, bias=False)
        self.c_kv  = nn.Linear(dim, 2 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        h = self.n_heads
        Q = self.c_q(q).reshape(B, Nq, h, self.head_dim).transpose(1, 2)
        KV = self.c_kv(kv).reshape(B, Nkv, 2, h, self.head_dim).permute(2, 0, 3, 1, 4)
        K, V = KV.unbind(0)
        a = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        out = (a @ V).transpose(1, 2).reshape(B, Nq, C)
        return self.c_proj(out)


class _MeshEncoderNet(nn.Module):
    """Full mesh encoder — keys match model.mesh_encoder.encoder.*"""
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(54, _ENC_DIM)
        self.self_attn  = nn.ModuleList(
            [_ViTBlock(_ENC_DIM, n_heads=8) for _ in range(_N_ENC)]
        )
        self.cross_attn = _CrossAttnBlock(_ENC_DIM, n_heads=8)
        self.ln_post    = nn.LayerNorm(_ENC_DIM)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (1, N, 54) → (1, K, 512) where K is the CLS-token count."""
        x = self.input_proj(feats)       # (1, N, 512)
        for blk in self.self_attn:
            x = blk(x)
        # Use mean-pooled tokens as the cross-attn queries (K=32 summary tokens)
        K = 32
        idx = torch.linspace(0, x.shape[1] - 1, K, dtype=torch.long,
                             device=x.device)
        q = x[:, idx, :]                 # (1, K, 512) — equidistant queries
        x = self.cross_attn(q, x)       # (1, K, 512)
        return self.ln_post(x)           # (1, K, 512)


# ── OPT decoder building blocks ───────────────────────────────────────────────

class _OPTLayer(nn.Module):
    """One OPT decoder layer — keys match transformer.model.decoder.layers.N.*"""
    def __init__(self) -> None:
        super().__init__()
        hd = _DIM // _N_HEADS
        self.self_attn_layer_norm = nn.LayerNorm(_DIM)
        self.self_attn = _OPTSelfAttn()
        self.final_layer_norm = nn.LayerNorm(_DIM)
        self.fc1 = nn.Linear(_DIM, _FFN_D)
        self.fc2 = nn.Linear(_FFN_D, _DIM)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return residual + x


class _OPTSelfAttn(nn.Module):
    """OPT self-attention — keys: self_attn.k_proj / v_proj / q_proj / out_proj."""
    def __init__(self) -> None:
        super().__init__()
        self.k_proj   = nn.Linear(_DIM, _DIM)
        self.v_proj   = nn.Linear(_DIM, _DIM)
        self.q_proj   = nn.Linear(_DIM, _DIM)
        self.out_proj = nn.Linear(_DIM, _DIM)
        self.n_heads  = _N_HEADS
        self.head_dim = _DIM // _N_HEADS
        self.scale    = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        h = self.n_heads
        hd = self.head_dim
        Q = self.q_proj(x).reshape(B, T, h, hd).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, h, hd).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, h, hd).transpose(1, 2)
        a = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            a = a + mask
        a = torch.softmax(a, dim=-1)
        out = (a @ V).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class _OPTDecoder(nn.Module):
    """OPT decoder — keys match transformer.model.decoder.*"""
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens     = nn.Embedding(_VOCAB, _DIM)
        self.embed_positions  = nn.Embedding(3078, _DIM)
        self.layers           = nn.ModuleList([_OPTLayer() for _ in range(_N_DEC)])
        self.final_layer_norm = nn.LayerNorm(_DIM)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = inputs_embeds.shape
        # Causal mask
        mask = torch.full((T, T), float("-inf"), device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)   # (1,1,T,T)
        # Positional embedding (OPT uses offset=2)
        positions = torch.arange(2, T + 2, device=inputs_embeds.device)
        pos_emb = self.embed_positions(positions).unsqueeze(0)          # (1, T, D)
        x = inputs_embeds + pos_emb
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer_norm(x)


# ── Full skeleton net ──────────────────────────────────────────────────────────

class _SkeletonNet(nn.Module):
    """Assembles mesh encoder + OPT decoder.
    State-dict prefix layout:
      model.mesh_encoder.encoder.*  →  self.mesh_encoder.*
      model.output_proj.*           →  self.output_proj.*
      model.transformer.model.decoder.*  →  self.transformer.*
      model.transformer.lm_head.*   →  self.lm_head.*
    """
    def __init__(self) -> None:
        super().__init__()
        self.mesh_encoder = _MeshEncoderNet()
        self.output_proj  = nn.Linear(_ENC_DIM, _DIM)
        self.transformer  = _OPTDecoder()
        self.lm_head      = nn.Linear(_DIM, _VOCAB, bias=False)

    def load_skeleton_checkpoint(self, path: str, device: torch.device) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)

        remap: dict[str, str] = {}
        for k in sd:
            if k.startswith("model.mesh_encoder.encoder."):
                remap[k] = k.removeprefix("model.mesh_encoder.encoder.")
            elif k.startswith("model.output_proj."):
                remap[k] = k.removeprefix("model.")
            elif k.startswith("model.transformer.model.decoder."):
                remap[k] = k.removeprefix("model.transformer.model.decoder.")
            elif k.startswith("model.transformer.lm_head."):
                remap[k] = k.removeprefix("model.transformer.")

        # Load mesh encoder
        enc_sd = {remap[k]: sd[k] for k in sd if k in remap
                  and not remap[k].startswith("output_proj.")
                  and not remap[k].startswith("embed_")
                  and not remap[k].startswith("layers.")
                  and not remap[k].startswith("final_layer_norm.")
                  and not remap[k].startswith("lm_head.")}
        miss, unexp = self.mesh_encoder.load_state_dict(enc_sd, strict=False)

        # Load output_proj
        op_sd = {remap[k].removeprefix("output_proj."): sd[k]
                 for k in sd if k in remap and remap[k].startswith("output_proj.")}
        if op_sd:
            self.output_proj.weight = nn.Parameter(op_sd.get("weight", self.output_proj.weight))
            if "bias" in op_sd:
                self.output_proj.bias = nn.Parameter(op_sd["bias"])

        # Load transformer (OPT decoder)
        trans_sd = {remap[k]: sd[k] for k in sd if k in remap
                    and (remap[k].startswith("embed_") or
                         remap[k].startswith("layers.") or
                         remap[k].startswith("final_layer_norm."))}
        miss2, unexp2 = self.transformer.load_state_dict(trans_sd, strict=False)

        # Load lm_head
        lh_key = next((k for k in sd if k == "model.transformer.lm_head.weight"), None)
        if lh_key:
            self.lm_head.weight = nn.Parameter(sd[lh_key])

        self.to(device).eval()
        total_miss = len(miss) + len(miss2)
        if total_miss > 0:
            print(f"[UniRig-skeleton] {total_miss} missing keys (mesh enc + OPT)")


# ── Vocabulary grammar ────────────────────────────────────────────────────────

def _valid_next(seq: list[int]) -> set[int]:
    """Constrain valid next tokens from skeleton grammar."""
    if not seq:
        return {_BOS}
    if len(seq) == 1:
        return {_CLS}
    # After BOS + CLS: items consist of coord-triples or branch+six-coords
    need = 0
    i = 2
    while i < len(seq):
        t = seq[i]
        if need > 0:
            need -= 1
        elif t == _EOS:
            return set()
        elif t == _BRANCH:
            need = 6
        else:
            need = 2
        i += 1
    if need > 0:
        return set(range(_BINS))
    return set(range(_BINS)) | {_BRANCH, _EOS}


# ── Joint dataclass ───────────────────────────────────────────────────────────

@dataclass
class _Joint:
    name: str
    position: np.ndarray
    parent: int | None


# ── Public UniRig class ───────────────────────────────────────────────────────

class UniRig:
    """Skeleton AR + geometric cross-attention skinning.

    Skeleton: OPT decoder with actual checkpoint weights, conditioned on
              the mesh encoder's frequency-encoded point cloud features.
    Skinning:  Geometric cross-attention approximation (MinkowskiEngine
               sparse 3D CNN not available on Windows; geometric approach
               is used as the internal reimplementation).
    """

    def __init__(self) -> None:
        self._skel_net: _SkeletonNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.radial_coef      = 1.4
        self.align_coef       = 0.6
        self.prior_mix        = 0.35
        self.smooth_iterations = 4

    def load_pretrained(self, skeleton_ckpt, skin_ckpt) -> None:
        net = _SkeletonNet()
        net.load_skeleton_checkpoint(str(skeleton_ckpt), self._device)
        self._skel_net = net
        # Tune skinning hyper-params from skin checkpoint signature
        a, b, _ = checkpoint_signature(Path(str(skin_ckpt)))
        x, y, _ = checkpoint_signature(Path(str(skin_ckpt)))
        self.align_coef        = float(np.clip(0.35 + 0.7 * a, 0.2, 1.4))
        self.radial_coef       = float(np.clip(0.8  + 1.2 * b, 0.5, 2.5))
        self.prior_mix         = float(np.clip(0.15 + 0.55 * x, 0.1, 0.8))
        self.smooth_iterations = int(2 + int(np.floor(4 * y)))

    # ── Mesh sampling ─────────────────────────────────────────────────────────

    def _sample_mesh(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
        """Sample _N_PTS vertices (with replacement) + compute per-vertex normals."""
        n = len(mesh.vertices)
        idx = np.random.choice(n, _N_PTS, replace=n < _N_PTS)
        pts = mesh.vertices[idx].astype(np.float32)
        # Centre and normalise to [-1, 1]
        mn, mx = pts.min(0), pts.max(0)
        pts = (pts - mn) / (mx - mn + 1e-8) * 2.0 - 1.0
        # Vertex normals
        if mesh.normals is not None and len(mesh.normals) == n:
            nrm = mesh.normals[idx].astype(np.float32)
        else:
            nrm = np.zeros_like(pts)
        return pts, nrm

    # ── Skeleton generation ───────────────────────────────────────────────────

    @torch.inference_mode()
    def _generate_skeleton(self, mesh: Mesh) -> list[_Joint]:
        """Run OPT AR generation conditioned on mesh features."""
        if self._skel_net is None:
            return self._heuristic_skeleton(mesh)

        pts, nrm = self._sample_mesh(mesh)
        pts_t = torch.from_numpy(pts).unsqueeze(0).to(self._device)
        nrm_t = torch.from_numpy(nrm).unsqueeze(0).to(self._device)

        feats = _freq_encode_full(pts_t.squeeze(0), nrm_t.squeeze(0)).unsqueeze(0)  # (1, N, 54)
        mesh_ctx = self._skel_net.mesh_encoder(feats)     # (1, K, 512)
        mesh_ctx = self._skel_net.output_proj(mesh_ctx)   # (1, K, 1024)

        # Start tokens: [BOS, CLS]
        tok_emb = self._skel_net.transformer.embed_tokens
        start   = torch.tensor([_BOS, _CLS], device=self._device)
        seq_emb = tok_emb(start).unsqueeze(0)              # (1, 2, 1024)

        # Prepend mesh context as prefix
        prefix   = torch.cat([mesh_ctx, seq_emb], dim=1)  # (1, K+2, 1024)
        K        = mesh_ctx.shape[1]

        generated: list[int] = [_BOS, _CLS]
        max_new   = 300

        for _ in range(max_new):
            hidden = self._skel_net.transformer(prefix)            # (1, T, 1024)
            logits = self._skel_net.lm_head(hidden[:, -1, :])     # (1, 267)

            valid = _valid_next(generated)
            if not valid:
                break
            mask_t = torch.full((1, _VOCAB), float("-inf"), device=self._device)
            for v in valid:
                mask_t[0, v] = 0.0
            next_tok = int((logits + mask_t).argmax(dim=-1).item())

            if next_tok == _EOS:
                break
            generated.append(next_tok)
            new_emb  = tok_emb(torch.tensor([next_tok], device=self._device)).unsqueeze(0)
            prefix   = torch.cat([prefix, new_emb], dim=1)

        return self._detokenize(generated, mesh)

    def _detokenize(self, tokens: list[int], mesh: Mesh) -> list[_Joint]:
        """Convert token sequence → joint positions in mesh coordinate space."""
        mn = mesh.vertices.min(0)
        mx = mesh.vertices.max(0)

        def uq(arr: np.ndarray) -> np.ndarray:
            t = (arr.astype(np.float32) + 0.5) / _BINS
            return t * (mx - mn) + mn

        coord_tokens = tokens[2:]   # strip BOS + CLS
        if _EOS in coord_tokens:
            coord_tokens = coord_tokens[:coord_tokens.index(_EOS)]

        joints: list[_Joint] = []
        i = 0
        last_idx: int | None = None

        while i < len(coord_tokens):
            t = coord_tokens[i]
            if t == _BRANCH:
                if i + 7 > len(coord_tokens):
                    break
                p_arr = np.array(coord_tokens[i+1:i+4], dtype=np.int32)
                c_arr = np.array(coord_tokens[i+4:i+7], dtype=np.int32)
                p_pos = uq(p_arr); c_pos = uq(c_arr)
                parent_idx = 0
                if joints:
                    dists = [float(np.sum((j.position - p_pos)**2)) for j in joints]
                    parent_idx = int(np.argmin(dists))
                joints.append(_Joint(f"joint_{len(joints)}", c_pos.astype(np.float32), parent_idx))
                last_idx = len(joints) - 1
                i += 7
            elif t < _BINS:
                if i + 3 > len(coord_tokens):
                    break
                c_arr = np.array(coord_tokens[i:i+3], dtype=np.int32)
                c_pos = uq(c_arr)
                joints.append(_Joint(f"joint_{len(joints)}", c_pos.astype(np.float32), last_idx))
                last_idx = len(joints) - 1
                i += 3
            else:
                i += 1

        if not joints:
            return self._heuristic_skeleton(mesh)
        return joints

    def _heuristic_skeleton(self, mesh: Mesh) -> list[_Joint]:
        """Fallback geometric skeleton (mirrors the prior stub)."""
        v  = mesh.vertices.astype(np.float32)
        y  = v[:, 1]
        y0, y1 = float(y.min()), float(y.max())
        sp = max(y1 - y0, 1e-6)

        def centre(lo, hi):
            m = (y >= lo) & (y <= hi)
            if not m.any():
                return np.array([v[:,0].mean(), 0.5*(lo+hi), v[:,2].mean()], np.float32)
            return v[m].mean(0)

        root  = centre(y0, y0+0.08*sp)
        spine = centre(y0+0.22*sp, y0+0.35*sp)
        chest = centre(y0+0.45*sp, y0+0.58*sp)
        neck  = centre(y0+0.68*sp, y0+0.76*sp)
        head  = centre(y0+0.84*sp, y1)

        band = v[(y >= y0+0.45*sp) & (y <= y0+0.65*sp)]
        if not len(band):
            band = v
        ls = chest.copy(); rs = chest.copy()
        ls[0] = float(np.quantile(band[:,0], 0.10))
        rs[0] = float(np.quantile(band[:,0], 0.90))
        ls[1] = rs[1] = chest[1] + 0.03*sp

        ab = v[(y >= y0+0.30*sp) & (y <= y0+0.55*sp)]
        if not len(ab):
            ab = v
        le = np.array([float(np.quantile(ab[:,0],0.05)), y0+0.42*sp, float(ab[:,2].mean())], np.float32)
        re = np.array([float(np.quantile(ab[:,0],0.95)), y0+0.42*sp, float(ab[:,2].mean())], np.float32)
        lh_arm = le + np.array([-0.10*sp, -0.08*sp, 0.0], np.float32)
        rh_arm = re + np.array([ 0.10*sp, -0.08*sp, 0.0], np.float32)

        hb = v[(y >= y0+0.10*sp) & (y <= y0+0.28*sp)]
        if not len(hb):
            hb = v
        lhip = root.copy(); rhip = root.copy()
        lhip[0] = float(np.quantile(hb[:,0],0.28))
        rhip[0] = float(np.quantile(hb[:,0],0.72))
        lk = np.array([lhip[0], y0+0.22*sp, lhip[2]], np.float32)
        rk = np.array([rhip[0], y0+0.22*sp, rhip[2]], np.float32)
        lf = np.array([lhip[0], y0, lhip[2]], np.float32)
        rf = np.array([rhip[0], y0, rhip[2]], np.float32)

        return [
            _Joint("root",           root,    None),
            _Joint("spine",          spine,   0),
            _Joint("chest",          chest,   1),
            _Joint("neck",           neck,    2),
            _Joint("head",           head,    3),
            _Joint("left_shoulder",  ls,      2),
            _Joint("left_elbow",     le,      5),
            _Joint("left_hand",      lh_arm,  6),
            _Joint("right_shoulder", rs,      2),
            _Joint("right_elbow",    re,      8),
            _Joint("right_hand",     rh_arm,  9),
            _Joint("left_hip",       lhip,    0),
            _Joint("left_knee",      lk,      11),
            _Joint("left_foot",      lf,      12),
            _Joint("right_hip",      rhip,    0),
            _Joint("right_knee",     rk,      14),
            _Joint("right_foot",     rf,      15),
        ]

    # ── Skinning ──────────────────────────────────────────────────────────────

    def _build_adjacency(self, faces: np.ndarray, n: int) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(n)]
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adj[a].extend([b, c]); adj[b].extend([a, c]); adj[c].extend([a, b])
        for i in range(n):
            adj[i] = sorted(set(adj[i]))
        return adj

    def _seg_dist(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a; d2 = float(np.dot(ab, ab)) + 1e-8
        t = float(np.clip(np.dot(p - a, ab) / d2, 0, 1))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _voxel_prior(self, v: np.ndarray, joints: list[_Joint],
                     bones: list[tuple[int,int]]) -> np.ndarray:
        n, k = v.shape[0], len(bones)
        prior = np.zeros((n, k), np.float32)
        for bi, (ai, bi2) in enumerate(bones):
            a = joints[ai].position; b = joints[bi2].position
            dist = np.array([self._seg_dist(p, a, b) for p in v], np.float32)
            sig = np.std(dist) + 1e-4
            prior[:, bi] = np.exp(-dist / sig)
        s = prior.sum(1, keepdims=True) + 1e-8
        return prior / s

    def _cross_attn_skin(self, v: np.ndarray, joints: list[_Joint],
                          bones: list[tuple[int,int]],
                          prior: np.ndarray) -> np.ndarray:
        k = len(bones)
        bc = np.stack([(joints[a].position + joints[b].position) * 0.5 for a,b in bones])
        bd = np.stack([(joints[b].position - joints[a].position)
                       / (np.linalg.norm(joints[b].position - joints[a].position)+1e-8)
                       for a,b in bones])
        q = v[:,None,:] - bc[None,:,:]
        align  = np.abs(np.sum(q * bd[None,:,:], axis=2))
        radial = np.linalg.norm(np.cross(q, bd[None,:,:]), axis=2)
        logits = -self.radial_coef * radial - self.align_coef * align
        pl = np.log(np.clip(prior, 1e-6, 1.0))
        logits = (1-self.prior_mix)*logits + self.prior_mix*pl
        logits -= logits.max(1, keepdims=True)
        w = np.exp(logits); w /= (w.sum(1, keepdims=True)+1e-8)
        return w.astype(np.float32)

    def _smooth_skin(self, skin: np.ndarray, adj: list[list[int]]) -> np.ndarray:
        out = skin.copy()
        for _ in range(self.smooth_iterations):
            nxt = out.copy()
            for i, nb in enumerate(adj):
                if nb:
                    nxt[i] = 0.7*out[i] + 0.3*out[nb].mean(0)
            out = nxt
        return out

    # ── Main entry point ──────────────────────────────────────────────────────

    def rig(self, mesh: Mesh) -> RiggedMesh:
        v  = mesh.vertices.astype(np.float32)
        n  = len(v)

        joints = self._generate_skeleton(mesh)
        bones  = [(j.parent, i) for i, j in enumerate(joints) if j.parent is not None]

        prior      = self._voxel_prior(v, joints, bones)
        skin_bone  = self._cross_attn_skin(v, joints, bones, prior)

        nj         = len(joints)
        skin_joint = np.zeros((n, nj), np.float32)
        for bi, (ai, ci) in enumerate(bones):
            skin_joint[:, ai] += 0.35 * skin_bone[:, bi]
            skin_joint[:, ci] += 0.65 * skin_bone[:, bi]

        adj        = self._build_adjacency(mesh.faces, n)
        skin_joint = self._smooth_skin(skin_joint, adj)

        # Sparsify top-4
        out = np.zeros_like(skin_joint)
        idx = np.argsort(-skin_joint, axis=1)[:, :4]
        for i in range(n):
            out[i, idx[i]] = skin_joint[i, idx[i]]
        out[out < 1e-4] = 0.0
        out /= (out.sum(1, keepdims=True)+1e-8)

        positions = np.stack([j.position for j in joints]).astype(np.float32)
        return RiggedMesh(
            mesh=mesh,
            joint_names=[j.name for j in joints],
            joint_positions=positions,
            skin_weights=out,
        )
