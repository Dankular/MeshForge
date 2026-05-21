"""
momask_server.py
────────────────────────────────────────────────────────────────────────────
Lightweight Flask inference server wrapping MoMask text-to-motion generation.
Runs on the Vast.ai instance. Exposes POST /generate → [T, 263] JSON.

Does NOT require SMPL body models — only the MoMask VQ-VAE checkpoints.

Deploy
──────
  1. Upload this file to /root/momask_server.py on the instance
  2. Install deps (see deploy_momask.sh)
  3. Run: python /root/momask_server.py --port 8765

Endpoint
────────
  POST /generate
  Body:  {"prompt": str, "num_frames": int, "seed": int}
  Reply: {"motion": [[T, 263] as nested list], "num_frames": T, "fps": 20}
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

# ── Flask ──────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify
except ImportError:
    sys.exit("pip install flask")

app = Flask(__name__)

# ── Global model state ──────────────────────────────────────────────────────
_model    = None
_mean     = None
_std      = None
_max_len  = 196   # max HumanML3D frames (~9.8 s at 20 fps)


def _load_model(momask_root: str, device: str = "cuda"):
    """Load MoMask model + normalisation stats into global state."""
    global _model, _mean, _std

    sys.path.insert(0, momask_root)

    import torch
    from models.mask_transformer.transformer import MaskTransformer
    from options.get_eval_option import get_opt

    # Load options from checkpoint directory
    opt_path = os.path.join(momask_root, "checkpoints", "t2m", "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns",
                            "opt.txt")
    opt      = get_opt(opt_path, device=device)

    # Load normalisation stats (from the HumanML3D dataset)
    stat_dir = os.path.join(momask_root, "checkpoints", "t2m",
                            "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    _mean = np.load(os.path.join(stat_dir, "meta", "mean.npy"))
    _std  = np.load(os.path.join(stat_dir, "meta", "std.npy"))

    # Load the transformer + VQ-VAE
    from models.mask_transformer.transformer import MaskTransformer
    from models.vq.model import RVQVAE
    import options.option_transformer as option_trans

    args = option_trans.get_args_parser()
    args = args.parse_args([])
    args.dataname   = "t2m"
    args.res_name   = "ter1"
    args.nb_code    = 512
    args.code_dim   = 512
    args.output_emb_width = 512
    args.nb_joints  = 22
    args.window_size = 64
    args.down_t     = 2
    args.stride_t   = 2
    args.width      = 512
    args.depth      = 3
    args.dilation_growth_rate = 3
    args.vq_act     = "relu"
    args.vq_norm    = None
    args.num_quantizers = 6

    net = RVQVAE(args,
                 263,
                 args.nb_code,
                 args.code_dim,
                 args.output_emb_width,
                 args.down_t,
                 args.stride_t,
                 args.width,
                 args.depth,
                 args.dilation_growth_rate,
                 args.vq_act,
                 args.vq_norm)

    # Load residual VQ-VAE weights
    vqvae_ckpt = os.path.join(momask_root, "checkpoints", "t2m", "Comp_v6_KLD005",
                              "net_last.pth")
    ckpt = torch.load(vqvae_ckpt, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    # Load mask transformer weights
    trans_ckpt_dir = os.path.join(momask_root, "checkpoints", "t2m",
                                  "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    trans = MaskTransformer(code_dim=opt.code_dim,
                            cond_mode="text",
                            latent_dim=opt.latent_dim,
                            ff_size=opt.ff_size,
                            num_layers=opt.num_layers,
                            num_heads=opt.num_heads,
                            dropout=opt.dropout,
                            clip_dim=512,
                            cond_drop_prob=opt.cond_drop_prob,
                            clip_version=opt.clip_version,
                            opt=opt)
    trans_ckpt = torch.load(os.path.join(trans_ckpt_dir, "net_last.pth"), map_location="cpu")
    trans.load_state_dict(trans_ckpt["trans"], strict=True)
    trans.eval().to(device)

    _model = (net, trans, opt, device)
    print(f"[momask_server] Model loaded on {device}")


def _generate(prompt: str, num_frames: int, seed: int) -> np.ndarray:
    """Run MoMask inference; return denormalised [T, 263] array."""
    import torch
    from utils.motion_process import recover_from_ric

    net, trans, opt, device = _model

    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    T = min(int(num_frames), _max_len)

    with torch.no_grad():
        # CLIP text encoding
        from models.mask_transformer.transformer import MaskTransformer
        cond_vector = trans.encode_text([prompt])   # [1, 77, 512]

        # MoMask iterative decoding
        mids = trans.generate(cond_vector, T // 4, temperature=1.0, topk_filter_thres=0.9,
                              gsample=True, force_mask=False)   # [1, T//4, nb_code]

        # Decode token sequence → motion features via RVQVAE decoder
        motion = net.forward_decoder(mids)   # [1, T, 263]
        motion = motion[0].cpu().numpy()     # [T, 263]

    # Denormalise
    motion = motion * _std + _mean
    return motion.astype(np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    prompt     = body.get("prompt", "a person walks forward")
    num_frames = int(body.get("num_frames", 120))
    seed       = int(body.get("seed", -1))

    if _model is None:
        return jsonify({"error": "model not loaded"}), 503

    try:
        motion = _generate(prompt, num_frames, seed)
        return jsonify({
            "motion":     motion.tolist(),
            "num_frames": int(motion.shape[0]),
            "fps":        20,
            "prompt":     prompt,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--momask-root", default="/root/momask-codes")
    parser.add_argument("--port",        type=int, default=8765)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--host",        default="0.0.0.0")
    args = parser.parse_args()

    print(f"[momask_server] Loading model from {args.momask_root} ...")
    _load_model(args.momask_root, args.device)

    print(f"[momask_server] Listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
