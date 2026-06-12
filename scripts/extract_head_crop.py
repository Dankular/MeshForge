"""Extract an RGBA head crop from a pre-rig snapshot (CLI wrapper).

The crop logic lives in avatar_pipeline.generators.pshuman_head; this
script recomputes the segmentation live because snapshots written before
the Goliath label-order fix carry mislabeled classes.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avatar_pipeline.generators.pshuman_head import extract_head_crop_rgba  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.35)
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        state = pickle.load(f)
    processed = state["processed"]

    import torch
    from avatar_pipeline.sapiens.human_parsing import HumanParser

    parser = HumanParser()
    parser.load_pretrained(None)
    if torch.cuda.is_available():
        parser._model = parser._model.to("cuda")
    seg = parser.parse(processed)

    rgba = extract_head_crop_rgba(processed, seg.labels, margin=args.margin)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(args.out)
    print(f"head crop {rgba.shape[1]}x{rgba.shape[0]} -> {args.out}")


if __name__ == "__main__":
    main()
