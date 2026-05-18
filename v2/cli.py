from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import get_preset
from .workflow import UnifiedWorkflow
from .check import check_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MeshForge V2 unified pipeline runner")

    subparsers = p.add_subparsers(dest="command", help="Command to run")

    # Pipeline command (default)
    pipeline_parser = subparsers.add_parser("run", help="Run the unified pipeline")
    pipeline_parser.add_argument("--image", required=True, help="Path to reference portrait image")
    pipeline_parser.add_argument("--output-dir", required=True, help="Directory for job outputs")
    pipeline_parser.add_argument(
        "--preset", default="quality", choices=["preview", "quality", "cinematic"]
    )
    pipeline_parser.add_argument("--seed", type=int, default=None)
    pipeline_parser.add_argument("--tex-seed", type=int, default=None)
    pipeline_parser.add_argument("--export-fbx", action="store_true")

    # Check command
    check_parser = subparsers.add_parser("check", help="Validate a model for pshuman rendering")
    check_parser.add_argument("model", help="Path to GLB model file")
    check_parser.add_argument(
        "--model-type",
        choices=["head", "body", "auto"],
        default="auto",
        help="Model type (auto-detected from filename if not specified)"
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    # Legacy support: if --image and --output-dir are provided without a command
    p.add_argument("--image", help=argparse.SUPPRESS)
    p.add_argument("--output-dir", help=argparse.SUPPRESS)
    p.add_argument(
        "--preset", default="quality", choices=["preview", "quality", "cinematic"],
        help=argparse.SUPPRESS
    )
    p.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--tex-seed", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--export-fbx", action="store_true", help=argparse.SUPPRESS)

    return p


def run_pipeline(args) -> int:
    cfg = get_preset(args.preset)
    if args.export_fbx:
        cfg = type(cfg)(**{**cfg.__dict__, "export_fbx": True})

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    tex_seed = (
        args.tex_seed if args.tex_seed is not None else random.randint(0, 2**31 - 1)
    )

    wf = UnifiedWorkflow(
        config=cfg,
        output_dir=Path(args.output_dir),
        seed=seed,
        tex_seed=tex_seed,
    )
    result = wf.run(Path(args.image))

    print(f"status={result['status']}")
    print(f"manifest={Path(args.output_dir) / 'manifest.json'}")
    if result["status"] != "completed":
        print(result.get("error", {}).get("message", "unknown error"))
        return 1
    return 0


def run_check(args) -> int:
    result = check_model(args.model, args.model_type)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["status"] != "ok":
            print(f"Error: {result['message']}")
            return 1

        model_type = "body" if any(r.get("location") for r in result.get("results", [])) else "head"
        print(f"✓ Model validation passed ({model_type})")

        if model_type == "body":
            print(f"\nAnatomy checks (Y range: {result['y_range'][0]:.3f} to {result['y_range'][1]:.3f}):")
            for r in result["results"]:
                print(f"  {r['location']:6s} (frac={r['fraction']:.2f}): "
                      f"x_mean={r['x_mean']:7.3f} x_std={r['x_std']:6.3f} "
                      f"z_mean={r['z_mean']:7.3f} z_std={r['z_std']:6.3f} "
                      f"n={r['points']}")
        else:
            for r in result["results"]:
                print(f"  {r['mesh']}: centroid={r['centroid']}, "
                      f"y_range=[{r['y_range'][0]:.3f}, {r['y_range'][1]:.3f}]")

    return 0


def main() -> int:
    args = build_parser().parse_args()

    # Legacy support: if --image and --output-dir provided, treat as pipeline run
    if args.image and args.output_dir and not args.command:
        args.command = "run"
    elif not args.command:
        build_parser().print_help()
        return 1

    if args.command == "run":
        return run_pipeline(args)
    elif args.command == "check":
        return run_check(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
