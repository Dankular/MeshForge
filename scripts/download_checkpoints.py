from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avatar_pipeline.checkpoints import (
    CHECKPOINTS,
    CheckpointSpec,
    cleanup_incomplete_files,
    download_checkpoint,
    resolve_dynamic_checkpoint_urls,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="checkpoints")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--names",
        type=str,
        default="",
        help="Comma separated checkpoint names. Empty means all.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    if args.list:
        for spec in CHECKPOINTS:
            print(f"{spec.name}: {spec.url}")
        return

    root = Path(args.root)
    if args.cleanup:
        removed = cleanup_incomplete_files(root)
        for p in removed:
            print(f"removed {p}")
        if not removed:
            print("no incomplete files found")
        if not args.names:
            return

    names = [x.strip() for x in args.names.split(",") if x.strip()] or None
    selected = set(names) if names is not None else None
    dynamic = resolve_dynamic_checkpoint_urls()
    downloaded = {}
    failures = {}
    for spec in CHECKPOINTS:
        if selected is not None and spec.name not in selected:
            continue
        eff = spec
        if spec.name in dynamic:
            eff = CheckpointSpec(
                name=spec.name,
                url=dynamic[spec.name],
                relative_path=spec.relative_path,
                sha256=spec.sha256,
            )
        try:
            downloaded[spec.name] = download_checkpoint(
                eff, root=root, force=args.force
            )
        except Exception as exc:
            failures[spec.name] = str(exc)

    for name, path in downloaded.items():
        print(f"{name} -> {path}")
    for name, err in failures.items():
        print(f"FAILED {name}: {err}")


if __name__ == "__main__":
    main()
