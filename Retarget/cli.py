"""
cli.py
Command-line interface for rig_retarget.

Usage:
    python -m rig_retarget.cli \\
        --source  walk.bvh \\
        --dest    unirig_character.glb \\
        --mapping radical2unirig.json \\
        --output  animated_character.glb \\
        [--fps 30] [--start 0] [--frames 100] [--step 1]

    # Calculate corrections only (no transfer):
    python -m rig_retarget.cli --calc-corrections \\
        --source walk.bvh --dest unirig_character.glb \\
        --mapping mymap.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="rig_retarget",
        description="Retarget animation from BVH/glTF source onto UniRig/glTF destination.",
    )
    p.add_argument("--source", required=True, help="Source animation file (.bvh or .glb/.gltf)")
    p.add_argument("--dest", required=True, help="Destination skeleton file (.glb/.gltf, UniRig output)")
    p.add_argument("--mapping", required=True, help="KeeMap-compatible JSON bone mapping file")
    p.add_argument("--output", default=None, help="Output animated .glb (default: dest_retargeted.glb)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=int, default=0, help="Start frame index (0-based)")
    p.add_argument("--frames", type=int, default=None, help="Number of frames to transfer (default: all)")
    p.add_argument("--step", type=int, default=1, help="Keyframe every N source frames")
    p.add_argument("--skin", type=int, default=0, help="Skin index in destination glTF")
    p.add_argument("--calc-corrections", action="store_true",
                   help="Auto-calculate bone corrections and update the mapping JSON, then exit.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    from .io.mapping import load_mapping, save_mapping, KeeMapSettings
    from .io.gltf_io import load_gltf, write_gltf_animation
    from .retarget import (
        calc_all_corrections, transfer_animation,
    )

    # -----------------------------------------------------------------------
    # Load mapping
    # -----------------------------------------------------------------------
    print(f"[*] Loading mapping  : {args.mapping}")
    settings, bone_items = load_mapping(args.mapping)

    # Override settings from CLI args
    settings.start_frame_to_apply = args.start
    settings.keyframe_every_n_frames = args.step

    # -----------------------------------------------------------------------
    # Load source animation
    # -----------------------------------------------------------------------
    src_path = Path(args.source)
    print(f"[*] Loading source   : {src_path}")

    if src_path.suffix.lower() == ".bvh":
        from .io.bvh import load_bvh
        src_anim = load_bvh(str(src_path))
        if args.verbose:
            print(f"    BVH: {src_anim.num_frames} frames, "
                  f"{src_anim.frame_time*1000:.1f} ms/frame, "
                  f"{len(src_anim.armature.pose_bones)} joints")
    elif src_path.suffix.lower() in (".glb", ".gltf"):
        # glTF source — load skeleton only; animation reading is TODO
        raise NotImplementedError(
            "glTF source animation reading is not yet implemented. "
            "Use a BVH file for the source animation."
        )
    else:
        print(f"[!] Unsupported source format: {src_path.suffix}", file=sys.stderr)
        sys.exit(1)

    if args.frames is not None:
        settings.number_of_frames_to_apply = args.frames
    else:
        settings.number_of_frames_to_apply = src_anim.num_frames - args.start

    # -----------------------------------------------------------------------
    # Load destination skeleton
    # -----------------------------------------------------------------------
    dst_path = Path(args.dest)
    print(f"[*] Loading dest     : {dst_path}")
    dst_arm = load_gltf(str(dst_path), skin_index=args.skin)
    if args.verbose:
        print(f"    Skeleton: {len(dst_arm.pose_bones)} bones")

    # -----------------------------------------------------------------------
    # Auto-correct pass (optional)
    # -----------------------------------------------------------------------
    if args.calc_corrections:
        print("[*] Calculating bone corrections ...")
        src_anim.apply_frame(args.start)
        calc_all_corrections(bone_items, src_anim.armature, dst_arm, settings)
        save_mapping(args.mapping, settings, bone_items)
        print(f"[*] Updated mapping saved → {args.mapping}")
        return

    # -----------------------------------------------------------------------
    # Transfer
    # -----------------------------------------------------------------------
    print(f"[*] Transferring {settings.number_of_frames_to_apply} frames "
          f"(start={settings.start_frame_to_apply}, step={settings.keyframe_every_n_frames}) ...")
    keyframes = transfer_animation(src_anim, dst_arm, bone_items, settings)
    print(f"[*] Generated {len(keyframes)} keyframes")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    out_path = args.output or str(dst_path.with_name(dst_path.stem + "_retargeted.glb"))
    print(f"[*] Writing output   : {out_path}")
    write_gltf_animation(str(dst_path), dst_arm, keyframes, out_path, fps=args.fps, skin_index=args.skin)
    print("[✓] Done")


if __name__ == "__main__":
    main()
