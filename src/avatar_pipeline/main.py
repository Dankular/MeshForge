from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from avatar_pipeline.pipeline import AvatarPipeline, PipelineConfig

app = typer.Typer()


@dataclass
class BatchJob:
    name: str
    image: str
    no_clothes: bool = False


def parse_jobs(text: str) -> list[BatchJob]:
    """One job per line: ``<run_name> | <image_path> [| no-clothes]``.
    Blank lines and ``#`` comments are skipped."""
    jobs: list[BatchJob] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"jobs line {lineno}: expected '<name> | <image> [| flags]', "
                f"got {raw!r}"
            )
        flags = {p.lower() for p in parts[2:] if p}
        unknown = flags - {"no-clothes"}
        if unknown:
            raise ValueError(f"jobs line {lineno}: unknown flags {unknown}")
        jobs.append(
            BatchJob(
                name=parts[0], image=parts[1],
                no_clothes="no-clothes" in flags,
            )
        )
    if not jobs:
        raise ValueError("jobs file contains no jobs")
    return jobs


@app.command()
def run(
    input_image: str,
    output_dir: str,
    snapshot: Optional[str] = typer.Option(
        None,
        "--snapshot",
        help="Path to a pre-rig state cache (pickle). Loads it if present, "
             "otherwise runs stages 1-5 normally and writes it for next time. "
             "Stale texture schemas rebake textures without rerunning TripoSG.",
    ),
    from_candid: bool = typer.Option(
        False,
        "--from-candid",
        help="INPUT_IMAGE is a candid photo (face or partial face+body), not "
             "a T-pose reference. Build the reference first with IP-Adapter "
             "FaceID PlusV2 + OpenPose ControlNet, save it to "
             "OUTPUT_DIR/tpose_reference.png, and run the pipeline on that.",
    ),
    full_pshuman: bool = typer.Option(
        True,
        "--full-pshuman/--triposg-body",
        help="PSHuman native full-figure SMPL-X-guided reconstruction for "
             "body AND head (default). --triposg-body restores the legacy "
             "TripoSG + head-transplant arm.",
    ),
    no_clothes: bool = typer.Option(
        False,
        "--no-clothes",
        help="Generate the T-pose reference in fitted athletic underwear "
             "instead of clothing — anatomy validation between candidates "
             "and a clean body shape for the reconstruction (only with "
             "--from-candid).",
    ),
):
    face_portrait: str | None = None
    if from_candid:
        # Runs and unloads before AvatarPipeline construction so its SD15
        # stack never competes with the pipeline's shared mmgp domain.
        from avatar_pipeline.preprocess.tpose_reference import build_tpose_reference

        portrait_path = (
            Path(output_dir) / "face_portrait.png" if full_pshuman else None
        )
        input_image = str(
            build_tpose_reference(
                input_image,
                Path(output_dir) / "tpose_reference.png",
                portrait_out=portrait_path,
                no_clothes=no_clothes,
            )
        )
        print(f"T-pose reference written to: {input_image}")
        if portrait_path is not None:
            face_portrait = str(portrait_path)

    pipeline = AvatarPipeline(PipelineConfig(full_pshuman=full_pshuman))
    result = pipeline.run(
        input_image=input_image, output_dir=output_dir,
        snapshot=snapshot, face_portrait=face_portrait,
    )
    print(f"Avatar exported to: {result}")


@app.command()
def batch(
    jobs_file: str,
    output_root: str = typer.Option("outputs", "--output-root"),
):
    """Queue multiple candid runs through ONE resident pipeline.

    Phase A builds every job's T-pose reference + face portrait (the SD15
    gate stack loads and frees per job — it cannot share VRAM with the
    Sapiens pose gate). Phase B constructs the AvatarPipeline once and
    serves every job from the same mmgp domain: the 17-model stack loads
    exactly once instead of once per run. A failed job is reported and
    skipped, never blocking the rest of the queue.
    """
    import traceback

    from avatar_pipeline.preprocess.tpose_reference import build_tpose_reference

    jobs = parse_jobs(Path(jobs_file).read_text(encoding="utf-8"))
    print(f"[batch] {len(jobs)} jobs: {', '.join(j.name for j in jobs)}")

    failures: list[str] = []
    prepared: list[tuple[BatchJob, str, str]] = []
    for job in jobs:
        out_dir = Path(output_root) / job.name
        print(f"\n[batch] === reference: {job.name} ===")
        try:
            portrait = out_dir / "face_portrait.png"
            reference = build_tpose_reference(
                job.image,
                out_dir / "tpose_reference.png",
                portrait_out=portrait,
                no_clothes=job.no_clothes,
            )
            prepared.append((job, str(reference), str(portrait)))
        except Exception:
            traceback.print_exc()
            failures.append(job.name)

    if prepared:
        pipeline = AvatarPipeline(PipelineConfig())
        for job, reference, portrait in prepared:
            out_dir = Path(output_root) / job.name
            print(f"\n[batch] === pipeline: {job.name} ===")
            try:
                result = pipeline.run(
                    input_image=reference,
                    output_dir=str(out_dir),
                    snapshot=str(out_dir / "state.pkl"),
                    face_portrait=portrait,
                )
                print(f"[batch] {job.name} -> {result}")
            except Exception:
                traceback.print_exc()
                failures.append(job.name)

    done = len(jobs) - len(failures)
    print(f"\n[batch] complete: {done}/{len(jobs)} succeeded")
    if failures:
        print(f"[batch] FAILED: {', '.join(failures)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
