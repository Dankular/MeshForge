from pathlib import Path
from typing import Optional

import typer

from avatar_pipeline.pipeline import AvatarPipeline, PipelineConfig

app = typer.Typer()


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

    pipeline = AvatarPipeline(
        PipelineConfig(full_pshuman=full_pshuman, face_portrait=face_portrait)
    )
    result = pipeline.run(input_image=input_image, output_dir=output_dir, snapshot=snapshot)
    print(f"Avatar exported to: {result}")


if __name__ == "__main__":
    app()
