import typer

from avatar_pipeline.pipeline import AvatarPipeline, PipelineConfig

app = typer.Typer()


@app.command()
def run(input_image: str, output_dir: str):
    pipeline = AvatarPipeline(PipelineConfig())
    result = pipeline.run(input_image=input_image, output_dir=output_dir)
    print(f"Avatar exported to: {result}")


if __name__ == "__main__":
    app()
