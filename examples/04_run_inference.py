"""Run SAM2-Contrails inference on a single video.

Given a folder of JPEG frames and pre-computed prompt masks, loads the
model from Hugging Face Hub, runs dense-prompt video inference, and saves
COCO RLE predictions.

Usage::

    # Run on the first GVCCS video (using prompts from example 03)
    uv run python examples/04_run_inference.py \
        --frames data/frames/20230930055430_20230930075430 \
        --prompts data/prompts_cocip_age5 \
        --video-id 20230930055430_20230930075430

    # With a local checkpoint
    uv run python examples/04_run_inference.py \
        --frames data/frames/20230930055430_20230930075430 \
        --prompts data/prompts_cocip_age5 \
        --video-id 20230930055430_20230930075430 \
        --checkpoint checkpoints/ternary.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import torch
import typer

log = structlog.get_logger()

app = typer.Typer(add_completion=False)


@app.command()
def main(
    frames: Annotated[
        Path, typer.Option(help="Directory of JPEG frames (output of example 01).")
    ],
    prompts: Annotated[
        Path, typer.Option(help="Prompt masks directory (output of example 03).")
    ],
    video_id: Annotated[
        str, typer.Option(help="Video identifier (e.g. 20230930055430_20230930075430).")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON. Default: results/<video_id>.json"),
    ] = None,
    config: Annotated[
        str, typer.Option(help="Model config: 'ternary' (recommended) or 'original'.")
    ] = "ternary",
    checkpoint: Annotated[
        Path | None,
        typer.Option(help="Local .pt checkpoint. Omit to auto-download from HF Hub."),
    ] = None,
    encoding: Annotated[
        str, typer.Option(help="Prompt encoding: 'ternary', 'age_weighted', or 'binary'.")
    ] = "ternary",
    score_threshold: Annotated[
        float, typer.Option(help="Minimum mask confidence score.")
    ] = 0.5,
    max_propagation_frames: Annotated[
        int, typer.Option(help="Max frames to propagate after last prompt (0 = unlimited).")
    ] = 50,
    device: Annotated[
        str, typer.Option(help="Compute device.")
    ] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Run contrail detection on a frame sequence."""
    import contrailtrack as ct

    output_json = output or Path(f"results/{video_id}.json")

    if not frames.exists():
        log.error("frames_not_found", path=str(frames))
        raise typer.Exit(code=1)
    if not prompts.exists():
        log.error("prompts_not_found", path=str(prompts))
        raise typer.Exit(code=1)

    # 1. Load model
    log.info("loading_model", config=config, device=device)
    model = ct.load_model(checkpoint=checkpoint, config=config, device=device)

    # 2. Load frames
    log.info("loading_frames", path=str(frames))
    frame_data, frame_names, h, w = ct.load_frames(frames, image_size=model.image_size)
    log.info("frames_loaded", n_frames=len(frame_names), resolution=f"{w}x{h}")

    # 3. Read prompts
    log.info("reading_prompts", encoding=encoding)
    prompt_data = ct.read_prompts(prompts, video_id, encoding=encoding)
    log.info("prompts_loaded", n_objects=len(prompt_data))

    if not prompt_data:
        log.warning("no_prompts_found", video_id=video_id)
        raise typer.Exit(code=1)

    # 4. Run inference
    log.info("running_inference", n_objects=len(prompt_data), n_frames=len(frame_names))
    predictions = ct.run_video(
        model=model,
        frames=frame_data,
        frame_names=frame_names,
        prompts=prompt_data,
        original_height=h,
        original_width=w,
        score_threshold=score_threshold,
        max_propagation_frames=max_propagation_frames,
        device=device,
    )

    # 5. Save results
    output_json.parent.mkdir(parents=True, exist_ok=True)
    ct.export_coco_json(
        predictions=predictions,
        video_name=video_id,
        frame_names=frame_names,
        height=h,
        width=w,
        output_path=output_json,
        metadata={"config": config, "encoding": encoding},
    )

    n_masks = sum(len(v) for v in predictions.values()) if isinstance(predictions, dict) else 0
    log.info("done", masks=n_masks, output=str(output_json))


if __name__ == "__main__":
    app()
