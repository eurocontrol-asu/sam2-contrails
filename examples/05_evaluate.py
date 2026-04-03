"""Evaluate predictions against GVCCS ground-truth annotations.

Usage::

    # Evaluate the first video
    uv run python examples/05_evaluate.py \
        --predictions results/20230930055430_20230930075430.json \
        --annotations /path/to/GVCCS/test/annotations.json

    # Skip slow segmentation mAP
    uv run python examples/05_evaluate.py \
        --predictions results/20230930055430_20230930075430.json \
        --annotations /path/to/GVCCS/test/annotations.json \
        --skip-segmentation
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import typer

log = structlog.get_logger()

app = typer.Typer(add_completion=False)


@app.command()
def main(
    predictions: Annotated[
        Path, typer.Option(help="Predictions JSON file.")
    ],
    annotations: Annotated[
        Path, typer.Option(help="GVCCS ground-truth annotations.json.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Evaluation output directory."),
    ] = None,
    iou_threshold: Annotated[
        float, typer.Option(help="IoU threshold for tracking/attribution.")
    ] = 0.25,
    skip_segmentation: Annotated[
        bool, typer.Option(help="Skip slow COCO mAP computation.")
    ] = False,
) -> None:
    """Evaluate contrail predictions against ground truth."""
    import contrailtrack as ct

    if not predictions.exists():
        log.error("predictions_not_found", path=str(predictions))
        raise typer.Exit(code=1)
    if not annotations.exists():
        log.error("annotations_not_found", path=str(annotations))
        raise typer.Exit(code=1)

    # Video name is inferred from the predictions filename
    video_name = predictions.stem
    eval_dir = output or Path(f"evaluation/{video_name}/")

    log.info("evaluating", video=video_name)
    results = ct.evaluate(
        predictions_path=predictions,
        gt_annotations=annotations,
        video_name=video_name,
        iou_threshold=iou_threshold,
        output_dir=eval_dir,
        skip_segmentation=skip_segmentation,
    )

    if "segmentation" in results:
        log.info("segmentation", mAP=round(results["segmentation"]["mAP"], 3))

    t = results.get("tracking", {}).get("metrics", {})
    if t:
        log.info(
            "tracking",
            detection_rate=round(t.get("detection_rate", 0), 3),
            completeness=round(t.get("mean_completeness", 0), 3),
            tiou=round(t.get("mean_tiou", 0), 3),
        )

    a = results.get("attribution", {}).get("metrics", {})
    if a:
        log.info(
            "attribution",
            precision=round(a.get("attribution_precision", 0), 3),
            assessed=a.get("n_assessed", 0),
        )

    log.info("done", output=str(eval_dir))


if __name__ == "__main__":
    app()
