"""Render a video overlaying prompts and predictions on the original frames.

Each flight gets a consistent colour. Prompts are shown as semi-transparent
overlays, predictions as filled contours with the flight ID label.

Usage::

    uv run python examples/06_render_video.py \
        --frames data/frames/20230930055430_20230930075430 \
        --prompts data/prompts_cocip_age5 \
        --predictions results/20230930055430_20230930075430.json \
        --video-id 20230930055430_20230930075430

    # Only predictions (no prompts)
    uv run python examples/06_render_video.py \
        --frames data/frames/20230930055430_20230930075430 \
        --predictions results/20230930055430_20230930075430.json \
        --video-id 20230930055430_20230930075430

    # Custom output and FPS
    uv run python examples/06_render_video.py \
        --frames data/frames/20230930055430_20230930075430 \
        --predictions results/20230930055430_20230930075430.json \
        --video-id 20230930055430_20230930075430 \
        -o my_video.mp4 --fps 10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import structlog
import typer
from pycocotools import mask as mask_utils

log = structlog.get_logger()

app = typer.Typer(add_completion=False)

# Distinct colours (BGR) for up to 20 flights; cycles beyond that.
PALETTE = [
    (230, 159, 0),   # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # green
    (240, 228, 66),   # yellow
    (0, 114, 178),    # blue
    (213, 94, 0),     # vermillion
    (204, 121, 167),  # pink
    (0, 0, 0),        # black
    (128, 128, 128),  # grey
    (255, 255, 255),  # white
    (0, 255, 0),      # lime
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # dark orange
    (128, 0, 255),    # purple
    (0, 128, 255),    # light blue
    (255, 0, 128),    # hot pink
    (128, 255, 0),    # chartreuse
    (64, 64, 192),    # muted red
    (192, 64, 64),    # muted blue
]


def _flight_color(flight_id: str, flight_ids: list[str]) -> tuple[int, int, int]:
    """Return a consistent BGR colour for this flight."""
    idx = flight_ids.index(flight_id) if flight_id in flight_ids else hash(flight_id)
    return PALETTE[idx % len(PALETTE)]


def _decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict to a binary mask."""
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(bool)


def _overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """Blend a coloured mask onto an image."""
    overlay = img.copy()
    overlay[mask] = (
        (1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def _draw_contours(img: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 1) -> np.ndarray:
    """Draw contours of a mask on an image."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def _label_mask(img: np.ndarray, mask: np.ndarray, text: str, color: tuple) -> np.ndarray:
    """Put a label at the centroid of a mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    cx, cy = int(xs.mean()), int(ys.mean())
    cv2.putText(img, text, (cx - 20, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


@app.command()
def main(
    frames: Annotated[Path, typer.Option(help="Directory of JPEG frames.")],
    predictions: Annotated[Path, typer.Option(help="Predictions JSON from example 04.")],
    video_id: Annotated[str, typer.Option(help="Video identifier.")],
    prompts: Annotated[
        Path | None,
        typer.Option(help="Prompts directory (optional, for overlay)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output video path. Default: results/<video_id>.mp4"),
    ] = None,
    fps: Annotated[int, typer.Option(help="Output video frame rate.")] = 5,
    prompt_alpha: Annotated[float, typer.Option(help="Prompt overlay opacity.")] = 0.3,
    pred_alpha: Annotated[float, typer.Option(help="Prediction overlay opacity.")] = 0.5,
    show_labels: Annotated[bool, typer.Option(help="Show flight ID labels.")] = True,
) -> None:
    """Render a video with prompt and prediction overlays."""
    if not frames.exists():
        log.error("frames_not_found", path=str(frames))
        raise typer.Exit(code=1)
    if not predictions.exists():
        log.error("predictions_not_found", path=str(predictions))
        raise typer.Exit(code=1)

    out_path = output or Path(f"results/{video_id}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load predictions
    with open(predictions) as f:
        pred_data = json.load(f)

    # Index predictions by frame_name
    preds_by_frame: dict[str, list[dict]] = {}
    for ann in pred_data.get("annotations", []):
        preds_by_frame.setdefault(ann["frame_name"], []).append(ann)

    # Collect all flight IDs for consistent colouring
    all_flight_ids = sorted({a["flight_id"] for a in pred_data.get("annotations", [])})

    # Load prompts if provided
    prompt_data = {}
    if prompts:
        from contrailtrack.data.prompt_reader import read_prompts
        prompt_data = read_prompts(prompts, video_id, encoding="binary")
        # Add prompt flight IDs to palette
        for fid in prompt_data:
            if fid not in all_flight_ids:
                all_flight_ids.append(fid)
        all_flight_ids = sorted(all_flight_ids)

    # Get sorted frame paths
    frame_paths = sorted(
        p for p in Path(frames).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )

    if not frame_paths:
        log.error("no_frames_found")
        raise typer.Exit(code=1)

    # Read first frame for dimensions
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    log.info("rendering", n_frames=len(frame_paths), n_flights=len(all_flight_ids))

    for frame_path in frame_paths:
        frame_name = frame_path.stem
        img = cv2.imread(str(frame_path))

        # Draw prompts (semi-transparent, no contours)
        if prompt_data:
            for fid, fid_prompts in prompt_data.items():
                if frame_name in fid_prompts:
                    prompt_mask = fid_prompts[frame_name] > 0
                    color = _flight_color(fid, all_flight_ids)
                    img = _overlay_mask(img, prompt_mask, color, alpha=prompt_alpha)

        # Draw predictions (filled + contours + labels)
        for ann in preds_by_frame.get(frame_name, []):
            fid = ann["flight_id"]
            mask = _decode_rle(ann["segmentation"])
            color = _flight_color(fid, all_flight_ids)
            img = _overlay_mask(img, mask, color, alpha=pred_alpha)
            img = _draw_contours(img, mask, color, thickness=2)
            if show_labels:
                img = _label_mask(img, mask, fid[:8], color)

        # Timestamp overlay
        cv2.putText(img, frame_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, frame_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    n_pred_frames = len(preds_by_frame)
    log.info("done", output=str(out_path), frames_with_predictions=n_pred_frames)


if __name__ == "__main__":
    app()
