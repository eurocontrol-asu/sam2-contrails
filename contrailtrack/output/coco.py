"""COCO RLE export for contrail predictions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as COCO RLE.

    Args:
        mask: bool or uint8 numpy array [H, W].

    Returns:
        {"size": [H, W], "counts": str}  (counts is base-64 RLE string)
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _compute_bbox(mask: np.ndarray):
    """Return [x, y, w, h] bounding box or None for an empty mask."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return [int(cols.min()), int(rows.min()),
            int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)]


def export_coco_json(
    predictions: dict,
    video_name: str,
    frame_names: list,
    height: int,
    width: int,
    output_path: str | Path,
    metadata: dict = None,
) -> Path:
    """Write predictions to a COCO RLE JSON file.

    Args:
        predictions: {frame_idx: {obj_id: {"mask": np.ndarray, "score": float}}}
                     as returned by run_video().
        video_name: Video identifier string (e.g., "00001").
        frame_names: Ordered list of frame name strings (no extension).
        height: Frame height in pixels (original, not resized).
        width: Frame width in pixels (original, not resized).
        output_path: Path to write the JSON file.
        metadata: Optional dict of extra fields to include in "info".

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_obj_ids = set()
    for frame_preds in predictions.values():
        all_obj_ids.update(frame_preds.keys())

    info = {
        "description": "contrailtrack SAM2 predictions",
        "version": "1.0",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if metadata:
        info.update(metadata)

    result = {
        "info": info,
        "video": {
            "name": video_name,
            "width": int(width),
            "height": int(height),
            "num_frames": len(frame_names),
            "num_objects": len(all_obj_ids),
        },
        "annotations": [],
    }

    ann_id = 1
    for frame_idx, frame_name in enumerate(frame_names):
        if frame_idx not in predictions:
            continue
        for obj_id, pred in predictions[frame_idx].items():
            mask = pred["mask"]
            area = int(np.sum(mask))
            if area == 0:
                continue
            bbox = _compute_bbox(mask)
            rle = encode_rle(mask)
            result["annotations"].append({
                "id": ann_id,
                "flight_id": str(obj_id),
                "frame_idx": int(frame_idx),
                "frame_name": frame_name,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "score": float(pred["score"]),
            })
            ann_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return output_path


def export_prompts_coco_json(
    prompts_dir: str | Path,
    video_id: str,
    frame_names: list[str],
    height: int,
    width: int,
    output_path: str | Path,
    metadata: dict = None,
) -> Path:
    """Export prompt masks to a COCO RLE JSON matching the predictions format.

    Reads the per-flight prompt PNGs from ``prompts_dir/{video_id}/{flight_id}/``
    and encodes each non-empty mask as a COCO RLE annotation with the same
    schema as ``export_coco_json`` (minus ``score``, plus ``age_weight``).

    Args:
        prompts_dir: Root prompts directory containing ``{video_id}/`` folders.
        video_id: Video identifier.
        frame_names: Ordered list of frame name strings (no extension).
        height: Frame height in pixels.
        width: Frame width in pixels.
        output_path: Path to write the JSON file.
        metadata: Optional dict of extra fields for "info".

    Returns:
        Path to the written file.
    """
    from PIL import Image

    prompts_dir = Path(prompts_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_dir = prompts_dir / video_id
    if not video_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {video_dir}")

    frame_name_to_idx = {fn: i for i, fn in enumerate(frame_names)}

    flight_dirs = sorted(
        d for d in video_dir.iterdir()
        if d.is_dir()
    )

    info = {
        "description": "contrailtrack prompt masks",
        "version": "1.0",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "prompts",
    }
    if metadata:
        info.update(metadata)

    result = {
        "info": info,
        "video": {
            "name": video_id,
            "width": int(width),
            "height": int(height),
            "num_frames": len(frame_names),
            "num_objects": len(flight_dirs),
        },
        "annotations": [],
    }

    ann_id = 1
    for flight_dir in flight_dirs:
        flight_id = flight_dir.name
        for prompt_path in sorted(flight_dir.glob("*_prompt.png")):
            frame_name = prompt_path.stem.replace("_prompt", "")
            frame_idx = frame_name_to_idx.get(frame_name)
            if frame_idx is None:
                continue

            mask_arr = np.array(Image.open(prompt_path))
            binary = mask_arr > 0
            area = int(binary.sum())
            if area == 0:
                continue

            age_weight = float(mask_arr[binary].mean()) / 255.0

            rle = encode_rle(binary)
            bbox = _compute_bbox(binary)
            result["annotations"].append({
                "id": ann_id,
                "flight_id": flight_id,
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "age_weight": round(age_weight, 4),
            })
            ann_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return output_path
