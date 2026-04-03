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
