"""Read per_object_data/ prompt PNG structures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


def list_objects(prompt_dir: str | Path, video_id: str) -> list[str]:
    """Return sorted list of object (flight) IDs present for this video."""
    vid_dir = Path(prompt_dir) / video_id
    if not vid_dir.exists():
        return []
    return sorted(
        d for d in os.listdir(vid_dir)
        if (vid_dir / d).is_dir()
    )


def read_prompts(
    prompt_dir: str | Path,
    video_id: str,
    encoding: Literal["binary", "age_weighted", "ternary"] = "ternary",
) -> dict:
    """Read per_object_data/ PNG structure into prompt arrays.

    Folder layout expected::

        {prompt_dir}/{video_id}/{flight_id}/{frame}_prompt.png
        {prompt_dir}/{video_id}/{frame}_all_prompts_union.png   (ternary only)

    Args:
        prompt_dir: Root of the per_object_data directory tree.
        video_id: Video identifier string (e.g. "20230930055430_20230930075430").
        encoding: Prompt encoding to apply.
            "binary"        -- raw prompt PNG as [0, 1] float
            "age_weighted"  -- same (age weighting already baked into the PNG values)
            "ternary"       -- positive (own flight) minus negative (competing flights):
                               np.where(own > 0, own, -union)

    Returns:
        {flight_id (str): {frame_name (str): np.ndarray float32}}
        Only frames where the prompt is non-zero are included.
    """
    prompt_dir = Path(prompt_dir)
    vid_dir = prompt_dir / video_id
    if not vid_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {vid_dir}")

    # Pre-load union masks for ternary encoding
    union_per_frame = {}
    if encoding == "ternary":
        for fname in os.listdir(vid_dir):
            if fname.endswith("_all_prompts_union.png"):
                frame_name = fname.replace("_all_prompts_union.png", "")
                arr = np.array(Image.open(vid_dir / fname)).astype(np.float32) / 255.0
                union_per_frame[frame_name] = arr

    obj_dirs = sorted(
        d for d in os.listdir(vid_dir)
        if (vid_dir / d).is_dir()
    )
    result = {}

    for flight_id in obj_dirs:
        obj_dir = vid_dir / flight_id
        obj_prompts = {}

        for fname in os.listdir(obj_dir):
            if not fname.endswith("_prompt.png"):
                continue
            frame_name = fname.replace("_prompt.png", "")
            arr = np.array(Image.open(obj_dir / fname)).astype(np.float32) / 255.0

            if not arr.any():
                continue

            if encoding == "ternary" and frame_name in union_per_frame:
                union = union_per_frame[frame_name]
                arr = np.where(arr > 0, arr, -union)

            obj_prompts[frame_name] = arr

        if obj_prompts:
            result[flight_id] = obj_prompts

    return result
