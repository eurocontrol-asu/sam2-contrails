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


def load_union_frames(prompt_dir: str | Path, video_id: str) -> dict:
    """Load all union mask PNGs for a video without loading per-object prompts.

    Separating union loading from object loading allows callers to load unions
    once and pass them into repeated ``read_prompts`` calls (e.g. one per object
    batch), avoiding redundant disk I/O.

    Returns:
        {frame_name: np.ndarray float32} — empty dict if the video directory
        does not exist or contains no union files.
    """
    vid_dir = Path(prompt_dir) / video_id
    if not vid_dir.exists():
        return {}
    union_per_frame = {}
    for fname in os.listdir(vid_dir):
        if fname.endswith("_all_prompts_union.png"):
            frame_name = fname.replace("_all_prompts_union.png", "")
            arr = np.array(Image.open(vid_dir / fname)).astype(np.float32) / 255.0
            union_per_frame[frame_name] = arr
    return union_per_frame


def read_prompts(
    prompt_dir: str | Path,
    video_id: str,
    encoding: Literal["binary", "age_weighted", "ternary"] = "ternary",
    include_negative_only: bool | None = None,
    max_negative_only_frames: int | None = None,
    obj_ids: list | None = None,
    union_per_frame: dict | None = None,
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
        include_negative_only: Whether to include frames where this object has no
            positive prompt but other objects are present (producing a pure-negative
            prompt = -union). Default ``None`` auto-enables for ternary encoding.
            Only effective when encoding is "ternary" and union files exist.
        max_negative_only_frames: Maximum number of negative-only frames to include
            after the last positive prompt. ``None`` means unlimited. Used to cap
            RAM usage — each frame is a full 1024×1024 float32 array.
        obj_ids: If provided, only load prompts for these object IDs. ``None``
            (default) loads all objects — preserves existing behaviour.
        union_per_frame: Pre-loaded union masks (from ``load_union_frames``).
            When provided, union files are not read from disk again. Useful
            when calling ``read_prompts`` once per batch to avoid redundant I/O.

    Returns:
        {flight_id (str): {frame_name (str): np.ndarray float32}}
        Only frames where the prompt is non-zero are included.
    """
    if include_negative_only is None:
        include_negative_only = (encoding == "ternary")

    prompt_dir = Path(prompt_dir)
    vid_dir = prompt_dir / video_id
    if not vid_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {vid_dir}")

    # Union masks: use caller-supplied cache or load from disk
    if union_per_frame is None:
        union_per_frame = {}
        if encoding == "ternary":
            for fname in os.listdir(vid_dir):
                if fname.endswith("_all_prompts_union.png"):
                    frame_name = fname.replace("_all_prompts_union.png", "")
                    arr = np.array(Image.open(vid_dir / fname)).astype(np.float32) / 255.0
                    union_per_frame[frame_name] = arr

    all_obj_dirs = sorted(
        d for d in os.listdir(vid_dir)
        if (vid_dir / d).is_dir()
    )
    if obj_ids is not None:
        obj_ids_set = set(obj_ids)
        obj_dirs = [d for d in all_obj_dirs if d in obj_ids_set]
    else:
        obj_dirs = all_obj_dirs

    result = {}

    for flight_id in obj_dirs:
        obj_dir = vid_dir / flight_id
        obj_prompts = {}

        for fname in os.listdir(obj_dir):
            if not fname.endswith("_prompt.png"):
                continue
            frame_name = fname.replace("_prompt.png", "")
            arr = np.array(Image.open(obj_dir / fname)).astype(np.float32) / 255.0

            if encoding == "ternary" and frame_name in union_per_frame:
                union = union_per_frame[frame_name]
                arr = np.where(arr > 0, arr, -union)

            if not arr.any():
                continue

            obj_prompts[frame_name] = arr

        if not obj_prompts:
            continue

        if include_negative_only and encoding == "ternary" and union_per_frame:
            last_positive_frame = max(obj_prompts.keys())
            positive_frames = set(obj_prompts.keys())

            neg_count = 0
            for frame_name in sorted(union_per_frame.keys()):
                if frame_name in positive_frames:
                    continue
                if frame_name <= last_positive_frame:
                    continue
                union = union_per_frame[frame_name]
                if not union.any():
                    continue
                if max_negative_only_frames is not None and neg_count >= max_negative_only_frames:
                    break
                obj_prompts[frame_name] = -union
                neg_count += 1

        result[flight_id] = obj_prompts

    return result
