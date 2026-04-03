"""Load video frame sequences for contrail inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_frames(
    folder: str | Path,
    image_size: int = 1024,
) -> tuple[torch.Tensor, list[str], int, int]:
    """Load a sorted JPEG frame sequence.

    Frames can be named with any convention (timestamps, integers, etc.) —
    they are sorted lexicographically by stem.

    Args:
        folder: Directory containing JPEG frames.
        image_size: Square size to resize frames to (must match model.image_size).

    Returns:
        (frames, frame_names, orig_H, orig_W) where:
          - frames: float32 tensor [T, 3, H, H] in [0, 1] range (no ImageNet norm)
          - frame_names: list of str stems without extension, sorted
          - orig_H, orig_W: original frame dimensions before resizing
    """
    folder = Path(folder)

    jpg_files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if not jpg_files:
        raise FileNotFoundError(f"No JPEG frames found in {folder}")

    frame_names = [p.stem for p in jpg_files]

    # Load first frame to get original dimensions
    with Image.open(jpg_files[0]) as im:
        orig_w, orig_h = im.size

    images = torch.zeros(len(jpg_files), 3, image_size, image_size, dtype=torch.float32)
    for i, img_path in enumerate(tqdm(jpg_files, desc="loading frames")):
        with Image.open(img_path) as im:
            arr = np.array(im.convert("RGB").resize((image_size, image_size)))
        images[i] = torch.from_numpy(arr / 255.0).permute(2, 0, 1)

    return images, frame_names, orig_h, orig_w
