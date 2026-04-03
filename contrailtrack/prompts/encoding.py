"""Prompt encoding functions for contrail masks.

These functions convert a raw age-weighted float mask [0, 1] (1 = newest,
0 = oldest / absent) into the prompt array fed to SAM2.

All three encodings are applied *per-object* at write time when generating
prompt PNGs, but the ternary combination (positive + negative) is assembled
from pre-saved per-object PNGs by read_prompts() using the union PNG.
"""

import numpy as np


def encode_binary(mask: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Convert a float mask to binary [0, 1].

    Any positive pixel becomes 1.0 regardless of age.

    Args:
        mask: float32 array [H, W] with values in [0, 1].
        threshold: Pixels below or equal to this value are set to 0.

    Returns:
        float32 array [H, W] with values in {0.0, 1.0}.
    """
    return (mask > threshold).astype(np.float32)


def encode_age_weighted(mask: np.ndarray, max_age_min: float = 5.0) -> np.ndarray:
    """Pass-through for age-weighted masks.

    The age weighting is already baked into the mask values by the geometry
    writer (newest = 1.0, at max_age = 0.0). This function just clips to [0, 1].

    Args:
        mask: float32 array [H, W] with values in [0, 1].
        max_age_min: Documented window used when the mask was generated.
                     Not applied here — present for API symmetry.

    Returns:
        float32 array clipped to [0, 1].
    """
    return np.clip(mask, 0.0, 1.0)


def encode_ternary(own: np.ndarray, union: np.ndarray) -> np.ndarray:
    """Combine a per-object mask with the all-objects union mask.

    Positive region: own flight prompt (age-weighted, > 0).
    Negative region: other flights' footprint not covered by this flight.

    Formula: np.where(own > 0, own, -union)

    This is the encoding used during inference by read_prompts(encoding="ternary").
    Use this function directly if you want to construct the ternary array from
    separately loaded own + union arrays.

    Args:
        own: float32 [H, W] age-weighted mask for the target flight, values in [0, 1].
        union: float32 [H, W] union of all flights' masks for this frame, values in [0, 1].

    Returns:
        float32 [H, W] with positive values where target flight is present
        and negative values (down to -1) where competing flights are present.
    """
    return np.where(own > 0, own, -union).astype(np.float32)
