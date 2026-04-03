"""Load trained SAM2-Contrails models.

Models can be loaded from a local checkpoint path or downloaded automatically
from the Hugging Face Hub:

    >>> model = load_model()                          # auto-download (ternary)
    >>> model = load_model(config="original")         # binary baseline
    >>> model = load_model(checkpoint="my.pt")        # local file
"""

from __future__ import annotations

from pathlib import Path

from sam2.build_sam import build_sam2

# ── Hugging Face repository ────────────────────────────────────────────────────
HF_REPO_ID = "ramondalmau/sam2-contrails"

# ── Inference config short-names ───────────────────────────────────────────────
# "ternary"  — trained with ternary prompt encoding (age-weighted + negative
#               signal).  Use for all age-weighted / negative-signal variants.
# "original" — binary baseline (positive-only prompt, 1-min window).
CONFIGS: dict[str, str] = {
    "original": "configs/sam2.1/sam2.1_hiera_b+_GVCCS_original.yaml",
    "ternary":  "configs/sam2.1/sam2.1_hiera_b+_GVCCS_ternary.yaml",
}

# Filenames as stored in the Hugging Face repository
HF_FILENAMES: dict[str, str] = {
    "ternary":  "checkpoints/ternary.pt",
    "original": "checkpoints/original.pt",
}


def load_model(
    checkpoint: str | Path | None = None,
    config: str = "ternary",
    device: str = "cuda",
    hf_repo_id: str = HF_REPO_ID,
):
    """Load a trained SAM2-Contrails model.

    If *checkpoint* is ``None``, the weights are downloaded automatically from
    the Hugging Face Hub (requires ``huggingface_hub``).  Downloaded files are
    cached in the default ``huggingface_hub`` cache directory so subsequent
    calls are instant.

    Args:
        checkpoint: Path to a local ``.pt`` checkpoint file, or ``None`` to
            download from Hugging Face Hub.
        config: Config short-name — ``"ternary"`` (default) or
            ``"original"`` — or a full YAML config path.  Must match the
            architecture that the checkpoint was trained with.
        device: Compute device: ``"cuda"`` (default) or ``"cpu"``.
        hf_repo_id: Hugging Face repository ID.  Override this only if you are
            hosting the weights under a different repository.

    Returns:
        SAM2Base model in eval mode, ready for inference.

    Examples:
        >>> from contrailtrack import load_model
        >>> model = load_model()                           # auto-download
        >>> model = load_model("checkpoints/ternary.pt")  # local file
        >>> model = load_model(config="original")          # binary baseline
    """
    if checkpoint is None:
        checkpoint = _download_from_hub(config=config, hf_repo_id=hf_repo_id)

    cfg   = CONFIGS.get(config, config)
    model = build_sam2(cfg, str(checkpoint), device=device, apply_postprocessing=False)
    model.eval()
    return model


def list_configs() -> dict[str, str]:
    """Return the available config short-names and their YAML paths."""
    return dict(CONFIGS)


# ── Private helpers ────────────────────────────────────────────────────────────

def _download_from_hub(config: str, hf_repo_id: str) -> Path:
    """Download a checkpoint from Hugging Face Hub and return the local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic weight download.\n"
            "Install it with:  pip install huggingface_hub\n"
            "Or provide an explicit checkpoint path:  load_model('path/to/checkpoint.pt')"
        ) from exc

    filename = HF_FILENAMES.get(config)
    if filename is None:
        available = list(HF_FILENAMES)
        raise ValueError(
            f"No Hugging Face checkpoint registered for config {config!r}.\n"
            f"Available configs: {available}.\n"
            "Pass an explicit checkpoint path instead."
        )

    import logging
    logging.getLogger(__name__).info("Downloading %r checkpoint from %s ...", config, hf_repo_id)
    local_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
    return Path(local_path)
