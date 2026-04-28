"""Visualize campaign predictions vs prompts for a single video window.

Produces a figure per top-detected object showing:
  Row 1: camera image + ternary prompt overlay (blue=own, orange=others)
  Row 2: camera image + prediction overlay (vermilion)

Usage::
    uv run python examples/visualize_campaign.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# ── Paths ────────────────────────────────────────────────────────────────────
VID = "20251231100000_20251231120000"
BASE = Path("/data/contrailnet/sam2")
FRAMES_DIR = BASE / "frames" / VID
PROMPTS_DIR = BASE / "prompts" / VID
PRED_PATH = BASE / "predictions" / f"{VID}.json"
OUT_DIR = BASE / "visualizations" / VID

# ── Colors (from paper_style.py) ────────────────────────────────────────────
PROMPT_BLUE = np.array([0.13, 0.47, 0.71])
NEG_ORANGE = np.array([0.90, 0.62, 0.0])
PRED_COLOR = np.array([213, 94, 0], dtype=np.uint8)
ALPHA_PROMPT = 0.75
ALPHA_PRED = 0.85

GRAY_VMIN, GRAY_VMAX = -30, 200


def load_gray(path):
    arr = np.array(Image.open(path).convert("L")).astype(np.float32)
    arr = (arr - GRAY_VMIN) / (GRAY_VMAX - GRAY_VMIN) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def to_rgb(gray):
    return np.stack([gray, gray, gray], axis=-1)


def blend_ternary_prompt(rgb, ternary, alpha=ALPHA_PROMPT):
    vis = rgb.astype(np.float32)
    pos = np.clip(ternary, 0, 1)
    neg = np.clip(-ternary, 0, 1)
    for c in range(3):
        vis[:, :, c] = np.where(
            pos > 1e-6,
            vis[:, :, c] * (1 - pos * alpha) + PROMPT_BLUE[c] * 255 * pos * alpha,
            vis[:, :, c],
        )
        vis[:, :, c] = np.where(
            neg > 1e-6,
            vis[:, :, c] * (1 - neg * alpha) + NEG_ORANGE[c] * 255 * neg * alpha,
            vis[:, :, c],
        )
    return np.clip(vis, 0, 255).astype(np.uint8)


def blend_mask(rgb, mask, color=PRED_COLOR, alpha=ALPHA_PRED):
    if mask is None or not mask.any():
        return rgb.copy()
    out = rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def load_ternary(obj_id, frame_name):
    pos_path = PROMPTS_DIR / obj_id / f"{frame_name}_prompt.png"
    union_path = PROMPTS_DIR / f"{frame_name}_all_prompts_union.png"
    pos = np.array(Image.open(pos_path).convert("L")).astype(np.float32) / 255.0 if pos_path.exists() else np.zeros((1024, 1024), np.float32)
    union = np.array(Image.open(union_path).convert("L")).astype(np.float32) / 255.0 if union_path.exists() else np.zeros((1024, 1024), np.float32)
    return np.where(pos > 0, pos, -union)


def main():
    # Load predictions
    with open(PRED_PATH) as f:
        data = json.load(f)

    # Group annotations by flight_id
    from collections import defaultdict
    preds_by_obj = defaultdict(list)
    for ann in data["annotations"]:
        preds_by_obj[ann["flight_id"]].append(ann)

    # Sort objects by number of predictions
    top_objects = sorted(preds_by_obj.keys(), key=lambda k: len(preds_by_obj[k]), reverse=True)

    # Build frame name list
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    frame_names = [p.stem for p in frame_paths]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Per-object stripcharts ───────────────────────────────────────────────
    for obj_id in top_objects[:8]:
        anns = sorted(preds_by_obj[obj_id], key=lambda a: a["frame_idx"])
        pred_frames = {a["frame_idx"]: a for a in anns}

        # Find all prompt frames for this object
        obj_prompt_dir = PROMPTS_DIR / obj_id
        if not obj_prompt_dir.exists():
            continue
        prompt_frame_names = sorted(f.stem.replace("_prompt", "") for f in obj_prompt_dir.glob("*_prompt.png"))

        # Union of all relevant frame indices (prompts + predictions + some margin)
        prompt_indices = [frame_names.index(fn) for fn in prompt_frame_names if fn in frame_names]
        pred_indices = list(pred_frames.keys())
        all_indices = sorted(set(prompt_indices + pred_indices))
        if not all_indices:
            continue

        # Extend range a bit on both sides for context
        idx_min = max(0, min(all_indices) - 2)
        idx_max = min(len(frame_names) - 1, max(all_indices) + 2)
        show_indices = list(range(idx_min, idx_max + 1))

        # Subsample to max 12 columns
        if len(show_indices) > 12:
            step = max(1, len(show_indices) // 12)
            show_indices = show_indices[::step][:12]

        n_cols = len(show_indices)
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 1.5, 3.2), dpi=150)
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        for col, frame_idx in enumerate(show_indices):
            fn = frame_names[frame_idx]
            frame_path = frame_paths[frame_idx]
            gray = load_gray(frame_path)
            rgb = to_rgb(gray)

            # Row 0: prompt overlay
            if fn in prompt_frame_names:
                ternary = load_ternary(obj_id, fn)
                row0 = blend_ternary_prompt(rgb, ternary)
            else:
                row0 = rgb.copy()

            # Row 1: prediction overlay
            if frame_idx in pred_frames:
                rle = pred_frames[frame_idx]["segmentation"]
                mask = mask_utils.decode(rle).astype(bool)
                score = pred_frames[frame_idx]["score"]
                row1 = blend_mask(rgb, mask)
            else:
                row1 = rgb.copy()
                score = None

            axes[0, col].imshow(row0)
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            axes[0, col].set_title(f"f{frame_idx}", fontsize=6, pad=2)
            if fn in prompt_frame_names:
                for spine in axes[0, col].spines.values():
                    spine.set_edgecolor("#0072B2")
                    spine.set_linewidth(2)

            axes[1, col].imshow(row1)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])
            if score is not None:
                axes[1, col].text(0.97, 0.05, f"{score:.2f}", transform=axes[1, col].transAxes,
                                  fontsize=7, fontweight="bold", color="white", ha="right", va="bottom",
                                  bbox=dict(fc="#D55E00", alpha=0.9, pad=1.5, boxstyle="round,pad=0.2", ec="none"))
                for spine in axes[1, col].spines.values():
                    spine.set_edgecolor("#D55E00")
                    spine.set_linewidth(2)

        axes[0, 0].set_ylabel("Prompt", fontsize=8, fontweight="bold")
        axes[1, 0].set_ylabel("Prediction", fontsize=8, fontweight="bold")
        fig.suptitle(f"{obj_id}  ({len(anns)} preds / {len(prompt_frame_names)} prompts)", fontsize=9, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = OUT_DIR / f"{obj_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {obj_id}: {out_path}")

    # ── All-objects summary for one frame ────────────────────────────────────
    # Pick a frame with the most predictions
    from collections import Counter
    frame_counts = Counter(a["frame_idx"] for a in data["annotations"])
    best_frame_idx = frame_counts.most_common(1)[0][0]
    fn = frame_names[best_frame_idx]

    gray = load_gray(frame_paths[best_frame_idx])
    rgb = to_rgb(gray)
    # Overlay all predictions for this frame
    all_pred = rgb.copy()
    all_prompt = rgb.copy()
    frame_anns = [a for a in data["annotations"] if a["frame_idx"] == best_frame_idx]
    for ann in frame_anns:
        mask = mask_utils.decode(ann["segmentation"]).astype(bool)
        all_pred = blend_mask(all_pred, mask)

    # Union prompt for this frame
    union_path = PROMPTS_DIR / f"{fn}_all_prompts_union.png"
    if union_path.exists():
        union = np.array(Image.open(union_path).convert("L")).astype(np.float32) / 255.0
        ternary_union = -union
        all_prompt = blend_ternary_prompt(rgb, ternary_union)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(rgb)
    axes[0].set_title(f"Camera (frame {best_frame_idx}, {fn})", fontsize=8)
    axes[1].imshow(all_prompt)
    axes[1].set_title(f"Union prompt ({len(frame_anns)} objects)", fontsize=8)
    axes[2].imshow(all_pred)
    axes[2].set_title(f"All predictions ({len(frame_anns)} masks)", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path = OUT_DIR / "all_objects_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Summary: {out_path}")


if __name__ == "__main__":
    main()
