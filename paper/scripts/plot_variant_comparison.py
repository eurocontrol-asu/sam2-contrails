#!/usr/bin/env python3
"""
Fig 4: 5-variant ablation grid on the same flight.

Layout: 5 rows (Binary | Age 5-min | Age 10-min | Ternary 5-min | Ternary 10-min)
        x N columns (frames evenly sampled from the contrail lifetime)
Each cell: image crop with prediction overlay + score.

Default example: flight 32f8ba84 (video 3)
  base=0%       (obj_id=99)
  age_5=?       (obj_id=102, placeholder — eval data pending)
  age=17.8%     (obj_id=102)
  ternary_5=88% (obj_id=100)
  ternary_10=7% (obj_id=102)

Usage:
    python paper/scripts/plot_variant_comparison.py
    python paper/scripts/plot_variant_comparison.py \\
        --pred_dirs outputs/base outputs/age_5 outputs/age outputs/ternary_5 outputs/ternary_10 \\
        --variant_names base age_5 age ternary_5 ternary_10 \\
        --obj_ids 99 102 102 100 102
"""

import argparse
import json
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
import paper_style as ps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# All prediction overlays use one consistent color; prompts use sky
_PRED_NP   = tuple(ps.PRED_COLOR / 255.0)
_PROMPT_NP = tuple(ps.PROMPT_COLOR / 255.0)
OVERLAY_ALPHA = ps.ALPHA_PRED


def _setup():
    pass  # publication.mplstyle already loaded by paper_style


def save(fig, name, out_dir):
    ps.save_figure(fig, name, out_dir)


# ── Data helpers ─────────────────────────────────────────────────────────────

def _blend(img_rgb, mask_bool, color_rgb, alpha):
    out = img_rgb.astype(float) / 255.0
    r, g, b = color_rgb
    out[mask_bool, 0] = (1 - alpha) * out[mask_bool, 0] + alpha * r
    out[mask_bool, 1] = (1 - alpha) * out[mask_bool, 1] + alpha * g
    out[mask_bool, 2] = (1 - alpha) * out[mask_bool, 2] + alpha * b
    return (out * 255).clip(0, 255).astype(np.uint8)


def load_predictions(pred_dir, video, obj_id):
    """Load predictions for a specific object from a variant's JSON."""
    pred_file = Path(pred_dir) / f"{video}.json"
    if not pred_file.exists():
        return {}
    with open(pred_file) as f:
        d = json.load(f)
    by_frame = {}
    for ann in d["annotations"]:
        if ann["object_id"] == obj_id:
            by_frame[ann["frame_name"]] = ann
    return by_frame


def decode_rle(ann):
    return mask_utils.decode(ann["segmentation"]).astype(bool)


def load_prompts(pod_dir, video, obj_id):
    """Return dict: frame_name -> prompt uint8 array."""
    obj_path = Path(pod_dir) / video / str(obj_id)
    if not obj_path.exists():
        return {}
    prompts = {}
    for f in obj_path.iterdir():
        if f.name.endswith("_prompt.png"):
            fn = f.name.replace("_prompt.png", "")
            prompts[fn] = np.array(Image.open(f).convert("L"))
    return prompts


def load_gt_masks(pod_dir, video, obj_id):
    """Return dict: frame_name -> bool GT mask."""
    obj_path = Path(pod_dir) / video / str(obj_id)
    if not obj_path.exists():
        return {}
    masks = {}
    for f in obj_path.iterdir():
        if f.name.endswith("_mask.png"):
            fn = f.name.replace("_mask.png", "")
            masks[fn] = np.array(Image.open(f).convert("L")) > 127
    return masks


def mask_bbox(mask_bool):
    rows = np.where(mask_bool.any(axis=1))[0]
    cols = np.where(mask_bool.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return rows[0], rows[-1], cols[0], cols[-1]


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(gvccs_dir, pred_dirs, variant_names, video, obj_ids,
                n_frames, out_dir, pad=50, selected_frames=None, pod_dirs=None):
    """obj_ids: list of int, one per variant (may differ across variants)."""
    gvccs = Path(gvccs_dir)
    img_base = gvccs / "img_folder"

    # Per-variant prompt directories (default: match encoding to correct pod dir)
    if pod_dirs is None:
        _pod_map = {
            "base":       gvccs / "per_object_data",
            "age_5":      gvccs / "per_object_data_age_5",
            "age":        gvccs / "per_object_data_age_10",
            "ternary_5":  gvccs / "per_object_data_age_5",
            "ternary_10": gvccs / "per_object_data_age_10",
        }
        pod_dirs = [_pod_map.get(n, gvccs / "per_object_data") for n in variant_names]

    # Load per-variant prompts using correct obj_id and pod dir
    variant_prompts = {}
    for name, pdir, oid in zip(variant_names, pod_dirs, obj_ids):
        variant_prompts[name] = load_prompts(pdir, video, oid)

    # Load GT masks (same across variants — use first available pod dir)
    gt_masks = {}
    for pdir, oid in zip(pod_dirs, obj_ids):
        gt_masks = load_gt_masks(pdir, video, oid)
        if gt_masks:
            break

    # Load predictions for each variant using its own obj_id
    variant_preds = {}
    for name, pdir, oid in zip(variant_names, pred_dirs, obj_ids):
        variant_preds[name] = load_predictions(pdir, video, oid)

    # Find frames where at least one variant has predictions OR prompts exist
    all_frames = set()
    for name in variant_names:
        all_frames |= set(variant_prompts[name].keys())
        all_frames |= set(variant_preds[name].keys())
    all_frames = sorted(all_frames)

    if not all_frames:
        print(f"ERROR: No frames found for video={video}, obj={obj_id}")
        return

    # Sample from frames where ALL variants have prompts so every column shows all rows.
    # Use intersection so binary (1-min window, narrowest coverage) is always shown.
    prompt_frames_per = [set(variant_prompts[n].keys())
                         for n in variant_names if len(variant_prompts[n]) > 5]
    if prompt_frames_per:
        prompt_frames = prompt_frames_per[0].intersection(*prompt_frames_per[1:])
    else:
        prompt_frames = set()
    pool = sorted(prompt_frames) if prompt_frames else all_frames
    n = min(n_frames, len(pool))
    idxs = np.round(np.linspace(0, len(pool) - 1, n)).astype(int)
    selected = [pool[i] for i in idxs]
    if selected_frames:
        selected = selected_frames
    n = len(selected)

    print(f"  Selected frames: {selected}")

    # Compute global crop region from union of all prompts and predictions
    H, W = 1024, 1024  # Default, will be overridden
    sample_img = ps.load_gray(img_base / video / f"{selected[0]}.jpg")
    H, W = sample_img.shape[:2]

    # Crop to ground truth + predictions only: the (much longer) 10-min prompt
    # strokes would otherwise zoom the panels out until the contrail vanishes.
    union_mask = np.zeros((H, W), dtype=bool)
    for fn in selected:
        for name in variant_names:
            if fn in variant_preds[name]:
                union_mask |= decode_rle(variant_preds[name][fn])
        if fn in gt_masks:
            union_mask |= gt_masks[fn]

    bb = mask_bbox(union_mask)
    if bb is None:
        # Use center crop
        r0, r1, c0, c1 = H//4, 3*H//4, W//4, 3*W//4
    else:
        r0, r1, c0, c1 = bb
        r0 = max(0, r0 - pad)
        r1 = min(H, r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(W, c1 + pad)

    # Create figure: (1 + n_variants) rows × n columns
    # Row 0: Image + GT  |  Rows 1…: one per variant (prompt + prediction)
    n_variants = len(variant_names)
    n_rows = n_variants + 1
    cell_h = ps.FIG_W_FULL / n   # square cells
    fig_h  = cell_h * n_rows
    fig, axes = plt.subplots(n_rows, n, figsize=(ps.FIG_W_FULL, fig_h),
                              constrained_layout=False,
                              gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    fig.subplots_adjust(left=0.18)
    if n == 1:
        axes = axes[:, np.newaxis]

    # Short row labels that fit in the margin at 7.2" width
    row_labels = ["Image + GT"] + [
        {"base":       "Binary",
         "age_5":      "Age-wtd.\n5 min",
         "age":        "Age-wtd.\n10 min",
         "ternary_5":  "+ neg.\n5 min",
         "ternary_10": "+ neg.\n10 min"}.get(vn, vn)
        for vn in variant_names
    ]

    for col, fn in enumerate(selected):
        # Load image crop
        img_gray = ps.load_gray(img_base / video / f"{fn}.jpg")
        img = np.stack([img_gray, img_gray, img_gray], axis=-1)
        img_crop = img[r0:r1, c0:c1]
        gt = gt_masks.get(fn)
        gt_crop = gt[r0:r1, c0:c1] if gt is not None else None

        # ── Row 0: Image + GT (solid outlined stroke) ────────────────────────
        ax0 = axes[0, col]
        vis0 = ps.overlay_mask(img_crop, gt_crop, tuple(ps.GT_COLOR),
                               crop_w=c1 - c0, frac=0.008)
        ax0.imshow(vis0, interpolation="bilinear")
        ps.clean_ax(ax0)
        f0 = int(selected[0])
        dt = (int(fn) - f0) * 0.5
        ax0.set_title(f"frame {int(fn)}" if col == 0 else f"+{dt:.1f} min",
                      fontsize=ps.FONT_COL_TITLE - 2, pad=2.5)
        if col == 0:
            ax0.set_ylabel(row_labels[0], fontsize=6.5, labelpad=6)

        # ── Rows 1…: per-variant prompt + prediction ─────────────────────────
        for vrow, name in enumerate(variant_names):
            ax = axes[vrow + 1, col]

            # Prompt rendering: ternary→blue+orange, age→blue gradient, binary→blue flat
            if name in ("ternary_5", "ternary_10"):
                ternary = ps.load_ternary_prompt(pod_dirs[vrow], video, obj_ids[vrow], fn)
                vis = ps.blend_ternary_prompt(img_crop, ternary[r0:r1, c0:c1],
                                              alpha=0.5)
            else:
                raw_prompt = variant_prompts[name].get(fn)
                if raw_prompt is not None:
                    m = raw_prompt.astype(float).max()
                    prompt_norm = raw_prompt.astype(float) / m if m > 0 else raw_prompt.astype(float)
                else:
                    prompt_norm = np.zeros_like(img_crop[:, :, 0], dtype=float)
                vis = ps.blend_prompt_blue(img_crop, prompt_norm[r0:r1, c0:c1])

            # Prediction overlay only (no GT underlay in variant rows — GT is in Row 0)
            pred_ann = variant_preds[name].get(fn)
            score_text = None
            if pred_ann is not None:
                pred_mask_crop = decode_rle(pred_ann)[r0:r1, c0:c1]
                score = pred_ann.get("score", 0)
                vis = ps.overlay_mask(vis, pred_mask_crop, tuple(ps.PRED_COLOR),
                                      crop_w=c1 - c0, frac=0.008)
                score_text = f"{score:.2f}"

            ax.imshow(vis, interpolation="bilinear")
            ps.clean_ax(ax)

            if score_text:
                badge_kw = {**ps.SCORE_BADGE_KW,
                            "fontsize": 5,
                            "bbox": {**ps.SCORE_BADGE_KW["bbox"], "pad": 0.8,
                                     "boxstyle": "round,pad=0.22"}}
                ax.text(0.97, 0.05, score_text,
                        transform=ax.transAxes, **badge_kw)

            if col == 0:
                ax.set_ylabel(row_labels[vrow + 1],
                              fontsize=6.5, labelpad=6)

    save(fig, "fig_variant_comparison", out_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Variant comparison figure")
    parser.add_argument("--gvccs_dir", type=Path,
                        default=Path("/data/common/TRAILVISION/GVCCS_V/test"))
    parser.add_argument("--pred_dirs", nargs="+", type=Path,
                        default=[Path("outputs/base"),
                                 Path("outputs/age_5"),
                                 Path("outputs/age_10"),
                                 Path("outputs/ternary_5"),
                                 Path("outputs/ternary_10")])
    parser.add_argument("--variant_names", nargs="+",
                        default=["base", "age_5", "age", "ternary_5", "ternary_10"])
    parser.add_argument("--video", default="00012")
    parser.add_argument("--obj_ids", nargs="+", type=int, default=[102, 102, 102, 102, 102],
                        help="Per-variant object IDs (one per variant, same order as --variant_names)")
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--selected_frames", nargs="+", default=None,
                        help="Explicit frame names to display (overrides n_frames)")
    parser.add_argument("--pod_dirs", nargs="+", type=Path, default=None,
                        help="Per-variant prompt dirs (default: inferred from variant name)")
    parser.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--pad", type=int, default=50)
    args = parser.parse_args()

    if len(args.obj_ids) != len(args.variant_names):
        raise ValueError(f"--obj_ids length ({len(args.obj_ids)}) must match "
                         f"--variant_names length ({len(args.variant_names)})")

    _setup()
    print(f"Generating variant comparison: video={args.video}, "
          f"variants={args.variant_names}, obj_ids={args.obj_ids}")
    make_figure(args.gvccs_dir, args.pred_dirs, args.variant_names,
                args.video, args.obj_ids, args.n_frames, args.out_dir,
                pad=args.pad, selected_frames=args.selected_frames,
                pod_dirs=args.pod_dirs)
    print("Done.")


if __name__ == "__main__":
    main()
