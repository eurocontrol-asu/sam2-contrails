#!/usr/bin/env python3
"""
Window comparison figure: 5-min vs 10-min prompt windows.

Case A — Flight 3363798b (video 5, obj 27): 10-min rescues detection.
Case B — Flight 33c16444 (video 7, obj 14): 5-min avoids contamination.

One standalone figure per case (fig_window_case_a / fig_window_case_b):
3 rows (image+GT, 5-min pred, 10-min pred) x 3 columns; square panels.

Usage:
    python paper/scripts/plot_window_comparison.py [--out_dir paper/figures]
"""

import argparse
import json
import sys as _sys
from pathlib import Path

_sys.path.insert(0, str(Path(__file__).parent))
import paper_style as ps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from scipy.ndimage import binary_dilation

# ── Consistent overlay colors ────────────────────────────────────────────────
_C_GT   = ps.GT_COLOR
_C_PRED = ps.PRED_COLOR
_PRED_NP = tuple(ps.PRED_COLOR / 255.0)

IOU_THR = ps.IOU_THR

# ── Data paths ───────────────────────────────────────────────────────────────
_IMG_DIR    = Path("/data/common/TRAILVISION/GVCCS_V/test/img_folder")
_PROMPT_DIR = Path("/data/common/TRAILVISION/GVCCS_V/test")
_PRED_DIR   = Path("outputs")
_GT_ANN     = Path("/data/common/CAMERA/datasets/GVCCS/test/annotations.json")


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load_image(vid, fidx):
    """Load image as brightened grayscale RGB using shared style."""
    gray = ps.load_gray(_IMG_DIR / vid / f"{fidx:05d}.jpg")
    return np.stack([gray, gray, gray], axis=-1)


def _load_ternary_prompt(vid, obj, fidx, wmin):
    """Load ternary prompt in [-1, 1]: positive=blue target, negative=magenta competing."""
    subdir = "per_object_data" if wmin == 1 else f"per_object_data_age_{wmin}"
    pod_dir = _PROMPT_DIR / subdir
    return ps.load_ternary_prompt(pod_dir, vid, obj, f"{fidx:05d}")


def _poly2mask(seg, H=1024, W=1024):
    rles = mask_util.frPyObjects(seg, H, W)
    return mask_util.decode(mask_util.merge(rles)).astype(bool)


def _rle2mask(seg):
    return (_poly2mask(seg) if isinstance(seg, list)
            else mask_util.decode(seg).astype(bool))


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    i = (a & b).sum(); u = (a | b).sum()
    return float(i / u) if u else 0.0


def _gt_mask(coco, images, fidx, flight):
    img_id = images[fidx]["id"]
    anns = [a for a in coco["annotations"]
            if a["image_id"] == img_id and a.get("flight_id") == flight]
    if not anns:
        return None
    m = np.zeros((1024, 1024), bool)
    for a in anns:
        m |= _rle2mask(a["segmentation"])
    return m


def _pred_index(variant, vid):
    with open(_PRED_DIR / variant / f"{vid}.json") as f:
        data = json.load(f)
    idx = {}
    for ann in data["annotations"]:
        k = (ann["frame_idx"], ann["object_id"])
        if k not in idx or ann["score"] > idx[k]["score"]:
            idx[k] = ann
    return idx


def _pred_mask_and_score(idx, fidx, obj):
    a = idx.get((fidx, obj))
    if a is None:
        return None, 0.0
    return _rle2mask(a["segmentation"]), a.get("score", 0.0)


# ── Rendering ────────────────────────────────────────────────────────────────

def _blend(rgb, mask, color, alpha):
    if mask is None or not mask.any():
        return rgb.copy()
    out = rgb.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.where(mask,
                                out[:, :, c] * (1-alpha) + color[c] * alpha,
                                out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_prompt(rgb, ternary, alpha=ps.ALPHA_PROMPT):
    """Blend ternary prompt onto RGB image (positive=blue, negative=magenta)."""
    if ternary is None:
        return rgb.copy()
    return ps.blend_ternary_prompt(rgb, ternary, alpha)


def _crop_region(masks, pad=52):
    """Return square (r0,r1,c0,c1) bounds covering all masks, padded."""
    region = np.zeros((1024, 1024), bool)
    for m in masks:
        if m is not None and m.any():
            region |= m
    return ps.square_crop_bounds(region, pad=pad)


# ── Case definitions ─────────────────────────────────────────────────────────

CASES = [
    dict(vid_str="00005", vid=5, obj=27, flight="3363798b",
         frames=[110, 127, 140],
         variants=["ternary_5", "ternary_10"],
         win=[5, 10], letter="A"),
    dict(vid_str="00007", vid=7, obj=14, flight="33c16444",
         frames=[0, 8, 19],
         variants=["ternary_5", "ternary_10"],
         win=[5, 10], letter="B"),
]


# ── Figure construction ──────────────────────────────────────────────────────

def make_case_figure(case, coco, out_dir):
    """One standalone figure per case: 3 rows (image+GT, 5-min, 10-min) ×
    3 frame columns, all panels square, at full text width so each panel
    stays ~1.5 in — a 6-row combined figure would not fit the elsarticle
    text block at that panel size."""
    frames = case["frames"]
    nf = len(frames)

    label_frac = 0.09
    unit_w = label_frac + nf
    fig_w = ps.FIG_W_FULL
    fig = plt.figure(figsize=(fig_w, fig_w * 3.0 / unit_w + 0.25))
    gs = fig.add_gridspec(
        nrows=3, ncols=nf + 1,
        width_ratios=[label_frac] + [1.0] * nf,
        hspace=0.03, wspace=0.03,
    )

    variant_labels = {5: "5-min\nwindow", 10: "10-min\nwindow"}

    vid_str = case["vid_str"]
    vid_int = case["vid"]
    obj     = case["obj"]
    frames  = case["frames"]

    vid_imgs = sorted([im for im in coco["images"] if im["video_id"] == vid_int],
                      key=lambda x: x["id"])

    pred_idxs = {v: _pred_index(v, vid_str) for v in case["variants"]}

    # Pre-collect all masks for crop computation
    gt_all = [_gt_mask(coco, vid_imgs, f, flight=case["flight"]) for f in frames]
    pred_all = [[_pred_mask_and_score(pred_idxs[v], f, obj)[0]
                 for f in frames] for v in case["variants"]]

    # Square crop region from union of GT + all predictions
    all_masks = list(gt_all)
    for pa in pred_all:
        all_masks.extend(pa)
    r0, r1, c0, c1 = _crop_region(all_masks)

    # ── Row 0: image + GT ────────────────────────────────────────
    ax_lbl = fig.add_subplot(gs[0, 0])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.5, "Image\n+ GT", transform=ax_lbl.transAxes,
                ha="center", va="center", fontsize=ps.FONT_ROW_LABEL,
                rotation=90, linespacing=1.3)

    for fi, fidx in enumerate(frames):
        ax = fig.add_subplot(gs[0, fi + 1])
        img = _load_image(vid_str, fidx)
        gt = gt_all[fi]
        vis = img[r0:r1, c0:c1].copy()
        if gt is not None:
            vis = ps.overlay_mask(vis, gt[r0:r1, c0:c1], tuple(_C_GT),
                                  crop_w=c1 - c0, frac=0.008)
        ax.imshow(vis, interpolation="bilinear")
        ps.clean_ax(ax)
        dt = (fidx - frames[0]) * 0.5
        ax.set_title(f"frame {fidx}" if fi == 0 else f"+{dt:.1f} min",
                     fontsize=ps.FONT_COL_TITLE, pad=2.0)

    # ── Rows 1 & 2: predictions ──────────────────────────────────
    for ri, (variant, wmin) in enumerate(zip(case["variants"], case["win"])):
        row = ri + 1
        label = variant_labels.get(wmin, f"{wmin}-min")

        ax_lbl2 = fig.add_subplot(gs[row, 0])
        ax_lbl2.axis("off")
        ax_lbl2.text(0.5, 0.5, label, transform=ax_lbl2.transAxes,
                     ha="center", va="center", fontsize=ps.FONT_ROW_LABEL,
                     rotation=90, linespacing=1.3)

        for fi, fidx in enumerate(frames):
            ax = fig.add_subplot(gs[row, fi + 1])
            img = _load_image(vid_str, fidx)
            pred, score = _pred_mask_and_score(pred_idxs[variant], fidx, obj)

            vis = img[r0:r1, c0:c1].copy()

            # Ternary prompt overlay, subdued: context only — the
            # prediction (vermilion) must dominate visually
            ternary = _load_ternary_prompt(vid_str, obj, fidx, wmin)
            if ternary is not None:
                vis = _blend_prompt(vis, ternary[r0:r1, c0:c1], alpha=0.35)

            # Prediction only — GT reference lives in the top row, so
            # every panel shows a single overlay color
            if pred is not None:
                vis = ps.overlay_mask(vis, pred[r0:r1, c0:c1], tuple(_C_PRED),
                                      crop_w=c1 - c0, frac=0.008)

            ax.imshow(vis, interpolation="bilinear")
            ps.clean_ax(ax)

            if pred is not None and score > 0.01:
                ax.text(0.97, 0.05, f"{score:.2f}",
                        transform=ax.transAxes, **ps.SCORE_BADGE_KW)
            else:
                kw = {**ps.SCORE_BADGE_KW,
                      "bbox": {**ps.SCORE_BADGE_KW["bbox"], "fc": "#666666"}}
                ax.text(0.97, 0.05, "no detection",
                        transform=ax.transAxes, **kw)

    ps.overlay_legend(fig, [
        (ps.GT_HEX, "ground truth"),
        (ps.PRED_HEX, "prediction"),
        (ps.POS_HEX, "target prompt"),
        (ps.NEG_HEX, "competing prompt"),
    ])
    name = f"fig_window_case_{case['letter'].lower()}"
    ps.save_figure(fig, name, out_dir)
    plt.close(fig)


def make_figure(cases, coco, out_dir):
    for case in cases:
        make_case_figure(case, coco, out_dir)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    print("Loading annotations...")
    with open(_GT_ANN) as f:
        coco = json.load(f)

    print("Rendering...")
    make_figure(CASES, coco, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
