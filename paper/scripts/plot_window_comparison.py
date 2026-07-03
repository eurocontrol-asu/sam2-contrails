#!/usr/bin/env python3
"""
Window comparison figure: 5-min vs 10-min prompt windows.

Case A — Flight 3363798b (video 5, obj 27): 10-min rescues detection.
Case B — Flight 33c16444 (video 7, obj 14): 5-min avoids contamination.

Layout per case: 3 rows (image+GT, 5-min pred, 10-min pred) x 5 columns.

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
    """Load ternary prompt in [-1, 1]: positive=blue target, negative=orange competing."""
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
    """Blend ternary prompt onto RGB image (positive=blue, negative=orange)."""
    if ternary is None:
        return rgb.copy()
    return ps.blend_ternary_prompt(rgb, ternary, alpha)


def _crop_region(masks, pad=52):
    """Return (r0,r1,c0,c1) bounding-box of all masks, padded."""
    H = W = 1024
    rows, cols = [], []
    for m in masks:
        if m is not None and m.any():
            r, c = np.where(m); rows += r.tolist(); cols += c.tolist()
    if not rows:
        return 300, 724, 300, 724
    r0, r1 = max(0, min(rows)-pad), min(H, max(rows)+pad)
    c0, c1 = max(0, min(cols)-pad), min(W, max(cols)+pad)
    return r0, r1, c0, c1


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

def make_figure(cases, coco, out_dir):
    nf = max(len(c["frames"]) for c in cases)
    ncase = len(cases)

    # Layout: per case = 3 rows (image+GT, 5-min, 10-min); cases separated by spacer
    row_h  = [1.0, 1.0, 1.0]
    spacer = [0.08]
    heights = (row_h + spacer) * ncase
    heights = heights[:-1]  # drop trailing spacer

    label_frac = 0.07
    fig = plt.figure(figsize=(ps.FIG_W_FULL, 4.9))
    gs = fig.add_gridspec(
        nrows=len(heights), ncols=nf + 1,
        height_ratios=heights,
        width_ratios=[label_frac] + [1.0] * nf,
        hspace=0.03, wspace=0.03,
    )

    variant_labels = {5: "5-min\nwindow", 10: "10-min\nwindow"}

    gs_row = 0
    for ci, case in enumerate(cases):
        vid_str = case["vid_str"]
        vid_int = case["vid"]
        obj     = case["obj"]
        flight  = case["flight"]
        frames  = case["frames"]

        vid_imgs = sorted([im for im in coco["images"] if im["video_id"] == vid_int],
                          key=lambda x: x["id"])

        pred_idxs = {v: _pred_index(v, vid_str) for v in case["variants"]}

        # Pre-collect all masks for crop computation
        gt_all = [_gt_mask(coco, vid_imgs, f, flight) for f in frames]
        pred_all = [[_pred_mask_and_score(pred_idxs[v], f, obj)[0]
                     for f in frames] for v in case["variants"]]

        # Crop region from union of GT + all predictions
        all_masks = list(gt_all)
        for pa in pred_all:
            all_masks.extend(pa)
        r0, r1, c0, c1 = _crop_region(all_masks)

        # ── Row 0: image + GT ────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[gs_row, 0])
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.5, "Image\n+ GT", transform=ax_lbl.transAxes,
                    ha="center", va="center", fontsize=ps.FONT_ROW_LABEL,
                    rotation=90, linespacing=1.3)

        for fi, fidx in enumerate(frames):
            ax = fig.add_subplot(gs[gs_row, fi + 1])
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
            if fi == 0:
                ps.panel_tag(ax, case["letter"])

        # ── Rows 1 & 2: predictions ──────────────────────────────────
        for ri, (variant, wmin) in enumerate(zip(case["variants"], case["win"])):
            row = gs_row + ri + 1
            label = variant_labels.get(wmin, f"{wmin}-min")

            ax_lbl2 = fig.add_subplot(gs[row, 0])
            ax_lbl2.axis("off")
            ax_lbl2.text(0.5, 0.5, label, transform=ax_lbl2.transAxes,
                         ha="center", va="center", fontsize=ps.FONT_ROW_LABEL,
                         rotation=90, linespacing=1.3)

            for fi, fidx in enumerate(frames):
                ax = fig.add_subplot(gs[row, fi + 1])
                img = _load_image(vid_str, fidx)
                gt = gt_all[fi]
                pred, score = _pred_mask_and_score(pred_idxs[variant], fidx, obj)

                vis = img[r0:r1, c0:c1].copy()

                # Ternary prompt overlay, subdued: context only — the
                # prediction (vermilion) must dominate visually
                ternary = _load_ternary_prompt(vid_str, obj, fidx, wmin)
                if ternary is not None:
                    vis = _blend_prompt(vis, ternary[r0:r1, c0:c1], alpha=0.35)

                # GT reference (solid, no outline) under the prediction
                if gt is not None:
                    vis = ps.overlay_mask(vis, gt[r0:r1, c0:c1], tuple(_C_GT),
                                          crop_w=c1 - c0, frac=0.006,
                                          outline=False)

                # Prediction: solid outlined stroke — the visual subject
                if pred is not None:
                    vis = ps.overlay_mask(vis, pred[r0:r1, c0:c1], tuple(_C_PRED),
                                          crop_w=c1 - c0, frac=0.008)

                ax.imshow(vis, interpolation="bilinear")
                ps.clean_ax(ax)

                # Score badge — smaller to match cell size
                if pred is not None and score > 0.01:
                    badge_kw = ps.SCORE_BADGE_KW
                    ax.text(0.97, 0.05, f"{score:.2f}",
                            transform=ax.transAxes, **badge_kw)

        gs_row += len(row_h) + len(spacer)

    ps.save_figure(fig, "fig_window_comparison", out_dir)
    plt.close(fig)


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
