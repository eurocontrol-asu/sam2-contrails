#!/usr/bin/env python3
"""
Fig: three characteristic failure modes of the best variant, on real frames.

  A  Prompt-following over-segmentation — flight 324b62ce (video 12): the
     model traces the full wind-drifted prompt instead of the short visible
     contrail; the mask covers the annotation but is far larger, failing
     the IoU match.
  B  Crossing contamination — flight 324b5124's prediction (video 12)
     covers 86% of neighbouring flight 324b65d8's contrail.
  C  Faint contrail missed — flight 33638739 (video 5): prompted on all 28
     annotated frames, never detected (prompt rejection false negative).

Usage (from a directory containing outputs/, per_object_data_age_5/):
    python paper/scripts/plot_failure_modes.py
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
from scipy.ndimage import binary_dilation

CASES = [
    dict(letter="A", title="Prompt-following\nover-segmentation",
         video="00012", frame="00064", gt_obj=105, pred_obj=105,
         prompt_obj=105, badge=True, crop="gt_pred"),
    dict(letter="B", title="Crossing\ncontamination", gt_on_top=False,
         video="00012", frame="00097", gt_obj=113, pred_obj=57,
         prompt_obj=57, badge=True),
    dict(letter="C", title="Faint contrail\nmissed",
         video="00005", frame="00043", gt_obj=33, pred_obj=None,
         prompt_obj=33, badge=False),
]


def load_pred(pred_dir, video, obj, frame):
    d = json.load(open(Path(pred_dir) / f"{video}.json"))
    for ann in d["annotations"]:
        if ann["object_id"] == obj and ann["frame_name"] == frame:
            return mask_utils.decode(ann["segmentation"]).astype(bool), \
                   float(ann.get("score", 0))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvccs_dir", type=Path,
                    default=Path("/data/common/TRAILVISION/GVCCS_V/test"))
    ap.add_argument("--pred_dir", type=Path, default=Path("outputs/ternary_5"))
    ap.add_argument("--pad", type=int, default=90)
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    g = args.gvccs_dir
    pod = g / "per_object_data_age_5"

    _w = ps.FIG_W_FULL
    fig, axes = plt.subplots(1, 3, figsize=(_w, _w / 3 + 0.2),
                             gridspec_kw={"wspace": 0.05})

    for ax, case in zip(axes, CASES):
        video, frame = case["video"], case["frame"]
        img_gray = ps.load_gray(g / "img_folder" / video / f"{frame}.jpg")
        img = np.stack([img_gray] * 3, axis=-1)

        gt_p = pod / video / str(case["gt_obj"]) / f"{frame}_mask.png"
        gt = np.array(Image.open(gt_p).convert("L")) > 127 if gt_p.exists() else None

        ternary = ps.load_ternary_prompt(pod, video, case["prompt_obj"], frame)

        pred, score = (None, None)
        if case["pred_obj"] is not None:
            pred, score = load_pred(args.pred_dir, video, case["pred_obj"], frame)

        vis = ps.blend_ternary_prompt(img, np.clip(ternary, 0, 1), alpha=0.9)
        layers = [("pred", pred), ("gt", gt)]
        if not case.get("gt_on_top", True):
            layers.reverse()
        for kind, m in layers:
            if m is None:
                continue
            color = ps.PRED_COLOR if kind == "pred" else ps.GT_COLOR
            it = 2 if kind == "pred" else 3
            vis = ps.blend_mask(vis, binary_dilation(m, iterations=it), color, 0.9)

        # Crop around everything relevant (or just GT+prediction when the
        # full prompt stroke would zoom the panel out too far)
        region = np.zeros_like(img_gray, dtype=bool)
        for m in (gt, pred):
            if m is not None:
                region |= m
        if case.get("crop") != "gt_pred":
            region |= ternary > 0.05
        ys, xs = np.nonzero(region)
        H, W = img_gray.shape
        r0, r1 = max(0, ys.min() - args.pad), min(H, ys.max() + args.pad)
        c0, c1 = max(0, xs.min() - args.pad), min(W, xs.max() + args.pad)
        side = max(r1 - r0, c1 - c0)
        r1, c1 = min(H, r0 + side), min(W, c0 + side)

        ax.imshow(vis[r0:r1, c0:c1], interpolation="bilinear")
        ps.clean_ax(ax)
        ax.set_title(case["title"], fontsize=ps.FONT_COL_TITLE - 3, pad=2.5)
        ax.text(0.04, 0.96, case["letter"], transform=ax.transAxes,
                fontsize=ps.FONT_PANEL_LETTER - 2, fontweight="bold",
                va="top", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
        if case["badge"] and score is not None:
            ax.text(0.96, 0.05, f"{score:.2f}", transform=ax.transAxes,
                    fontsize=5.5, fontweight="bold", color="white",
                    ha="right", va="bottom",
                    bbox=dict(fc=ps.PRED_HEX, alpha=0.92,
                              boxstyle="round,pad=0.25", ec="none"))
        elif not case["badge"]:
            ax.text(0.96, 0.05, "no detection", transform=ax.transAxes,
                    fontsize=5.5, fontweight="bold", color="white",
                    ha="right", va="bottom",
                    bbox=dict(fc="#666666", alpha=0.92,
                              boxstyle="round,pad=0.25", ec="none"))

    ps.save_figure(fig, "fig_failure_modes", args.out_dir)


if __name__ == "__main__":
    main()
