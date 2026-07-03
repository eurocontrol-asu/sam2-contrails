#!/usr/bin/env python3
"""
Graphical abstract: the whole pipeline in one image, built from a real
test-set frame (video 2, frame 176).

  [camera image] -> [physics-derived prompts] -> [attributed contrail masks]

Usage (from a directory containing per_object_data_age_5/, outputs/):
    python paper/scripts/plot_graphical_abstract.py
"""

import argparse
import json
from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).parent))
import paper_style as ps
import plot_attribution_scene as pas

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvccs_dir", type=Path,
                    default=Path("/data/common/TRAILVISION/GVCCS_V/test"))
    ap.add_argument("--pred_dir", type=Path, default=Path("outputs/ternary_5"))
    ap.add_argument("--video", default="00002")
    ap.add_argument("--frame", default="00176")
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    g = args.gvccs_dir
    pod = g / "per_object_data_age_5"
    video, frame = args.video, args.frame

    img_gray = ps.load_gray(g / "img_folder" / video / f"{frame}.jpg")
    img = np.stack([img_gray] * 3, axis=-1)

    obj_ids = pas.load_all_objects(pod, video, frame)
    colors = [pas._hex_to_rgb(pas._FLIGHT_PALETTE[i % len(pas._FLIGHT_PALETTE)])
              for i in range(len(obj_ids))]
    prompts = [pas.load_prompt_mask(pod, video, o, frame) for o in obj_ids]
    preds = [pas.load_pred_mask(args.pred_dir, video, o, frame) for o in obj_ids]

    prompt_vis, pred_vis = img.copy(), img.copy()
    n_pred = 0
    for rgb, pm, dm in zip(colors, prompts, preds):
        if pm is not None and np.any(pm > 0):
            prompt_vis = pas._blend_color(prompt_vis, pm, rgb, alpha=0.85)
        if dm is not None and dm.any():
            from scipy.ndimage import binary_dilation
            pred_vis = pas._blend_color(pred_vis,
                                        binary_dilation(dm, iterations=3),
                                        rgb, alpha=0.95)
            n_pred += 1

    titles = [
        "1. All-sky camera video\n(30-second frames)",
        f"2. Physics predicts contrail positions\n({len(obj_ids)} flights: ADS-B + weather)",
        f"3. SAM2 confirms or rejects\n({n_pred} confirmed, identity built in)",
    ]
    panels = [img_gray, prompt_vis, pred_vis]

    _w = ps.FIG_W_FULL
    fig, axes = plt.subplots(1, 3, figsize=(_w, _w / 3 + 0.42),
                             gridspec_kw={"wspace": 0.16})

    for ax, panel, title in zip(axes, panels, titles):
        kw = {"cmap": "gray", "vmin": 0, "vmax": 255} if panel.ndim == 2 else {}
        ax.imshow(panel, interpolation="bilinear", **kw)
        ps.clean_ax(ax)
        ax.set_title(title, fontsize=ps.FONT_COL_TITLE, pad=3, linespacing=1.35)

    # arrows between panels (figure coordinates)
    for x in (0.352, 0.672):
        fig.text(x, 0.44, "$\\rightarrow$", fontsize=15, ha="center",
                 va="center", color="#333333")

    fig.suptitle("A contrail cannot exist without a flight: "
                 "detection, tracking, and attribution in one pass",
                 fontsize=7.5, fontweight="bold", y=1.06)

    ps.save_figure(fig, "graphical_abstract", args.out_dir)


if __name__ == "__main__":
    main()
