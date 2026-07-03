#!/usr/bin/env python3
"""
Fig 2: Prompt encoding comparison on real data.

One flight from the GVCCS test set, shown at a single frame under the three
prompt encodings actually used in the paper:

  A  camera image + ground truth annotation
  B  binary prompt (presence only, 1-min window)
  C  age-weighted prompt (5-min window; bright = fresh, dim = old)
  D  age-weighted + negative signal (5-min; magenta = competing flights)

Default example: video 5, object 64, frame 104 — a young contrail whose
prompt crosses several other flights' prompts.

Usage:
    python paper/scripts/plot_prompt_comparison.py
    python paper/scripts/plot_prompt_comparison.py --video 00005 --obj 64 --frame 00104
"""

import argparse
from pathlib import Path

import sys as _sys

_sys.path.insert(0, str(Path(__file__).parent))
import paper_style as ps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def load_float(p):
    if not Path(p).exists():
        return None
    arr = np.array(Image.open(p).convert("L")).astype(np.float32)
    m = arr.max()
    return arr / m if m > 0 else arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvccs_dir", type=Path,
                    default=Path("/data/common/TRAILVISION/GVCCS_V/test"))
    ap.add_argument("--video", default="00005")
    ap.add_argument("--obj", type=int, default=64)
    ap.add_argument("--frame", default="00104")
    ap.add_argument("--pad", type=int, default=60)
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    g = args.gvccs_dir
    video, obj, frame = args.video, args.obj, args.frame

    img_gray = ps.load_gray(g / "img_folder" / video / f"{frame}.jpg")
    img = np.stack([img_gray] * 3, axis=-1)

    binary = load_float(g / "per_object_data" / video / str(obj) / f"{frame}_prompt.png")
    age = load_float(g / "per_object_data_age_5" / video / str(obj) / f"{frame}_prompt.png")
    union = load_float(g / "per_object_data_age_5" / video / f"{frame}_all_prompts_union.png")
    gt = load_float(g / "per_object_data_age_5" / video / str(obj) / f"{frame}_mask.png")

    ternary = np.where(age > 0, age, -(union if union is not None else 0))

    # Crop to the neighbourhood of the flight's prompt + nearby negative signal
    own = age > 0
    near = binary_dilation(own, iterations=110)
    region = own | ((np.abs(ternary) > 0) & near)
    # exactly square crop for consistent panels
    r0, r1, c0, c1 = ps.square_crop_bounds(region, pad=args.pad)

    def crop(a):
        return a[r0:r1, c0:c1]

    # Panel A: image + GT (dilated for visibility)
    vis_a = img.copy()
    if gt is not None:
        vis_a = ps.overlay_mask(vis_a, gt > 0.5, tuple(ps.GT_COLOR),
                                crop_w=700, frac=0.008)

    # Panel B: binary prompt — flat blue tint
    vis_b = ps.blend_prompt_blue(img, (binary > 0).astype(float) if binary is not None
                                 else np.zeros_like(img_gray, dtype=float))

    # Panel C: age-weighted prompt — blue gradient
    vis_c = ps.blend_prompt_blue(img, age)

    # Panel D: ternary — blue gradient + magenta negative
    vis_d = ps.blend_ternary_prompt(img, ternary)

    panels = [crop(v) for v in (vis_a, vis_b, vis_c, vis_d)]
    titles = ["Image + ground truth", "Binary (1 min)",
              "Age-weighted (5 min)", "+ negative signal (5 min)"]

    _w = ps.FIG_W_FULL
    fig, axes2d = plt.subplots(2, 2, figsize=(_w, _w + 0.35),
                               constrained_layout=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.04, hspace=0.14)
    axes = axes2d.ravel()
    for ax, panel, letter, title in zip(axes, panels, "ABCD", titles):
        ax.imshow(panel, interpolation="bilinear")
        ps.clean_ax(ax)
        ax.set_title(title, fontsize=ps.FONT_COL_TITLE, pad=2.5)
        ps.panel_tag(ax, letter)

    ps.overlay_legend(fig, [
        (ps.GT_HEX, "ground truth"),
        (ps.POS_HEX, "target prompt"),
        (ps.NEG_HEX, "competing prompt"),
    ])
    ps.save_figure(fig, "fig_prompts", args.out_dir)


if __name__ == "__main__":
    main()
