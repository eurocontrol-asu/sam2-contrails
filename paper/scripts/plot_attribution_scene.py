#!/usr/bin/env python3
"""
Fig 3: Attribution in action.

Three-panel figure for a single video frame containing multiple simultaneous
contrails.  Uses the paper-wide semantic colors (cyan = prompt, vermilion =
prediction); flight identity is carried by numbered markers that match
between panels B and C.

  Panel A: raw camera image
  Panel B: all prompt footprints (cyan) with flight numbers
  Panel C: all predictions (vermilion) with matching flight numbers

Default: video 2, frame 176 (12 prompted flights, 8 with predictions), ternary_5.

Usage:
    python paper/scripts/plot_attribution_scene.py
    python paper/scripts/plot_attribution_scene.py \\
        --video 00002 --frame 00176 --pred_dir outputs/ternary_5
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
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def load_all_objects(pod_dir, video, frame):
    """Find all object folders that have a prompt for the given frame."""
    pod = Path(pod_dir) / video
    if not pod.exists():
        return []
    objects = []
    for obj_dir in sorted(pod.iterdir()):
        if not obj_dir.is_dir() or not obj_dir.name.isdigit():
            continue
        prompt_path = obj_dir / f"{frame}_prompt.png"
        if prompt_path.exists():
            objects.append(int(obj_dir.name))
    return objects


def load_prompt_mask(pod_dir, video, obj_id, frame):
    p = Path(pod_dir) / video / str(obj_id) / f"{frame}_prompt.png"
    if not p.exists():
        return None
    arr = np.array(Image.open(p).convert("L")).astype(np.float32)
    m = arr.max()
    return arr / m if m > 0 else arr  # normalized [0,1], preserving age gradient


def load_pred_mask(pred_dir, video, obj_id, frame):
    pred_file = Path(pred_dir) / f"{video}.json"
    if not pred_file.exists():
        return None
    with open(pred_file) as f:
        d = json.load(f)
    for ann in d["annotations"]:
        if ann["object_id"] == obj_id and ann["frame_name"] == frame:
            return mask_utils.decode(ann["segmentation"]).astype(bool)
    return None


def make_figure(gvccs_dir, pred_dir, video, frame, out_dir):
    gvccs = Path(gvccs_dir)
    pod   = gvccs / "per_object_data_age_5"

    # Load camera image
    img_path = gvccs / "img_folder" / video / f"{frame}.jpg"
    img_gray = ps.load_gray(img_path)
    img = np.stack([img_gray, img_gray, img_gray], axis=-1)  # grayscale as RGB for overlay

    # Discover all objects prompted in this frame
    obj_ids = load_all_objects(pod, video, frame)
    if not obj_ids:
        print(f"WARNING: no prompted objects found for video={video} frame={frame}")
        return

    print(f"  Found {len(obj_ids)} prompted objects: {obj_ids}")

    # Load all prompt and prediction masks
    prompts  = [load_prompt_mask(pod, video, oid, frame) for oid in obj_ids]
    preds    = [load_pred_mask(pred_dir, video, oid, frame) for oid in obj_ids]

    n_preds = sum(1 for p in preds if p is not None)
    print(f"  Predictions found: {n_preds}/{len(obj_ids)}")

    # ── Build composite overlays ──────────────────────────────────────────────
    # Paper-wide semantic colors: every prompt cyan, every prediction
    # vermilion, drawn as solid outlined strokes exactly like the sibling
    # figures. Flight identity is carried by the numbered markers instead.
    prompt_vis = img.copy()
    pred_vis   = img.copy()

    for prompt_mask, pred_mask in zip(prompts, preds):
        if prompt_mask is not None and np.any(prompt_mask > 0):
            prompt_vis = ps.overlay_mask(prompt_vis, prompt_mask > 0,
                                         tuple(ps.POS_COLOR),
                                         crop_w=1024, frac=0.004)
        if pred_mask is not None and pred_mask.any():
            pred_vis = ps.overlay_mask(pred_vis, pred_mask,
                                       tuple(ps.PRED_COLOR),
                                       crop_w=1024, frac=0.004)

    # ── Numeric labels: consecutive numbers for flights visible in B or C ────
    def _candidates(mask):
        """Candidate label positions along the stroke (quantiles of the
        dominant axis), so overlapping labels can spread out."""
        ys, xs = np.nonzero(mask > 0.15 if mask.dtype != bool else mask)
        if len(ys) == 0:
            return []
        order = np.argsort(ys if np.ptp(ys) >= np.ptp(xs) else xs)
        qs = [0.5, 0.3, 0.7, 0.15, 0.85, 0.02, 0.98]
        idx = [order[int(q * (len(order) - 1))] for q in qs]
        return [(float(xs[i]), float(ys[i])) for i in idx]

    label_info = []  # (number, prompt_cands, pred_cands)
    num = 0
    for prompt_mask, pred_mask in zip(prompts, preds):
        p_c = _candidates(prompt_mask) if prompt_mask is not None else []
        d_c = _candidates(pred_mask) if (pred_mask is not None and pred_mask.any()) else []
        if not p_c and not d_c:
            continue
        num += 1
        label_info.append((num, p_c, d_c))

    # ── Figure layout: 3 panels side by side, full double-column width ───────
    _w = ps.FIG_W_FULL
    _h = round(_w / 3.08, 2)
    fig, axes = plt.subplots(1, 3, figsize=(_w, _h),
                              gridspec_kw={"wspace": 0.04})

    panel_letters = ["A", "B", "C"]
    panel_titles  = ["Camera image", "Per-flight prompts",
                     "Attributed predictions"]
    panels        = [img_gray, prompt_vis, pred_vis]

    for ax, letter, title, panel_img in zip(axes, panel_letters, panel_titles,
                                            panels):
        kw = {"cmap": "gray", "vmin": 0, "vmax": 255} if panel_img.ndim == 2 else {}  # already stretched by load_gray
        ax.imshow(panel_img, interpolation="bilinear", **kw)
        ps.clean_ax(ax)
        ax.set_title(title, fontsize=ps.FONT_COL_TITLE, pad=2.5)
        ps.panel_tag(ax, letter)

    H, W = img_gray.shape
    margin = 50  # ≥ badge radius (~40 data px) so circles are never clipped
    placed = {1: [], 2: []}  # per-panel list of placed label centers

    def _place(ax_idx, cands):
        """Pick the candidate farthest from already placed labels."""
        best_xy, best_d = None, -1.0
        for x, y in cands:
            x = float(np.clip(x, margin, W - margin))
            y = float(np.clip(y, margin, H - margin))
            d = min((np.hypot(x - px, y - py) for px, py in placed[ax_idx]),
                    default=np.inf)
            if d > best_d:
                best_d, best_xy = d, (x, y)
            if d == np.inf:
                break
        placed[ax_idx].append(list(best_xy))
        return best_xy

    def _spread(labels, min_d=95):
        """Push overlapping markers apart so no two circles collide, even
        when the margin clamp funnels several onto the same edge."""
        for _ in range(60):
            moved = False
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    dx = labels[j][0] - labels[i][0]
                    dy = labels[j][1] - labels[i][1]
                    d = float(np.hypot(dx, dy))
                    if d < min_d:
                        if d < 1e-6:
                            dx, dy, d = 0.0, 1.0, 1.0
                        shift = (min_d - d) / 2
                        ux, uy = dx / d, dy / d
                        labels[i][0] -= ux * shift; labels[i][1] -= uy * shift
                        labels[j][0] += ux * shift; labels[j][1] += uy * shift
                        moved = True
            for lab in labels:
                lab[0] = float(np.clip(lab[0], margin, W - margin))
                lab[1] = float(np.clip(lab[1], margin, H - margin))
            if not moved:
                break

    # Neutral numbered markers (white circle, dark text) so the semantic
    # overlay colors stay unambiguous; the same number marks the same
    # flight in panels B and C. Place all markers first, then relax
    # collisions before drawing.
    to_draw = {1: [], 2: []}
    for number, p_c, d_c in label_info:
        for ax_idx, cands in ((1, p_c), (2, d_c)):
            if not cands:
                continue
            _place(ax_idx, cands)
            to_draw[ax_idx].append(number)

    for ax_idx, ax in ((1, axes[1]), (2, axes[2])):
        _spread(placed[ax_idx])
        for (x, y), number in zip(placed[ax_idx], to_draw[ax_idx]):
            ax.text(x, y, str(number), fontsize=5.5, fontweight="bold",
                    color="#222222", ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white",
                              ec="#555555", lw=0.5, alpha=0.95))

    ps.overlay_legend(fig, [
        (ps.POS_HEX, "prompt"),
        (ps.PRED_HEX, "prediction"),
    ])
    ps.save_figure(fig, "fig_attribution_scene", out_dir)
    plt.close(fig)
    n_pred = sum(1 for _, _, d in label_info if d)
    rejected = [str(n) for n, _, d in label_info if not d]
    print(f"  {len(label_info)} flights labelled; {n_pred} with predictions.")
    print(f"  Flights without prediction (prompt rejection): {', '.join(rejected) or 'none'}")


def main():
    parser = argparse.ArgumentParser(description="Attribution in action (Fig 3)")
    parser.add_argument("--gvccs_dir", type=Path,
                        default=Path("/data/common/TRAILVISION/GVCCS_V/test"))
    parser.add_argument("--pred_dir", type=Path, default=Path("outputs/ternary_5"))
    parser.add_argument("--video",  default="00002")
    parser.add_argument("--frame",  default="00176")
    parser.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    print(f"Generating attribution scene: video={args.video}, frame={args.frame}")
    make_figure(args.gvccs_dir, args.pred_dir, args.video, args.frame, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
