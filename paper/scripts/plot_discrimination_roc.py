#!/usr/bin/env python3
"""
Fig: prompt discrimination — score histogram + ROC curve.

Panel A: per-flight maximum prediction score, split by whether the flight has
         a visible (annotated) contrail; log-scale counts; 0.5 threshold line.
Panel B: ROC of accepting a flight by its max score, with AUC and the 0.5
         operating point highlighted.

Reads scores directly from outputs/ternary_5/<video>.json and ground truth
presence from per_object_data_age_5 (same logic as analyze_stats.compute_roc).

Usage (from a directory containing outputs/ and per_object_data_age_5/):
    python paper/scripts/plot_discrimination_roc.py
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).parent))
import paper_style as ps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOGT_GRAY = "#888888"


def collect_scores(pred_dir: Path, pod_dir: Path):
    scores, labels = [], []
    for vid_dir in sorted(os.listdir(pod_dir)):
        pred_file = pred_dir / f"{vid_dir}.json"
        if not pred_file.exists():
            continue
        d = json.load(open(pred_file))
        by_obj = defaultdict(list)
        for ann in d["annotations"]:
            by_obj[int(ann["object_id"])].append(float(ann.get("score", 0)))
        for obj_folder in sorted(os.listdir(pod_dir / vid_dir)):
            obj_path = pod_dir / vid_dir / obj_folder
            if not obj_path.is_dir() or not obj_folder.isdigit():
                continue
            files = os.listdir(obj_path)
            if not any(f.endswith("_prompt.png") for f in files):
                continue
            scores.append(max(by_obj.get(int(obj_folder), [0.0])))
            labels.append(int(any(f.endswith("_mask.png") for f in files)))
    return np.array(scores), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, default=Path("outputs/ternary_5"))
    ap.add_argument("--pod_dir", type=Path, default=Path("per_object_data_age_5"))
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    scores, labels = collect_scores(args.pred_dir, args.pod_dir)
    print(f"{len(scores)} prompted flights: {labels.sum()} with GT, "
          f"{(1 - labels).sum()} without")

    fig, (ax_h, ax_r) = plt.subplots(
        1, 2, figsize=(ps.FIG_W_HALF, 1.7),
        gridspec_kw={"wspace": 0.5, "width_ratios": [1.2, 1.0]})
    for a in (ax_h, ax_r):
        a.tick_params(labelsize=6)
        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)

    # ── Panel A: histogram ────────────────────────────────────────────────
    bins = np.linspace(0, 1, 26)
    ax_h.hist(scores[labels == 0], bins=bins, color=NOGT_GRAY, alpha=0.75,
              label="no visible contrail")
    ax_h.hist(scores[labels == 1], bins=bins, color=ps.GT_HEX, alpha=0.8,
              label="visible contrail")
    ax_h.axvline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)))
    ax_h.set_yscale("log")
    ax_h.set_xlabel("Max prediction score")
    ax_h.set_ylabel("Flights")
    ax_h.set_xlim(0, 1)
    ax_h.legend(loc="upper right", fontsize=6, handlelength=1.2,
                borderaxespad=0.2)

    # ── Panel B: ROC ──────────────────────────────────────────────────────
    order = np.argsort(-scores)
    l = labels[order]
    P, N = l.sum(), (1 - l).sum()
    tpr = np.concatenate([[0], np.cumsum(l) / P])
    fpr = np.concatenate([[0], np.cumsum(1 - l) / N])
    auc = float(np.trapezoid(tpr, fpr))

    thr = 0.5
    tp = ((scores >= thr) & (labels == 1)).sum()
    fp = ((scores >= thr) & (labels == 0)).sum()
    op_tpr, op_fpr = tp / P, fp / N

    ax_r.plot(fpr, tpr, color=ps.VARIANT_COLOR["ternary_5"], linewidth=1.3)
    ax_r.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.6,
              linestyle=(0, (2, 2)))
    ax_r.plot([op_fpr], [op_tpr], "o", color=ps.PRED_HEX, markersize=4,
              markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    ax_r.annotate(f"score $\\geq$ 0.5\nTPR {op_tpr:.1%}\nFPR {op_fpr:.1%}",
                  (op_fpr, op_tpr), textcoords="offset points",
                  xytext=(12, -13), fontsize=6, ha="left", va="top")
    ax_r.text(0.97, 0.07, f"AUC = {auc:.2f}", transform=ax_r.transAxes,
              fontsize=6.5, ha="right", va="bottom")
    ax_r.set_xlabel("False-positive rate")
    ax_r.set_ylabel("True-positive rate")
    ax_r.set_xlim(-0.02, 1)
    ax_r.set_ylim(0, 1.02)

    for ax, letter in ((ax_h, "A"), (ax_r, "B")):
        ax.text(-0.28, 1.04, letter, transform=ax.transAxes,
                fontsize=ps.FONT_PANEL_LETTER, fontweight="bold",
                va="bottom", ha="left")

    ps.save_figure(fig, "fig_discrimination", args.out_dir)
    print(f"AUC={auc:.4f} TPR@0.5={op_tpr:.4f} FPR@0.5={op_fpr:.4f}")


if __name__ == "__main__":
    main()
