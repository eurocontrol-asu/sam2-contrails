#!/usr/bin/env python3
"""
Fig: Average Precision vs IoU threshold for all five prompt variants
(single-column figure, sized at final print width).

Data source: evaluation/<variant>/evaluation_results.json
             (segmentation.ap_per_iou_threshold), falling back to
             paper/scripts/fig4_ap_data.csv when JSONs are unavailable.

Usage:
    python paper/scripts/plot_ablation.py [--eval_root evaluation]
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


def load_ap_curves(eval_root: Path):
    """Return {variant: (thresholds, ap_values)} from evaluation JSONs."""
    curves = {}
    for var in ps.VARIANT_ORDER:
        p = eval_root / var / "evaluation_results.json"
        if not p.exists():
            continue
        with open(p) as f:
            seg = json.load(f)["segmentation"]
        pairs = sorted((float(k), v) for k, v in seg["ap_per_iou_threshold"].items())
        thr = np.array([t for t, _ in pairs])
        ap = np.array([a for _, a in pairs])
        curves[var] = (thr, ap)
    return curves


def load_ap_csv(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    col = {
        "base": "Binary",
        "age_5": "Age-weighted 5min",
        "age_10": "Age-weighted 10min",
        "ternary_5": "Ternary 5min",
        "ternary_10": "Ternary 10min",
    }
    thr = df["IoU_threshold"].values
    return {v: (thr, df[c].values) for v, c in col.items() if c in df}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, default=Path("evaluation"))
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    curves = load_ap_curves(args.eval_root)
    if len(curves) < 5:
        curves = load_ap_csv(Path(__file__).parent / "fig4_ap_data.csv")

    fig, ax = plt.subplots(figsize=(ps.FIG_W_HALF, 2.6))

    for var in ps.VARIANT_ORDER:
        thr, apv = curves[var]
        ax.plot(
            thr,
            apv,
            color=ps.VARIANT_COLOR[var],
            linestyle=ps.VARIANT_LS[var],
            marker=ps.VARIANT_MARKER[var],
            markersize=3,
            linewidth=1.4 if var == "ternary_5" else 1.1,
            label=ps.VARIANT_SHORT[var],
            zorder=3 if var == "ternary_5" else 2,
        )

    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Average precision")
    ax.set_xlim(0.24, 0.76)
    ax.set_ylim(0, 0.75)
    ax.set_xticks(np.arange(0.25, 0.80, 0.10))
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.legend(loc="upper right", handlelength=2.2, labelspacing=0.35)

    ps.save_figure(fig, "fig_ablation_ap_curves", args.out_dir)


if __name__ == "__main__":
    main()
