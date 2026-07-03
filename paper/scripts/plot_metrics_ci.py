#!/usr/bin/env python3
"""
Fig: headline metrics with video-level bootstrap 95% confidence intervals.

Four panels (detection rate, track completeness, temporal IoU, attribution
precision), each showing the five prompt variants as dot-and-whisker rows.
Reads paper/scripts/stats_results.json produced by analyze_stats.py.

Usage:
    python paper/scripts/plot_metrics_ci.py [--stats paper/scripts/stats_results.json]
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

PANELS = [
    ("detection_rate", "Detection rate"),
    ("completeness", "Track completeness"),
    ("temporal_iou", "Temporal IoU"),
    ("attribution_precision", "Attribution precision"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path,
                    default=Path(__file__).parent / "stats_results.json")
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text())["variants"]

    order = list(reversed(ps.VARIANT_ORDER))  # binary at bottom
    y = np.arange(len(order))

    fig, axes2d = plt.subplots(2, 2,
                               figsize=(ps.FIG_W_FULL, 2.6), sharey=True,
                               constrained_layout=False)
    fig.subplots_adjust(left=0.28, right=0.965, top=0.92, bottom=0.1,
                        wspace=0.1, hspace=0.75)
    axes = axes2d.ravel()

    for ax, (key, title) in zip(axes, PANELS):
        for yi, var in zip(y, order):
            p = stats[var]["point"][key]
            lo, hi = stats[var]["ci95"][key]
            c = ps.VARIANT_COLOR[var]
            ax.errorbar(p, yi, xerr=[[p - lo], [hi - p]], fmt="o",
                        color=c, ecolor=c, elinewidth=1.1, capsize=2.2,
                        markersize=4 if var == "ternary_5" else 3.2,
                        markeredgecolor="white", markeredgewidth=0.4)
            ax.annotate(f"{p:.3f}" if p < 1 else f"{p:.2f}",
                        (hi, yi), textcoords="offset points", xytext=(3.5, 0),
                        ha="left", va="center", fontsize=5.5, color="#333333")
        ax.set_title(title, fontsize=ps.FONT_COL_TITLE, pad=4)
        ax.grid(axis="x", linewidth=0.3, color="#dddddd")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6)
        ax.margins(x=0.12)

    for ax in (axes2d[0, 0], axes2d[1, 0]):
        ax.set_yticks(y)
        ax.set_yticklabels([ps.VARIANT_SHORT[v] for v in order], fontsize=ps.FONT_ROW_LABEL)
        ax.set_ylim(-0.6, len(order) - 0.2)

    ps.save_figure(fig, "fig_metrics_ci", args.out_dir)


if __name__ == "__main__":
    main()
