#!/usr/bin/env python3
"""
Fig: per-flight track-completeness distributions.

Panel A: survival curves — fraction of the 610 annotated flights whose track
         completeness meets or exceeds each threshold, for all five variants.
Panel B: paired per-flight comparison, binary baseline vs best variant, with
         win/tie/loss counts. Shows WHERE the aggregate differences come from.

Usage (from a directory containing evaluation/):
    python paper/scripts/plot_completeness_dist.py
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
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, default=Path("evaluation"))
    ap.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()

    comp = {}
    for var in ps.VARIANT_ORDER:
        df = pd.read_csv(args.eval_root / var / "tracking_per_flight.csv")
        comp[var] = df.set_index("flight_id")["completeness"]

    fig, (ax_e, ax_s) = plt.subplots(
        1, 2, figsize=(ps.FIG_W_FULL, 2.1),
        gridspec_kw={"wspace": 0.35, "width_ratios": [1.15, 1.0]})

    # ── Panel A: survival curves ─────────────────────────────────────────
    thr = np.linspace(0, 1, 201)
    for var in ps.VARIANT_ORDER:
        c = comp[var].values
        frac = [(c >= t).mean() for t in thr]
        ax_e.plot(thr, frac, color=ps.VARIANT_COLOR[var],
                  linestyle=ps.VARIANT_LS[var],
                  linewidth=1.5 if var == "ternary_5" else 1.0,
                  label=ps.VARIANT_SHORT[var],
                  zorder=3 if var == "ternary_5" else 2)
    ax_e.set_xlabel("Track completeness threshold")
    ax_e.set_ylabel("Fraction of flights $\\geq$ threshold")
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.legend(loc="lower left", fontsize=5.2, handlelength=2.0,
                labelspacing=0.3)

    # ── Panel B: paired scatter, binary vs best ──────────────────────────
    j = pd.concat([comp["base"].rename("b"), comp["ternary_5"].rename("t")],
                  axis=1).dropna()
    d = j["t"] - j["b"]
    win, loss = (d > 0.02).sum(), (d < -0.02).sum()
    tie = len(j) - win - loss
    up = d > 0.02
    dn = d < -0.02
    mid = ~(up | dn)
    ax_s.scatter(j["b"][mid], j["t"][mid], s=4, color="#bbbbbb", alpha=0.5,
                 linewidths=0, label=f"within $\\pm$0.02 ({tie})")
    ax_s.scatter(j["b"][up], j["t"][up], s=5, color=ps.VARIANT_COLOR["ternary_5"],
                 alpha=0.65, linewidths=0, label=f"best variant higher ({win})")
    ax_s.scatter(j["b"][dn], j["t"][dn], s=5, color=ps.PRED_HEX, alpha=0.65,
                 linewidths=0, label=f"binary higher ({loss})")
    ax_s.plot([0, 1], [0, 1], color="#888888", linewidth=0.6,
              linestyle=(0, (2, 2)), zorder=1)
    ax_s.set_xlabel("Completeness, binary (1 min)")
    ax_s.set_ylabel("Completeness, age-wtd.\n+ neg. (5 min)")
    ax_s.set_xlim(-0.03, 1.03)
    ax_s.set_ylim(-0.03, 1.03)
    ax_s.set_aspect("equal")
    ax_s.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
                fontsize=5.0, handletextpad=0.15, borderaxespad=0.0,
                labelspacing=0.25, markerscale=1.4, frameon=False)

    for ax, letter in ((ax_e, "A"), (ax_s, "B")):
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.text(-0.22, 1.04, letter, transform=ax.transAxes,
                fontsize=ps.FONT_PANEL_LETTER - 2, fontweight="bold",
                va="bottom", ha="left")

    ps.save_figure(fig, "fig_completeness_dist", args.out_dir)
    print(f"win={win} tie={tie} loss={loss} median diff={d.median():.3f}")


if __name__ == "__main__":
    main()
