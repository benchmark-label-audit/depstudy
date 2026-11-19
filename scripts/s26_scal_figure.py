"""Scalability figure from results/scalability.csv: wall-clock time per method vs. panel size (log-log),
MMLU and HellaSwag; x marks a run above the 40-minute limit.

  python scripts/s26_scal_figure.py  ->  results/fig_scal.pdf
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from s22_figures import COL, LAB, MUTED, GRID  # noqa: E402

M = {"mv": "mv", "crh": "crh", "ds_onecoin": "ds_onecoin_lo", "ds_full": "ds_full_lo", "accu_depen": "accu_depen_lo",
     "irt4pl": "irt4pl", "crowdfm": "crowdfm"}


def main():
    d = pd.read_csv(os.path.join(ROOT, "results", "scalability.csv"))
    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6.2, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 2.35), sharey=True)
    for ax, b in zip(axes, ("mmlu", "hellaswag")):
        for m, key in M.items():
            x = d[(d.bench == b) & (d.method == m)].sort_values("n")
            if not len(x):
                continue
            ok = x[~x.timeout.astype(bool)]
            ax.plot(ok.n, ok.sec, color=COL[key], marker="o", ms=2.6, lw=1.2, label=LAB[key])
            to = x[x.timeout.astype(bool)]
            if len(to):
                ax.plot(to.n, [2400] * len(to), marker="x", color=COL[key], ms=5, mew=1.4, ls="none")
        ax.axhline(2400, color=MUTED, lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1000"])
        ax.set_title({"mmlu": "MMLU (5,631 items)", "hellaswag": "HellaSwag (10,036 items)"}[b])
        ax.set_xlabel("models")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("wall-clock seconds")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(os.path.join(ROOT, "results", "fig_scal.pdf")); fig.savefig(os.path.join(ROOT, "results", "fig_scal.png"), dpi=200)


if __name__ == "__main__":
    main()
