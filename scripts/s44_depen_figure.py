"""Figure: shift of the log odds against the key that copy detection (DEPEN) applies relative to ACCU, by item group,
averaged over the panels of each dependence ladder.  python scripts/s44_depen_figure.py -> results/fig_depen.pdf"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from s42_figures_v2 import INK, MUTED, GRID, style  # noqa: E402

RES = os.path.join(ROOT, "results")


def main():
    q = pd.read_csv(os.path.join(RES, "depen_discount.csv"))
    q = q[q.family == "dose"]
    fig, ax = plt.subplots(figsize=(3.35, 1.6))
    benches = [("mmlu", "MMLU"), ("gsm8k", "GSM8K"), ("hellaswag", "HellaSwag")]
    groups = [("delta_key_errors", "key errors", "#2a78d6"), ("delta_other_defects", "ill-posed", "#9fb8d9"),
              ("delta_clean", "clean items", "#c9c8c4")]
    w = 0.26
    for k, (col, lab, c) in enumerate(groups):
        vals, errs = [], []
        for b, _ in benches:
            x = q[q.bench == b][col].dropna()
            vals.append(x.mean() if len(x) else np.nan); errs.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
        ax.bar(np.arange(3) + (k - 1) * w, vals, width=w - 0.03, color=c, label=lab, yerr=errs, error_kw={"lw": 0.7, "ecolor": MUTED})
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in benches])
    ax.set_ylabel("log-odds shift\n(Depen $-$ Accu)")
    style(ax)
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_depen.pdf")); fig.savefig(os.path.join(RES, "fig_depen.png"), dpi=200)


if __name__ == "__main__":
    main()
