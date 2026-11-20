"""Review-budget view: share of key errors (and of all defects) found within the first k items of the review
queue, for selected panels and methods, from cached item scores. Light computation (sorting only).

Superseded: the paper's Figure 4 comes from s42_figures_v2.py. Outputs go to results/archive/ so that a rerun
leaves the paper's figure in place.

  python scripts/s24_budget.py  ->  results/archive/budget.csv, results/archive/fig_budget_s24.pdf
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results", "archive")
SEL = [("full", "crh"), ("full", "ds_onecoin"), ("full", "ds_onecoin_lo"), ("small_strong", "crh"), ("small_strong", "ds_onecoin_lo")]
KS = [50, 100, 200, 500, 1000]


def items(bench):
    f = sorted(glob.glob(os.path.join(DATA_ROOT, "matrix_cache", f"{bench}_*_items.parquet")))[0]
    return pd.read_parquet(f)


def main():
    from s10_size_capability import key_mask
    rows, curves = [], {}
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        t = items(bench)
        kpos, _ = key_mask(bench, t)
        y = t.defective.values.astype(bool)
        for panel, m in SEL:
            p = os.path.join(CACHE, f"{bench}__single_{panel}__{m}.npy")
            if not os.path.exists(p):
                continue
            s = np.load(p)
            order = np.argsort(-s, kind="stable")
            ck, cd = np.cumsum(kpos[order]), np.cumsum(y[order])
            curves[(bench, panel, m)] = ck / kpos.sum()
            for k in KS:
                rows.append({"bench": bench, "panel": panel, "method": m, "k": k, "share_items": k / len(s),
                             "key_recall": ck[k - 1] / kpos.sum(), "defect_recall": cd[k - 1] / y.sum(), "precision": cd[k - 1] / k})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "budget.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["bench", "panel", "method"], columns="k", values="key_recall").round(3).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from s22_figures import COL, LAB, INK, MUTED, GRID  # noqa: F401
    plt.rcParams.update({"font.size": 7, "legend.fontsize": 6.3, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.33, 2.55))
    n = 5631
    kk = np.arange(1, 1001)
    spec = [(("mmlu", "small_strong", "ds_onecoin_lo"), COL["ds_onecoin_lo"], "-", "DS-1, 100 strongest"),
            (("mmlu", "small_strong", "crh"), COL["crh"], "-", "CRH, 100 strongest"),
            (("mmlu", "full", "ds_onecoin_lo"), COL["ds_onecoin_lo"], "--", "DS-1, all 2,999"),
            (("mmlu", "full", "crh"), COL["crh"], "--", "CRH, all 2,999"),
            (("mmlu", "full", "ds_onecoin"), MUTED, ":", "DS-1 probability, all 2,999")]
    for key, c, ls, lab in spec:
        ax.plot(kk, curves[key][:1000], color=c, ls=ls, lw=1.3, label=lab)
    ax.plot(kk, kk / n, color=MUTED, lw=0.8, ls="-.", label="random order")
    ax.set_xlabel("items reviewed (of 5,631)"); ax.set_ylabel("share of MMLU wrong keys found")
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_budget_s24.pdf")); fig.savefig(os.path.join(RES, "fig_budget_s24.png"), dpi=200)


if __name__ == "__main__":
    main()
