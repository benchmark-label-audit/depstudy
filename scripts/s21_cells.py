"""Lineage x dependence cells, all methods (probability-scored originals, log-odds variants, CrowdFM): cell means,
the dependence effect (dep_high - dep_low, averaged over the two lineage levels), the lineage effect
(lineage_high - lineage_low, averaged over the two dependence levels), and the interaction, with 95% intervals
from a bootstrap over the 20 panels of each cell (B = 2000), Holm-adjusted within each benchmark x metric.

  python scripts/s21_cells.py  ->  results/cells_effects.csv, results/cells_means.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RES = os.path.join(ROOT, "results")
METHODS = ["mv", "crh", "ds_onecoin_lo", "accu_nodep_lo", "accu_depen_lo", "ds_full_lo", "irt4pl", "crowdfm", "ds_onecoin"]
CELLS = ["lineage_high__dep_high", "lineage_high__dep_low", "lineage_low__dep_high", "lineage_low__dep_low"]


def load():
    g = pd.concat([pd.read_parquet(os.path.join(RES, "gate_scores.parquet")), pd.read_parquet(os.path.join(RES, "hellaswag_scores.parquet"))])
    l = pd.read_parquet(os.path.join(RES, "logodds_scores.parquet"))
    l = l[l.family == "gate"]
    c = pd.read_parquet(os.path.join(RES, "crowdfm_scores.parquet"))
    cols = ["bench", "kind", "condition", "rep", "method", "ap", "pk"]
    d = pd.concat([g[cols], l[cols], c[c.method == "crowdfm"][cols]])
    return d[(d.kind == "cell") & d.method.isin(METHODS)]


def effects(M):
    """M: dict cell -> array of rep values. Returns dep, lineage, interaction effects."""
    m = {k: v.mean() for k, v in M.items()}
    dep = ((m["lineage_high__dep_high"] - m["lineage_high__dep_low"]) + (m["lineage_low__dep_high"] - m["lineage_low__dep_low"])) / 2
    lin = ((m["lineage_high__dep_high"] - m["lineage_low__dep_high"]) + (m["lineage_high__dep_low"] - m["lineage_low__dep_low"])) / 2
    inter = (m["lineage_high__dep_high"] - m["lineage_high__dep_low"]) - (m["lineage_low__dep_high"] - m["lineage_low__dep_low"])
    return dep, lin, inter


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def main():
    d = load()
    rng = np.random.default_rng(7)
    rows, means = [], []
    for (bench, method), grp in d.groupby(["bench", "method"]):
        for metric in ("pk", "ap"):
            M = {c: grp[grp.condition == c].sort_values("rep")[metric].values for c in CELLS}
            if any(len(v) == 0 for v in M.values()):
                continue
            rnd = grp[grp.condition == "random"][metric].values
            for c in CELLS + ["random"]:
                v = M.get(c, rnd)
                means.append({"bench": bench, "method": method, "metric": metric, "cell": c, "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": len(v)})
            est = effects(M)
            boot = np.array([effects({c: v[rng.integers(0, len(v), len(v))] for c, v in M.items()}) for _ in range(2000)])
            for k, name in enumerate(("dependence", "lineage", "interaction")):
                se = boot[:, k].std(ddof=1)
                lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
                p = float(2 * norm.sf(abs(est[k]) / se)) if se > 0 else 1.0
                rows.append({"bench": bench, "method": method, "metric": metric, "effect": name, "estimate": est[k], "ci_lo": lo, "ci_hi": hi, "p": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for (b, m), grp in df.groupby(["bench", "metric"]):
        df.loc[grp.index, "p_holm"] = holm(grp.p.values)
    df.to_csv(os.path.join(RES, "cells_effects.csv"), index=False)
    pd.DataFrame(means).to_csv(os.path.join(RES, "cells_means.csv"), index=False)
    pd.set_option("display.width", 220)
    x = df[df.effect != "interaction"].copy()
    x["v"] = x.apply(lambda r: f"{r.estimate*100:+.1f} [{r.ci_lo*100:+.1f},{r.ci_hi*100:+.1f}]{'*' if r.p_holm < 0.05 else ''}", axis=1)
    print(x.pivot_table(index=["bench", "metric", "method"], columns="effect", values="v", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
