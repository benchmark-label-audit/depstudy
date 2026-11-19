"""Holm correction for the RQ4 dependence experiments.

Families follow Section 4.3 of the paper: research question x benchmark x metric, here split by design (cells,
dose ladder, MMLU split-half ladders, capability-matched curation). Within a family every method's primary effect is
one hypothesis: dependence_main and lineage_main (cells), top_minus_bottom_level (ladders), curated_minus_matched
(curation). Bootstrap p-values come from the two-way bootstrap (B = 500) of s35 / s37.

  python scripts/s48_rq4_holm.py  ->  results/rq4_holm.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def main():
    parts = []
    c = pd.read_csv(os.path.join(RES, "cells_v2_effects.csv"))
    c = c[c.effect.isin(["dependence_main", "lineage_main"])].assign(design="cells")
    parts.append(c)
    d = pd.read_csv(os.path.join(RES, "dose_effects.csv"))
    d = d[d.effect == "top_minus_bottom_level"].assign(design="dose")
    parts.append(d)
    for suf, name in (("", "splitdose_p400"), ("_p200", "splitdose_p200")):
        p = os.path.join(RES, f"splitdose_effects{suf}.csv")
        if os.path.exists(p):
            s = pd.read_csv(p)
            s = s[s.effect == "top_minus_bottom_level"].assign(design=name, bench="mmlu")
            parts.append(s)
    m = pd.read_csv(os.path.join(RES, "capmatch_effects.csv"))
    m = m[m.effect == "curated_minus_matched"].assign(design="capmatch_" + m.curation)
    parts.append(m)
    A = pd.concat(parts, ignore_index=True)
    # families hold the methods the paper reports: the key-anchored copy-test variants are diagnostics, and on GSM8K
    # DS-1 is the fixed-n_f version
    A = A[~A.method.isin(["accu_depen_key_lo", "accu_depen_nu_key_lo"])]
    A = A[~((A.bench == "gsm8k") & (A.method == "ds_onecoin_lo"))]
    A["p_boot"] = A.p_boot.clip(lower=1.0 / 500)  # resolution of B = 500
    A["p_holm"] = np.nan
    for _, idx in A.groupby(["design", "bench", "metric", "effect"]).groups.items():
        A.loc[idx, "p_holm"] = holm(A.loc[idx, "p_boot"].values)
    cols = ["design", "bench", "metric", "effect", "method", "estimate", "lo", "hi", "p_boot", "p_holm"]
    A[cols].to_csv(os.path.join(RES, "rq4_holm.csv"), index=False)
    pd.set_option("display.width", 250)
    print(A[cols].round(3).to_string())


if __name__ == "__main__":
    main()
