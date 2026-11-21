"""Interval of the share of accuracy error that verification removes (review item S11).

The share is 1 - mean_j |acc_j^k - acc_j^*| / mean_j |acc_j^0 - acc_j^*| over the models j of the population (s49).
Two 95% intervals are computed: an item bootstrap stratified by defect status (B = 1,000), which re-selects the top k
items in every resample and is the interval the paper reports, and a model bootstrap (B = 2,000). The audit ranking, the
budget k and the verified fixes stay those of s49 (DS-1 on the strong panel, main correction "both").

  python scripts/s60_budget_ci.py  ->  results/budget_ci.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s33_rankshift as S33  # noqa: E402
import s49_downstream as S49  # noqa: E402

RES = os.path.join(ROOT, "results")
B = 2000


def item_interval(C0, C1, fix, score, k, defective, rng, B_items=1000):
    """Item bootstrap stratified by defect status (as for AP): each resample keeps the item count and the defect count,
    verifies its k top-ranked entries, and recomputes the share. Weights w_i count the copies of item i; v_i counts the
    copies among the top k."""
    n = len(score)
    rek = np.array([f == "rekey" for f in fix]); drp = np.array([f == "drop" for f in fix])
    A_star = np.where(rek[:, None], C1, C0).astype(np.float64); A_star[drp] = 0.0
    C0d = C0.astype(np.float64)
    order = np.argsort(-score, kind="stable")
    pos, neg = np.flatnonzero(defective), np.flatnonzero(~defective)
    out = np.empty(B_items)
    for b in range(B_items):
        I = np.concatenate([pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]])
        w = np.bincount(I, minlength=n).astype(np.float64)
        wo = w[order]; cum = np.cumsum(wo)
        v = np.zeros(n); v[order] = np.clip(k - (cum - wo), 0, wo)
        a0 = (w @ C0d) / w.sum()
        a_star = (w @ A_star) / (w.sum() - w[drp].sum())
        ak = ((w - v) @ C0d + v @ A_star) / (w.sum() - v[drp].sum())
        out[b] = 1 - np.abs(ak - a_star).mean() / np.abs(a0 - a_star).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    rows = []
    rng = np.random.default_rng(20261122)
    for bench in ("mmlu", "gsm8k"):
        t, mods, C0, C1, fix = S33.correctness(bench)
        n = len(t)
        k = S33.K[bench]
        allsel = np.flatnonzero([x is not None for x in fix])
        Cs, keep_s = S33.apply(C0, C1, fix, allsel)
        s_star = S33.scores(Cs, keep_s)
        e0 = np.abs(S33.scores(C0, np.ones(n, bool)) - s_star)
        for m, panel in (("DS", "small_strong"), ("crh", "small_strong")):
            m = S49.DS[bench] if m == "DS" else m
            sc = np.load(os.path.join(S33.CACHE, f"{bench}__single_{panel}__{m}.npy"))
            sel = np.argsort(-sc, kind="stable")[:k]
            C, keep = S33.apply(C0, C1, fix, sel)
            ek = np.abs(S33.scores(C, keep) - s_star)
            est = 1 - ek.mean() / e0.mean()
            J = rng.integers(0, len(e0), size=(B, len(e0)))
            boot = 1 - ek[J].mean(1) / e0[J].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ilo, ihi = item_interval(C0, C1, fix, sc, k, t.defective.values.astype(bool), rng)
            rows.append({"bench": bench, "audit": f"{m}@{panel}", "k": k, "share_removed": float(est), "lo_models": float(lo), "hi_models": float(hi),
                         "lo_items": ilo, "hi_items": ihi, "n_models": int(len(e0))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "budget_ci.csv"), index=False)


if __name__ == "__main__":
    main()
