"""Diagnosis of the IRT-4PL replication gap on Land and Bikel's records (review item S9).

(a) Item sampling: 95% bootstrap interval of the Spearman correlation between our IRT-4PL scores and their released
    delta_li on the released items (B = 2,000).
(b) Model panel: Land and Bikel fit about 110 models, the Every Eval Ever records hold 66 to 69. We refit our IRT-4PL
    on random 60% subsets of the models (the overlap of 66 of 110) and on disjoint halves, and report the Spearman
    correlation of the refitted scores with the full-panel scores and between the halves, on the same released items
    (50 draws each).
(c) Reference provenance: their weak-reference mislabels were reviewed either because their delta_li was positive or
    because of another flag; top-50 hits of IRT-4PL and majority voting are split by this origin.

  python scripts/s62_irt_gap.py  ->  results/irt_gap.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
import s14_landbikel as L  # noqa: E402

RES = os.path.join(ROOT, "results")


def main():
    lb = pd.read_parquet(os.path.join(L.EEE, "landbikel_flagged.parquet"))
    rng = np.random.default_rng(6262)
    out = {}
    for bench, short in L.BMAP.items():
        items, models, X, g, Y, M = L.matrix(bench)
        k = g >= 0
        items = [s for s, kk in zip(items, k) if kk]; X, g, Y, M = X[k], g[k], Y[k], M[k]
        key = pd.Series(range(len(items)), index=[L.question_key(s) for s in items]); key = key[~key.index.duplicated(keep=False)]
        sub = lb[lb.benchmark == short].copy(); sub["row"] = [key.get(L.question_key(c)) for c in sub.sample_content]
        sub = sub.dropna(subset=["row"]); sub["row"] = sub.row.astype(int)
        has = sub.delta_li.notna().values
        rows, delta = sub.row.values[has], sub.delta_li.values[has]
        full = Mt.irt_ceiling_lr(X, g, Y, M)
        rho = spearmanr(full[rows], delta)[0]
        boot = []
        for _ in range(2000):
            j = rng.integers(0, len(rows), len(rows))
            if len(np.unique(delta[j])) > 2:
                boot.append(spearmanr(full[rows[j]], delta[j])[0])
        n_m = X.shape[1]
        sub_full, sub_theirs, halves = [], [], []
        for _ in range(50):
            pick = rng.choice(n_m, int(round(0.6 * n_m)), replace=False)
            s = Mt.irt_ceiling_lr(X[:, pick], g, Y[:, pick], M[:, pick])
            sub_full.append(spearmanr(s[rows], full[rows])[0]); sub_theirs.append(spearmanr(s[rows], delta)[0])
            perm = rng.permutation(n_m); a_, b_ = perm[: n_m // 2], perm[n_m // 2:]
            sa = Mt.irt_ceiling_lr(X[:, a_], g, Y[:, a_], M[:, a_]); sb = Mt.irt_ceiling_lr(X[:, b_], g, Y[:, b_], M[:, b_])
            halves.append(spearmanr(sa[rows], sb[rows])[0])
        mv = Mt.majority_vote(X, g)
        mis = sub[sub.weak_ref_label == "mislabel"]
        prov = {}
        for origin, m_ in (("delta_gt0", mis[mis.flagged_by_delta_li_gt_0.fillna(False)]), ("other_flag", mis[~mis.flagged_by_delta_li_gt_0.fillna(False)])):
            y = np.zeros(len(items), int); y[m_.row.values] = 1
            prov[origin] = {"n": int(len(m_)), "irt_top50": float(Mt.precision_at_k_expected(full, y, 50) * 50),
                            "mv_top50": float(Mt.precision_at_k_expected(mv, y, 50) * 50)}
        q = lambda v: [float(np.nanpercentile(v, p)) for p in (10, 50, 90)]  # noqa: E731
        out[short] = {"models": int(n_m), "scored_items": int(len(rows)), "rho_ours_theirs": float(rho),
                      "rho_item_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                      "rho_subset60_vs_full_p10_p50_p90": q(sub_full), "rho_subset60_vs_theirs_p10_p50_p90": q(sub_theirs),
                      "rho_disjoint_halves_p10_p50_p90": q(halves), "subjective_among_scored": int((sub.weak_ref_label.values[has] == "subjective").sum()),
                      "mislabel_provenance": prov}
        print(short, json.dumps(out[short], indent=1), flush=True)
    json.dump(out, open(os.path.join(RES, "irt_gap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
