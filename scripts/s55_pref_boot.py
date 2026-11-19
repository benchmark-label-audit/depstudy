"""Bootstrap interval of the AP difference between the rank average (CRH, DS-1, IRT-4PL) and IRT-4PL on Land and
Bikel's LLM-judge preference benchmarks against their weak reference labels (Section 5.1). Stratified item bootstrap
(positives and negatives resampled separately), B = 2,000. Writes results/pref_boot.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
import s43_tables as T  # noqa: E402
import s14_landbikel as L  # noqa: E402
import s41_pref_audit as P  # noqa: E402

B = 2000


def main():
    lb = pd.read_parquet(os.path.join(L.EEE, "landbikel_flagged.parquet"))
    rng = np.random.default_rng(1)
    out = {}
    for b, (short, K) in P.BENCH.items():
        d = pd.read_parquet(os.path.join(P.EEE, f"{b}.parquet")); d["c"] = d.answer.map(P.code)
        items = sorted(d.sample_id.unique()); models = sorted(d.model.unique())
        ii = {s: i for i, s in enumerate(items)}; jj = {m: j for j, m in enumerate(models)}
        X = np.full((len(items), len(models)), -1, np.int32); ok = d.c >= 0
        X[d.sample_id[ok].map(ii).values, d.model[ok].map(jj).values] = d.c[ok].values
        g = np.zeros(len(items), np.int32); M = X >= 0; Y = (X == 0).astype(np.float32)
        sub = lb[lb.benchmark == short]
        mis = sub[sub.weak_ref_label == "mislabel"].id.map(ii).dropna().astype(int).values
        y = np.zeros(len(items), int); y[mis] = 1
        sc = T.scores(X, g, Y, M, K)
        e, r = sc["ensemble"], sc["irt"]
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        diffs = []
        for _ in range(B):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            diffs.append(Mt.average_precision_expected(e[idx], y[idx]) - Mt.average_precision_expected(r[idx], y[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[short] = {"ap_rank_average": Mt.average_precision_expected(e, y), "ap_irt": Mt.average_precision_expected(r, y),
                      "diff": Mt.average_precision_expected(e, y) - Mt.average_precision_expected(r, y),
                      "lo": float(lo), "hi": float(hi), "n_pos": int(y.sum()), "n_items": int(len(y)), "B": B}
        print(short, out[short], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "pref_boot.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
