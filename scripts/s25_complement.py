"""Do methods find the same defects? Overlap of the top-k review queues across methods, and rank-average
ensembles, on the single panels, from cached item scores.

  python scripts/s25_complement.py  ->  results/complement.csv
"""
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}
M = ["crh", "ds_onecoin_lo", "irt4pl", "crowdfm"]


def main():
    from s10_size_capability import key_mask
    rows = []
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        t = pd.read_parquet(sorted(glob.glob(os.path.join(DATA_ROOT, "matrix_cache", f"{bench}_*_items.parquet")))[0])
        y = t.defective.values.astype(int)
        kpos, clean = key_mask(bench, t); keep = kpos | clean
        k = PK[bench]
        for panel in ("full", "small_strong"):
            S = {m: np.load(os.path.join(CACHE, f"{bench}__single_{panel}__{m}.npy")) for m in M
                 if os.path.exists(os.path.join(CACHE, f"{bench}__single_{panel}__{m}.npy"))}
            top = {m: set(np.argsort(-s, kind="stable")[:k]) for m, s in S.items()}
            for a, b in itertools.combinations(S, 2):
                inter = top[a] & top[b]
                rows.append({"bench": bench, "panel": panel, "kind": "overlap", "name": f"{a}&{b}", "value": len(inter) / k,
                             "defects_in_both": int(y[list(inter)].sum()) if inter else 0,
                             "defects_only_a": int(y[list(top[a] - top[b])].sum()), "defects_only_b": int(y[list(top[b] - top[a])].sum())})
            R = {m: rankdata(s) / len(s) for m, s in S.items()}
            for combo in (("ds_onecoin_lo", "irt4pl"), ("crh", "irt4pl"), ("crh", "ds_onecoin_lo", "irt4pl")):
                if all(c in R for c in combo):
                    e = np.mean([R[c] for c in combo], axis=0)
                    rows.append({"bench": bench, "panel": panel, "kind": "ensemble", "name": "+".join(combo), "ap": Mt.average_precision(e, y),
                                 "ap_key": Mt.average_precision(e[keep], kpos[keep].astype(int)), "pk": Mt.precision_at_k(e, y, k)})
            for m, s in S.items():
                rows.append({"bench": bench, "panel": panel, "kind": "single", "name": m, "ap": Mt.average_precision(s, y),
                             "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)), "pk": Mt.precision_at_k(s, y, k)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "complement.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df[df.kind == "overlap"][["bench", "panel", "name", "value", "defects_in_both", "defects_only_a", "defects_only_b"]].round(2).to_string(index=False))
    print(df[df.kind != "overlap"].pivot_table(index=["bench", "panel"], columns="name", values="ap").round(3).to_string())
    print(df[df.kind != "overlap"].pivot_table(index=["bench", "panel"], columns="name", values="ap_key").round(3).to_string())


if __name__ == "__main__":
    main()
