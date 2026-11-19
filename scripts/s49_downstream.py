"""Downstream estimand of the verification budget.

For models j = 1..m, acc_j^0 is the accuracy under the shipped labels and acc_j^* under the corrected benchmark.
Correction applies one of three fixes to the verified defects: rekey (key errors receive the verified answer),
drop (the other defects are removed), or both (the paper's main definition, as GSM8K-Platinum does).
The accuracy error is E_0 = mean_j |acc_j^0 - acc_j^*|. Verifying the top-k items of an audit ranking applies the
same fix to the defects among those k items only, giving acc_j^k, and the share of the error removed is
1 - mean_j |acc_j^k - acc_j^*| / E_0. Random verification draws k items uniformly (1,000 draws; mean and 95% range).

  python scripts/s49_downstream.py  ->  results/downstream_estimand.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s33_rankshift as S33  # noqa: E402

RES = os.path.join(ROOT, "results")
# DS-1 as in the paper: K - 1 false values on MMLU, a fixed n_false = 10 on GSM8K
DS = {"mmlu": "ds_onecoin_lo", "gsm8k": "ds_onecoin_nf10_lo"}
AUDITS = [("DS", "small_strong"), ("crh", "small_strong"), ("DS", "full")]


def main():
    rows = []
    rng = np.random.default_rng(20261126)
    for bench in ("mmlu", "gsm8k"):
        t, mods, C0, C1, fix = S33.correctness(bench)
        n = len(t)
        k = S33.K[bench]
        for variant in ("both", "rekey", "drop"):
            f = np.array([x if (variant == "both" or x == variant) else None for x in fix], dtype=object)
            allsel = np.flatnonzero([x is not None for x in f])
            Cs, keep_s = S33.apply(C0, C1, f, allsel)
            s_star = S33.scores(Cs, keep_s)
            s0 = S33.scores(C0, np.ones(n, bool))
            e0 = float(np.abs(s0 - s_star).mean())

            def removed(sel):
                C, keep = S33.apply(C0, C1, f, sel)
                return float(1 - np.abs(S33.scores(C, keep) - s_star).mean() / e0)
            base = {"bench": bench, "variant": variant, "k": k, "n_fixed": int(len(allsel)), "E0": e0}
            for m, panel in AUDITS:
                m = DS[bench] if m == "DS" else m
                sc = np.load(os.path.join(S33.CACHE, f"{bench}__single_{panel}__{m}.npy"))
                sel = np.argsort(-sc, kind="stable")[:k]
                rows.append({**base, "audit": f"{m}@{panel}", "share_removed": removed(sel)})
            rr = np.array([removed(rng.choice(n, k, replace=False)) for _ in range(1000)])
            rows.append({**base, "audit": "random", "share_removed": float(rr.mean()),
                         "lo": float(np.percentile(rr, 2.5)), "hi": float(np.percentile(rr, 97.5))})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "downstream_estimand.csv"), index=False)
    pd.set_option("display.width", 250)
    print(d.round(4).to_string())


if __name__ == "__main__":
    main()
