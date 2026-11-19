"""Leave-one-benchmark-out check of the rank-average default.

The paper's default averages the normalized ranks of CRH, DS-1, and IRT-4PL. Here every 2- and 3-method subset of
seven methods (MV, CRH, CATD, DS-1, GLAD, IRT-4PL, CrowdFM) is scored on the four single panels of each benchmark
(expected AP against all defects); for each held-out benchmark the subset with the highest mean AP on the other two
benchmarks is selected and evaluated on the held-out one, next to the fixed default and the best single method.

  python scripts/s50_lobo.py  ->  results/lobo_ensemble.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
import s39_stats_v2 as S39  # noqa: E402

RES = os.path.join(ROOT, "results")
PANELS = ["full", "corr_dedup", "small_strong", "small_diverse"]
POOL = ["mv", "crh", "catd", "DS", "glad_lo", "irt4pl", "crowdfm"]
DEFAULT = ("crh", "DS", "irt4pl")


def main():
    import s19_logodds as S19
    rank = {}
    ys = {}
    for b in ("mmlu", "gsm8k", "hellaswag"):
        t = S19.data(b)[0]
        ys[b] = t.defective.values.astype(int)
        for p in PANELS:
            for m in POOL:
                mm = S39.DS[b] if m == "DS" else m
                s = S39.load(b, f"single_{p}", mm)
                if s is not None:
                    rank[(b, p, m)] = pd.Series(s).rank(method="average").values / len(s)
    def ap(b, combo):
        vals = []
        for p in PANELS:
            parts = [rank.get((b, p, m)) for m in combo]
            if any(x is None for x in parts):
                return np.nan
            vals.append(Mt.average_precision_expected(np.mean(parts, axis=0), ys[b]))
        return float(np.mean(vals))
    combos = [c for k in (1, 2, 3) for c in itertools.combinations(POOL, k)]
    table = {(b, c): ap(b, c) for b in ys for c in combos}
    rows = []
    for held in ys:
        train = [b for b in ys if b != held]
        multi = [c for c in combos if len(c) > 1]
        score = {c: np.nanmean([table[(b, c)] for b in train]) for c in multi}
        best = max(score, key=score.get)
        single = max([c for c in combos if len(c) == 1], key=lambda c: table[(held, c)])
        rows.append({"held_out": held, "selected": "+".join(best), "selected_ap": table[(held, best)],
                     "default_ap": table[(held, DEFAULT)], "best_single": single[0], "best_single_ap": table[(held, single)],
                     "default_rank_among_multi_on_train": int(sorted(score.values(), reverse=True).index(score[DEFAULT]) + 1),
                     "n_multi": len(multi)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "lobo_ensemble.csv"), index=False)


if __name__ == "__main__":
    main()
