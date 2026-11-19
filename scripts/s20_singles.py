"""Single-panel analysis from cached item scores (s17, s19, s15b): per benchmark x panel x method, AP, P@k,
key-error AP, error-type lifts, and the saturation of probability scores.

  python scripts/s20_singles.py  ->  results/singles_table.parquet, results/error_types_v2.parquet, results/saturation.parquet
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.data import matrices  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}
METHODS = ["mv", "crh", "catd", "ds_onecoin", "ds_onecoin_nf10", "accu_nodep", "ds_onecoin_lo", "ds_onecoin_nf10_lo", "accu_nodep_lo", "ds_full_lo",
           "accu_depen_lo", "glad_lo", "irt4pl", "irt4pl_miss", "crowdfm", "crowdfm_lo", "crowdfm_block100",
           "crowdfm_block100_lo", "entropy"]
SINGLES = ["full", "corr_dedup", "small_strong", "small_diverse", "offbench_strong"]


def main():
    from s4_panels import eligible_models
    from s9_error_types import types_of
    from s10_size_capability import key_mask
    models = eligible_models(None)
    rows, et_rows, sat_rows = [], [], []
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        t = matrices(models, bench)[0]
        y = t.defective.values.astype(int)
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        tys, tclean = types_of(bench, t)
        for cond in SINGLES:
            for m in METHODS:
                p = os.path.join(CACHE, f"{bench}__single_{cond}__{m}.npy")
                if not os.path.exists(p):
                    continue
                s = np.load(p)
                rows.append({"bench": bench, "panel": cond, "method": m, "ap": Mt.average_precision(s, y),
                             "pk": Mt.precision_at_k(s, y, PK[bench]), "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int))})
                for ty, pos in tys.items():
                    kk = pos | tclean
                    yy = pos[kk].astype(int)
                    ap = Mt.average_precision(s[kk], yy)
                    et_rows.append({"bench": bench, "panel": cond, "method": m, "type": ty, "n_pos": int(yy.sum()),
                                    "base": float(yy.mean()), "ap": ap, "lift": ap / float(yy.mean())})
                if m in ("ds_onecoin", "ds_onecoin_nf10", "accu_nodep", "crowdfm", "crowdfm_block100", "mv", "crh"):  # ds_onecoin_nf10: GSM8K, s53
                    # saturated = the audit score 1 - posterior(key) is exactly 0 or 1 in double precision;
                    # distinct values are counted exactly, so the two columns use the same definition
                    sat_rows.append({"bench": bench, "panel": cond, "method": m,
                                     "frac_saturated": float(((s == 0.0) | (s == 1.0)).mean()),
                                     "frac_score_one": float((s == 1.0).mean()),
                                     "frac_score_zero": float((s == 0.0).mean()),
                                     "distinct": int(len(np.unique(s))),
                                     "largest_tie_group": int(np.unique(s, return_counts=True)[1].max())})
    pd.DataFrame(rows).to_parquet(os.path.join(RES, "singles_table.parquet"))
    pd.DataFrame(et_rows).to_parquet(os.path.join(RES, "error_types_v2.parquet"))
    pd.DataFrame(sat_rows).to_parquet(os.path.join(RES, "saturation.parquet"))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for met in ("ap", "pk", "ap_key"):
        print(met)
        print(d.pivot_table(index=["bench", "panel"], columns="method", values=met).round(3).to_string())
    print(pd.DataFrame(sat_rows).pivot_table(index=["bench", "panel"], columns="method", values="frac_saturated").round(3).to_string())


if __name__ == "__main__":
    main()
