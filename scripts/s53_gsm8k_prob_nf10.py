"""DS-1 ranked by probability on GSM8K with the paper's fixed n_false = 10 (the log-odds row of Table 6 uses the same
n_false), on the four single panels. Writes item scores `gsm8k__single_<panel>__ds_onecoin_nf10.npy` and
results/gsm8k_prob_nf10.csv with tie-aware expected AP and key-error AP. A check recomputes the stored per-item-count
probability row of singles_v2.parquet with the same AP code."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
import s19_logodds as S19  # noqa: E402
from s10_size_capability import key_mask  # noqa: E402
from s28_newmethods import CACHE  # noqa: E402

RES = os.path.join(ROOT, "results")


def main():
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]["gsm8k"]["singles"]
    t, mods, X, g, Y, M = S19.data("gsm8k")
    idx = {m: i for i, m in enumerate(mods)}
    y = t.defective.values.astype(int)
    kpos, clean = key_mask("gsm8k", t)
    keep = kpos | clean
    ref = pd.read_parquet(os.path.join(RES, "singles_v2.parquet"))
    rows = []
    for panel in ("full", "corr_dedup", "small_strong", "small_diverse"):
        c = np.array([idx[m] for m in P[panel]["models"]])
        p = os.path.join(CACHE, f"gsm8k__single_{panel}__ds_onecoin_nf10.npy")
        if os.path.exists(p):
            s = np.load(p)
        else:
            s = np.asarray(Mt.dawid_skene_onecoin(X[:, c], g, K=None, n_false=10)[0], np.float64)
            np.save(p, s)
        old = np.load(os.path.join(CACHE, f"gsm8k__single_{panel}__ds_onecoin.npy"))
        r = ref[(ref.bench == "gsm8k") & (ref.panel == panel) & (ref.method == "ds_onecoin")].iloc[0]
        rows.append(dict(panel=panel, n_models=len(c),
                         ap=Mt.average_precision_expected(s, y),
                         ap_key=Mt.average_precision_expected(s[keep], kpos[keep].astype(int)),
                         distinct_scores=int(len(np.unique(s))),
                         check_old_ap=Mt.average_precision_expected(old, y), stored_old_ap=float(r.ap)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "gsm8k_prob_nf10.csv"), index=False)


if __name__ == "__main__":
    main()
