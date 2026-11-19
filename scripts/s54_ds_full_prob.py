"""Full Dawid-Skene ranked by probability (1 - posterior of the key) against log-odds ranking on the MMLU full and
strong panels: the probability-saturation check of Section 5.4 for the confusion-matrix model. Writes
results/ds_full_prob.csv and item scores `mmlu__single_<panel>__ds_full.npy`."""
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
from s28_newmethods import CACHE  # noqa: E402

RES = os.path.join(ROOT, "results")


def main():
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]["mmlu"]["singles"]
    t, mods, X, g, Y, M = S19.data("mmlu")
    idx = {m: i for i, m in enumerate(mods)}
    y = t.defective.values.astype(int)
    rows = []
    for panel in ("full", "small_strong"):
        c = np.array([idx[m] for m in P[panel]["models"]])
        p = os.path.join(CACHE, f"mmlu__single_{panel}__ds_full.npy")
        s = np.load(p) if os.path.exists(p) else np.asarray(Mt.dawid_skene_full(X[:, c], g, 4)[0], np.float64)
        np.save(p, s)
        lo = np.load(os.path.join(CACHE, f"mmlu__single_{panel}__ds_full_lo.npy"))
        rows.append(dict(bench="mmlu", panel=panel, ap_prob=Mt.average_precision_expected(s, y),
                         ap_logodds=Mt.average_precision_expected(lo, y), distinct_prob=int(len(np.unique(s)))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "ds_full_prob.csv"), index=False)


if __name__ == "__main__":
    main()
