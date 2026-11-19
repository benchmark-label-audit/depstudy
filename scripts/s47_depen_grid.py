"""Copy-model settings of DEPEN on capable panels.

The default copy model of Dong et al. (copy rate c = .8, first-round error rate eps = .2) flags almost every pair of
capable models as copiers. This grid varies c and eps jointly, with uniform and NonUni false values and with the copying
test anchored on the inferred truth or on the shipped key, and records the share of source pairs flagged
(P(dep) > .5), AP, and key-error AP on the strong single panels, the HELM populations, and the extreme levels of the
GSM8K and HellaSwag dependence ladders.

  python scripts/s47_depen_grid.py --workers 4  ->  results/depen_grid.csv
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RES = os.path.join(ROOT, "results")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
CS = [0.8, 0.9, 0.95, 0.99]
EPS = [0.05, 0.2]


def panels():
    out = []
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    P["hellaswag"] = json.load(open(os.path.join(RES, "panels_hellaswag.json")))["benches"]["hellaswag"]
    for b in ("mmlu", "gsm8k", "hellaswag"):
        out.append((b, "single_small_strong", b, P[b]["singles"]["small_strong"]["models"]))
        out.append((b, "single_small_diverse", b, P[b]["singles"]["small_diverse"]["models"]))
    DP = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"]
    for b in ("gsm8k", "hellaswag", "mmlu"):
        for lv in ("level0", "level5"):
            for r, p in enumerate(DP[b]["dose"][lv]["reps"][:5]):
                out.append((b, f"dose_{lv}_{r}", b, p["models"]))
    import s6_sources as S6
    for b in ("mmlu", "gsm8k"):
        mods = S6.load(b)[1]
        out.append((b, "src_helm_all_0", "src_" + b, [m for m in mods if m.startswith("helm:")]))
    return out


def job(task):
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    import s19_logodds as S19
    bench, tag, src, models = task
    t, mods, X, g, Y, M = S19.data(src)
    idx = {m: i for i, m in enumerate(mods)}
    c_ = np.array([idx[m] for m in models if m in idx])
    Xp = X[:, c_]
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    y = t.defective.values.astype(int)
    K = 4 if bench != "gsm8k" else None
    iu = np.triu_indices(len(c_), 1)
    rows = []
    s0 = Mt.accu_depen(Xp, g, NF[bench], use_dependence=False, K=K, logodds=True)[0]
    base = {"bench": bench, "tag": tag, "n": len(c_), "ap_accu": Mt.average_precision_expected(s0, y),
            "keyap_accu": Mt.average_precision_expected(s0[keep], kpos[keep].astype(int))}
    for c in CS:
        for eps in EPS:
            for nonuni in (False, True):
                for anchor in ("inferred", "key"):
                    kw = dict(c=c, eps=eps, nonuni=nonuni)
                    if anchor == "key":
                        kw["truth_override"] = g
                    s, post, acc, D = Mt.accu_depen(Xp, g, NF[bench], K=K, logodds=True, **kw)
                    rows.append({**base, "c": c, "eps": eps, "nonuni": nonuni, "anchor": anchor,
                                 "flag": float((D[iu] > 0.5).mean()), "ap": Mt.average_precision_expected(s, y),
                                 "keyap": Mt.average_precision_expected(s[keep], kpos[keep].astype(int))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    T = panels()
    print("panels", len(T), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(job, t) for t in T])):
            rows += f.result()
            print(i + 1, len(T), rows[-1]["bench"], rows[-1]["tag"], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "depen_grid.csv"), index=False)


if __name__ == "__main__":
    main()
