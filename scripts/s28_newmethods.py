"""Added baselines and parameterizations on every panel family.

New methods
  glad_lo            multi-class GLAD (Whitehill et al. 2009), log-odds score; n_false = 10 on GSM8K
  catd               CATD (Li et al. PVLDB 2014), chi-square lower-bound weights, support score
  entropy            answer entropy, a key-blind disagreement baseline
  ds_onecoin_nf10_lo one-coin DS with a fixed n_false = 10 on GSM8K (same as ACCU), instead of |V_i| - 1
  irt4pl_miss        IRT-4PL with unparseable GSM8K answers treated as missing, as every other method treats them

New panel
  single_offbench_strong  the 100 models most capable on the other benchmarks (the size x capability ranking);
                          every method, so that the main table holds a strong panel chosen without the
                          evaluation benchmark's answers

Panels and seeds come from s19.tasks() (gate singles and cells, size x capability, cross-source), so every tag
matches the existing item-score cache. Item scores go to $DEPSTUDY_DATA/item_scores/<bench>__<tag>__<method>.npy and one
checkpoint per panel to results/ckpt_new/.

  python scripts/s28_newmethods.py --workers 6 [--only singles|cells|sizecap|sources] [--collect]
      ->  results/newmethods_scores.parquet
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CK = os.path.join(RES, "ckpt_new")
CACHE = os.path.join(DATA_ROOT, "item_scores")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}
PK_SRC = {"mmlu": 200, "gsm8k": 50}
NEW = ["glad_lo", "catd", "entropy"]
GSM_EXTRA = ["ds_onecoin_nf10_lo"]
OLD_ALL = ["mv", "crh", "ds_onecoin", "ds_onecoin_lo", "ds_full_lo", "accu_nodep_lo", "accu_depen_lo", "irt4pl"]


def _K(bench):
    return 4 if bench in ("mmlu", "hellaswag") else None


def score(method, X, g, Y, M, bench):
    from depstudy import methods as Mt
    K = _K(bench)
    nf = NF[bench] if bench == "gsm8k" else None
    if method == "glad_lo":
        return Mt.glad(X, g, K=K, n_false=nf, logodds=True)[0]
    if method == "catd":
        return Mt.catd(X, g)[0]
    if method == "entropy":
        return Mt.answer_entropy(X, g)
    if method == "ds_onecoin_nf10_lo":
        return Mt.dawid_skene_onecoin(X, g, K=None, n_false=10, logodds=True)[0]
    if method == "irt4pl_miss":
        return Mt.irt_ceiling_lr(X, g, (X == g[:, None]).astype(np.float32), X >= 0)
    if method == "mv":
        return Mt.majority_vote(X, g)
    if method == "crh":
        return Mt.crh(X, g)[0]
    if method == "ds_onecoin":
        return Mt.dawid_skene_onecoin(X, g, K=K)[0]
    if method == "ds_onecoin_lo":
        return Mt.dawid_skene_onecoin(X, g, K=K, logodds=True)[0]
    if method == "ds_full_lo":
        return Mt.dawid_skene_full(X, g, 4, logodds=True)[0]
    if method == "accu_nodep_lo":
        return Mt.accu_depen(X, g, NF[bench], use_dependence=False, K=K, logodds=True)[0]
    if method == "accu_depen_lo":
        return Mt.accu_depen(X, g, NF[bench], K=K, logodds=True)[0]
    if method == "accu_depen_nu_lo":
        return Mt.accu_depen(X, g, NF[bench], K=K, logodds=True, nonuni=True)[0]
    if method == "accu_depen_key_lo":
        return Mt.accu_depen(X, g, NF[bench], K=K, logodds=True, truth_override=g)[0]
    if method == "accu_depen_nu_key_lo":
        return Mt.accu_depen(X, g, NF[bench], K=K, logodds=True, nonuni=True, truth_override=g)[0]
    if method == "irt4pl":
        return Mt.irt_ceiling_lr(X, g, Y, M)
    raise KeyError(method)


def job(task):
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    import s19_logodds as S19
    family, bench, kind, cond, rep, tag, models, methods = task
    src = ("src_" + bench) if family == "sources" else bench
    t, mods, X, g, Y, M = S19.data(src)
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    y = t.defective.values.astype(int)
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    pk = (PK_SRC if family == "sources" else PK)[bench]
    rows = []
    for m in methods:
        p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
        t0 = time.time()
        if os.path.exists(p):
            s = np.load(p); sec = float("nan")
        else:
            s = np.asarray(score(m, X[:, c], g, Y[:, c], M[:, c], bench), np.float64); sec = time.time() - t0
            np.save(p + ".tmp.npy", s); os.replace(p + ".tmp.npy", p)
        rows.append({"family": family, "bench": bench, "kind": kind, "condition": cond, "rep": rep, "n_models": int(len(c)),
                     "method": m, "ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, pk),
                     "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)), "sec": sec})
    key = f"{family}__{bench}__{kind}__{cond}__{rep}"
    with open(os.path.join(CK, key + ".json.tmp"), "w") as fh:
        json.dump(rows, fh)
    os.replace(os.path.join(CK, key + ".json.tmp"), os.path.join(CK, key + ".json"))
    return key, len(c)


def tasks():
    import s19_logodds as S19
    import s10_size_capability as S10
    out = []
    for family, bench, kind, cond, rep, tag, models, _ in S19.tasks():
        ms = NEW + (GSM_EXTRA if bench == "gsm8k" else [])
        if family == "gate" and kind == "single":
            ms = ms + (["irt4pl_miss"] if bench == "gsm8k" else [])
        if family in ("gate", "sizecap", "sources"):
            ms = ms + ["mv", "crh"]
        out.append((family, bench, kind, cond, rep, tag, models, ms))
    common, cap = S10.capability()
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)[:100]
        ms = OLD_ALL + NEW
        if b == "gsm8k":
            ms = [m for m in ms if m != "ds_full_lo"] + GSM_EXTRA + ["irt4pl_miss"]
        out.append(("gate", b, "single", "offbench_strong", 0, "single_offbench_strong", order, ms))
    return out


def group_of(t):
    if t[0] == "gate":
        return "singles" if t[2] == "single" else "cells"
    return t[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--heavy-workers", dest="heavy_workers", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    os.makedirs(CK, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if not a.collect:
        T = tasks()
        if a.only:
            T = [t for t in T if group_of(t) in a.only.split(",")]
        todo = [t for t in T if not os.path.exists(os.path.join(CK, f"{t[0]}__{t[1]}__{t[2]}__{t[3]}__{t[4]}.json"))]
        heavy = sorted([t for t in todo if len(t[6]) >= 1000], key=lambda t: -len(t[6]))
        light = [t for t in todo if len(t[6]) < 1000]
        print("tasks", len(T), "todo light", len(light), "heavy", len(heavy), flush=True)
        for group, w in ((light, a.workers), (heavy, a.heavy_workers)):
            if not group:
                continue
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(job, t) for t in group]
                for i, f in enumerate(as_completed(futs)):
                    key, n = f.result()
                    if i % 25 == 0 or n >= 1000:
                        print(i + 1, "/", len(group), key, n, time.strftime("%H:%M:%S"), flush=True)
    rows = [r for f in glob.glob(os.path.join(CK, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RES, "newmethods_scores.parquet"))
    print("rows", len(df), flush=True)


if __name__ == "__main__":
    main()
