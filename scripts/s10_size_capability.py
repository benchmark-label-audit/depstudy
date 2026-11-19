"""Panel size x independently measured capability (exploratory; addresses selection circularity).

Capability of a model for evaluation benchmark E is its accuracy on OTHER benchmarks (MMLU <- mean of GSM8K and
HellaSwag; GSM8K <- MMLU; HellaSwag <- MMLU), so panel selection never uses E's labels. For each size n we compare
the top-n models by that capability with random n-model panels (10 repetitions). Metrics: AP and P@k for all
defects and for key errors only (MMLU wrong_groundtruth, GSM8K revised, HellaSwag wrong_answer).

  python scripts/s10_size_capability.py --workers 6  ->  results/ckpt_sizecap/*.json, results/size_capability.parquet
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
NFALSE = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}
SIZES = [10, 30, 100, 300, 1000]
CK = os.path.join(ROOT, "results", "ckpt_sizecap")
_D = {}


def _K(bench):
    from depstudy.data import n_options
    return n_options(bench)


def key_mask(bench, t):
    if bench == "mmlu":
        return (t.error_type == "wrong_groundtruth").values, (t.error_type == "ok").values
    if bench == "gsm8k":
        return (t.status == "revised").values, t.status.isin(["consensus", "verified"]).values
    w = t.wrong_answer.fillna(False).values.astype(bool)
    return w, ~w & ~t.multiple_correct.fillna(False).values.astype(bool)


def _init():
    from depstudy.data import matrices
    from s4_panels import eligible_models
    models = eligible_models(None)
    for b in ("mmlu", "gsm8k", "hellaswag"):
        _D[b] = matrices(models, b)


def capability():
    from depstudy.data import matrices
    from s4_panels import eligible_models
    models = eligible_models(None)
    acc = {}
    for b in ("mmlu", "gsm8k", "hellaswag"):
        t, mods, X, g, Y, M = matrices(models, b)
        acc[b] = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
    common = sorted(set(acc["mmlu"].index) & set(acc["gsm8k"].index) & set(acc["hellaswag"].index))
    cap = {"mmlu": (acc["gsm8k"][common] + acc["hellaswag"][common]) / 2, "gsm8k": acc["mmlu"][common],
           "hellaswag": acc["mmlu"][common]}
    return common, cap


def job(spec):
    from depstudy import methods as Mt
    bench, kind, n, rep, models = spec
    t, mods, X, g, Y, M = _D[bench]
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    Xp, Yp, Mp = X[:, c], Y[:, c], M[:, c]
    y_all = t.defective.values.astype(int)
    kpos, clean = key_mask(bench, t)
    meths = [("mv", lambda: Mt.majority_vote(Xp, g)), ("crh", lambda: Mt.crh(Xp, g)[0]),
             ("accu_nodep", lambda: Mt.accu_depen(Xp, g, NFALSE[bench], use_dependence=False, K=_K(bench))[0])]
    if len(c) <= 1000:
        meths.append(("ds_onecoin", lambda: Mt.dawid_skene_onecoin(Xp, g, K=_K(bench))[0]))
    if len(c) <= 300:
        meths.append(("irt4pl", lambda: Mt.irt_ceiling_lr(Xp, g, Yp, Mp)))
    rows = []
    for name, f in meths:
        s = f()
        keep = kpos | clean
        rows.append({"bench": bench, "kind": kind, "n": int(len(c)), "rep": rep, "method": name,
                     "ap": Mt.average_precision(s, y_all), "pk": Mt.precision_at_k(s, y_all, PK[bench]),
                     "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)),
                     "base_key": float(kpos[keep].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(CK, exist_ok=True)
    common, cap = capability()
    rng = np.random.default_rng(20261117)
    specs = []
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)
        for n in SIZES:
            specs.append((b, "top_capability", n, 0, order[:n]))
            for r in range(10):
                specs.append((b, "random", n, r, list(rng.choice(common, n, replace=False))))
        specs.append((b, "all", len(common), 0, common))
    k = lambda s: f"{s[0]}__{s[1]}__{s[2]}__{s[3]}"
    todo = [s for s in specs if not os.path.exists(os.path.join(CK, k(s) + ".json"))]
    print("to run", len(todo), "of", len(specs), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        futs = {ex.submit(job, s): s for s in todo}
        for i, f in enumerate(as_completed(futs)):
            with open(os.path.join(CK, k(futs[f]) + ".json.tmp"), "w") as fh:
                json.dump(f.result(), fh)
            os.replace(os.path.join(CK, k(futs[f]) + ".json.tmp"), os.path.join(CK, k(futs[f]) + ".json"))
            if i % 20 == 0:
                print(i, len(todo), k(futs[f]), flush=True)
    rows = [r for s in specs for r in json.load(open(os.path.join(CK, k(s) + ".json")))]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "results", "size_capability.parquet"))
    pd.set_option("display.width", 250)
    for metric in ("pk", "ap_key"):
        print(f"\n==== {metric} ====")
        print(df.groupby(["bench", "method", "kind", "n"])[metric].mean().unstack(["kind"]).round(3).to_string())


if __name__ == "__main__":
    main()
