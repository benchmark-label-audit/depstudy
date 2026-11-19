"""One-factor hyperparameter sensitivity.

Every method's settings are varied one at a time around the defaults used in the paper, on the full population and
the small-strong panel of MMLU and GSM8K. On GSM8K the number of false values n_f of DS-1, ACCU and GLAD is varied
as well, including the per-item |V_i| - 1 parameterization. DEPEN and IRT run on the small-strong panels only
(on the full population one DEPEN run takes 17 minutes).

  python scripts/s30_sensitivity.py --workers 6  ->  results/sensitivity.parquet
"""
import argparse
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

RES = os.path.join(ROOT, "results")
CK = os.path.join(RES, "ckpt_sens_v2")  # expected AP under random ties
PK = {"mmlu": 200, "gsm8k": 100}

# (method, parameter, values, default)
GRID = [
    ("ds_onecoin_lo", "prior_acc", [0.5, 0.6, 0.7, 0.8, 0.9], 0.7),
    ("ds_onecoin_lo", "iters", [10, 20, 50, 100, 200], 50),
    ("ds_full_lo", "smooth", [0.1, 0.3, 1.0, 3.0, 10.0], 1.0),
    ("ds_full_lo", "iters", [10, 20, 50, 100], 50),
    ("crh", "iters", [1, 5, 10, 20, 50], 20),
    ("catd", "alpha", [0.01, 0.05, 0.1, 0.2], 0.05),
    ("accu_nodep_lo", "iters", [2, 5, 10, 20, 50], 5),
    ("glad_lo", "prior", [0.01, 0.1, 1.0, 10.0], 1.0),
    ("glad_lo", "iters", [10, 20, 40, 100], 40),
    ("accu_depen_lo", "alpha", [0.05, 0.1, 0.2, 0.4], 0.2),
    ("accu_depen_lo", "c", [0.5, 0.65, 0.8, 0.95], 0.8),
    ("accu_depen_lo", "iters", [2, 5, 10, 20], 5),
    ("accu_depen_lo", "eps", [0.05, 0.1, 0.2, 0.4], 0.2),
    ("irt4pl", "min_r", [0.0, 0.1, 0.2, 0.3], 0.2),
]
GSM_NF = [("ds_onecoin_lo", "n_false", ["per_item", 1, 3, 10, 30, 100], 10),
          ("accu_nodep_lo", "n_false", [1, 3, 10, 30, 100], 10),
          ("glad_lo", "n_false", ["per_item", 3, 10, 30], 10)]
SMALL_ONLY = {"accu_depen_lo", "irt4pl"}


def run_one(method, param, value, X, g, Y, M, bench):
    from depstudy import methods as Mt
    K = 4 if bench == "mmlu" else None
    nf_default = 3 if bench == "mmlu" else 10
    kw = {param: value}
    if method == "ds_onecoin_lo":
        if param == "n_false":
            return Mt.dawid_skene_onecoin(X, g, K=K, logodds=True, n_false=None if value == "per_item" else value)[0]
        return Mt.dawid_skene_onecoin(X, g, K=K, logodds=True, n_false=None if bench == "mmlu" else nf_default, **kw)[0]
    if method == "ds_full_lo":
        return Mt.dawid_skene_full(X, g, 4, logodds=True, **kw)[0]
    if method == "crh":
        return Mt.crh(X, g, **kw)[0]
    if method == "catd":
        return Mt.catd(X, g, **kw)[0]
    if method == "accu_nodep_lo":
        nf = value if param == "n_false" else nf_default
        it = value if param == "iters" else 5
        return Mt.accu_depen(X, g, nf, iters=it, use_dependence=False, K=K, logodds=True)[0]
    if method == "accu_depen_lo":
        args = {"iters": 5, "alpha": 0.2, "c": 0.8}; args.update(kw)
        return Mt.accu_depen(X, g, nf_default, K=K, logodds=True, **args)[0]
    if method == "glad_lo":
        if param == "n_false":
            return Mt.glad(X, g, K=K, n_false=None if value == "per_item" else value)[0]
        return Mt.glad(X, g, K=K, n_false=None if bench == "mmlu" else nf_default, **kw)[0]
    if method == "irt4pl":
        return Mt.irt_ceiling_lr(X, g, Y, M, min_r=value)
    raise KeyError(method)


def job(task):
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    import s19_logodds as S19
    bench, panel, models, method, param, value, default = task
    t, mods, X, g, Y, M = S19.data(bench)
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    y = t.defective.values.astype(int)
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    t0 = time.time()
    s = np.asarray(run_one(method, param, value, X[:, c], g, Y[:, c], M[:, c], bench), np.float64)
    fin = s[np.isfinite(s)]
    s = np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))
    row = {"bench": bench, "panel": panel, "method": method, "param": param, "value": str(value), "default": str(default),
           "is_default": str(value) == str(default), "ap": Mt.average_precision_expected(s, y),
           "pk": Mt.precision_at_k_expected(s, y, PK[bench]),
           "ap_key": Mt.average_precision_expected(s[keep], kpos[keep].astype(int)), "sec": time.time() - t0}
    key = f"{bench}__{panel}__{method}__{param}__{value}"
    with open(os.path.join(CK, key + ".json"), "w") as fh:
        json.dump(row, fh)
    return key


def tasks():
    out = []
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    for bench in ("mmlu", "gsm8k"):
        for panel in ("small_strong", "full"):
            models = P[bench]["singles"][panel]["models"]
            grid = list(GRID) + (GSM_NF if bench == "gsm8k" else [])
            for method, param, values, default in grid:
                if bench == "gsm8k" and method == "ds_full_lo":
                    continue
                if panel == "full" and method in SMALL_ONLY:
                    continue
                for v in values:
                    out.append((bench, panel, models, method, param, v, default))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(CK, exist_ok=True)
    T = [t for t in tasks() if not os.path.exists(os.path.join(CK, f"{t[0]}__{t[1]}__{t[3]}__{t[4]}__{t[5]}.json"))]
    T.sort(key=lambda t: t[1] == "full")  # small panels first
    print("todo", len(T), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, t) for t in T]
        for i, f in enumerate(as_completed(futs)):
            k = f.result()
            if i % 20 == 0:
                print(i + 1, len(T), k, time.strftime("%H:%M:%S"), flush=True)
    rows = [json.load(open(os.path.join(CK, f))) for f in os.listdir(CK) if f.endswith(".json")]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RES, "sensitivity.parquet"))
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["bench", "method", "param", "value"], columns="panel", values="ap").round(3).to_string())


if __name__ == "__main__":
    main()
