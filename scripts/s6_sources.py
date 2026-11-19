"""Exploratory cross-source analysis (post hoc): a small panel of strong models from independent vendors (HELM)
versus panels from the open-model leaderboard (v1), and their unions.

  python scripts/s6_sources.py --workers 8  ->  results/sources_scores.parquet, results/sources_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PK = {"mmlu": 200, "gsm8k": 50}
NFALSE = {"mmlu": 3, "gsm8k": 10}
_D = {}


def _K(bench):
    from depstudy.data import n_options
    return n_options(bench)


def load(bench):
    """Combined v1 + HELM matrices on their common items (cached on D:)."""
    from depstudy.data import CACHE
    p = os.path.join(CACHE, f"sources_{bench}.pkl")
    if os.path.exists(p):
        import pickle
        return pickle.load(open(p, "rb"))
    out = _load(bench)
    import pickle
    os.makedirs(CACHE, exist_ok=True)
    with open(p + ".tmp", "wb") as fh:
        pickle.dump(out, fh)
    os.replace(p + ".tmp", p)
    return out


def _load(bench):
    from depstudy.data import assemble, helm_long, long_table, truth
    from s4_panels import eligible_models
    v1 = eligible_models(None)
    t = truth(bench)
    lt_v1 = long_table(v1, bench)
    lt_h = helm_long("helm_" + bench)
    items = set(lt_h.item_id) & set(lt_v1.item_id)
    t = t[t.item_id.isin(items)].reset_index(drop=True)
    lt = pd.concat([lt_v1, lt_h], ignore_index=True)
    models = v1 + sorted(lt_h.model.unique())
    tt, mods, X, g, Y, M = assemble(t, lt, models, "mc4" if bench == "mmlu" else "num", 0.0)
    return tt, mods, X, g, Y, M


def _init():
    for b in ("mmlu", "gsm8k"):
        _D[b] = load(b)


def run(spec):
    from depstudy import methods as Mt
    bench, cond, rep, models = spec
    t, mods, X, g, Y, M = _D[bench]
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    Xp, Yp, Mp = X[:, c], Y[:, c], M[:, c]
    y = t.defective.values.astype(int)
    out = []
    meths = [("mv", lambda: Mt.majority_vote(Xp, g)), ("ds_onecoin", lambda: Mt.dawid_skene_onecoin(Xp, g, K=_K(bench))[0]),
             ("crh", lambda: Mt.crh(Xp, g)[0]), ("irt4pl", lambda: Mt.irt_ceiling_lr(Xp, g, Yp, Mp)),
             ("accu_nodep", lambda: Mt.accu_depen(Xp, g, NFALSE[bench], use_dependence=False, K=_K(bench))[0])]
    if bench == "mmlu":
        meths.append(("ds_full", lambda: Mt.dawid_skene_full(Xp, g, 4)[0]))
    if len(c) <= 1500:
        meths.append(("accu_depen", lambda: Mt.accu_depen(Xp, g, NFALSE[bench], K=_K(bench))[0]))
    for name, f in meths:
        t0 = time.time(); s = f()
        out.append({"bench": bench, "condition": cond, "rep": rep, "n_models": int(len(c)), "method": name,
                    "ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, PK[bench]),
                    "sec": time.time() - t0, "scores": s.astype(np.float32).tobytes()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--heavy-workers", dest="heavy_workers", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(20261116)
    specs = []
    for bench in ("mmlu", "gsm8k"):
        t, mods, X, g, Y, M = load(bench)
        helm = [m for m in mods if m.startswith("helm:")]
        v1 = [m for m in mods if not m.startswith("helm:")]
        acc = (Y * M).sum(0) / np.maximum(M.sum(0), 1)
        acc_s = pd.Series(acc, index=mods)
        k = len(helm)
        v1_strong_k = list(acc_s[v1].sort_values(ascending=False).index[:k])
        v1_strong_100 = list(acc_s[v1].sort_values(ascending=False).index[:100])
        specs += [(bench, "helm_all", 0, helm), (bench, "v1_strong_k", 0, v1_strong_k),
                  (bench, "helm_plus_v1_strong100", 0, helm + v1_strong_100), (bench, "helm_plus_v1_all", 0, helm + v1),
                  (bench, "v1_all", 0, v1)]
        for r in range(a.reps):
            specs.append((bench, "v1_random_k", r, list(rng.choice(v1, k, replace=False))))
        vendor = pd.Series([m.split(":", 1)[1].split("_", 1)[0] for m in helm], index=helm)
        vc = vendor.value_counts()
        n_sub = 20
        top_v = list(vc.index[:3])
        for r in range(a.reps):
            conc = [m for m in helm if vendor[m] in top_v]
            specs.append((bench, "helm_vendor_concentrated", r, list(rng.choice(conc, min(n_sub, len(conc)), replace=False))))
            div = []
            for v in rng.permutation(vc.index):
                pool = [m for m in helm if vendor[m] == v]
                div.append(pool[rng.integers(len(pool))])
                if len(div) == n_sub:
                    break
            specs.append((bench, "helm_vendor_diverse", r, div))
        print(bench, "items", len(t), "defective", int(t.defective.sum()), "helm", len(helm), "v1", len(v1),
              "vendors", len(vc), "top vendors", dict(vc.head(5)), flush=True)
    import pickle
    from concurrent.futures import as_completed
    ck = os.path.join(ROOT, "results", "ckpt_sources")
    os.makedirs(ck, exist_ok=True)
    key = lambda s: f"{s[0]}__{s[1]}__{s[2]}"
    t0 = time.time()
    for phase, workers in (("light", a.workers), ("heavy", a.heavy_workers)):
        todo = [s for s in specs if (len(s[3]) >= 1000) == (phase == "heavy") and not os.path.exists(os.path.join(ck, key(s) + ".pkl"))]
        print("phase", phase, "to run", len(todo), flush=True)
        if not todo:
            continue
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
            futs = {ex.submit(run, s): s for s in todo}
            for i, f in enumerate(as_completed(futs)):
                tmp = os.path.join(ck, key(futs[f]) + ".pkl.tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump(f.result(), fh)
                os.replace(tmp, os.path.join(ck, key(futs[f]) + ".pkl"))
                if i % 10 == 0:
                    print(phase, i, len(todo), round(time.time() - t0), key(futs[f]), flush=True)
    rows = [r for s in specs for r in pickle.load(open(os.path.join(ck, key(s) + ".pkl"), "rb"))]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "results", "sources_scores.parquet"))
    summ = df.groupby(["bench", "condition", "method"]).agg(ap=("ap", "mean"), pk=("pk", "mean"), n=("n_models", "mean"),
                                                           reps=("rep", "nunique")).reset_index()
    json.dump(summ.to_dict(orient="records"), open(os.path.join(ROOT, "results", "sources_report.json"), "w"), indent=1)
    print(summ.to_string())


if __name__ == "__main__":
    main()
