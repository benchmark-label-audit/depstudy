"""Log-odds scoring for the probabilistic methods (DS-1, full DS, ACCU, DEPEN) on every panel.

Ranking by 1 - posterior(key) saturates when thousands of sources multiply their likelihoods: on the full v1
population 99.9% of MMLU items receive posterior 0 or 1 in double precision, and ties are broken by item order.
Ranking by the log posterior odds against the key (methods.logodds_against_key) orders items exactly as the
exact posterior does. This script recomputes the four methods with log-odds scores ("<method>_lo") on the gate
panels (singles and cells, three benchmarks), the size x capability panels, and the cross-source panels,
reconstructing every panel from the same panel files and seeded generators as s5/s5b, s10, and s6.

Every finished task is written to results/ckpt_logodds/<key>.json; item scores are cached in
$DEPSTUDY_DATA/item_scores/<bench>__<tag>__<method>.npy for the bootstrap (s17).

  python scripts/s19_logodds.py --workers 6 [--phase light|heavy|all|collect]  ->  results/logodds_scores.parquet
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
CK = os.path.join(RES, "ckpt_logodds")
CACHE = os.path.join(DATA_ROOT, "item_scores")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
PK = {"gate": {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}, "sizecap": {"mmlu": 200, "gsm8k": 100, "hellaswag": 200},
      "sources": {"mmlu": 200, "gsm8k": 50}}
HEAVY = 1000
_D = {}


def _K(bench):
    return 4 if bench in ("mmlu", "hellaswag") else None


def data(src):
    if src not in _D:
        if src.startswith("src_"):
            import s6_sources as S6
            _D[src] = S6.load(src[4:])
        else:
            from depstudy.data import matrices
            from s4_panels import eligible_models
            _D[src] = matrices(eligible_models(None), src)
    return _D[src]


def score(method, X, g, bench):
    from depstudy import methods as Mt
    K = _K(bench)
    if method == "ds_onecoin_lo":
        return Mt.dawid_skene_onecoin(X, g, K=K, logodds=True)[0]
    if method == "ds_full_lo":
        return Mt.dawid_skene_full(X, g, 4, logodds=True)[0]
    if method == "accu_nodep_lo":
        return Mt.accu_depen(X, g, NF[bench], use_dependence=False, K=K, logodds=True)[0]
    if method == "accu_depen_lo":
        return Mt.accu_depen(X, g, NF[bench], K=K, logodds=True)[0]
    raise KeyError(method)


def job(task):
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    family, bench, kind, cond, rep, tag, models, methods = task
    src = ("src_" + bench) if family == "sources" else bench
    t, mods, X, g, Y, M = data(src)
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    y = t.defective.values.astype(int)
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    rows = []
    for m in methods:
        p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
        t0 = time.time()
        if os.path.exists(p):
            s = np.load(p); sec = float("nan")
        else:
            s = score(m, X[:, c], g, bench); sec = time.time() - t0
            np.save(p + ".tmp.npy", s); os.replace(p + ".tmp.npy", p)
        rows.append({"family": family, "bench": bench, "kind": kind, "condition": cond, "rep": rep, "n_models": int(len(c)),
                     "method": m, "ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, PK[family][bench]),
                     "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)), "sec": sec})
    key = f"{family}__{bench}__{kind}__{cond}__{rep}"
    json.dump(rows, open(os.path.join(CK, key + ".json"), "w"))
    return key, len(c)


def tasks():
    out = []
    # gate panels (s5 / s5b): singles and lineage x dependence cells
    for bench, pfile in (("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json")):
        P = json.load(open(os.path.join(RES, pfile)))["benches"][bench]
        base = ["ds_onecoin_lo", "accu_nodep_lo"] + (["ds_full_lo"] if bench != "gsm8k" else [])
        for name, s in P["singles"].items():
            ms = base + (["accu_depen_lo"] if s["size"] < 2999 else [])
            out.append(("gate", bench, "single", name, 0, f"single_{name}", s["models"], ms))
        for name, reps in P["cells"].items():
            for r, p in enumerate(reps):
                out.append(("gate", bench, "cell", name, r, f"cell_{name}_{r}", p["models"], base + ["accu_depen_lo"]))
    # size x capability (s10 generator)
    import s10_size_capability as S10
    common, cap = S10.capability()
    rng = np.random.default_rng(20261117)
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)
        for n in S10.SIZES:
            out.append(("sizecap", b, "top_capability", str(n), 0, f"size_top_capability_{n}_0", order[:n], ["ds_onecoin_lo", "accu_nodep_lo"]))
            for r in range(10):
                out.append(("sizecap", b, "random", str(n), r, f"size_random_{n}_{r}", list(rng.choice(common, n, replace=False)), ["ds_onecoin_lo", "accu_nodep_lo"]))
        out.append(("sizecap", b, "all", str(len(common)), 0, "size_all", common, ["ds_onecoin_lo", "accu_nodep_lo"]))
    # cross-source (s6 generator)
    import s6_sources as S6
    rng = np.random.default_rng(20261116)
    for bench in ("mmlu", "gsm8k"):
        t, mods, X, g, Y, M = S6.load(bench)
        helm = [m for m in mods if m.startswith("helm:")]
        v1 = [m for m in mods if not m.startswith("helm:")]
        acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
        k = len(helm)
        v1_strong_k = list(acc[v1].sort_values(ascending=False).index[:k])
        v1_strong_100 = list(acc[v1].sort_values(ascending=False).index[:100])
        specs = [("helm_all", 0, helm), ("v1_strong_k", 0, v1_strong_k), ("helm_plus_v1_strong100", 0, helm + v1_strong_100),
                 ("helm_plus_v1_all", 0, helm + v1), ("v1_all", 0, v1)]
        for r in range(20):
            specs.append(("v1_random_k", r, list(rng.choice(v1, k, replace=False))))
        vendor = pd.Series([m.split(":", 1)[1].split("_", 1)[0] for m in helm], index=helm)
        vc = vendor.value_counts(); top_v = list(vc.index[:3])
        for r in range(20):
            conc = [m for m in helm if vendor[m] in top_v]
            specs.append(("helm_vendor_concentrated", r, list(rng.choice(conc, min(20, len(conc)), replace=False))))
            div = []
            for v in rng.permutation(vc.index):
                pool = [m for m in helm if vendor[m] == v]
                div.append(pool[rng.integers(len(pool))])
                if len(div) == 20:
                    break
            specs.append(("helm_vendor_diverse", r, div))
        base = ["ds_onecoin_lo", "accu_nodep_lo"] + (["ds_full_lo"] if bench == "mmlu" else [])
        for name, r, ms in specs:
            out.append(("sources", bench, "source", name, r, f"src_{name}_{r}", ms, base + (["accu_depen_lo"] if len(ms) <= 1500 else [])))
    return out


def done(t):
    """A task counts as finished only when its checkpoint and every item-score file it writes exist."""
    ck = os.path.join(CK, f"{t[0]}__{t[1]}__{t[2]}__{t[3]}__{t[4]}.json")
    return os.path.exists(ck) and all(os.path.exists(os.path.join(CACHE, f"{t[1]}__{t[5]}__{m}.npy")) for m in t[7])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--heavy-workers", dest="heavy_workers", type=int, default=3)
    ap.add_argument("--phase", default="all", choices=["light", "heavy", "all", "collect"])
    a = ap.parse_args()
    os.makedirs(CK, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    if a.phase != "collect":
        todo = [t for t in tasks() if not done(t)]
        light = [t for t in todo if len(t[6]) < HEAVY]
        heavy = sorted([t for t in todo if len(t[6]) >= HEAVY], key=lambda t: -len(t[6]))
        print("tasks: light", len(light), "heavy", len(heavy), flush=True)
        groups = []
        if a.phase in ("light", "all"):
            groups.append((light, a.workers))
        if a.phase in ("heavy", "all"):
            groups.append((heavy, a.heavy_workers))
        for group, w in groups:
            if not group:
                continue
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(job, t) for t in group]
                for i, f in enumerate(as_completed(futs)):
                    key, n = f.result()
                    if i % 50 == 0 or n >= HEAVY:
                        print(i + 1, "/", len(group), key, n, flush=True)
    rows = [r for f in glob.glob(os.path.join(CK, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RES, "logodds_scores.parquet"))
    print("rows", len(df), flush=True)


if __name__ == "__main__":
    main()
