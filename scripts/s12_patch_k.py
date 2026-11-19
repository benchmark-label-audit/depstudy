"""Patch: recompute one-coin DS and ACCU (with and without dependence) on the multiple-choice benchmarks with
a fixed option count K = 4 (the earlier implementation used the per-item number of observed options, which
deviates from ZenCrowd/one-coin DS; found when reproducing Zheng et al. 2017). Other methods are unaffected.

Checkpoints patched in place: ckpt_gate (mmlu), ckpt_hellaswag, ckpt_sizecap (mmlu, hellaswag),
ckpt_sources (mmlu). Panel memberships are reconstructed from the same panel files / seeded generators.

  python scripts/s12_patch_k.py --workers 8
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RES = os.path.join(ROOT, "results")
PK = {"mmlu": 200, "hellaswag": 200}
NF = {"mmlu": 3, "hellaswag": 3}
_D = {}


def _init():
    from depstudy.data import matrices
    from s4_panels import eligible_models
    models = eligible_models(None)
    for b in ("mmlu", "hellaswag"):
        _D[b] = matrices(models, b)
    import s6_sources
    _D["src_mmlu"] = s6_sources.load("mmlu")


def scores(src, bench, models, which):
    from depstudy import methods as Mt
    t, mods, X, g, Y, M = _D[src]
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    Xp = X[:, c]
    out = {}
    for name in which:
        t0 = time.time()
        if name == "ds_onecoin":
            s = Mt.dawid_skene_onecoin(Xp, g, K=4)[0]
        elif name == "accu_nodep":
            s = Mt.accu_depen(Xp, g, NF[bench], use_dependence=False, K=4)[0]
        else:
            s = Mt.accu_depen(Xp, g, NF[bench], K=4)[0]
        out[name] = (s, time.time() - t0)
    return t, out


def patch(task):
    from depstudy import methods as Mt
    kind, path, bench, models = task
    if kind == "sources":
        rows = pickle.load(open(path, "rb"))
    else:
        rows = json.load(open(path, encoding="utf-8"))
    which = [r["method"] for r in rows if r["method"] in ("ds_onecoin", "accu_nodep", "accu_depen")]
    if not which:
        return path, 0
    stored_n = rows[0].get("n", rows[0].get("n_models"))
    if stored_n is not None and int(stored_n) != len(models):  # reconstructed panel must match the stored one
        raise RuntimeError(f"panel size mismatch for {path}: stored {stored_n}, reconstructed {len(models)}")
    t, out = scores("src_mmlu" if kind == "sources" else bench, bench, models, which)
    y = t.defective.values.astype(int)
    for r in rows:
        if r["method"] not in out:
            continue
        s, sec = out[r["method"]]
        r["ap"] = Mt.average_precision(s, y)
        r["pk"] = Mt.precision_at_k(s, y, PK[bench] if kind != "sources" else 200)
        r["sec"] = sec
        if kind == "sizecap":
            from s10_size_capability import key_mask
            kpos, clean = key_mask(bench, t)
            keep = kpos | clean
            r["ap_key"] = Mt.average_precision(s[keep], kpos[keep].astype(int))
        if kind == "sources":
            r["scores"] = s.astype(np.float32).tobytes()
        r["k_fixed"] = True
    tmp = path + ".tmp"
    if kind == "sources":
        pickle.dump(rows, open(tmp, "wb"))
    else:
        json.dump(rows, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, path)
    return path, len(out)


def tasks():
    out = []
    pan = json.load(open(os.path.join(RES, "panels.json")))["benches"]["mmlu"]
    panh = json.load(open(os.path.join(RES, "panels_hellaswag.json")))["benches"]["hellaswag"]
    for d, P, b in (("ckpt_gate", pan, "mmlu"), ("ckpt_hellaswag", panh, "hellaswag")):
        for f in glob.glob(os.path.join(RES, d, f"{b}__*.json")):
            parts = os.path.basename(f)[:-5].split("__")  # condition names themselves contain "__"
            kind, rep, name = parts[1], parts[-1], "__".join(parts[2:-1])
            models = P["cells"][name][int(rep)]["models"] if kind == "cell" else P["singles"][name]["models"]
            out.append(("gate", f, b, models))
    # size x capability: same generator as s10
    import s10_size_capability as S10
    common, cap = S10.capability()
    rng = np.random.default_rng(20261117)
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)
        specs = []
        for n in S10.SIZES:
            specs.append((b, "top_capability", n, 0, order[:n]))
            for r in range(10):
                specs.append((b, "random", n, r, list(rng.choice(common, n, replace=False))))
        specs.append((b, "all", len(common), 0, common))
        if b == "gsm8k":
            continue
        for s in specs:
            f = os.path.join(RES, "ckpt_sizecap", f"{s[0]}__{s[1]}__{s[2]}__{s[3]}.json")
            if os.path.exists(f):
                out.append(("sizecap", f, b, s[4]))
    # cross-source: same generator as s6 (mmlu only)
    import pandas as pd
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
        specs = [(bench, "helm_all", 0, helm), (bench, "v1_strong_k", 0, v1_strong_k),
                 (bench, "helm_plus_v1_strong100", 0, helm + v1_strong_100), (bench, "helm_plus_v1_all", 0, helm + v1),
                 (bench, "v1_all", 0, v1)]
        for r in range(20):
            specs.append((bench, "v1_random_k", r, list(rng.choice(v1, k, replace=False))))
        vendor = pd.Series([m.split(":", 1)[1].split("_", 1)[0] for m in helm], index=helm)
        vc = vendor.value_counts(); top_v = list(vc.index[:3])
        for r in range(20):
            conc = [m for m in helm if vendor[m] in top_v]
            specs.append((bench, "helm_vendor_concentrated", r, list(rng.choice(conc, min(20, len(conc)), replace=False))))
            div = []
            for v in rng.permutation(vc.index):
                pool = [m for m in helm if vendor[m] == v]
                div.append(pool[rng.integers(len(pool))])
                if len(div) == 20:
                    break
            specs.append((bench, "helm_vendor_diverse", r, div))
        if bench != "mmlu":
            continue
        for s in specs:
            f = os.path.join(RES, "ckpt_sources", f"{s[0]}__{s[1]}__{s[2]}.pkl")
            if os.path.exists(f):
                out.append(("sources", f, "mmlu", s[3]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    todo = tasks()
    print("files to patch", len(todo), flush=True)
    light = [t for t in todo if len(t[3]) < 1000]
    heavy = [t for t in todo if len(t[3]) >= 1000]
    for group, w in ((light, a.workers), (heavy, 3)):
        with ProcessPoolExecutor(max_workers=w, initializer=_init) as ex:
            futs = [ex.submit(patch, t) for t in group]
            for i, f in enumerate(as_completed(futs)):
                p, n = f.result()
                if i % 25 == 0:
                    print(i, len(group), os.path.basename(p), n, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
