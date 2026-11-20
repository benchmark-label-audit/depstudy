"""Rescore every copy-detection (DEPEN) panel with the corrected implementation.

The earlier DEPEN runs started the copying test from the majority-vote truth, so values shared by copiers were
counted as shared true values and copying went undetected (Dong et al.'s Table 1 example gave P(S3~S4) = 0.2).
Dong et al. start from "every value true with probability 1 - eps"; methods.accu_depen now does the same and
reproduces their example (s45 ref). This script

  archive   moves the old caches <bench>__<tag>__accu_depen_lo.npy to item_scores/_depen_v0/
  run       rescores every archived (bench, tag) panel, plus the full populations, with
              accu_depen_lo     DEPEN, uniform false values (Dong et al. 2009, alpha .2, c .8, eps .2)
              accu_depen_nu_lo  DEPEN-NonUni: the copying test uses the empirical same-false-value rate
            and writes per-panel copying diagnostics to results/depen_diag/<bench>__<tag>.json
            (mean P(dep), share of pairs with P(dep) > .5, empirical same-false rate, the uniform rate 1/n_f,
            within- vs cross-family P(dep) for panels with declared lineage); D matrices of single panels go to
            DATA_ROOT/depen_D/.

  python scripts/s46_depen_rerun.py archive
  python scripts/s46_depen_rerun.py run --workers 6 [--only small|full]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CACHE = os.path.join(DATA_ROOT, "item_scores")
OLD = os.path.join(CACHE, "_depen_v0")
DIAG = os.path.join(RES, "depen_diag")
DMAT = os.path.join(DATA_ROOT, "depen_D")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
METHODS = {"accu_depen_lo": dict(), "accu_depen_nu_lo": dict(nonuni=True)}
# key-anchored variants: the copying test counts shared values against the shipped key (the data under audit)
METHODS_KEY = {"accu_depen_key_lo": dict(key_anchor=True), "accu_depen_nu_key_lo": dict(nonuni=True, key_anchor=True)}
VARIANT = os.environ.get("DEPEN_VARIANT", "")
if VARIANT == "key":
    METHODS = METHODS_KEY
    DIAG = DIAG + "_key"


def base_bench(b):
    return "mmlu" if b.startswith("mmluhalf") else ("gsm8k" if b.startswith("gsm8k") else b)


def archive():
    os.makedirs(OLD, exist_ok=True)
    n = 0
    for p in glob.glob(os.path.join(CACHE, "*__accu_depen_lo.npy")):
        shutil.move(p, os.path.join(OLD, os.path.basename(p))); n += 1
    print("archived", n, "to", OLD)


def tasks():
    """(cache_bench, tag, data_kind, data_arg, models, eval_half) for every panel that has an archived DEPEN cache,
    plus the full populations."""
    import s19_logodds as S19
    import s28_newmethods as S28
    import s29_dose as S29
    import s31_gsm8k_flex as S31
    out = {}
    for family, bench, kind, cond, rep, tag, models, ms in S19.tasks() + S28.tasks():
        if "accu_depen_lo" in ms or (family == "gate" and kind == "single"):
            src = ("src_" + bench) if family == "sources" else bench
            out[(bench, tag)] = ("s19", src, list(models), None)
    for family, bench, cond, rep, tag, models, ms in S29.tasks():
        if "accu_depen_lo" in ms:
            out[(bench, tag)] = ("s19", bench, list(models), None)
    for suf in ("", "_p200"):
        P = json.load(open(os.path.join(RES, f"splitdose_panels{suf}.json")))
        for f, F in enumerate(P["folds"]):
            for lv, L in F["dose"].items():
                for r, p in enumerate(L["reps"]):
                    out[(f"mmluhalf{1 - f}", f"splitdose{suf}_f{f}_{lv}_{r}")] = ("half", "mmlu", list(p["models"]), 1 - f)
    for family, cond, rep, tag, models, ms in S31.tasks():
        if "accu_depen_lo" in ms or family == "single":
            out[("gsm8kflex", tag)] = ("flex_" + family, "gsm8kflex", list(models), None)
    archived = {tuple(os.path.basename(p)[:-len("__accu_depen_lo.npy")].split("__", 1)) for p in glob.glob(os.path.join(OLD, "*.npy"))}
    if not archived:
        # no archive (fresh setup): the shipped diagnostics in results/depen_diag*/ list the rescored panels
        archived = {tuple(os.path.basename(p)[:-len(".json")].split("__", 1)) for p in glob.glob(os.path.join(DIAG, "*.json"))}
    keep = {k: v for k, v in out.items() if k in archived or k[1] == "single_full"}
    missing = archived - set(out)
    if missing:
        print("archived panels without a task:", sorted(missing)[:10], len(missing))
    return keep


_D = {}


def load(kind, arg, half):
    key = (kind, arg, half)
    if key not in _D:
        import s19_logodds as S19
        if kind == "s19":
            _D[key] = S19.data(arg)
        elif kind == "half":
            import s37_mmlu_splitdose as S37
            t, mods, X, g, Y, M = S19.data("mmlu")
            E = S37.halves(t)[half]
            _D[key] = (t[E].reset_index(drop=True), mods, X[E], g[E], Y[E], M[E])
        elif kind == "flex_single":
            import s31_gsm8k_flex as S31
            _D[key] = S31.single_data()
        else:
            import s31_gsm8k_flex as S31
            _D[key] = S31.source_data()
    return _D[key]


def job(item):
    from depstudy import methods as Mt
    (cbench, tag), (kind, arg, models, half) = item
    bench = base_bench(cbench)
    t, mods, X, g, Y, M = load(kind, arg, half)
    idx = {m: i for i, m in enumerate(mods)}
    names = [m for m in models if m in idx]
    c = np.array([idx[m] for m in names])
    K = 4 if bench in ("mmlu", "hellaswag") else None
    Xp = X[:, c]
    diag = {"bench": cbench, "tag": tag, "n_models": int(len(c)), "n_false": NF[bench], "uniform_rate": 1.0 / NF[bench]}
    t0 = time.time()
    for m, kw in METHODS.items():
        p = os.path.join(CACHE, f"{cbench}__{tag}__{m}.npy")
        kw = dict(kw); nonuni = kw.get("nonuni", False)
        if kw.pop("key_anchor", False):
            kw["truth_override"] = g
        s, post, acc, D = Mt.accu_depen(Xp, g, NF[bench], K=K, logodds=True, **kw)
        np.save(p + ".tmp.npy", np.asarray(s, np.float64)); os.replace(p + ".tmp.npy", p)
        iu = np.triu_indices(len(c), 1)
        d = D[iu]
        lab = ("nu" if nonuni else "uni") + ("_key" if VARIANT == "key" else "")
        diag[f"meanD_{lab}"] = float(d.mean())
        diag[f"shareD50_{lab}"] = float((d > 0.5).mean())
        diag[f"same_false_rate_{lab}"] = float(Mt.same_false_rate(Xp, post.argmax(1)))
        if tag.startswith("single_") or tag.startswith("cell_"):
            try:
                import pandas as pd
                lin = pd.read_parquet(os.path.join(ROOT, "data", "lineage.parquet")).set_index("model_dir")
                fam = np.array([str(lin.family.get(m)).lower() if m in lin.index else "none" for m in names])
                same = (fam[iu[0]] == fam[iu[1]]) & (fam[iu[0]] != "none") & (fam[iu[0]] != "nan")
                diag[f"n_samefam_pairs"] = int(same.sum())
                diag[f"meanD_samefam_{lab}"] = float(d[same].mean()) if same.any() else None
                diag[f"meanD_crossfam_{lab}"] = float(d[~same].mean())
            except Exception as e:  # noqa: BLE001
                diag["lineage_error"] = str(e)
        if tag.startswith("single_"):
            os.makedirs(DMAT, exist_ok=True)
            np.save(os.path.join(DMAT, f"{cbench}__{tag}__{lab}.npy"), D.astype(np.float16))
    diag["sec"] = time.time() - t0
    os.makedirs(DIAG, exist_ok=True)
    json.dump(diag, open(os.path.join(DIAG, f"{cbench}__{tag}.json"), "w"))
    return cbench, tag, round(diag["sec"], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["archive", "run"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", choices=["small", "full"], default=None)
    a = ap.parse_args()
    if a.step == "archive":
        archive(); return
    T = tasks()
    # a panel counts as finished only when its diagnostics and every item-score file it writes exist
    items = [(k, v) for k, v in T.items() if not (os.path.exists(os.path.join(DIAG, f"{k[0]}__{k[1]}.json"))
             and all(os.path.exists(os.path.join(CACHE, f"{k[0]}__{k[1]}__{m}.npy")) for m in METHODS))]
    if a.only == "small":
        items = [x for x in items if len(x[1][2]) <= 400]
    elif a.only == "full":
        items = [x for x in items if len(x[1][2]) > 400]
    # group by data source so each worker loads one matrix family
    items.sort(key=lambda x: (x[1][0], x[1][1], str(x[1][3]), x[0][1]))
    print("todo", len(items), "of", len(T), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, it) for it in items]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if i % 25 == 0 or len(items) < 30:
                print(i + 1, len(items), r, time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
