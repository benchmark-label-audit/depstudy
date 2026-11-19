"""GSM8K with the lenient answer extraction.

The v1 harness rule ('#### <number>') leaves 38% of the v1 GSM8K answers unparseable. This script rebuilds the
GSM8K answer matrix with the 'flexible-extract' rule of later lm-evaluation-harness versions (last number in the
generation; s27) and reruns every method on (a) the four single panels and the off-benchmark strong panel, with the
model lists unchanged, plus the 100 models most accurate under the lenient rule, and (b) the cross-source panels
with HELM, whose GSM8K answers are already extracted with a last-number rule (s3c). Benchmark tag: 'gsm8kflex'.

  python scripts/s31_gsm8k_flex.py --workers 6  ->  results/gsm8kflex_scores.parquet, results/gsm8kflex_stats.json
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
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CK = os.path.join(RES, "ckpt_gsm8kflex")
CACHE = os.path.join(DATA_ROOT, "item_scores")
B = "gsm8kflex"
ALL = ["mv", "crh", "ds_onecoin_lo", "ds_onecoin_nf10_lo", "accu_nodep_lo", "glad_lo", "catd", "entropy", "irt4pl", "irt4pl_miss"]
_D = {}


def single_data():
    if "single" not in _D:
        from depstudy.data import matrices
        from s4_panels import eligible_models
        _D["single"] = matrices(eligible_models(None), B)
    return _D["single"]


def source_data():
    if "source" not in _D:
        from depstudy.data import CACHE as MC, assemble, gsm8k_flex_long, helm_long, truth
        from s4_panels import eligible_models
        p = os.path.join(MC, "sources_gsm8kflex.pkl")
        if os.path.exists(p):
            _D["source"] = pickle.load(open(p, "rb"))
        else:
            v1 = eligible_models(None)
            t = truth("gsm8k")
            lt_v1 = gsm8k_flex_long(v1)
            lt_h = helm_long("helm_gsm8k")
            items = set(lt_h.item_id) & set(lt_v1.item_id)
            t = t[t.item_id.isin(items)].reset_index(drop=True)
            lt = pd.concat([lt_v1, lt_h], ignore_index=True)
            out = assemble(t, lt, v1 + sorted(lt_h.model.unique()), "num", 0.0)
            pickle.dump(out, open(p, "wb"))
            _D["source"] = out
    return _D["source"]


def job(task):
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    import s28_newmethods as S28
    family, cond, rep, tag, models, methods = task
    t, mods, X, g, Y, M = single_data() if family == "single" else source_data()
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    y = t.defective.values.astype(int)
    kpos, clean = key_mask("gsm8k", t)
    keep = kpos | clean
    pk = 100 if family == "single" else 50
    rows = []
    for m in methods:
        p = os.path.join(CACHE, f"{B}__{tag}__{m}.npy")
        t0 = time.time()
        if os.path.exists(p):
            s = np.load(p); sec = float("nan")
        else:
            s = np.asarray(S28.score(m, X[:, c], g, Y[:, c], M[:, c], "gsm8k"), np.float64); sec = time.time() - t0
            np.save(p + ".tmp.npy", s); os.replace(p + ".tmp.npy", p)
        rows.append({"family": family, "bench": B, "condition": cond, "rep": rep, "n_models": int(len(c)), "method": m,
                     "n_items": int(len(t)), "ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, pk),
                     "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)), "sec": sec})
    key = f"{family}__{cond}__{rep}"
    with open(os.path.join(CK, key + ".json.tmp"), "w") as fh:
        json.dump(rows, fh)
    os.replace(os.path.join(CK, key + ".json.tmp"), os.path.join(CK, key + ".json"))
    return key, len(c)


def tasks():
    import s10_size_capability as S10
    out = []
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]["gsm8k"]["singles"]
    t, mods, X, g, Y, M = single_data()
    acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
    for name, s in P.items():
        ms = ALL + (["accu_depen_lo"] if s["size"] <= 400 else [])
        out.append(("single", name, 0, f"single_{name}", s["models"], ms))
    out.append(("single", "small_strong_flex", 0, "single_small_strong_flex", list(acc.sort_values(ascending=False).index[:100]),
                ALL + ["accu_depen_lo"]))
    common, cap = S10.capability()
    out.append(("single", "offbench_strong", 0, "single_offbench_strong", list(cap["gsm8k"].sort_values(ascending=False).index[:100]),
                ALL + ["accu_depen_lo"]))
    # cross-source panels: same generator as s6 / s19 (seed 20261116), v1 ranked by lenient accuracy
    ts, ms_, Xs, gs, Ys, Ms = source_data()
    helm = [m for m in ms_ if m.startswith("helm:")]
    v1 = [m for m in ms_ if not m.startswith("helm:")]
    accs = pd.Series((Ys * Ms).sum(0) / np.maximum(Ms.sum(0), 1), index=ms_)
    k = len(helm)
    rng = np.random.default_rng(20261116)
    specs = [("helm_all", 0, helm), ("v1_strong_k", 0, list(accs[v1].sort_values(ascending=False).index[:k])),
             ("helm_plus_v1_strong100", 0, helm + list(accs[v1].sort_values(ascending=False).index[:100])),
             ("helm_plus_v1_all", 0, helm + v1), ("v1_all", 0, v1)]
    for r in range(20):
        specs.append(("v1_random_k", r, list(rng.choice(v1, k, replace=False))))
    src_methods = ["mv", "crh", "ds_onecoin_lo", "ds_onecoin_nf10_lo", "accu_nodep_lo", "glad_lo", "catd", "entropy", "irt4pl"]
    for name, r, models in specs:
        out.append(("source", name, r, f"src_{name}_{r}", models, src_methods + (["accu_depen_lo"] if len(models) <= 400 else [])))
    return out


def stats():
    """Answer-extraction statistics: unparseable shares and accuracy under both rules."""
    import s19_logodds as S19
    from s4_panels import eligible_models
    t1, m1, X1, g1, Y1, M1 = S19.data("gsm8k")
    t2, m2, X2, g2, Y2, M2 = single_data()
    a1 = (Y1 * M1).sum(0) / np.maximum(M1.sum(0), 1)
    a2 = (Y2 * M2).sum(0) / np.maximum(M2.sum(0), 1)
    s = {"models_strict": len(m1), "models_flex": len(m2), "items_strict": len(t1), "items_flex": len(t2),
         "unparseable_strict": float(((X1 < 0) & M1).sum() / M1.sum()), "unparseable_flex": float(((X2 < 0) & M2).sum() / M2.sum()),
         "median_acc_strict": float(np.median(a1)), "median_acc_flex": float(np.median(a2)),
         "models_below_2pct_strict": int((a1 < 0.02).sum()), "models_below_2pct_flex": int((a2 < 0.02).sum())}
    common = sorted(set(m1) & set(m2))
    i1 = {m: i for i, m in enumerate(m1)}; i2 = {m: i for i, m in enumerate(m2)}
    from scipy.stats import spearmanr
    s["spearman_acc_strict_flex"] = float(spearmanr(a1[[i1[m] for m in common]], a2[[i2[m] for m in common]])[0])
    json.dump(s, open(os.path.join(RES, "gsm8kflex_stats.json"), "w"), indent=1)
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    from s4_panels import eligible_models
    from depstudy.data import GSM_RAW
    missing = [m for m in eligible_models(None) if not os.path.exists(os.path.join(GSM_RAW, m + ".parquet"))]
    print("models without raw generations", len(missing), flush=True)
    if missing and not a.collect:
        print(missing[:10]); print("run s27 first; matrices are cached by model list"); return
    os.makedirs(CK, exist_ok=True)
    if not a.collect:
        stats()
        T = [t for t in tasks() if not os.path.exists(os.path.join(CK, f"{t[0]}__{t[1]}__{t[2]}.json"))]
        light = [t for t in T if len(t[4]) < 1000]
        heavy = [t for t in T if len(t[4]) >= 1000]
        print("todo", len(light), len(heavy), flush=True)
        for group, w in ((light, a.workers), (heavy, max(2, a.workers // 2))):
            if not group:
                continue
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(job, t) for t in group]
                for i, f in enumerate(as_completed(futs)):
                    key, n = f.result()
                    print(i + 1, len(group), key, n, time.strftime("%H:%M:%S"), flush=True)
    rows = [r for f in glob.glob(os.path.join(CK, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RES, "gsm8kflex_scores.parquet"))
    pd.set_option("display.width", 250)
    d = df[df.rep == 0]
    for met in ("ap", "ap_key"):
        print(met); print(d.pivot_table(index=["family", "condition"], columns="method", values=met).round(3).to_string())


if __name__ == "__main__":
    main()
