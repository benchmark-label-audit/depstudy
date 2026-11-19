"""CrowdFM on (1) Land & Bikel's benchmarks from Every Eval Ever (66-69 models, inside CrowdFM's 20-100 worker
pretraining range): recall of their weak-reference mislabel items in the top-k, next to the other methods in
results/landbikel_repeat.json; (2) item scores of the single gate panels, cached for the paired bootstrap
($DEPSTUDY_DATA/item_scores/<bench>__single_<cond>__crowdfm[_block100].npy).

  <crowdfm-venv>/python -I scripts/s15b_crowdfm_extra.py  ->  results/crowdfm_landbikel.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.data import matrices, n_options  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402
from crowdfm_adapter import crowdfm_scores  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")


def landbikel():
    import s14_landbikel as L
    from scipy.stats import spearmanr
    lb = pd.read_parquet(os.path.join(L.EEE, "landbikel_flagged.parquet"))
    prev = json.load(open(os.path.join(ROOT, "results", "landbikel_repeat.json")))
    out = {}
    for bench, short in L.BMAP.items():
        items, models, X, g, Y, M = L.matrix(bench)
        keep = g >= 0
        items = [s for s, k in zip(items, keep) if k]; X, g = X[keep], g[keep]
        key = pd.Series(range(len(items)), index=[L.question_key(s) for s in items])
        key = key[~key.index.duplicated(keep=False)]
        sub = lb[lb.benchmark == short].copy()
        sub["row"] = [key.get(L.question_key(c)) for c in sub.sample_content]
        sub = sub.dropna(subset=["row"]); sub["row"] = sub.row.astype(int)
        s = crowdfm_scores(X, g, 4)
        mis = sub[sub.weak_ref_label == "mislabel"].row.values
        order = np.argsort(-s)
        rec = {f"crowdfm@{k}": int(np.isin(order[:k], mis).sum()) for k in (20, 50, 100)}
        has = sub.delta_li.notna()
        out[short] = {"models": len(models), "items": len(items), "weak_ref_mislabel_items": int(len(mis)),
                      "recall_of_mislabel_in_topk": rec, "others": prev[short]["recall_of_mislabel_in_topk"],
                      "spearman_crowdfm_vs_their_delta": round(float(spearmanr(s[sub.row[has]], sub.delta_li[has])[0]), 3)}
        print(short, json.dumps(out[short]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "crowdfm_landbikel.json"), "w"), indent=1)


def singles():
    from s4_panels import eligible_models
    models = eligible_models(None)
    os.makedirs(CACHE, exist_ok=True)
    for bench, pfile in (("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json")):
        t, mods, X, g, Y, M = matrices(models, bench)
        idx = {m: i for i, m in enumerate(mods)}
        P = json.load(open(os.path.join(ROOT, "results", pfile)))["benches"][bench]["singles"]
        K = n_options(bench) if bench in ("mmlu", "hellaswag") else None
        for cond, spec in P.items():
            c = np.array([idx[m] for m in spec["models"] if m in idx])
            for variant, wb in (("crowdfm", None), ("crowdfm_block100", 100)):
                if wb and len(c) <= 100:
                    continue
                p = os.path.join(CACHE, f"{bench}__single_{cond}__{variant}.npy")
                if not os.path.exists(p):
                    np.save(p, crowdfm_scores(X[:, c], g, K, worker_block=wb))
            print(bench, cond, "cached", flush=True)


if __name__ == "__main__":
    landbikel()
    singles()
