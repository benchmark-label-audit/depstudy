"""Audit of Land & Bikel's preference benchmarks with LLM judges as sources (RewardBench 2: 4-way; JudgeBench: pairwise),
from the Every Eval Ever records fetched by s36. Answers: chosen[0] = key (code 0), rejected[i] = code i + 1, ties missing.

Evaluation against Land & Bikel's released labels: recall of their weak-reference 'mislabel' items within the top k
of each method's ranking, AP against them, and the Spearman correlation of our IRT score with their delta_li.

  python scripts/s41_pref_audit.py  ->  results/pref_audit.json
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

EEE = os.path.join(DATA_ROOT, "eee")
BENCH = {"reward-bench-2": ("RB2", 4), "judgebench": ("JB", 2)}


def code(a):
    if a == "chosen[0]":
        return 0
    m = re.fullmatch(r"rejected\[(\d+)\]", str(a))
    return int(m.group(1)) + 1 if m else -1


def main():
    lb = pd.read_parquet(os.path.join(EEE, "landbikel_flagged.parquet"))
    out = {}
    for b, (short, K) in BENCH.items():
        d = pd.read_parquet(os.path.join(EEE, f"{b}.parquet"))
        d["c"] = d.answer.map(code)
        items = sorted(d.sample_id.unique()); models = sorted(d.model.unique())
        ii = {s: i for i, s in enumerate(items)}; jj = {m: j for j, m in enumerate(models)}
        X = np.full((len(items), len(models)), -1, np.int32)
        ok = d.c >= 0
        X[d.sample_id[ok].map(ii).values, d.model[ok].map(jj).values] = d.c[ok].values
        g = np.zeros(len(items), np.int32)
        M = X >= 0
        Y = (X == 0).astype(np.float32)
        nf = K - 1
        scores = {"mv": Mt.majority_vote(X, g), "crh": Mt.crh(X, g)[0], "catd": Mt.catd(X, g)[0],
                  "ds_onecoin_lo": Mt.dawid_skene_onecoin(X, g, K=K, logodds=True)[0],
                  "glad_lo": Mt.glad(X, g, K=K)[0], "ds_full_lo": Mt.dawid_skene_full(X, g, K, logodds=True)[0],
                  "accu_depen_lo": Mt.accu_depen(X, g, nf, K=K, logodds=True)[0],
                  "irt4pl": Mt.irt_ceiling_lr(X, g, Y, M), "entropy": Mt.answer_entropy(X, g)}
        sub = lb[lb.benchmark == short].copy()
        sub["row"] = sub.id.map(ii)
        matched = sub.dropna(subset=["row"]).copy(); matched["row"] = matched.row.astype(int)
        mis = matched[matched.weak_ref_label == "mislabel"].row.values
        pos = np.zeros(len(items), int); pos[mis] = 1
        res = {"models": len(models), "items": len(items), "lb_rows": int(len(sub)), "lb_rows_matched": int(len(matched)),
               "mislabel_matched": int(len(mis)), "base_rate": float(pos.mean()), "methods": {}}
        for m, s in scores.items():
            s = np.asarray(s, np.float64)
            fin = s[np.isfinite(s)]
            s = np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))
            rk = {}
            for k in (20, 50, 100, 200):
                o = np.argsort(-s, kind="stable")
                rk[f"recall@{k}"] = int(pos[o[:k]].sum())
            res["methods"][m] = {"ap": Mt.average_precision_expected(s, pos), **rk,
                                 "p_at_50_expected": Mt.precision_at_k_expected(s, pos, 50)}
        has = matched.delta_li.notna()
        res["spearman_irt_vs_their_delta"] = float(spearmanr(scores["irt4pl"][matched.row[has]], matched.delta_li[has])[0]) if has.sum() > 3 else None
        res["spearman_n"] = int(has.sum())
        out[short] = res
        print(short, json.dumps({k: v for k, v in res.items() if k != "methods"}), flush=True)
        print(pd.DataFrame(res["methods"]).T.round(3).to_string(), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "pref_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
