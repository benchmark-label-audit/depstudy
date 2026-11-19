"""Repeat-then-extend on the closest recent work (Land & Bikel, EMNLP 2026): their benchmarks' instance-level
records from Every Eval Ever (GSM-MC 68, MATH-MC 69, GPQA-Diamond 66 models; they used about 110), their released
IRT scores (delta_li) for flagged items, and their GPT-5.4 weak-reference labels.

Repeat: our 4PL ceiling likelihood-ratio on the same items; agreement with their delta_li on the released items;
where their flagged items land in our ranking.
Extend: every method's recall of their weak-reference "mislabel" items within its top-k.

  python scripts/s14_landbikel.py  ->  results/landbikel_repeat.json + printed tables
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
from depstudy import methods as Mt  # noqa: E402
from depstudy.textnorm import norm_key  # noqa: E402

from depstudy.hf import DATA_ROOT  # noqa: E402

EEE = os.path.join(DATA_ROOT, "eee")  # written by s13_eee_fetch.py
BMAP = {"gsm-mc": "GSM", "math-mc": "MATH", "gpqa-diamond": "GPQA"}


def matrix(bench):
    d = pd.read_parquet(os.path.join(EEE, f"{bench}.parquet"))
    items = sorted(d.sample_id.unique()); models = sorted(d.model.unique())
    ii = {s: i for i, s in enumerate(items)}; mj = {m: j for j, m in enumerate(models)}
    X = np.full((len(items), len(models)), -1, np.int32)
    code = {"A": 0, "B": 1, "C": 2, "D": 3}
    ok = d.answer.isin(code)
    X[d.sample_id[ok].map(ii).values, d.model[ok].map(mj).values] = d.answer[ok].map(code).values
    gold = d.drop_duplicates("sample_id").set_index("sample_id").gold.reindex(items).map(code).fillna(-1).values.astype(np.int32)
    M = np.zeros_like(X, bool)
    M[d.sample_id.map(ii).values, d.model.map(mj).values] = True  # attempted
    Y = (X == gold[:, None]).astype(np.float32)
    return items, models, X, gold, Y, M


def question_key(text):
    """Question text from Land & Bikel's sample_content ([USER]\\n<question>\\n\\n[A]...) or EEE sample_id (prefix:question)."""
    t = str(text)
    if t.startswith("[USER]"):
        t = t[len("[USER]"):].split("\n\n[A]")[0]
    elif ":" in t[:20]:
        t = t.split(":", 1)[1]
    return norm_key(t)[:200]


def main():
    lb = pd.read_parquet(os.path.join(EEE, "landbikel_flagged.parquet"))
    out = {}
    for bench, short in BMAP.items():
        items, models, X, g, Y, M = matrix(bench)
        keep = g >= 0
        items = [s for s, k in zip(items, keep) if k]; X, g, Y, M = X[keep], g[keep], Y[keep], M[keep]
        key = pd.Series(range(len(items)), index=[question_key(s) for s in items])
        key = key[~key.index.duplicated(keep=False)]
        sub = lb[lb.benchmark == short].copy()
        sub["row"] = [key.get(question_key(c)) for c in sub.sample_content]
        sub = sub.dropna(subset=["row"]); sub["row"] = sub.row.astype(int)
        scores = {"irt4pl": Mt.irt_ceiling_lr(X, g, Y, M), "mv": Mt.majority_vote(X, g), "crh": Mt.crh(X, g)[0],
                  "ds_onecoin": Mt.dawid_skene_onecoin(X, g, K=4)[0], "ds_full": Mt.dawid_skene_full(X, g, 4)[0],
                  "accu_nodep": Mt.accu_depen(X, g, 3, use_dependence=False, K=4)[0], "accu_depen": Mt.accu_depen(X, g, 3, K=4)[0],
                  "ds_onecoin_lo": Mt.dawid_skene_onecoin(X, g, K=4, logodds=True)[0], "ds_full_lo": Mt.dawid_skene_full(X, g, 4, logodds=True)[0],
                  "accu_depen_lo": Mt.accu_depen(X, g, 3, K=4, logodds=True)[0],
                  "accu_nodep_lo": Mt.accu_depen(X, g, 3, use_dependence=False, K=4, logodds=True)[0],
                  "glad_lo": Mt.glad(X, g, K=4)[0], "catd": Mt.catd(X, g)[0], "entropy": Mt.answer_entropy(X, g)}
        ours = scores["irt4pl"]
        has = sub.delta_li.notna()
        rho = spearmanr(ours[sub.row[has]], sub.delta_li[has])[0] if has.sum() > 3 else float("nan")
        flagged = sub[sub.flagged_by_delta_li_gt_0.fillna(False)]
        rank = pd.Series(ours).rank(ascending=False)
        n_pos_ours = int((ours > 0).sum())
        mis = sub[sub.weak_ref_label == "mislabel"].row.values
        rec = {}
        for m, s in scores.items():
            order = np.argsort(-s)
            for k in (20, 50, 100):
                rec[f"{m}@{k}"] = int(np.isin(order[:k], mis).sum())
        out[short] = {"models": len(models), "items": len(items), "landbikel_rows_matched": int(len(sub)),
                      "landbikel_rows_with_delta": int(has.sum()), "spearman_n_items": int(has.sum()),
                      "flagged_by_delta_gt0": int(len(flagged)), "spearman_ours_vs_their_delta": round(float(rho), 3),
                      "their_flagged_with_our_delta_gt0": int((ours[flagged.row] > 0).sum()),
                      "our_items_with_delta_gt0": n_pos_ours,
                      "median_rank_of_their_flagged_in_ours": float(rank[flagged.row].median()) if len(flagged) else None,
                      "weak_ref_mislabel_items": int(len(mis)), "recall_of_mislabel_in_topk": rec}
        print(short, json.dumps(out[short], indent=1), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "landbikel_repeat.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
