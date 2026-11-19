"""Repeat-then-extend check: run our implementations on the public crowdsourcing datasets of Zheng et al.
("Truth Inference in Crowdsourcing: Is the Problem Solved?", PVLDB 2017) and compare inferred-truth accuracy with
their published Table 6 (complete data). Data: datasets.zip from the authors' project page, extracted under D:.

  python scripts/s11_repeat_zheng.py  ->  results/repeat_zheng.csv + printed comparison
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

# Zheng et al.'s datasets (https://github.com/TsinghuaDatabaseGroup/CrowdTruthInference), unpacked here
BASE = os.path.join(DATA_ROOT, "zheng2017_datasets", "raw", "datasets")
DATASETS = {"D_Product": "d_jn-product", "D_PosSent": "d_sentiment", "S_Rel": "s4_Relevance", "S_Adult": "s5_AdultContent"}
# Zheng et al. 2017, Table 6 (complete data): accuracy %, and F1 % for the decision-making datasets
PUBLISHED = {
    "MV": {"D_Product": (89.66, 59.05), "D_PosSent": (93.31, 92.85), "S_Rel": (54.19, None), "S_Adult": (36.04, None)},
    "ZC": {"D_Product": (92.80, 63.59), "D_PosSent": (95.10, 94.60), "S_Rel": (48.21, None), "S_Adult": (35.34, None)},
    "D&S": {"D_Product": (93.66, 71.59), "D_PosSent": (96.00, 95.66), "S_Rel": (61.30, None), "S_Adult": (36.05, None)},
    "PM": {"D_Product": (89.81, 59.34), "D_PosSent": (95.04, 94.53), "S_Rel": (59.02, None), "S_Adult": (36.50, None)},
    "GLAD": {"D_Product": (92.20, 60.17), "D_PosSent": (95.20, 94.71), "S_Rel": (53.59, None), "S_Adult": (36.47, None)},
    "CATD": {"D_Product": (92.66, 65.92), "D_PosSent": (95.50, 95.07), "S_Rel": (45.32, None), "S_Adult": (36.23, None)},
}
OURS = {"MV": "majority vote", "ZC": "one-coin DS", "ZC_peritem": "one-coin DS, per-item n_f", "D&S": "full-confusion DS", "PM": "CRH", "GLAD": "GLAD", "CATD": "CATD"}


def load(folder):
    a = pd.read_csv(os.path.join(BASE, folder, "answer.csv"))
    a.columns = ["question", "worker", "answer"]
    t = pd.read_csv(os.path.join(BASE, folder, "truth.csv"))
    t.columns = ["question", "truth"]
    a["question"] = a.question.astype(str); t["question"] = t.question.astype(str)
    labels = sorted(set(a.answer.unique()) | set(t.truth.unique()))
    code = {l: i for i, l in enumerate(labels)}
    qs = sorted(a.question.unique()); ws = sorted(a.worker.astype(str).unique())
    qi = {q: i for i, q in enumerate(qs)}; wi = {w: j for j, w in enumerate(ws)}
    X = np.full((len(qs), len(ws)), -1, np.int32)
    X[a.question.map(qi).values, a.worker.astype(str).map(wi).values] = a.answer.map(code).values
    truth = t[t.question.isin(qi)]
    return X, len(labels), np.array([qi[q] for q in truth.question]), truth.truth.map(code).values, code


def f1(pred, true, pos):
    tp = ((pred == pos) & (true == pos)).sum(); fp = ((pred == pos) & (true != pos)).sum(); fn = ((pred != pos) & (true == pos)).sum()
    return 200 * tp / max(2 * tp + fp + fn, 1)


def main():
    rows = []
    for name, folder in DATASETS.items():
        X, K, ti, tv, code = load(folder)
        g = np.zeros(X.shape[0], np.int32)  # no shipped key here; only inferred truth is evaluated
        res = {}
        t0 = time.time(); C = Mt.onehot_counts(X, g); res["MV"] = (C.argmax(1), time.time() - t0)
        t0 = time.time(); _, post, _ = Mt.dawid_skene_onecoin(X, g, K=K); res["ZC"] = (post.argmax(1), time.time() - t0)
        # per-item variant of the one-coin model (Section 5.1): candidates are the values observed on the item. These
        # datasets ship no key, so the candidate set is anchored on an observed answer instead of the placeholder g.
        g_obs = np.array([r[r >= 0][0] if (r >= 0).any() else 0 for r in X], np.int32)
        t0 = time.time(); _, post, _ = Mt.dawid_skene_onecoin(X, g_obs, K=None); res["ZC_peritem"] = (post.argmax(1), time.time() - t0)
        t0 = time.time(); _, post = Mt.dawid_skene_full(X, g, K); res["D&S"] = (post.argmax(1), time.time() - t0)
        t0 = time.time(); _, w = Mt.crh(X, g); res["PM"] = (Mt.onehot_counts(X, g, w).argmax(1), time.time() - t0)
        t0 = time.time(); _, post, _, _ = Mt.glad(X, g, K=K, logodds=False); res["GLAD"] = (post.argmax(1), time.time() - t0)
        t0 = time.time(); _, w = Mt.catd(X, g); res["CATD"] = (Mt.onehot_counts(X, g, w).argmax(1), time.time() - t0)
        n_truth = int(len(pd.read_csv(os.path.join(BASE, folder, "truth.csv"))))
        pos = code.get(1, 1)
        for m, (pred, sec) in res.items():
            acc = 100 * (pred[ti] == tv).mean()
            pub_acc, pub_f1 = PUBLISHED["ZC" if m == "ZC_peritem" else m][name]
            row = {"dataset": name, "items": X.shape[0], "workers": X.shape[1], "answers": int((X >= 0).sum()), "method": m,
                   "ours": OURS[m], "acc_ours": round(acc, 2), "acc_published": pub_acc, "diff_acc": round(acc - pub_acc, 2), "sec": round(sec, 2)}
            # Zheng et al. score every truth item; on S_Adult 1,184 of the 1,517 truth items received no answer,
            # so a method's accuracy there is that of a random label (1/K). Expected accuracy under that protocol:
            row["truth_items"] = n_truth
            row["truth_items_answered"] = int(len(ti))
            row["acc_ours_all_truth"] = round(100 * ((pred[ti] == tv).sum() + (n_truth - len(ti)) / K) / n_truth, 2)
            row["diff_acc_all_truth"] = round(row["acc_ours_all_truth"] - pub_acc, 2)
            if pub_f1 is not None:
                fo = f1(pred[ti], tv, pos)
                row.update(f1_ours=round(fo, 2), f1_published=pub_f1, diff_f1=round(fo - pub_f1, 2))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "repeat_zheng.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
