"""Downstream effect of label defects on the leaderboard.

Full correction: key errors receive the expert answer (MMLU-Redux 'correct_answer'; GSM8K-Platinum revised key) and
the other defective items are removed, as GSM8K-Platinum removes them. Each v1 model is re-scored and the ranking
compared with the ranking under the shipped keys: Kendall tau over all models and over the top 100, rank
displacement, and changes in the top 10.

Audit correction: annotators review the top-k items of an audit ranking (k = 200 on MMLU, 100 on GSM8K) and fix only
the verified defects among them. The share of the full correction's displacement that the audit recovers is
reported per method, against random review (mean of 200 random k-subsets).

  python scripts/s33_rankshift.py  ->  results/rankshift.json, results/rankshift_audit.parquet
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
PREV = os.path.join(ROOT, "data", "external")
K = {"mmlu": 200, "gsm8k": 100}
AUDITS = [("mv", "full"), ("crh", "full"), ("ds_onecoin_lo", "full"), ("glad_lo", "full"), ("irt4pl", "full"),
          ("crh", "small_strong"), ("ds_onecoin_lo", "small_strong"), ("crh", "offbench_strong"),
          ("ds_onecoin_lo", "offbench_strong"), ("glad_lo", "offbench_strong")]


def correctness(bench):
    """C0: correctness under shipped keys (items x models, NaN = no answer counted as incorrect);
    C1: correctness under corrected keys; fix: item index -> 'rekey' | 'drop' | None."""
    import s19_logodds as S19
    t, mods, X, g, Y, M = S19.data(bench)
    n = len(t)
    if bench == "mmlu":
        red = pd.read_parquet(os.path.join(PREV, "mmlu_redux_all.parquet"))
        red["k"] = red.groupby("subject").cumcount()
        red["item_id"] = red.subject + "__" + red.k.map(lambda x: f"{x:03d}")
        ca = red.set_index("item_id").correct_answer.reindex(t.item_id)
        lut = {"A": 0, "B": 1, "C": 2, "D": 3, "0": 0, "1": 1, "2": 2, "3": 3}
        newkey = np.array([lut.get(str(v).strip(), -1) if isinstance(v, (str, int, np.integer)) else -1 for v in ca.values])
        C0 = (X == g[:, None]).astype(np.float32)
        C1 = C0.copy()
        fix = np.array([None] * n, dtype=object)
        et = t.error_type.values
        for i in range(n):
            if et[i] == "ok":
                continue
            if et[i] == "wrong_groundtruth" and newkey[i] >= 0 and newkey[i] != g[i]:
                C1[i] = (X[i] == newkey[i]).astype(np.float32); fix[i] = "rekey"
            else:
                fix[i] = "drop"
    else:
        from depstudy.data import long_table
        lt = long_table(mods, "gsm8k")
        lt = lt[lt.item_id.isin(set(t.item_id))]
        ii = pd.Series(np.arange(n), index=t.item_id); jj = pd.Series(np.arange(len(mods)), index=mods)
        A = np.full((n, len(mods)), None, dtype=object)
        A[ii.reindex(lt.item_id).values, jj.reindex(lt.model).values] = lt.answer.values
        gold = np.asarray(t.gold.astype(str).to_numpy(), dtype=object)
        gp = np.asarray(t.gold_platinum.to_numpy(), dtype=object)
        C0 = (A == gold[:, None]).astype(np.float32)
        C1 = C0.copy()
        fix = np.array([None] * n, dtype=object)
        st = t.status.values
        for i in range(n):
            if st[i] == "revised" and isinstance(gp[i], str) and gp[i] != gold[i]:
                C1[i] = (A[i] == gp[i]).astype(np.float32); fix[i] = "rekey"
            elif st[i] in ("removed", "revised"):
                fix[i] = "drop"
    return t, mods, C0, C1, fix


def scores(C, keep):
    return C[keep].mean(0)


def rank(s):
    return pd.Series(-s).rank(method="min").values


def compare(s0, s1, top=100):
    r0, r1 = rank(s0), rank(s1)
    top0 = np.argsort(-s0)[:top]
    d = np.abs(r0 - r1)
    return {"tau_all": float(kendalltau(s0, s1)[0]), "tau_top100": float(kendalltau(s0[top0], s1[top0])[0]),
            "mean_disp_top100": float(d[top0].mean()), "max_disp_top100": float(d[top0].max()),
            "share_top100_moved_ge5": float((d[top0] >= 5).mean()),
            "top10_changed": int(len(set(np.argsort(-s0)[:10]) ^ set(np.argsort(-s1)[:10])) // 2),
            "disp_all": float(d.sum())}


def apply(C0, C1, fix, sel):
    """Correct only the defective items in sel."""
    C = C0.copy()
    keep = np.ones(C0.shape[0], bool)
    for i in sel:
        if fix[i] == "rekey":
            C[i] = C1[i]
        elif fix[i] == "drop":
            keep[i] = False
    return C, keep


def main():
    out, rows = {}, []
    rng = np.random.default_rng(20261120)
    for bench in ("mmlu", "gsm8k"):
        t, mods, C0, C1, fix = correctness(bench)
        n = len(t)
        defect = np.array([f is not None for f in fix])
        s0 = scores(C0, np.ones(n, bool))
        Cf, keepf = apply(C0, C1, fix, np.flatnonzero(defect))
        s1 = scores(Cf, keepf)
        full = compare(s0, s1)
        full.update(n_rekey=int((fix == "rekey").sum()), n_drop=int((fix == "drop").sum()), n_models=len(mods),
                    mean_acc_change=float((s1 - s0).mean()), max_acc_change=float(np.abs(s1 - s0).max()))
        out[bench] = {"full_correction": full}
        r0, r1 = rank(s0), rank(s1)
        full_disp = np.abs(r0 - r1).sum()
        k = K[bench]
        top1 = np.argsort(-s1)[:100]                     # top 100 under the corrected keys
        tau0 = float(kendalltau(s0[top1], s1[top1])[0])
        err0 = float(np.abs(s0 - s1).mean())
        out[bench]["full_correction"]["tau_top100_corrected_order"] = tau0

        def audit_row(name, sel):
            C, keep = apply(C0, C1, fix, sel)
            s = scores(C, keep)
            rs = rank(s)
            resid = np.abs(rs - r1).sum()
            tau = float(kendalltau(s[top1], s1[top1])[0])
            return {"bench": bench, "audit": name, "k": k, "defects_found": int(defect[sel].sum()),
                    "share_defects_found": float(defect[sel].sum() / defect.sum()),
                    "share_disp_recovered": float(1 - resid / full_disp),
                    # accuracy error removed: 1 - mean |acc_audit - acc_corrected| / mean |acc_shipped - acc_corrected|
                    "share_error_removed": float(1 - np.abs(s - s1).mean() / err0),
                    "tau_top100": tau, "tau_top100_gap_closed": float((tau - tau0) / (1 - tau0)),
                    "tau_to_full": float(kendalltau(s, s1)[0])}
        for m, panel in AUDITS:
            p = os.path.join(CACHE, f"{bench}__single_{panel}__{m}.npy")
            if not os.path.exists(p):
                continue
            sc = np.load(p)
            sel = np.argsort(-sc, kind="stable")[:k]
            rows.append(audit_row(f"{m}@{panel}", sel))
        rr = [audit_row("random", rng.choice(n, k, replace=False)) for _ in range(200)]
        rows.append({"bench": bench, "audit": "random", "k": k, **{c: float(np.mean([r[c] for r in rr]))
                     for c in ("defects_found", "share_defects_found", "share_disp_recovered", "share_error_removed",
                               "tau_top100", "tau_top100_gap_closed", "tau_to_full")}})
        print(bench, full, flush=True)
    json.dump(out, open(os.path.join(RES, "rankshift.json"), "w"), indent=1)
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(RES, "rankshift_audit.parquet"))
    pd.set_option("display.width", 250)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
