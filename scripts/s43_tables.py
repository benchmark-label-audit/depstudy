"""LaTeX rows for the supplementary tables of the paper (tie-aware expected metrics throughout).

  tab:lb     Land & Bikel's benchmarks: expected number of weak-reference mislabels within the top 50 (multiple
             choice) and AP against them (preference benchmarks)
  tab:sens   one-factor hyperparameter sensitivity (AP range over the tested values)

  python scripts/s43_tables.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402

RES = os.path.join(ROOT, "results")
LAB = {"mv": r"\mv", "crh": r"\crh", "catd": r"\catd", "ds": r"\dsone", "glad": r"\glad", "depen": r"\depen", "irt": r"\irt",
       "entropy": "Entropy", "ensemble": "Rank average"}


def fin(s):
    s = np.asarray(s, np.float64); f = s[np.isfinite(s)]
    return np.where(s == np.inf, f.max() + 1, np.where(s == -np.inf, f.min() - 1, s))


def scores(X, g, Y, M, K):
    sc = {"mv": Mt.majority_vote(X, g), "crh": Mt.crh(X, g)[0], "catd": Mt.catd(X, g)[0],
          "ds": Mt.dawid_skene_onecoin(X, g, K=K, logodds=True)[0], "glad": Mt.glad(X, g, K=K)[0],
          "depen": Mt.accu_depen(X, g, K - 1, K=K, logodds=True)[0], "irt": Mt.irt_ceiling_lr(X, g, Y, M),
          "entropy": Mt.answer_entropy(X, g)}
    sc = {k: fin(v) for k, v in sc.items()}
    sc["ensemble"] = np.mean([pd.Series(sc[k]).rank().values / len(g) for k in ("crh", "ds", "irt")], axis=0)
    return sc


def lb_table():
    import s14_landbikel as L
    import s41_pref_audit as P
    lb = pd.read_parquet(os.path.join(L.EEE, "landbikel_flagged.parquet"))
    cols = {}
    for bench, short in L.BMAP.items():
        items, models, X, g, Y, M = L.matrix(bench)
        k = g >= 0
        items = [s for s, kk in zip(items, k) if kk]; X, g, Y, M = X[k], g[k], Y[k], M[k]
        key = pd.Series(range(len(items)), index=[L.question_key(s) for s in items]); key = key[~key.index.duplicated(keep=False)]
        sub = lb[lb.benchmark == short].copy(); sub["row"] = [key.get(L.question_key(c)) for c in sub.sample_content]
        mis = sub.dropna(subset=["row"]); mis = mis[mis.weak_ref_label == "mislabel"].row.astype(int).values
        y = np.zeros(len(items), int); y[mis] = 1
        sc = scores(X, g, Y, M, 4)
        cols[short] = {m: Mt.precision_at_k_expected(s, y, 50) * 50 for m, s in sc.items()}
        cols[short + "_n"] = len(mis)
    for b, (short, K) in P.BENCH.items():
        d = pd.read_parquet(os.path.join(P.EEE, f"{b}.parquet")); d["c"] = d.answer.map(P.code)
        items = sorted(d.sample_id.unique()); models = sorted(d.model.unique())
        ii = {s: i for i, s in enumerate(items)}; jj = {m: j for j, m in enumerate(models)}
        X = np.full((len(items), len(models)), -1, np.int32); ok = d.c >= 0
        X[d.sample_id[ok].map(ii).values, d.model[ok].map(jj).values] = d.c[ok].values
        g = np.zeros(len(items), np.int32); M = X >= 0; Y = (X == 0).astype(np.float32)
        sub = lb[lb.benchmark == short]; mis = sub[sub.weak_ref_label == "mislabel"].id.map(ii).dropna().astype(int).values
        y = np.zeros(len(items), int); y[mis] = 1
        sc = scores(X, g, Y, M, K)
        cols[short] = {m: Mt.average_precision_expected(s, y) for m, s in sc.items()}
    json.dump(cols, open(os.path.join(RES, "landbikel_table.json"), "w"), indent=1)
    print("% tab:lb rows: method & GSM & MATH & GPQA (mislabels in top 50) & RB2 & JB (AP)")
    print(f"% n mislabel: GSM {cols['GSM_n']}, MATH {cols['MATH_n']}, GPQA {cols['GPQA_n']}")
    for m in ("mv", "crh", "catd", "ds", "glad", "depen", "irt", "entropy", "ensemble"):
        print(f"    {LAB[m]} & {cols['GSM'][m]:.1f} & {cols['MATH'][m]:.1f} & {cols['GPQA'][m]:.1f} & "
              f"{cols['RB2'][m]:.3f} & {cols['JB'][m]:.3f} \\\\")


def sens_table():
    d = pd.read_parquet(os.path.join(RES, "sensitivity.parquet"))
    name = {"ds_onecoin_lo": r"\dsone", "ds_full_lo": r"\dsfull", "crh": r"\crh", "catd": r"\catd", "accu_nodep_lo": r"\accu",
            "glad_lo": r"\glad", "accu_depen_lo": r"\depen", "irt4pl": r"\irt"}
    pname = {"iters": "iterations", "prior_acc": "initial $p_j$", "smooth": "smoothing", "alpha": r"$\alpha$", "c": "copy rate $c$", "eps": r"first-round $\varepsilon$",
             "prior": "prior precision", "min_r": "item filter $r_{\\min}$", "n_false": "$n_f$"}
    print("% tab:sens rows: method & setting & tested & MMLU full & MMLU strong & GSM8K strong")
    for (m, p), g in d.groupby(["method", "param"], sort=False):
        vals = sorted(g.value.unique(), key=lambda v: (v == "per_item", float(v) if v != "per_item" else 0))
        vals = [("$|V_i|-1$" if v == "per_item" else v) for v in vals]
        cells = []
        for b, pan in (("mmlu", "full"), ("mmlu", "small_strong"), ("gsm8k", "small_strong")):
            x = g[(g.bench == b) & (g.panel == pan)]
            cells.append(f"{x.ap.min():.3f}--{x.ap.max():.3f}".replace("0.", ".") if len(x) else "--")
        print(f"    {name[m]} & {pname[p]} & {', '.join(vals)} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    lb_table()
    sens_table()
