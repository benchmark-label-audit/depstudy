"""Analysis of the dependence experiments.

AP is the expected AP under random ordering of ties. Intervals come from a two-way bootstrap: each replicate
resamples items (stratified by defect status) (one resample shared by every panel) and,
independently, the panels within each cell or dose level. p-values are two-sided bootstrap tail shares.

  cells     main effects of lineage and dependence, their interaction, and the four simple effects, per method
  dose      OLS slope of AP on the panel's mean residual correlation (per +0.05), and highest minus lowest level
  capmatch  curated panel (deduplicated, small-diverse) minus capability-matched random panels of the same size
  discount  DEPEN minus ACCU log-odds score, by item group (key errors, other defects, clean), per cell and level

  python scripts/s35_dep_analysis.py --B 500  ->  results/cells_v2_effects.csv, results/dose_effects.csv,
                                                  results/capmatch_effects.csv, results/depen_discount.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
METHODS = ["mv", "crh", "catd", "ds_onecoin_lo", "ds_onecoin_nf10_lo", "ds_full_lo", "accu_nodep_lo", "accu_depen_lo", "accu_depen_nu_lo", "accu_depen_key_lo", "accu_depen_nu_key_lo",
           "glad_lo", "irt4pl"]
CELLS = ["lineage_high__dep_high", "lineage_high__dep_low", "lineage_low__dep_high", "lineage_low__dep_low"]


def load(bench, tag, m):
    p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
    if not os.path.exists(p):
        return None
    s = np.load(p); fin = s[np.isfinite(s)]
    return np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))


def ap_rows(S, y, I):
    """AP of each score vector (rows of S) on each item resample (rows of I): returns (n_vec, B)."""
    return np.stack([np.concatenate([Mt.average_precision_expected_batch(s[I[a:a + 250]], y[I[a:a + 250]])
                                     for a in range(0, len(I), 250)]) for s in S])


def pval(d):
    d = np.asarray(d)
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def summarize(name, est, boot):
    return {"effect": name, "estimate": float(est), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "p_boot": pval(boot)}


class Items:
    def __init__(self, bench, B, seed):
        import s19_logodds as S19
        from s10_size_capability import key_mask
        t = S19.data(bench)[0]
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        rng = np.random.default_rng(seed)
        self.y = {"ap": t.defective.values.astype(int), "ap_key": kpos.astype(int)}
        self.sub = {"ap": np.arange(len(t)), "ap_key": np.flatnonzero(keep)}
        self.I = {}
        for k, v in self.sub.items():
            yy = self.y[k][v]
            pos, neg = v[yy == 1], v[yy == 0]
            self.I[k] = np.concatenate([pos[rng.integers(0, len(pos), size=(B, len(pos)))],
                                        neg[rng.integers(0, len(neg), size=(B, len(neg)))]], axis=1)
        self.kpos, self.clean, self.defect = kpos, clean, t.defective.values.astype(bool)

    def point(self, s, metric):
        sub = self.sub[metric]
        return Mt.average_precision_expected(s[sub], self.y[metric][sub])

    def boot(self, S, metric):
        return ap_rows(S, self.y[metric], self.I[metric])


def cells(bench, P, items, B, rng):
    rows, disc = [], []
    for m in METHODS:
        vec = {c: [load(bench, f"cell_{c}_{r}", m) for r in range(len(P["cells"][c]))] for c in CELLS + ["random"]}
        if any(v is None for vs in vec.values() for v in vs):
            continue
        for metric in ("ap", "ap_key"):
            pt = {c: np.array([items.point(s, metric) for s in vs]) for c, vs in vec.items()}
            bt = {c: items.boot(np.stack(vs), metric) for c, vs in vec.items()}       # (n_panels, B)
            # panel resampling within each cell, independent of the item resample
            bm = {}
            for c, M in bt.items():
                J = rng.integers(0, M.shape[0], size=(B, M.shape[0]))
                bm[c] = M[J, np.arange(B)[:, None]].mean(1)
            mean = {c: v.mean() for c, v in pt.items()}
            hh, hl, lh, ll = CELLS
            defs = {
                "dependence_main": (lambda x: (x[hh] + x[lh] - x[hl] - x[ll]) / 2),
                "lineage_main": (lambda x: (x[hh] + x[hl] - x[lh] - x[ll]) / 2),
                "interaction": (lambda x: (x[hh] - x[hl]) - (x[lh] - x[ll])),
                "dependence_in_lineage_high": (lambda x: x[hh] - x[hl]),
                "dependence_in_lineage_low": (lambda x: x[lh] - x[ll]),
                "lineage_in_dependence_high": (lambda x: x[hh] - x[lh]),
                "lineage_in_dependence_low": (lambda x: x[hl] - x[ll]),
                "best_cell_minus_random": (lambda x: np.max(np.stack([np.asarray(x[c]) for c in CELLS]), axis=0) - x["random"]),
            }
            for name, f in defs.items():
                rows.append({"bench": bench, "method": m, "metric": metric, **summarize(name, f(mean), f(bm))})
            for c in CELLS + ["random"]:
                rows.append({"bench": bench, "method": m, "metric": metric, **summarize(f"mean_{c}", mean[c], bm[c])})
    # DEPEN discount by item group
    for c in CELLS + ["random"]:
        for r in range(len(P["cells"][c])):
            a, d = load(bench, f"cell_{c}_{r}", "accu_nodep_lo"), load(bench, f"cell_{c}_{r}", "accu_depen_lo")
            if a is None or d is None:
                continue
            delta = d - a
            ok = np.isfinite(delta)
            disc.append({"bench": bench, "family": "cell", "condition": c, "rep": r,
                         "delta_key_errors": float(delta[ok & items.kpos].mean()),
                         "delta_other_defects": float(delta[ok & items.defect & ~items.kpos].mean()),
                         "delta_clean": float(delta[ok & items.clean].mean()),
                         "top200_overlap": len(set(np.argsort(-a)[:200]) & set(np.argsort(-d)[:200])) / 200})
    return rows, disc


def dose(bench, DP, items, B, rng, manip):
    rows, disc = [], []
    levels = sorted(DP["dose"].keys())
    for m in METHODS:
        vecs, xs, lv = [], [], []
        for li, L in enumerate(levels):
            for r, p in enumerate(DP["dose"][L]["reps"]):
                s = load(bench, f"dose_{L}_{r}", m)
                if s is None:
                    continue
                vecs.append(s); xs.append(p["mean_R_design"]); lv.append(li)
        if len(vecs) < 20:
            continue
        xs, lv = np.array(xs), np.array(lv)
        xe = None
        if manip is not None:
            mm = manip[(manip.bench == bench) & (manip.kind == "dose")].set_index(["condition", "rep"])
            xe = np.array([mm.loc[(L, r), "R_eval"] for L in levels for r in range(len(DP["dose"][L]["reps"]))
                           if load(bench, f"dose_{L}_{r}", m) is not None])
        for metric in ("ap", "ap_key"):
            pt = np.array([items.point(s, metric) for s in vecs])
            bt = items.boot(np.stack(vecs), metric)                          # (n_panels, B)

            def slope(y, x):
                x0 = x - x.mean()
                return float((x0 * (y - y.mean())).sum() / (x0 ** 2).sum()) * 0.05
            est = slope(pt, xs)
            bs = np.empty(B); bd = np.empty(B)
            for b in range(B):
                J = np.concatenate([rng.choice(np.flatnonzero(lv == k), (lv == k).sum()) for k in range(len(levels))])
                bs[b] = slope(bt[J, b], xs[J])
                bd[b] = bt[J[lv[J] == len(levels) - 1], b].mean() - bt[J[lv[J] == 0], b].mean()
            rows.append({"bench": bench, "method": m, "metric": metric, "x": "R_design", **summarize("slope_per_0.05", est, bs)})
            rows.append({"bench": bench, "method": m, "metric": metric, "x": "level",
                         **summarize("top_minus_bottom_level", pt[lv == len(levels) - 1].mean() - pt[lv == 0].mean(), bd)})
            if xe is not None:
                rows.append({"bench": bench, "method": m, "metric": metric, "x": "R_eval",
                             **summarize("slope_per_0.05", slope(pt, xe), np.array([slope(bt[:, b], xe) for b in range(B)]))})
            for k, L in enumerate(levels):
                rows.append({"bench": bench, "method": m, "metric": metric, "x": "level",
                             **summarize(f"mean_{L}", pt[lv == k].mean(), bt[lv == k].mean(0)),
                             "R_design_mean": float(xs[lv == k].mean())})
    for L in levels:
        for r in range(len(DP["dose"][L]["reps"])):
            a, d = load(bench, f"dose_{L}_{r}", "accu_nodep_lo"), load(bench, f"dose_{L}_{r}", "accu_depen_lo")
            if a is None or d is None:
                continue
            delta = d - a; ok = np.isfinite(delta)
            disc.append({"bench": bench, "family": "dose", "condition": L, "rep": r,
                         "delta_key_errors": float(delta[ok & items.kpos].mean()),
                         "delta_other_defects": float(delta[ok & items.defect & ~items.kpos].mean()),
                         "delta_clean": float(delta[ok & items.clean].mean()),
                         "top200_overlap": len(set(np.argsort(-a)[:200]) & set(np.argsort(-d)[:200])) / 200})
    return rows, disc


def capmatch(bench, DP, items, B, rng):
    rows = []
    for name, C in DP["curation_matched"].items():
        for m in METHODS:
            cur = load(bench, f"single_{name}", m)
            nrep = min(len(C["reps"]), 8) if name == "corr_dedup" else len(C["reps"])  # s29 CAP_REPS_LARGE
            mt = [load(bench, f"capmatch_{name}_{r}", m) for r in range(nrep)]
            mt = [s for s in mt if s is not None]
            if cur is None or len(mt) < 5:
                continue
            for metric in ("ap", "ap_key"):
                pc = items.point(cur, metric); pm = np.array([items.point(s, metric) for s in mt])
                bc = items.boot(cur[None], metric)[0]; bm = items.boot(np.stack(mt), metric)
                J = rng.integers(0, len(mt), size=(B, len(mt)))
                bmm = bm[J, np.arange(B)[:, None]].mean(1)
                rows.append({"bench": bench, "curation": name, "method": m, "metric": metric, "curated": pc,
                             "matched_mean": float(pm.mean()), "acc_curated": C["acc_mean"],
                             "acc_matched": float(np.mean([p["acc_mean"] for p in C["reps"]])),
                             **summarize("curated_minus_matched", pc - pm.mean(), bc - bmm)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    rng = np.random.default_rng(20261121)
    DPall = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"]
    mp = os.path.join(RES, "dependence_manip.parquet")
    manip = pd.read_parquet(mp) if os.path.exists(mp) else None
    C, D, K, Q = [], [], [], []
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        P = json.load(open(os.path.join(RES, "panels_hellaswag.json" if bench == "hellaswag" else "panels.json")))["benches"][bench]
        items = Items(bench, a.B, 20261121 + len(bench))
        r1, d1 = cells(bench, P, items, a.B, rng); C += r1; Q += d1
        if bench in DPall:
            r2, d2 = dose(bench, DPall[bench], items, a.B, rng, manip); D += r2; Q += d2
            K += capmatch(bench, DPall[bench], items, a.B, rng)
        print(bench, "done", flush=True)
    pd.DataFrame(C).to_csv(os.path.join(RES, "cells_v2_effects.csv"), index=False)
    pd.DataFrame(D).to_csv(os.path.join(RES, "dose_effects.csv"), index=False)
    pd.DataFrame(K).to_csv(os.path.join(RES, "capmatch_effects.csv"), index=False)
    pd.DataFrame(Q).to_csv(os.path.join(RES, "depen_discount.csv"), index=False)
    pd.set_option("display.width", 250)
    c = pd.DataFrame(C)
    print(c[c.effect.isin(["dependence_main", "interaction", "dependence_in_lineage_high", "dependence_in_lineage_low"])
            & (c.metric == "ap")].round(4).to_string())
    d = pd.DataFrame(D)
    if len(d):
        print(d[d.effect.isin(["slope_per_0.05", "top_minus_bottom_level"])].round(4).to_string())
    if K:
        print(pd.DataFrame(K).round(4).to_string())
    if Q:
        print(pd.DataFrame(Q).groupby(["bench", "family", "condition"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
