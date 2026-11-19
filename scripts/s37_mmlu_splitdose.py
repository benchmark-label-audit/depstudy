"""MMLU dependence dose ladder with design and evaluation on disjoint halves of the subjects.

The 57 subjects are split into two halves, stratified by subject group (seeded). In fold f, panels are built only
from the answers on the design half D_f: the pool is the 400 most accurate models on D_f, panels of 100 models are
annealed to target mean residual correlations (2PL fit of the whole population on D_f) while matching the pool's
accuracy mean, accuracy SD and subject-group accuracies on D_f. Every method is then scored on the evaluation half
E_f, whose answers took no part in building the panels. Using MMLU itself for the design half lets the ladder reach
the dependence of the small-strong panel (residual correlation 0.41 on MMLU) without circularity.

Two folds x 6 levels x 10 panels. Effects use a two-way bootstrap (items of E_f x panels within level, B = 500).

  python scripts/s37_mmlu_splitdose.py build|run|analyze  ->  results/splitdose_panels.json, results/splitdose_effects.csv
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CACHE = os.path.join(DATA_ROOT, "item_scores")
POOL = int(os.environ.get("DOSE_POOL", "400"))  # 200: a second ladder whose pool is close to the small-strong panel
SUF = "" if POOL == 400 else f"_p{POOL}"
CK = os.path.join(RES, "ckpt_splitdose" + SUF)
N, LEVELS, REPS, SEED = 100, 6, 10, 20261122
METHODS = ["mv", "crh", "catd", "ds_onecoin_lo", "ds_full_lo", "accu_nodep_lo", "accu_depen_lo", "accu_depen_nu_lo", "accu_depen_key_lo", "accu_depen_nu_key_lo", "glad_lo", "irt4pl"]


def halves(t):
    from depstudy.panels import subject_group
    rng = np.random.default_rng(SEED)
    subj = sorted(t.subject.unique())
    grp = {s: subject_group(s) for s in subj}
    A = []
    for k in range(4):
        ss = [s for s in subj if grp[s] == k]
        rng.shuffle(ss)
        A += ss[: len(ss) // 2]
    a = t.subject.isin(set(A)).values
    return [a, ~a]          # fold 0 designs on half A and evaluates on B; fold 1 the reverse


def build():
    import s19_logodds as S19
    from s29_dose import anneal_target, _features
    from depstudy.panels import residual_corr
    t, mods, X, g, Y, M = S19.data("mmlu")
    H = halves(t)
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "folds": []}
    for f in range(2):
        D, E = H[f], H[1 - f]
        tD = t[D].reset_index(drop=True)
        F, tolm = _features("mmlu", tD, Y[D], M[D], np.arange(len(mods)))
        acc = F[:, 0]
        pool = np.argsort(-acc)[:POOL]
        Rfull = residual_corr(Y[D], M[D])
        Rp = Rfull[np.ix_(pool, pool)]
        Fp = F[pool]
        target = {"mean": Fp.mean(0), "sd": Fp[:, 0].std()}
        tol = {"mean": tolm, "sd": 0.02}
        loc = np.arange(POOL)
        strong = np.argsort(-acc)[:N]
        R_strong = float(Rfull[np.ix_(strong, strong)].sum() / (N * (N - 1)))
        lo = min((anneal_target(N, loc, Rp, Fp, target, tol, np.random.default_rng(rng.integers(1 << 31)), sign=-1) for _ in range(3)),
                 key=lambda r: r["mean_R"] + 10 * r["penalty"])
        hi = max((anneal_target(N, loc, Rp, Fp, target, tol, np.random.default_rng(rng.integers(1 << 31)), sign=+1) for _ in range(3)),
                 key=lambda r: r["mean_R"] - 10 * r["penalty"])
        levels = np.linspace(lo["mean_R"], hi["mean_R"], LEVELS)
        dose = {}
        for li, tr in enumerate(levels):
            reps = []
            for r in range(REPS):
                best = None
                for attempt in range(4):
                    c = anneal_target(N, loc, Rp, Fp, target, tol, np.random.default_rng(rng.integers(1 << 31)), target_R=float(tr))
                    if c["penalty"] == 0 and (best is None or abs(c["mean_R"] - tr) < abs(best["mean_R"] - tr)):
                        best = c
                    if best is not None and abs(best["mean_R"] - tr) < 0.003:
                        break
                if best is not None:
                    reps.append({"models": [mods[pool[i]] for i in best["models"]], "mean_R_design": best["mean_R"],
                                 "acc_mean": float(acc[pool[best["models"]]].mean())})
            dose[f"level{li}"] = {"target_R": float(tr), "reps": reps}
            print("fold", f, "level", li, round(float(tr), 3), len(reps), flush=True)
        out["folds"].append({"design_items": int(D.sum()), "eval_items": int(E.sum()), "R_small_strong_design": R_strong,
                             "pool_acc_mean": float(acc[pool].mean()), "dose": dose})
        json.dump(out, open(os.path.join(RES, f"splitdose_panels{SUF}.json"), "w"))


def job(task):
    import s19_logodds as S19
    import s28_newmethods as S28
    from s10_size_capability import key_mask
    f, lv, r, models = task
    t, mods, X, g, Y, M = S19.data("mmlu")
    E = halves(t)[1 - f]
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models])
    tE = t[E].reset_index(drop=True)
    y = tE.defective.values.astype(int)
    kpos, clean = key_mask("mmlu", tE)
    keep = kpos | clean
    rows = []
    for m in METHODS:
        p = os.path.join(CACHE, f"mmluhalf{1 - f}__splitdose{SUF}_f{f}_{lv}_{r}__{m}.npy")
        if os.path.exists(p):
            s = np.load(p)
        else:
            s = np.asarray(S28.score(m, X[E][:, c], g[E], Y[E][:, c], M[E][:, c], "mmlu"), np.float64)
            np.save(p + ".tmp.npy", s); os.replace(p + ".tmp.npy", p)
        rows.append({"fold": f, "level": lv, "rep": r, "method": m, "ap": Mt.average_precision_expected(s, y),
                     "ap_key": Mt.average_precision_expected(s[keep], kpos[keep].astype(int))})
    json.dump(rows, open(os.path.join(CK, f"f{f}__{lv}__{r}.json"), "w"))
    return f, lv, r


def run(workers):
    os.makedirs(CK, exist_ok=True)
    P = json.load(open(os.path.join(RES, f"splitdose_panels{SUF}.json")))
    T = [(f, lv, r, p["models"]) for f, F in enumerate(P["folds"]) for lv, L in F["dose"].items() for r, p in enumerate(L["reps"])
         if not os.path.exists(os.path.join(CK, f"f{f}__{lv}__{r}.json"))]
    print("todo", len(T), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(job, t) for t in T])):
            if i % 20 == 0:
                print(i + 1, len(T), fu.result(), time.strftime("%H:%M:%S"), flush=True)


def analyze(B=500):
    """Two-way bootstrap over evaluation items and panels within level; folds pooled by averaging per level."""
    import s19_logodds as S19
    from s10_size_capability import key_mask
    from s17_bootstrap import ap_batch
    t, mods, X, g, Y, M = S19.data("mmlu")
    H = halves(t)
    P = json.load(open(os.path.join(RES, f"splitdose_panels{SUF}.json")))
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for m in METHODS:
        for metric in ("ap", "ap_key"):
            per_fold = []
            for f, F in enumerate(P["folds"]):
                tE = t[H[1 - f]].reset_index(drop=True)
                kpos, clean = key_mask("mmlu", tE)
                if metric == "ap":
                    sub = np.arange(len(tE)); yv = tE.defective.values.astype(int)
                else:
                    sub = np.flatnonzero(kpos | clean); yv = kpos.astype(int)
                ps, ng = sub[yv[sub] == 1], sub[yv[sub] == 0]
                I = np.concatenate([ps[rng.integers(0, len(ps), size=(B, len(ps)))], ng[rng.integers(0, len(ng), size=(B, len(ng)))]], axis=1)
                lv_names = sorted(F["dose"])
                S, lv, xs = [], [], []
                for k, L in enumerate(lv_names):
                    for r, p in enumerate(F["dose"][L]["reps"]):
                        s = np.load(os.path.join(CACHE, f"mmluhalf{1 - f}__splitdose{SUF}_f{f}_{L}_{r}__{m}.npy"))
                        S.append(s); lv.append(k); xs.append(p["mean_R_design"])
                S = np.stack(S); lv = np.array(lv); xs = np.array(xs)
                pt = np.array([Mt.average_precision_expected(s[sub], yv[sub]) for s in S])
                fin = lambda s: np.where(s == np.inf, s[np.isfinite(s)].max() + 1, np.where(s == -np.inf, s[np.isfinite(s)].min() - 1, s))
                S = np.stack([fin(s) for s in S])
                pt = np.array([Mt.average_precision_expected(s[sub], yv[sub]) for s in S])
                bt = np.stack([np.concatenate([Mt.average_precision_expected_batch(s[I[a:a + 250]], yv[I[a:a + 250]])
                                               for a in range(0, B, 250)]) for s in S])
                per_fold.append((pt, bt, lv, xs))
            def effect(pt, lv):
                return pt[lv == lv.max()].mean() - pt[lv == 0].mean()
            def slope(pt, xs):
                x0 = xs - xs.mean(); return float((x0 * (pt - pt.mean())).sum() / (x0 ** 2).sum()) * 0.05
            est_d = np.mean([effect(pt, lv) for pt, bt, lv, xs in per_fold])
            est_s = np.mean([slope(pt, xs) for pt, bt, lv, xs in per_fold])
            bd, bs = np.empty(B), np.empty(B)
            for b in range(B):
                ds, ss = [], []
                for pt, bt, lv, xs in per_fold:
                    J = np.concatenate([rng.choice(np.flatnonzero(lv == k), (lv == k).sum()) for k in range(lv.max() + 1)])
                    ds.append(effect(bt[J, b], lv[J])); ss.append(slope(bt[J, b], xs[J]))
                bd[b], bs[b] = np.mean(ds), np.mean(ss)
            for name, est, bb in (("top_minus_bottom_level", est_d, bd), ("slope_per_0.05", est_s, bs)):
                rows.append({"method": m, "metric": metric, "effect": name, "estimate": float(est),
                             "lo": float(np.percentile(bb, 2.5)), "hi": float(np.percentile(bb, 97.5)),
                             "p_boot": float(min(1, 2 * min((bb <= 0).mean(), (bb >= 0).mean())))})
            for k in range(LEVELS):
                rows.append({"method": m, "metric": metric, "effect": f"mean_level{k}",
                             "estimate": float(np.mean([pt[lv == k].mean() for pt, bt, lv, xs in per_fold])),
                             "R_design": float(np.mean([xs[lv == k].mean() for pt, bt, lv, xs in per_fold]))})
        print(m, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, f"splitdose_effects{SUF}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(d[d.effect.str.startswith(("top", "slope"))].round(4).to_string(index=False))
    print(d[d.effect.str.startswith("mean")].pivot_table(index=["metric", "effect"], columns="method", values="estimate").round(3).to_string())
    print([round(F["R_small_strong_design"], 3) for F in P["folds"]], [(round(F["dose"]["level0"]["target_R"], 3), round(F["dose"][f"level{LEVELS-1}"]["target_R"], 3)) for F in P["folds"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "run", "analyze"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"build": build, "run": lambda: run(a.workers), "analyze": analyze}[a.phase]()


if __name__ == "__main__":
    main()
