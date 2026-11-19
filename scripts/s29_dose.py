"""Dependence experiments.

D2  Dose ladder at high capability. Pool: the 400 most accurate models on the evaluation benchmark E. Panels of
    100 models are annealed to a target mean residual correlation, estimated on the design benchmark D from the
    whole-population 2PL fit as for the cells (GSM8K for MMLU, MMLU for GSM8K and HellaSwag), while matching the pool's accuracy mean (0.01), accuracy SD (0.02) and,
    on MMLU, the four subject-group accuracies (0.015). Six target levels span the attainable range, which reaches
    the dependence of the small-strong panel; 10 panels per level.
D1  Manipulation check on E: every cell, dose, and single panel is re-measured with residual correlation and the
    same-wrong rate computed on the evaluation benchmark itself.
D3  Same-wrong rate: among items two models both answer incorrectly, the share on which they give the same wrong
    answer (the copying evidence used by DEPEN and by correlated-error studies).
D7  Capability-matched curation: for the deduplicated and small-diverse panels, 20 random panels of the same size
    matched to the curated panel's accuracy mean, SD and (MMLU) subject-group accuracies.

  python scripts/s29_dose.py build      -> results/dose_panels.json
  python scripts/s29_dose.py run --workers 6   -> results/ckpt_dose/*.json, item scores in the cache
  python scripts/s29_dose.py manip      -> results/dependence_manip.parquet
  python scripts/s29_dose.py collect    -> results/dose_scores.parquet
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
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CK = os.path.join(RES, "ckpt_dose")
CACHE = os.path.join(DATA_ROOT, "item_scores")
DESIGN = {"mmlu": "gsm8k", "gsm8k": "mmlu", "hellaswag": "mmlu"}  # as in s4 and s4b
POOL, N, LEVELS, REPS, CUR_REPS = 400, 100, 6, 10, 20
CAP_REPS_LARGE = 8  # matched panels scored for the deduplicated panels (of the 20 drawn)
SEED = 20261118
PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}


def methods_for(bench, n):
    ms = ["mv", "crh", "ds_onecoin_lo", "accu_nodep_lo", "glad_lo", "catd"]
    if bench == "gsm8k":
        ms.append("ds_onecoin_nf10_lo")
    else:
        ms.append("ds_full_lo")
    if n <= 400:
        ms += ["accu_depen_lo", "irt4pl"]
    return ms


def same_wrong(X, g):
    """(n_m, n_m) share of jointly wrong items on which two models give the same wrong answer, and the counts."""
    wrong = (X >= 0) & (X != g[:, None])
    from scipy import sparse
    W = wrong.astype(np.float32)
    both = W.T @ W
    V = int(X.max()) + 1
    # sparse (item x value) x model indicator of wrong answers; S^T S counts shared wrong values per model pair
    ii, jj = np.nonzero(wrong)
    A = sparse.csr_matrix((np.ones(len(ii), np.float32), (ii * V + X[ii, jj], jj)), shape=(X.shape[0] * V, X.shape[1]))
    same = (A.T @ A).toarray()
    S = same / np.maximum(both, 1)
    np.fill_diagonal(S, 0.0)
    return S, both


def panel_mean(Mx, idx):
    idx = np.asarray(idx)
    n = len(idx)
    return float(Mx[np.ix_(idx, idx)].sum() / (n * (n - 1)))


def anneal_target(n, allowed, R, feat, target, tol, rng, target_R=None, sign=0, steps=8000, lam=5.0, T0=0.02):
    """n models from `allowed` with mean pairwise R close to target_R (or maximal for sign=+1, minimal for -1, or
    unconstrained for sign=0 and target_R=None), with feature means / SD matched to target within tol."""
    allowed = np.asarray(allowed)
    P = list(rng.choice(allowed, n, replace=False))
    inP = np.zeros(R.shape[0], bool); inP[P] = True

    def pen(Pi):
        F = feat[Pi]
        dm = np.abs(F.mean(0) - target["mean"]) / tol["mean"]
        ds = np.abs(F[:, 0].std() - target["sd"]) / tol["sd"]
        return float(np.maximum(dm - 1, 0).sum() + max(ds - 1, 0))

    npairs = n * (n - 1) / 2
    S = R[np.ix_(P, P)].sum() / 2

    def obj(S_, Pi):
        m = S_ / npairs
        if target_R is not None:
            f = -abs(m - target_R) * 10
        else:
            f = sign * m
        return f - lam * pen(Pi)
    cur = obj(S, P)
    for t in range(steps):
        T = T0 * (1 - t / steps) + 1e-5
        k = rng.integers(n); a = P[k]
        cand = allowed[~inP[allowed]]
        b = cand[rng.integers(len(cand))]
        others = P[:k] + P[k + 1:]
        dS = R[b, others].sum() - R[a, others].sum()
        P2 = others + [b]
        new = obj(S + dS, P2)
        if new >= cur or rng.random() < np.exp((new - cur) / T):
            P = P2; S += dS; cur = new
            inP[a] = False; inP[b] = True
    return {"models": [int(x) for x in P], "mean_R": float(S / npairs), "penalty": pen(P)}


def _mats():
    import s19_logodds as S19
    return {b: S19.data(b) for b in ("mmlu", "gsm8k", "hellaswag")}


def _features(bench, t, Y, M, cols):
    from depstudy.panels import subject_group
    acc = (Y[:, cols] * M[:, cols]).sum(0) / np.maximum(M[:, cols].sum(0), 1)
    if bench == "mmlu":
        grp = t.subject.map(subject_group).values
        feats = [acc] + [(Y[grp == k][:, cols] * M[grp == k][:, cols]).sum(0) / np.maximum(M[grp == k][:, cols].sum(0), 1)
                         for k in range(4)]
        tol = np.array([0.01, 0.015, 0.015, 0.015, 0.015])
    else:
        feats = [acc]; tol = np.array([0.01])
    return np.stack(feats, 1), tol


def build():
    from depstudy.panels import residual_corr
    mats = _mats()
    rng = np.random.default_rng(SEED)
    out = {"pool": POOL, "n": N, "levels": LEVELS, "reps": REPS, "seed": SEED, "benches": {}}
    for E, D in DESIGN.items():
        tE, mE, XE, gE, YE, ME = mats[E]
        tD, mD, XD, gD, YD, MD = mats[D]
        iE = {m: i for i, m in enumerate(mE)}; iD = {m: i for i, m in enumerate(mD)}
        common = [m for m in mE if m in iD]
        cE = np.array([iE[m] for m in common]); cD = np.array([iD[m] for m in common])
        F, tolm = _features(E, tE, YE, ME, cE)
        acc = F[:, 0]
        # ---------------------------------------------------------------- D2 dose ladder in the top-capability pool
        pool = np.argsort(-acc)[:POOL]
        # residuals from the 2PL fit on the whole population, as for the cells and the single panels (s4)
        Rfull = residual_corr(YD[:, cD], MD[:, cD])
        Rp = Rfull[np.ix_(pool, pool)]
        Fp = F[pool]
        target = {"mean": Fp.mean(0), "sd": Fp[:, 0].std()}
        tol = {"mean": tolm, "sd": 0.02}
        loc = np.arange(POOL)
        lo = min((anneal_target(N, loc, Rp, Fp, target, tol, np.random.default_rng(rng.integers(1 << 31)), sign=-1)
                  for _ in range(3)), key=lambda r: r["mean_R"] + 10 * r["penalty"])
        hi = max((anneal_target(N, loc, Rp, Fp, target, tol, np.random.default_rng(rng.integers(1 << 31)), sign=+1)
                  for _ in range(3)), key=lambda r: r["mean_R"] - 10 * r["penalty"])
        levels = np.linspace(lo["mean_R"], hi["mean_R"], LEVELS)
        dose = {}
        for li, tr in enumerate(levels):
            reps = []
            for r in range(REPS):
                best = None
                for attempt in range(4):
                    cand = anneal_target(N, loc, Rp, Fp, target, tol, np.random.default_rng(rng.integers(1 << 31)), target_R=float(tr))
                    if cand["penalty"] == 0 and (best is None or abs(cand["mean_R"] - tr) < abs(best["mean_R"] - tr)):
                        best = cand
                    if best is not None and abs(best["mean_R"] - tr) < 0.003:
                        break
                if best is None:
                    continue
                Pm = [common[pool[i]] for i in best["models"]]
                reps.append({"models": Pm, "mean_R_design": best["mean_R"], "acc_mean": float(acc[pool[best["models"]]].mean()),
                             "acc_sd": float(acc[pool[best["models"]]].std())})
            dose[f"level{li}"] = {"target_R": float(tr), "reps": reps}
            print(E, "level", li, round(float(tr), 3), "reps", len(reps),
                  [round(x["mean_R_design"], 3) for x in reps][:4], flush=True)
        # ---------------------------------------------------------------- D7 capability-matched random panels
        P0 = json.load(open(os.path.join(RES, "panels_hellaswag.json" if E == "hellaswag" else "panels.json")))["benches"][E]["singles"]
        cur = {}
        idx_c = {m: i for i, m in enumerate(common)}
        allc = np.arange(len(common))
        Rz = np.zeros((len(common), len(common)), np.float32)  # unused objective; sign 0
        for name in ("corr_dedup", "small_diverse"):
            sel = np.array([idx_c[m] for m in P0[name]["models"] if m in idx_c])
            tgt = {"mean": F[sel].mean(0), "sd": F[sel, 0].std()}
            reps = []
            for r in range(CUR_REPS):
                cand = anneal_target(len(sel), allc, Rz, F, tgt, tol, np.random.default_rng(rng.integers(1 << 31)), sign=0,
                                     steps=4000 if len(sel) > 500 else 8000)
                if cand["penalty"] == 0:
                    reps.append({"models": [common[i] for i in cand["models"]], "acc_mean": float(acc[cand["models"]].mean())})
            cur[name] = {"size": int(len(sel)), "acc_mean": float(acc[sel].mean()), "reps": reps}
            print(E, "curation", name, len(sel), "matched reps", len(reps), flush=True)
        out["benches"][E] = {"design": D, "pool_acc_mean": float(acc[pool].mean()), "lo": lo["mean_R"], "hi": hi["mean_R"],
                             "dose": dose, "curation_matched": cur}
        json.dump(out, open(os.path.join(RES, "dose_panels.json"), "w"))


def tasks():
    P = json.load(open(os.path.join(RES, "dose_panels.json")))
    out = []
    for E, B in P["benches"].items():
        for lv, L in B["dose"].items():
            for r, p in enumerate(L["reps"]):
                out.append(("dose", E, lv, r, f"dose_{lv}_{r}", p["models"], methods_for(E, len(p["models"]))))
        for name, C in B["curation_matched"].items():
            for r, p in enumerate(C["reps"]):
                if name == "corr_dedup" and r >= CAP_REPS_LARGE:  # large panels vary little between draws
                    continue
                ms = ["mv", "crh", "ds_onecoin_lo", "accu_nodep_lo", "glad_lo", "catd"] + (["ds_onecoin_nf10_lo"] if E == "gsm8k" else [])
                out.append(("capmatch", E, name, r, f"capmatch_{name}_{r}", p["models"], ms))
    return out


def job(task):
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    import s19_logodds as S19
    import s28_newmethods as S28
    family, bench, cond, rep, tag, models, methods = task
    t, mods, X, g, Y, M = S19.data(bench)
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    y = t.defective.values.astype(int)
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    rows = []
    for m in methods:
        p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
        t0 = time.time()
        if os.path.exists(p):
            s = np.load(p); sec = float("nan")
        else:
            s = np.asarray(S28.score(m, X[:, c], g, Y[:, c], M[:, c], bench), np.float64); sec = time.time() - t0
            np.save(p + ".tmp.npy", s); os.replace(p + ".tmp.npy", p)
        rows.append({"family": family, "bench": bench, "condition": cond, "rep": rep, "n_models": int(len(c)), "method": m,
                     "ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, PK[bench]),
                     "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)), "sec": sec})
    key = f"{family}__{bench}__{cond}__{rep}"
    with open(os.path.join(CK, key + ".json.tmp"), "w") as fh:
        json.dump(rows, fh)
    os.replace(os.path.join(CK, key + ".json.tmp"), os.path.join(CK, key + ".json"))
    return key


def run(workers):
    os.makedirs(CK, exist_ok=True)
    T = [t for t in tasks() if not os.path.exists(os.path.join(CK, f"{t[0]}__{t[1]}__{t[2]}__{t[3]}.json"))]
    light = [t for t in T if len(t[5]) < 1000]
    heavy = [t for t in T if len(t[5]) >= 1000]
    print("todo", len(light), len(heavy), flush=True)
    for group, w in ((light, workers), (heavy, workers)):
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(job, t) for t in group]
            for i, f in enumerate(as_completed(futs)):
                k = f.result()
                if i % 25 == 0:
                    print(i + 1, len(group), k, time.strftime("%H:%M:%S"), flush=True)


def manip():
    """D1 / D3: dependence of every cell, dose, single, and matched panel measured on the evaluation benchmark
    (residual correlation and same-wrong rate) and on the design benchmark (same-wrong rate; residual correlation
    as stored)."""
    from depstudy.panels import residual_corr
    mats = _mats()
    rows = []
    DP = json.load(open(os.path.join(RES, "dose_panels.json")))
    for E, D in DESIGN.items():
        tE, mE, XE, gE, YE, ME = mats[E]
        tD, mD, XD, gD, YD, MD = mats[D]
        iE = {m: i for i, m in enumerate(mE)}; iD = {m: i for i, m in enumerate(mD)}
        common = [m for m in mE if m in iD]
        ci = {m: k for k, m in enumerate(common)}
        cE = np.array([iE[m] for m in common]); cD = np.array([iD[m] for m in common])
        t0 = time.time()
        R_E = residual_corr(YE[:, cE], ME[:, cE])
        R_D = residual_corr(YD[:, cD], MD[:, cD])
        SW_E, _ = same_wrong(XE[:, cE], gE)
        SW_D, _ = same_wrong(XD[:, cD], gD)
        print(E, "dependence matrices", round(time.time() - t0), "s", flush=True)
        P0 = json.load(open(os.path.join(RES, "panels_hellaswag.json" if E == "hellaswag" else "panels.json")))["benches"][E]
        panels = []
        for name, s in P0["singles"].items():
            panels.append(("single", name, 0, s["models"]))
        for name, reps in P0["cells"].items():
            for r, p in enumerate(reps):
                panels.append(("cell", name, r, p["models"]))
        for lv, L in DP["benches"][E]["dose"].items():
            for r, p in enumerate(L["reps"]):
                panels.append(("dose", lv, r, p["models"]))
        for name, C in DP["benches"][E]["curation_matched"].items():
            for r, p in enumerate(C["reps"]):
                panels.append(("capmatch", name, r, p["models"]))
        accE = (YE[:, cE] * ME[:, cE]).sum(0) / np.maximum(ME[:, cE].sum(0), 1)
        for kind, name, r, models in panels:
            idx = np.array([ci[m] for m in models if m in ci])
            if len(idx) > 1500:  # full-size panels: sample 1,500 models for the pairwise means
                idx = np.random.default_rng(0).choice(idx, 1500, replace=False)
            rows.append({"bench": E, "kind": kind, "condition": name, "rep": r, "n": len(models),
                         "R_design_full_pool": panel_mean(R_D, idx), "R_eval": panel_mean(R_E, idx),
                         "samewrong_design": panel_mean(SW_D, idx), "samewrong_eval": panel_mean(SW_E, idx),
                         "acc_mean": float(accE[idx].mean())})
    pd.DataFrame(rows).to_parquet(os.path.join(RES, "dependence_manip.parquet"))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(d.groupby(["bench", "kind", "condition"])[["R_design_full_pool", "R_eval", "samewrong_design", "samewrong_eval", "acc_mean"]].mean().round(3).to_string())


def collect():
    rows = [r for f in glob.glob(os.path.join(CK, "*.json")) for r in json.load(open(f))]
    pd.DataFrame(rows).to_parquet(os.path.join(RES, "dose_scores.parquet"))
    print("rows", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "run", "manip", "collect"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.phase == "build":
        build()
    elif a.phase == "run":
        run(a.workers)
    elif a.phase == "manip":
        manip()
    else:
        collect()


if __name__ == "__main__":
    main()
