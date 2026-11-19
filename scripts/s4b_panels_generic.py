"""Generic copy of s4_panels.py (benchmark pairs and output file as arguments; used for extensions).
Step 4: construct the Phase 1 panels. Uses no defect annotations.

For each evaluation benchmark E (mmlu, gsm8k), residual dependence is estimated on the *other* benchmark.
Cells: lineage {low, high} x residual dependence {low, high}, plus random panels; R repetitions each.
Single panels: full population, correlation-deduplicated (Land & Bikel style), small-strong, small-diverse.

  python scripts/s4_panels.py --n 100 --reps 20  ->  results/panels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.data import matrices  # noqa: E402
from depstudy.panels import anneal_panel, hhi, residual_corr, subject_group  # noqa: E402

def eligible_models(cutoff_n: int | None):
    """Analysis population: models among the first 3,000 in the random fetch order that were fetched
    successfully and whose parse agrees with the leaderboard's correctness flags on >= 98% of rows."""
    from depstudy.data import fetch_records
    recs = fetch_records()
    order = [l.strip() for l in open(os.path.join(ROOT, "data", "fetch_order_first3000.txt"), encoding="utf-8") if l.strip()]
    if cutoff_n:
        order = order[:cutoff_n]
    return [m for m in order if m in recs and recs[m].get("lb_consistency", 0) >= 0.98]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cutoff", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--pairs", default="mmlu:gsm8k,gsm8k:mmlu")
    ap.add_argument("--out", default="panels.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    models = eligible_models(a.cutoff or None)
    lin = pd.read_parquet(os.path.join(ROOT, "data", "lineage.parquet")).set_index("model_dir")
    out = {"n": a.n, "reps": a.reps, "seed": a.seed, "eligible": len(models), "benches": {}}
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    benches = sorted({b for p in pairs for b in p})
    mats = {b: matrices(models, b) for b in benches}
    common = sorted(set.intersection(*[set(mats[b][1]) for b in benches]))
    prepared = {}
    for E, D in pairs:
        tE, mE, XE, gE, YE, ME = mats[E]
        tD, mD, XD, gD, YD, MD = mats[D]
        idxE = {m: i for i, m in enumerate(mE)}; idxD = {m: i for i, m in enumerate(mD)}
        pool = [m for m in common]
        cE = np.array([idxE[m] for m in pool]); cD = np.array([idxD[m] for m in pool])
        R = residual_corr(YD[:, cD], MD[:, cD])
        acc = (YE[:, cE] * ME[:, cE]).sum(0) / np.maximum(ME[:, cE].sum(0), 1)
        if E == "mmlu":
            grp = tE.subject.map(subject_group).values
            feats = [acc] + [(YE[grp == k][:, cE] * ME[grp == k][:, cE]).sum(0) / np.maximum(ME[grp == k][:, cE].sum(0), 1)
                             for k in range(4)]
            tol_mean = np.array([0.01, 0.015, 0.015, 0.015, 0.015])
        else:
            feats = [acc]; tol_mean = np.array([0.01])
        F = np.stack(feats, 1)
        # family ids are case-insensitive on Hugging Face (deviation D3: normalise case)
        famraw = [lin.family.get(m) for m in pool]
        fam = np.array([f.lower() if isinstance(f, str) else f"__undeclared__{m}" for f, m in zip(famraw, pool)], dtype=object)
        declared = np.array([isinstance(f, str) and not f.startswith("MULTI:") for f in famraw])
        dec_idx = np.nonzero(declared)[0]
        fam_sizes = pd.Series(fam[dec_idx]).value_counts()
        prepared[E] = dict(D=D, pool=pool, R=R, acc=acc, F=F, fam=fam, declared=declared, dec_idx=dec_idx,
                           fam_sizes=fam_sizes, tol_mean=tol_mean, YE=YE[:, cE])

    def build_cells(E, n):
        P = prepared[E]
        fam, dec_idx, F, R, acc = P["fam"], P["dec_idx"], P["F"], P["R"], P["acc"]
        top, tot = [], 0  # high-lineage candidate set: the fewest largest families with combined size >= 2n
        for f, c in P["fam_sizes"].items():
            top.append(f); tot += c
            if tot >= 2 * n and len(top) >= 3:
                break
        hi_allowed = [i for i in dec_idx if fam[i] in set(top)]
        lo_allowed = list(dec_idx)
        target = {"mean": F[hi_allowed].mean(0), "sd": F[hi_allowed][:, 0].std()}
        tol = {"mean": P["tol_mean"], "sd": 0.02}
        specs = {("high", "high"): (hi_allowed, n, +1), ("high", "low"): (hi_allowed, n, -1),
                 ("low", "high"): (lo_allowed, 2, +1), ("low", "low"): (lo_allowed, 2, -1)}
        cells, ok = {}, True
        for (lg, dp), (allowed, cap, sign) in specs.items():
            reps = []
            for r in range(a.reps):
                res = None
                for attempt in range(3):  # a repetition counts only with zero matching penalty
                    cand = anneal_panel(n, allowed, fam, cap, R, F, target, tol, sign, np.random.default_rng(rng.integers(1 << 31)))
                    if cand is not None and cand["penalty"] == 0:
                        res = cand; break
                if res is None:
                    ok = False; break
                reps.append({"models": [P["pool"][i] for i in res["models"]], "mean_R": res["mean_R"], "penalty": res["penalty"],
                             "hhi": hhi(list(fam[res["models"]])), "acc_mean": float(acc[res["models"]].mean()),
                             "acc_sd": float(acc[res["models"]].std())})
            cells[f"lineage_{lg}__dep_{dp}"] = reps
            if not ok:
                break
        return ok, cells, top, hi_allowed, target

    n = a.n
    while True:
        built = {E: build_cells(E, n) for E in prepared}
        fails = [E for E, b in built.items() if not b[0]]
        print("n", n, "failed cells in", fails, flush=True)
        if not fails or n <= 10:
            break
        n -= 10
    out["n"] = n
    for E, P in prepared.items():
        ok, cells, top, hi_allowed, target = built[E]
        pool, R, acc, fam, declared = P["pool"], P["R"], P["acc"], P["fam"], P["declared"]
        rand = []
        for r in range(a.reps):
            Pi = rng.choice(len(pool), n, replace=False)
            rand.append({"models": [pool[i] for i in Pi], "mean_R": float(R[np.ix_(Pi, Pi)].sum() / (n * (n - 1))),
                         "hhi": hhi(list(fam[Pi])), "acc_mean": float(acc[Pi].mean()), "acc_sd": float(acc[Pi].std())})
        cells["random"] = rand
        # single panels; correlation dedup on the evaluation benchmark's correctness (vectorised)
        order = np.argsort(-acc)
        Yc = P["YE"].astype(np.float64)
        sd = Yc.std(0)
        Z = (Yc - Yc.mean(0)) / np.where(sd > 0, sd, 1)
        C = (Z.T @ Z) / Yc.shape[0]
        zero = sd == 0  # identical constant correctness vectors are exact duplicates
        C[np.ix_(zero, zero)] = 1.0
        np.fill_diagonal(C, 0.0)
        sel = []
        for i in order:
            if not sel or C[i, sel].max() < 0.8:
                sel.append(i)
        best_per_fam, seen = [], set()
        for i in order:
            if declared[i] and fam[i] not in seen:
                seen.add(fam[i]); best_per_fam.append(i)
        singles = {"full": list(range(len(pool))), "corr_dedup": sel, "small_strong": list(order[:n]),
                   "small_diverse": best_per_fam[:n]}
        out["benches"][E] = {
            "design_bench": P["D"], "pool": len(pool), "declared": int(declared.sum()), "families_declared": int(len(P["fam_sizes"])),
            "top_families": top, "hi_candidates": len(hi_allowed), "target": {"mean": target["mean"].tolist(), "sd": float(target["sd"])},
            "cells": cells,
            "singles": {k: {"models": [pool[i] for i in v], "size": len(v), "hhi": hhi(list(fam[v])),
                            "mean_R": float(R[np.ix_(v, v)].sum() / max(len(v) * (len(v) - 1), 1)),
                            "acc_mean": float(acc[v].mean())} for k, v in singles.items()},
        }
        print(E, {k: len(v) for k, v in cells.items()}, {k: len(v) for k, v in singles.items()}, flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", a.out), "w"))


if __name__ == "__main__":
    main()
