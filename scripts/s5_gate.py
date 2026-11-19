"""Step 5: run every method on every panel and evaluate against the defect annotations.

  python scripts/s5_gate.py --workers 24  ->  results/gate_scores.parquet (one row per panel x method)
  python scripts/s5_gate.py --report      ->  results/gate_report.json (gate contrasts)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PK = {"mmlu": 200, "gsm8k": 100}
NFALSE = {"mmlu": 3, "gsm8k": 10}
_M = {}


def _K(bench):
    from depstudy.data import n_options
    return n_options(bench)


def _all_models(benches):
    pan = json.load(open(os.path.join(ROOT, "results", "panels.json")))
    allm = set()
    for b in benches:
        for c in pan["benches"][b]["cells"].values():
            for r in c:
                allm |= set(r["models"])
        for s in pan["benches"][b]["singles"].values():
            allm |= set(s["models"])
    return sorted(allm)


def _init(benches):
    from depstudy.data import matrices
    allm = _all_models(benches)
    for b in benches:
        t, models, X, g, Y, M = matrices(allm, b)
        _M[b] = (t, {m: i for i, m in enumerate(models)}, X, g, Y, M)


def run_methods(bench, models, skip_depen=False):
    from depstudy import methods as Mt
    t, idx, X, g, Y, M = _M[bench]
    cols = np.array([idx[m] for m in models if m in idx])
    Xp, Yp, Mp = X[:, cols], Y[:, cols], M[:, cols]
    y = t.defective.values.astype(int)
    out = {}
    def rec(name, s, sec):
        out[name] = {"ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, PK[bench]), "sec": sec}
    for name, f in [("mv", lambda: Mt.majority_vote(Xp, g)),
                    ("ds_onecoin", lambda: Mt.dawid_skene_onecoin(Xp, g, K=_K(bench))[0]),
                    ("crh", lambda: Mt.crh(Xp, g)[0]),
                    ("irt4pl", lambda: Mt.irt_ceiling_lr(Xp, g, Yp, Mp)),
                    ("accu_nodep", lambda: Mt.accu_depen(Xp, g, NFALSE[bench], use_dependence=False, K=_K(bench))[0])]:
        t0 = time.time(); s = f(); rec(name, s, time.time() - t0)
    if bench == "mmlu":
        t0 = time.time(); s = Mt.dawid_skene_full(Xp, g, 4)[0]; rec("ds_full", s, time.time() - t0)
    if not skip_depen:
        t0 = time.time(); s = Mt.accu_depen(Xp, g, NFALSE[bench], K=_K(bench))[0]; rec("accu_depen", s, time.time() - t0)
    return out, len(cols)


def job(spec):
    bench, kind, name, rep, models, skip = spec
    res, n = run_methods(bench, models, skip)
    return [{"bench": bench, "kind": kind, "condition": name, "rep": rep, "n_models": n, "method": m, **v} for m, v in res.items()]


CKPT = os.path.join(ROOT, "results", "ckpt_gate")
HEAVY_MIN_MODELS = 1000  # panels this large run in the low-concurrency phase


def _key(spec):
    b, kind, name, rep, models, skip = spec
    return f"{b}__{kind}__{name}__{rep}"


def main():
    """Every finished job is written to results/ckpt_gate/<key>.json immediately, so an interrupted run
    resumes where it stopped. --phase light runs panels under HEAVY_MIN_MODELS models with many workers;
    --phase heavy runs the large single panels with few workers (memory); --phase collect assembles."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--phase", default="all", choices=["light", "heavy", "all", "collect"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    os.makedirs(CKPT, exist_ok=True)
    pan = json.load(open(os.path.join(ROOT, "results", "panels.json")))
    specs = []
    for b, B in pan["benches"].items():
        for cname, reps in B["cells"].items():
            for r, p in enumerate(reps):
                specs.append((b, "cell", cname, r, p["models"], False))
        for sname, s in B["singles"].items():
            specs.append((b, "single", sname, 0, s["models"], sname == "full" and len(s["models"]) > 1500))
    if a.phase != "collect":
        heavy = [s for s in specs if len(s[4]) >= HEAVY_MIN_MODELS]
        light = [s for s in specs if len(s[4]) < HEAVY_MIN_MODELS]
        todo = {"light": light, "heavy": heavy, "all": light + heavy}[a.phase]
        todo = [s for s in todo if not os.path.exists(os.path.join(CKPT, _key(s) + ".json"))]
        benches = list(pan["benches"])
        from depstudy.data import matrices
        for b in benches:  # build the cached matrices once before the workers start
            matrices(_all_models(benches), b)
        print("phase", a.phase, "to run", len(todo), "of", len(specs), flush=True)
        t0 = time.time()
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(benches,)) as ex:
            futs = {ex.submit(job, s): s for s in todo}
            for i, f in enumerate(as_completed(futs)):
                rows = f.result()
                tmp = os.path.join(CKPT, _key(futs[f]) + ".json.tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(rows, fh)
                os.replace(tmp, os.path.join(CKPT, _key(futs[f]) + ".json"))
                if i % 10 == 0:
                    print(i, len(todo), round(time.time() - t0), _key(futs[f]), flush=True)
        print("phase done", round(time.time() - t0), flush=True)
    done = [os.path.join(CKPT, _key(s) + ".json") for s in specs]
    have = [p for p in done if os.path.exists(p)]
    print("checkpoints", len(have), "of", len(specs), flush=True)
    if len(have) == len(specs):
        rows = [r for p in have for r in json.load(open(p, encoding="utf-8"))]
        pd.DataFrame(rows).to_parquet(os.path.join(ROOT, "results", "gate_scores.parquet"))
        print("collected", len(rows), "rows", flush=True)


def report():
    df = pd.read_parquet(os.path.join(ROOT, "results", "gate_scores.parquet"))
    rep = {}
    rng = np.random.default_rng(7)
    for b in df.bench.unique():
        d = df[(df.bench == b) & (df.kind == "cell")]
        rep[b] = {}
        for m in d.method.unique():
            x = d[d.method == m]
            cell = {c: x[x.condition == c].pk.values for c in x.condition.unique()}
            def mean(c):
                return float(np.mean(cell[c])) if c in cell and len(cell[c]) else float("nan")
            def boot(diff_fn, B=2000):
                vals = []
                for _ in range(B):
                    s = {c: rng.choice(v, len(v)) for c, v in cell.items() if len(v)}
                    vals.append(diff_fn(s))
                return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            def dep_effect(s):
                return (np.mean(s["lineage_high__dep_high"]) + np.mean(s["lineage_low__dep_high"])
                        - np.mean(s["lineage_high__dep_low"]) - np.mean(s["lineage_low__dep_low"])) / 2
            def lin_effect(s):
                return (np.mean(s["lineage_high__dep_high"]) + np.mean(s["lineage_high__dep_low"])
                        - np.mean(s["lineage_low__dep_high"]) - np.mean(s["lineage_low__dep_low"])) / 2
            need = ["lineage_high__dep_high", "lineage_high__dep_low", "lineage_low__dep_high", "lineage_low__dep_low"]
            ok = all(c in cell and len(cell[c]) for c in need)
            rep[b][m] = {"cell_means_pk": {c: mean(c) for c in cell},
                         "dep_effect_pk": float(dep_effect(cell)) if ok else None,
                         "dep_effect_ci": boot(dep_effect) if ok else None,
                         "lineage_effect_pk": float(lin_effect(cell)) if ok else None,
                         "lineage_effect_ci": boot(lin_effect) if ok else None}
        s = df[(df.bench == b) & (df.kind == "single")]
        rep[b]["singles"] = {f"{r.condition}|{r.method}": {"pk": r.pk, "ap": r.ap, "n": int(r.n_models), "sec": r.sec}
                             for r in s.itertuples()}
        best = d.groupby(["condition", "method"]).pk.mean().reset_index()
        rep[b]["best_method_per_cell"] = best.loc[best.groupby("condition").pk.idxmax()].set_index("condition").method.to_dict()
    json.dump(rep, open(os.path.join(ROOT, "results", "gate_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1)[:6000])


if __name__ == "__main__":
    main()
