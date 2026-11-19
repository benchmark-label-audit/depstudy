"""CrowdFM (ICLR 2026) as an additional baseline, run in the D: venv with torch (GPU), on: the gate panels of the
three v1 benchmarks, the cross-source panels (HELM vs v1), and Land & Bikel's benchmarks (Every Eval Ever).
Also a worker_block=100 variant on panels larger than its pretraining range (20-100 workers).

  <crowdfm-venv>/python -I scripts/s15_crowdfm.py  ->  results/crowdfm_scores.parquet
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.data import matrices, n_options  # noqa: E402
from crowdfm_adapter import crowdfm_scores  # noqa: E402

PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}
OUT = os.path.join(ROOT, "results", "crowdfm_scores.parquet")


def evaluate(s, t, bench):
    from s10_size_capability import key_mask
    y = t.defective.values.astype(int)
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    return {"ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, PK[bench]),
            "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int))}


def run_panel(D, bench, models, label, rows):
    t, mods, X, g, Y, M = D
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    for variant, wb in (("crowdfm", None), ("crowdfm_block100", 100)):
        if wb and len(c) <= 100:
            continue
        t0 = time.time()
        s = crowdfm_scores(X[:, c], g, n_options(bench), worker_block=wb)
        rows.append({"bench": bench, **label, "n_models": int(len(c)), "method": variant, "sec": time.time() - t0, **evaluate(s, t, bench)})


def main():
    from s4_panels import eligible_models
    models = eligible_models(None)
    rows = []
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT); rows = prev.to_dict(orient="records")
        done = {(r["bench"], r["kind"], r["condition"], r["rep"]) for r in rows}
    def save():
        pd.DataFrame(rows).to_parquet(OUT)
    for bench, pfile in (("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json")):
        D = matrices(models, bench)
        P = json.load(open(os.path.join(ROOT, "results", pfile)))["benches"][bench]
        for name, s in P["singles"].items():
            lab = {"kind": "single", "condition": name, "rep": 0}
            if (bench, "single", name, 0) not in done:
                run_panel(D, bench, s["models"], lab, rows); save()
        for cname, reps in P["cells"].items():
            for r, p in enumerate(reps):
                lab = {"kind": "cell", "condition": cname, "rep": r}
                if (bench, "cell", cname, r) not in done:
                    run_panel(D, bench, p["models"], lab, rows)
            save()
            print(bench, cname, "done", flush=True)
    # cross-source panels (same generator as s6)
    import s6_sources as S6
    rng = np.random.default_rng(20261116)
    for bench in ("mmlu", "gsm8k"):
        D = S6.load(bench)
        t, mods, X, g, Y, M = D
        helm = [m for m in mods if m.startswith("helm:")]
        v1 = [m for m in mods if not m.startswith("helm:")]
        acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
        k = len(helm)
        v1_strong_k = list(acc[v1].sort_values(ascending=False).index[:k])
        v1_strong_100 = list(acc[v1].sort_values(ascending=False).index[:100])
        specs = [("helm_all", 0, helm), ("v1_strong_k", 0, v1_strong_k), ("helm_plus_v1_strong100", 0, helm + v1_strong_100),
                 ("helm_plus_v1_all", 0, helm + v1), ("v1_all", 0, v1)]
        for r in range(20):
            choice = list(rng.choice(v1, k, replace=False))
            if r < 5:
                specs.append(("v1_random_k", r, choice))
        for name, rep, ms in specs:
            if (bench, "source", name, rep) not in done:
                run_panel(D, bench, ms, {"kind": "source", "condition": name, "rep": rep}, rows)
        save()
        print("sources", bench, "done", flush=True)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
