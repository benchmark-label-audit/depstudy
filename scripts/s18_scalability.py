"""Scalability: wall-clock time and peak memory of every method as the population grows, on MMLU (5,631 items)
and HellaSwag (10,036 items), n in {10, 30, 100, 300, 1000, 2999} models (seeded random subsets, nested).

Each (benchmark, n, method) runs in a fresh subprocess with only its panel matrix loaded; the subprocess reports
wall-clock seconds, the working set after loading the panel, the peak working set, and (CrowdFM) peak GPU memory.
Runs over TIMEOUT (40 min) seconds are recorded as timeouts. Runs under 60 s are repeated three times (median reported).
Run on an otherwise idle machine.

  <crowdfm-venv>/python -I scripts/s18_scalability.py   ->  results/scalability.csv
  ... s18_scalability.py --one <npz> <bench> <method>                            (subprocess entry point)
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

SIZES = [10, 30, 100, 300, 1000, 2999]
METHODS = ["mv", "crh", "catd", "glad", "ds_onecoin", "ds_full", "accu_nodep", "accu_depen", "irt4pl", "crowdfm", "crowdfm_block100"]
NF = {"mmlu": 3, "hellaswag": 3}
TIMEOUT = 2400
DIR = os.path.join(DATA_ROOT, "scal")
OUT = os.path.join(ROOT, "results", "scalability.csv")


def one(npz, bench, method):
    import psutil
    proc = psutil.Process()
    d = np.load(npz)
    X, g, Y, M = d["X"], d["g"], d["Y"], d["M"].astype(bool)
    K = 4
    gpu = None
    if method.startswith("crowdfm"):
        import torch
        from crowdfm_adapter import crowdfm_scores
        torch.cuda.reset_peak_memory_stats()
        _ = crowdfm_scores(X[:50, :5], g[:50], K)  # load weights / CUDA context outside the timed region
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    from depstudy import methods as Mt
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    if method == "mv":
        Mt.majority_vote(X, g)
    elif method == "crh":
        Mt.crh(X, g)
    elif method == "catd":
        Mt.catd(X, g)
    elif method == "glad":
        Mt.glad(X, g, K=K, logodds=True)
    elif method == "ds_onecoin":
        Mt.dawid_skene_onecoin(X, g, K=K, logodds=True)
    elif method == "ds_full":
        Mt.dawid_skene_full(X, g, K, logodds=True)
    elif method == "accu_nodep":
        Mt.accu_depen(X, g, NF[bench], use_dependence=False, K=K, logodds=True)
    elif method == "accu_depen":
        Mt.accu_depen(X, g, NF[bench], K=K, logodds=True)
    elif method == "irt4pl":
        Mt.irt_ceiling_lr(X, g, Y, M)
    elif method == "crowdfm":
        crowdfm_scores(X, g, K); torch.cuda.synchronize()
    elif method == "crowdfm_block100":
        crowdfm_scores(X, g, K, worker_block=100); torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", None) or getattr(mi, "peak_rss", None) or mi.rss
    if method.startswith("crowdfm"):
        gpu = torch.cuda.max_memory_allocated() / 2**20
    print("RESULT " + json.dumps({"sec": sec, "rss_after_load_mb": rss0 / 2**20, "peak_mb": peak / 2**20, "gpu_peak_mb": gpu}))


def prepare():
    from depstudy.data import matrices
    from s4_panels import eligible_models
    os.makedirs(DIR, exist_ok=True)
    models = eligible_models(None)
    for bench in ("mmlu", "hellaswag"):
        t, mods, X, g, Y, M = matrices(models, bench)
        perm = np.random.default_rng(20261118).permutation(len(mods))  # nested subsets: first n of one permutation
        for n in SIZES:
            p = os.path.join(DIR, f"{bench}_{n}.npz")
            if not os.path.exists(p):
                c = np.sort(perm[:n])
                np.savez(p, X=X[:, c], g=g, Y=Y[:, c].astype(np.float32), M=M[:, c])
        del X, Y, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=3)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--benches", default="mmlu,hellaswag")
    a = ap.parse_args()
    if a.one:
        return one(*a.one)
    import pandas as pd
    prepare()
    rows = pd.read_csv(OUT).to_dict(orient="records") if os.path.exists(OUT) else []
    done = {(r["bench"], int(r["n"]), r["method"]) for r in rows}
    import psutil
    hw = {"cpu": platform.processor(), "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    try:
        import torch
        hw["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    json.dump(hw, open(os.path.join(ROOT, "results", "scalability_hw.json"), "w"), indent=1)
    for bench in a.benches.split(","):
        for method in a.methods.split(","):
            timed_out = False
            for n in SIZES:
                if (bench, n, method) in done:
                    timed_out = any(r["timeout"] for r in rows if r["bench"] == bench and r["method"] == method and int(r["n"]) == n)
                    continue
                if method == "ds_full" and bench != "mmlu":
                    continue
                if timed_out:  # larger n would also exceed the limit
                    rows.append({"bench": bench, "n": n, "method": method, "timeout": True, "sec": np.nan, "rss_after_load_mb": np.nan,
                                 "peak_mb": np.nan, "gpu_peak_mb": np.nan, "reps": 0, "skipped_after_timeout": True})
                    pd.DataFrame(rows).to_csv(OUT, index=False)
                    continue
                res, reps = [], 0
                while reps < 3:
                    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--one", os.path.join(DIR, f"{bench}_{n}.npz"), bench, method]
                    try:
                        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=ROOT)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        break
                    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                    if not line:
                        print("FAILED", bench, n, method, p.stderr[-800:], flush=True)
                        break
                    res.append(json.loads(line[0][7:])); reps += 1
                    if res[-1]["sec"] >= 60:
                        break
                if timed_out:
                    rows.append({"bench": bench, "n": n, "method": method, "timeout": True, "sec": np.nan, "rss_after_load_mb": np.nan,
                                 "peak_mb": np.nan, "gpu_peak_mb": np.nan, "reps": 0, "skipped_after_timeout": False})
                elif res:
                    med = {k: float(np.median([r[k] for r in res])) if res[0][k] is not None else np.nan for k in res[0]}
                    rows.append({"bench": bench, "n": n, "method": method, "timeout": False, **med, "reps": len(res), "skipped_after_timeout": False})
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(bench, n, method, rows[-1].get("sec"), "s", rows[-1].get("peak_mb"), "MB", "timeout" if timed_out else "", flush=True)


if __name__ == "__main__":
    main()
