"""Fetch v1 HellaSwag (10-shot) per-sample predictions for the analysis population.

Answer = argmax of log-likelihood normalised by continuation length in characters, which is how the
leaderboard's acc_norm scores HellaSwag; the raw argmax is stored too. Rows are kept in validation order
(item index = row index), which is verified against the validation labels when the truth table is built.

  python scripts/s3b_hellaswag.py --workers 8 --shard i --nshards k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s3_fetch as S  # noqa: E402
from depstudy.hf import DATA_ROOT, file_path_from_name, readme_config_splits  # noqa: E402

OUT = os.path.join(DATA_ROOT, "preds_hellaswag")


def _cols_hs(names):
    text = "query" if "query" in names else "example"
    return [c for c in (text, "predictions", "metrics", "acc", "acc_norm") if c in names]


S._cols = _cols_hs  # the shared readers project these columns instead


def parse(df):
    """Store the four raw log-likelihoods, the leaderboard's acc / acc_norm flags and a text key per row.
    Newer files leave `choices`/`gold` empty, so length normalisation and gold come from the validation set
    when the truth table is joined (s0b_extra_truth.py)."""
    from depstudy.textnorm import norm_key
    preds = [S._as_list(p) for p in df["predictions"]]
    P = np.full((len(df), 4), np.nan, dtype=np.float32)
    for i, p in enumerate(preds):
        if len(p) == 4:
            P[i] = np.asarray(p, dtype=np.float32)
    if "metrics" in df.columns:
        acc = [m.get("acc") if isinstance(m, dict) else None for m in df["metrics"]]
        accn = [m.get("acc_norm") if isinstance(m, dict) else None for m in df["metrics"]]
    else:
        acc = list(df["acc"]) if "acc" in df.columns else [None] * len(df)
        accn = list(df["acc_norm"]) if "acc_norm" in df.columns else [None] * len(df)
    text = df["query"] if "query" in df.columns else df["example"]
    f = lambda v: None if v is None else float(str(v) in ("True", "1", "1.0") or (not isinstance(v, str) and float(v) > 0.5))
    return pd.DataFrame({"row": np.arange(len(df)), "p0": P[:, 0], "p1": P[:, 1], "p2": P[:, 2], "p3": P[:, 3],
                         "lb_acc": [f(v) for v in acc], "lb_acc_norm": [f(v) for v in accn],
                         "key": [norm_key(t)[:80] for t in text]})


def fetch(model_dir, cand):
    dst = os.path.join(OUT, model_dir + ".parquet")
    if os.path.exists(dst):
        return {"model_dir": model_dir, "status": "cached"}
    t0 = time.time()
    try:
        path = file_path_from_name(cand[-1].split("\t", 1)[1])
        tabs = S.lfs_tables(model_dir, {"hs": path}) if S.API.room(3) else S.resolve_projected(model_dir, {"hs": path})
        df = tabs.get("hs")
        if df is None or len(df) < 10000:
            df = S.resolve_table(model_dir, cand, 10000, skip_latest=df is not None)
        if df is None:
            return {"model_dir": model_dir, "status": "missing"}
        out = parse(df)
        out.to_parquet(dst)
        return {"model_dir": model_dir, "status": "ok", "rows": int(len(out)), "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        return {"model_dir": model_dir, "status": "error", "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    S.API = S.Budget(320 // a.nshards); S.RES = S.Budget(2800 // a.nshards)
    os.makedirs(OUT, exist_ok=True)
    order = [l.strip() for l in open(os.path.join(ROOT, "data", "fetch_order_first3000.txt"), encoding="utf-8") if l.strip()]
    jobs = []
    for m in order[a.shard::a.nshards]:
        t = open(os.path.join(DATA_ROOT, "readme", m + ".md"), encoding="utf-8").read()
        cand = readme_config_splits(t).get("harness_hellaswag_10", [])
        if cand:
            jobs.append((m, cand))
    log = os.path.join(DATA_ROOT, f"fetch_hellaswag_shard{a.shard}.jsonl")
    lock = threading.Lock()
    print("models", len(jobs), flush=True)
    t0 = time.time()
    def run(j):
        rec = fetch(*j)
        with lock, open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(run, jobs)):
            if i % 50 == 0:
                print(i, round(time.time() - t0), rec, flush=True)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
