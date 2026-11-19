"""Refetch the raw GSM8K generations of every model in the v1 population and store them with two answer
extractions: the strict v1 harness rule ('#### <number>', the rule used throughout) and the lenient
'flexible-extract' rule of later lm-evaluation-harness versions (the last number in the generation).

The fetch reuses the routes of s3_fetch (LFS batch or /resolve/ with column-projected CDN reads). For each model
it reads the same newest GSM8K details file as s3_fetch and checks that the strict extraction reproduces the
stored answer; models whose strict extraction disagrees on more than 2% of items are flagged.

  python scripts/s27_gsm8k_raw.py --workers 12  ->  $DEPSTUDY_DATA/preds_gsm8k_raw/<model>.parquet, gsm8k_raw_log.jsonl
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s3_fetch as F  # noqa: E402
from depstudy.hf import DATA_ROOT, file_path_from_name  # noqa: E402
from depstudy.textnorm import canon_number, gsm8k_pred, norm_key  # noqa: E402

OUT = os.path.join(DATA_ROOT, "preds_gsm8k_raw")
LOG = os.path.join(DATA_ROOT, "gsm8k_raw_log.jsonl")
# lm-evaluation-harness gsm8k 'flexible-extract': regex (-?[$0-9.,]{2,})|(-?[0-9]+), last match
_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def flexible(gen):
    ms = _FLEX.findall(str(gen))
    if not ms:
        return None
    a, b = ms[-1]
    return canon_number((a or b).strip().rstrip("."))


def fetch_one(model_dir, files, gg):
    dst = os.path.join(OUT, model_dir + ".parquet")
    if os.path.exists(dst):
        return {"model_dir": model_dir, "status": "cached"}
    t0 = time.time()
    try:
        cand = files.get("gsm8k") or []
        latest = file_path_from_name(cand[-1].split("\t", 1)[1]) if cand else None
        if not latest:
            return {"model_dir": model_dir, "status": "nofile"}
        route = None
        while route is None:
            if F.API.room(3):
                route = "lfs"
            elif F.RES.room(1):
                route = "resolve"
            else:
                time.sleep(1.0)
        tables = F.lfs_tables(model_dir, {"gsm8k": latest}) if route == "lfs" else F.resolve_projected(model_dir, {"gsm8k": latest})
        df = tables.get("gsm8k")
        if df is not None and len(df) < 1300:
            df = F.resolve_table(model_dir, cand, 1300, skip_latest=True)
        elif df is None:
            df = F.resolve_table(model_dir, cand, 1300)
        if df is None:
            return {"model_dir": model_dir, "status": "missing", "route": route}
        col = "example" if "example" in df.columns else "question"
        rows = []
        for ex, preds in zip(df[col], df["predictions"]):
            hit = gg.get(norm_key(ex))
            if hit is None:
                continue
            pl = F._as_list(preds)
            gen = str(pl[0]) if pl else ""
            rows.append((hit[0], gen, gsm8k_pred(gen), flexible(gen)))
        out = pd.DataFrame(rows, columns=["item_id", "generation", "strict", "flexible"]).drop_duplicates("item_id")
        old = pd.read_parquet(os.path.join(F.OUT, model_dir + ".parquet"), columns=["bench", "item_id", "answer"])
        old = old[old.bench == "gsm8k"].set_index("item_id").answer
        j = out.set_index("item_id").strict.reindex(old.index)
        agree = float(((j == old) | (j.isna() & old.isna())).mean()) if len(old) else float("nan")
        out.to_parquet(dst)
        return {"model_dir": model_dir, "status": "ok", "route": route, "n": len(out), "strict_agrees_stored": round(agree, 4),
                "frac_strict_none": round(float(out.strict.isna().mean()), 4),
                "frac_flex_none": round(float(out.flexible.isna().mean()), 4), "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        return {"model_dir": model_dir, "status": "error", "error": repr(e)[:300], "trace": traceback.format_exc()[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    F.API = F.Budget(320)
    F.RES = F.Budget(2800)
    os.makedirs(OUT, exist_ok=True)
    from s4_panels import eligible_models
    models = eligible_models(None)
    man = pd.read_parquet(os.path.join(ROOT, "data", "manifest.parquet")).set_index("model_dir")
    mm, gg, size = F.load_truth()
    jobs = [(m, json.loads(man.files[m])) for m in models if m in man.index]
    if a.limit:
        jobs = jobs[:a.limit]
    print("models", len(jobs), flush=True)
    t0 = time.time()
    lock = threading.Lock()

    def run(j):
        rec = fetch_one(j[0], j[1], gg)
        with lock, open(LOG, "a", encoding="utf-8") as log:
            log.write(json.dumps(rec) + "\n")
        return rec
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(run, jobs)):
            if i % 50 == 0:
                print(i, round(time.time() - t0), {k: rec.get(k) for k in ("model_dir", "status", "route", "seconds", "strict_agrees_stored")}, flush=True)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
