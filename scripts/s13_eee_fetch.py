"""Fetch the Every Eval Ever (evaleval/EEE_datastore) instance-level records for the factual multiple-choice
benchmarks used by Land & Bikel (2026): GSM-MC, MATH-MC, GPQA-Diamond. One samples file per model.

Output (on D:): eee/<bench>.parquet with (model, sample_id, answer, gold, correct), and Land & Bikel's released
flagged items (Writer/IRT-mislabeled-items) as eee/landbikel_flagged.parquet.

  python scripts/s13_eee_fetch.py
"""
import io
import json
import os
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import DATA_ROOT  # noqa: E402

OUT = os.path.join(DATA_ROOT, "eee")
API = "https://huggingface.co/api/datasets/evaleval/EEE_datastore/tree/main/"
RES = "https://huggingface.co/datasets/evaleval/EEE_datastore/resolve/main/"
BENCHES = ["gsm-mc", "math-mc", "gpqa-diamond"]
_tls = threading.local()


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session(); s.trust_env = False; s.headers["User-Agent"] = "research-script"
        _tls.s = s
    return s


def tree(p):
    return sess().get(API + p, timeout=60).json()


def sample_files(bench):
    files = []
    for org in tree("data/" + bench):
        if org["type"] != "directory":
            continue
        for m in tree(org["path"]):
            if m["type"] != "directory":
                continue
            for f in tree(m["path"]):
                if f["path"].endswith("_samples.jsonl"):
                    files.append(f["path"])
    return files


def parse(path):
    r = sess().get(RES + path, timeout=120)
    if r.status_code != 200:
        return []
    rows = []
    for line in r.text.splitlines():
        try:
            x = json.loads(line)
        except Exception:
            continue
        ans = None
        for a in x.get("answer_attribution") or []:
            if a.get("is_terminal"):
                ans = a.get("extracted_value")
        ref = (x.get("input") or {}).get("reference") or [None]
        ev = x.get("evaluation") or {}
        rows.append((x.get("model_id"), x.get("sample_id"), None if ans is None else str(ans).strip()[:1].upper(),
                     None if ref[0] is None else str(ref[0]).strip()[:1].upper(), ev.get("is_correct")))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for b in BENCHES:
        files = sample_files(b)
        with ThreadPoolExecutor(max_workers=16) as ex:
            rows = [r for part in ex.map(parse, files) for r in part]
        df = pd.DataFrame(rows, columns=["model", "sample_id", "answer", "gold", "correct"]).drop_duplicates(["model", "sample_id"])
        df.to_parquet(os.path.join(OUT, f"{b}.parquet"))
        chk = df.dropna(subset=["correct", "answer"])
        stats[b] = {"files": len(files), "models": int(df.model.nunique()), "items": int(df.sample_id.nunique()), "rows": int(len(df)),
                    "parse_agrees_with_is_correct": round(float(((chk.answer == chk.gold) == chk.correct.astype(bool)).mean()), 4)}
        print(b, stats[b], flush=True)
    z = sess().get("https://huggingface.co/datasets/Writer/IRT-mislabeled-items/resolve/main/mislabeled_items.jsonl.zip", timeout=120).content
    zf = zipfile.ZipFile(io.BytesIO(z))
    name = [n for n in zf.namelist() if n.endswith(".jsonl")][0]
    lb = pd.DataFrame([json.loads(l) for l in zf.read(name).decode("utf-8").splitlines() if l.strip()])
    lb.to_parquet(os.path.join(OUT, "landbikel_flagged.parquet"))
    stats["landbikel"] = {"rows": int(len(lb)), "benchmarks": lb.benchmark.value_counts().to_dict()}
    json.dump(stats, open(os.path.join(OUT, "eee_stats.json"), "w"), indent=1)
    print(stats["landbikel"])


if __name__ == "__main__":
    main()
