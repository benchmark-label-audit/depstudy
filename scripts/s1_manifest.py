"""Step 1: inventory of Open LLM Leaderboard v1 per-model details repositories.

Lists every open-llm-leaderboard-old/details_* dataset, downloads each README (resolve endpoint, cached on
D:), and records the data files available for the 57 MMLU subjects (5-shot) and GSM8K (5-shot).

  python scripts/s1_manifest.py   ->  data/manifest.parquet, data/manifest_stats.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import DATA_ROOT, details_base, get_bytes, readme_config_splits  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "readme")


def list_repos() -> list[str]:
    ids, url = [], "https://huggingface.co/api/datasets?author=open-llm-leaderboard-old&limit=1000"
    import urllib.request
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-script"}), timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
            link = r.headers.get("Link", "") or ""
        ids += [d["id"].split("/", 1)[1] for d in data if "/details_" in d["id"]]
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return sorted(set(i[len("details_"):] for i in ids))


def readme(model_dir: str) -> str | None:
    p = os.path.join(CACHE, model_dir + ".md")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return open(p, encoding="utf-8").read()
    b = get_bytes(details_base(model_dir) + "README.md")
    if b is None:
        return None
    t = b.decode("utf-8", errors="replace")
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)
    return t


def inspect(model_dir: str, subjects: list[str]) -> dict:
    t = readme(model_dir)
    if t is None:
        return {"model_dir": model_dir, "readme": False}
    cfg = readme_config_splits(t)
    files = {}
    for s in subjects:
        files["mmlu:" + s] = cfg.get(f"harness_hendrycksTest_{s}_5", [])
    files["gsm8k"] = cfg.get("harness_gsm8k_5", [])
    n_mmlu = sum(1 for s in subjects if files["mmlu:" + s])
    return {"model_dir": model_dir, "readme": True, "n_mmlu_subjects": n_mmlu, "has_gsm8k": bool(files["gsm8k"]),
            "files": json.dumps(files)}


def main():
    os.makedirs(CACHE, exist_ok=True)
    subjects = sorted(pd.read_parquet(os.path.join(ROOT, "data", "truth_mmlu.parquet")).subject.unique())
    assert len(subjects) == 57, len(subjects)
    repos = list_repos()
    print("repos", len(repos), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, r in enumerate(ex.map(lambda m: inspect(m, subjects), repos)):
            rows.append(r)
            if i % 500 == 0:
                print(i, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "data", "manifest.parquet"))
    ok = df.readme.fillna(False)
    complete = ok & df.n_mmlu_subjects.fillna(0).eq(57) & df.has_gsm8k.fillna(False)
    stats = {"repos": int(len(df)), "readme_ok": int(ok.sum()),
             "all_57_mmlu": int((ok & df.n_mmlu_subjects.fillna(0).eq(57)).sum()),
             "gsm8k": int((ok & df.has_gsm8k.fillna(False)).sum()), "complete_mmlu_and_gsm8k": int(complete.sum())}
    json.dump(stats, open(os.path.join(ROOT, "data", "manifest_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
