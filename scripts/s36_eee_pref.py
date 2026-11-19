"""Preference benchmarks of Land & Bikel (2026) from Every Eval Ever: RewardBench, RewardBench 2 and JudgeBench, with
LLMs acting as pairwise / 4-way judges. One samples file per judge model; the answer is the response the judge
prefers ('chosen[0]' is the key, 'rejected[i]' the alternatives).

Fetch (throttled tree walk with back-off on HTTP rate limits) -> eee/<bench>.parquet (model, sample_id, answer, gold)
in the same layout as s13, with sample ids normalised to Land & Bikel's 'task.index' form.

  python scripts/s36_eee_pref.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s13_eee_fetch as E  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

OUT = os.path.join(DATA_ROOT, "eee")
BENCHES = ["reward-bench", "reward-bench-2", "judgebench"]


def tree(p):
    delay = 30
    for _ in range(12):
        try:
            t = E.sess().get(E.API + p, timeout=60).json()
        except Exception:
            t = None
        if isinstance(t, list):
            time.sleep(1.0)
            return t
        print("  waiting on", p, str(t)[:80], flush=True)
        time.sleep(delay); delay = min(delay * 2, 600)
    return []


def walk(p, depth=0):
    out = []
    for e in tree(p):
        if e.get("type") == "directory" and depth < 5:
            out += walk(e["path"], depth + 1)
        elif e.get("path", "").endswith("_samples.jsonl"):
            out.append(e["path"])
    return out


def val(x):
    if isinstance(x, str):
        try:
            import ast
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def parse(path):
    for attempt in range(6):
        try:
            r = E.sess().get(E.RES + path, timeout=180)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(20 * (attempt + 1))
    else:
        return []
    rows = []
    for line in r.text.splitlines():
        try:
            x = json.loads(line)
        except Exception:
            continue
        ans = None
        for a in val(x.get("answer_attribution")) or []:
            if isinstance(a, dict) and a.get("is_terminal"):
                ans = a.get("extracted_value")
        inp = val(x.get("input")) or {}
        ref = (inp.get("reference") if isinstance(inp, dict) else None) or [None]
        ev = val(x.get("evaluation")) or {}
        sid = str(x.get("sample_id")).replace(":", ".")
        rows.append((x.get("model_id"), sid, None if ans is None else str(ans).strip(), None if ref[0] is None else str(ref[0]).strip(),
                     ev.get("is_correct") if isinstance(ev, dict) else None))
    return rows


def main():
    stats = {}
    for b in BENCHES:
        files = walk("data/" + b)
        print(b, "files", len(files), flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = [r for part in ex.map(parse, files) for r in part]
        df = pd.DataFrame(rows, columns=["model", "sample_id", "answer", "gold", "correct"]).drop_duplicates(["model", "sample_id"])
        df.to_parquet(os.path.join(OUT, f"{b}.parquet"))
        chk = df.dropna(subset=["correct", "answer"])
        stats[b] = {"files": len(files), "models": int(df.model.nunique()), "items": int(df.sample_id.nunique()), "rows": int(len(df)),
                    "answers": df.answer.value_counts().head(6).to_dict(),
                    "parse_agrees_with_is_correct": round(float(((chk.answer == chk.gold) == chk.correct.astype(bool)).mean()), 4) if len(chk) else None}
        print(b, stats[b], flush=True)
    json.dump(stats, open(os.path.join(OUT, "eee_pref_stats.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
