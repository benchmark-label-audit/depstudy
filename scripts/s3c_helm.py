"""Fetch HELM per-instance predictions for MMLU (project 'mmlu') and GSM8K (project 'lite') from the public
crfm-helm-public bucket, and match them to the MMLU-Redux and GSM8K-Platinum truth tables.

A model evaluated in several releases keeps its latest release per (model, subject). Output (on D:):
helm/helm_mmlu.parquet and helm/helm_gsm8k.parquet with (model, item_id, answer, helm_correct), plus
helm/helm_stats.json.

  python scripts/s3c_helm.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import DATA_ROOT  # noqa: E402
from depstudy.textnorm import canon_number, mmlu_key, norm_key  # noqa: E402

OUT = os.path.join(DATA_ROOT, "helm")
GCS = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
OBJ = "https://storage.googleapis.com/crfm-helm-public/"
_tls = threading.local()


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session(); s.trust_env = False; s.headers["User-Agent"] = "research-script"
        _tls.s = s
    return s


def ls(prefix):
    out, tok = [], None
    while True:
        p = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if tok:
            p["pageToken"] = tok
        r = sess().get(GCS, params=p, timeout=60).json()
        out += r.get("prefixes", []); tok = r.get("nextPageToken")
        if not tok:
            return out


def ver(rel):
    return tuple(int(x) for x in re.findall(r"\d+", rel.rstrip("/").split("/")[-1]))


def get_json(path):
    for _ in range(4):
        try:
            r = sess().get(OBJ + requests.utils.quote(path, safe="/"), timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
    return None


def latest_runs(project, prefix):
    """(model, subject) -> run path, keeping the latest release."""
    best = {}
    for rel in sorted(ls(f"{project}/benchmark_output/runs/"), key=ver):
        for run in ls(rel):
            name = run.rstrip("/").split("/")[-1]
            if not name.startswith(prefix) or "model=" not in name:
                continue
            model = name.split("model=")[1].split(",")[0]
            subj = re.search(r"subject=([a-z_]+)", name)
            best[(model, subj.group(1) if subj else "")] = run
    return best


def gsm_answer(text):
    """Final number in the completion, matching HELM's final_number_exact_match (agreement 0.989 on average
    over the 94 models, versus 0.985 for an 'answer is X' rule)."""
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text))
    return canon_number(nums[-1]) if nums else None


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(OUT, exist_ok=True)
    tm = pd.read_parquet(os.path.join(ROOT, "data", "truth_mmlu.parquet"))
    tg = pd.read_parquet(os.path.join(ROOT, "data", "truth_gsm8k.parquet"))
    mkey = {k: i for k, i, a in zip(tm.match_key, tm.item_id, tm.ambiguous_key) if isinstance(k, str) and not a}
    gkey = dict(zip(tg.match_key, tg.item_id))
    stats = json.load(open(os.path.join(OUT, "helm_stats.json"))) if os.path.exists(os.path.join(OUT, "helm_stats.json")) else {}
    if only == "gsm8k":
        runs = {}
    else:
        runs = latest_runs("mmlu", "mmlu:")
    print("mmlu runs", len(runs), "models", len({m for m, _ in runs}), flush=True)
    def one_mmlu(item):
        (model, subj), run = item
        inst = get_json(run + "instances.json"); pred = get_json(run + "display_predictions.json")
        if not inst or not pred:
            return []
        pmap = {p["instance_id"]: p for p in pred if p.get("train_trial_index", 0) == 0}
        rows = []
        for x in inst:
            refs = x.get("references", [])
            if len(refs) != 4 or x["id"] not in pmap:
                continue
            choices = [r["output"]["text"] for r in refs]
            iid = mkey.get(mmlu_key(subj, x["input"]["text"], choices))
            if iid is None:
                continue
            letter = pmap[x["id"]]["predicted_text"].strip()[:1].upper()
            correct = "ABCD"[[i for i, r in enumerate(refs) if "correct" in r.get("tags", [])][0]] if any("correct" in r.get("tags", []) for r in refs) else None
            rows.append((model, iid, letter if letter in "ABCD" and letter else None, correct))
        return rows
    if only != "gsm8k":
        mmlu_part(runs, one_mmlu, tm, stats)
    gsm_part(tg, gkey, stats)


def mmlu_part(runs, one_mmlu, tm, stats):
    with ThreadPoolExecutor(max_workers=32) as ex:
        rows = [r for part in ex.map(one_mmlu, runs.items()) for r in part]
    hm = pd.DataFrame(rows, columns=["model", "item_id", "answer", "helm_key"]).drop_duplicates(["model", "item_id"])
    gold = tm.set_index("item_id").gold
    hm["key_agrees_with_mmlu"] = hm.helm_key == hm.item_id.map(gold)
    hm.to_parquet(os.path.join(OUT, "helm_mmlu.parquet"))
    stats["mmlu"] = {"runs": len(runs), "models": int(hm.model.nunique()), "rows": int(len(hm)),
                     "items_covered": int(hm.item_id.nunique()), "helm_key_agrees_with_mmlu_key": round(float(hm.key_agrees_with_mmlu.mean()), 4),
                     "per_model_items_median": float(hm.groupby("model").size().median())}
    print(stats["mmlu"], flush=True)


def gsm_part(tg, gkey, stats):
    runs = latest_runs("lite", "gsm:")
    print("gsm runs", len(runs), flush=True)
    def one_gsm(item):
        (model, _), run = item
        inst = get_json(run + "instances.json"); pred = get_json(run + "display_predictions.json")
        if not inst or not pred:
            return []
        pmap = {p["instance_id"]: p for p in pred if p.get("train_trial_index", 0) == 0}
        out = []
        for x in inst:
            iid = gkey.get(norm_key(x["input"]["text"]))
            if iid is None or x["id"] not in pmap:
                continue
            st = pmap[x["id"]].get("stats", {})
            out.append((model, iid, gsm_answer(pmap[x["id"]]["predicted_text"]), st.get("final_number_exact_match", st.get("exact_match"))))
        return out
    with ThreadPoolExecutor(max_workers=32) as ex:
        rows = [r for part in ex.map(one_gsm, runs.items()) for r in part]
    hg = pd.DataFrame(rows, columns=["model", "item_id", "answer", "helm_exact_match"]).drop_duplicates(["model", "item_id"])
    gg = tg.set_index("item_id").gold
    chk = hg.dropna(subset=["helm_exact_match"])
    hg.to_parquet(os.path.join(OUT, "helm_gsm8k.parquet"))
    stats["gsm8k"] = {"runs": len(runs), "models": int(hg.model.nunique()), "rows": int(len(hg)), "items_covered": int(hg.item_id.nunique()),
                      "parse_agrees_with_helm_exact_match": round(float(((chk.answer == chk.item_id.map(gg)) == (chk.helm_exact_match > 0.5)).mean()), 4)}
    print(stats["gsm8k"], flush=True)
    json.dump(stats, open(os.path.join(OUT, "helm_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
