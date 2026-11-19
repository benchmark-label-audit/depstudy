"""Step 0: ground-truth tables for the two independently annotated benchmarks.

MMLU: MMLU-Redux 2.0 annotations (copied from the earlier study); defective = any annotated error type,
'expert' items excluded. GSM8K: GSM8K-Platinum (MadryLab), which drops ambiguous or inconsistent test
questions and revises some keys; defective = dropped or revised.

  python scripts/s0_truth.py   ->  data/truth_mmlu.parquet, data/truth_gsm8k.parquet, data/truth_stats.json
"""
from __future__ import annotations

import io
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import ds_parquet_urls, get_bytes  # noqa: E402
from depstudy.textnorm import gsm8k_gold, mmlu_key, norm_key  # noqa: E402

PREV = os.path.join(ROOT, "data", "external")  # MMLU-Redux 2.0 inputs


def mmlu():
    items = pd.DataFrame([json.loads(l) for l in open(os.path.join(PREV, "mmlu_redux_items.jsonl"), encoding="utf-8")])
    red = pd.read_parquet(os.path.join(PREV, "mmlu_redux_all.parquet"))
    red["k"] = red.groupby("subject").cumcount()
    red["item_id"] = red.subject + "__" + red.k.map(lambda x: f"{x:03d}")
    red["match_key"] = [mmlu_key(s, q, c) for s, q, c in zip(red.subject, red.question, red.choices)]
    df = items[["item_id", "subject", "gold", "error_type"]].merge(red[["item_id", "match_key"]], on="item_id", how="left")
    df["excluded"] = df.error_type.eq("expert")
    df["defective"] = ~df.error_type.isin(["ok", "expert"])
    dup = df.match_key.duplicated(keep=False)
    df["ambiguous_key"] = dup
    return df


def gsm8k():
    def load(ds, config, split):
        for f in ds_parquet_urls(ds):
            if f["config"] == config and f["split"] == split:
                return pd.read_parquet(io.BytesIO(get_bytes(f["url"])))
        raise RuntimeError(f"{ds} {config}/{split} not found")
    orig = load("openai/gsm8k", "main", "test").reset_index(drop=True)
    plat = load("madrylab/gsm8k-platinum", "main", "test")
    orig["item_id"] = [f"gsm8k__{i:04d}" for i in range(len(orig))]
    orig["match_key"] = orig.question.map(norm_key)
    orig["gold"] = orig.answer.map(gsm8k_gold)
    plat["match_key"] = plat.question.map(norm_key)
    plat["gold_platinum"] = plat.answer.map(gsm8k_gold)
    m = orig.merge(plat[["match_key", "gold_platinum", "cleaning_status"]], on="match_key", how="left")
    m["status"] = m.cleaning_status.fillna("removed")
    m["defective"] = m.status.isin(["removed", "revised"])
    m["excluded"] = False
    m["ambiguous_key"] = m.match_key.duplicated(keep=False)
    unmatched_plat = len(set(plat.match_key) - set(orig.match_key))
    return m[["item_id", "match_key", "gold", "gold_platinum", "status", "defective", "excluded", "ambiguous_key"]], unmatched_plat


def main():
    out = os.path.join(ROOT, "data")
    a = mmlu()
    b, unmatched = gsm8k()
    a.to_parquet(os.path.join(out, "truth_mmlu.parquet"))
    b.to_parquet(os.path.join(out, "truth_gsm8k.parquet"))
    stats = {
        "mmlu": {"items": int(len(a)), "excluded_expert": int(a.excluded.sum()),
                 "evaluated": int((~a.excluded).sum()), "defective": int(a.defective.sum()),
                 "error_types": a.error_type.value_counts().to_dict(), "ambiguous_keys": int(a.ambiguous_key.sum())},
        "gsm8k": {"items": int(len(b)), "status": b.status.value_counts().to_dict(), "defective": int(b.defective.sum()),
                  "platinum_rows_unmatched_to_original": int(unmatched), "ambiguous_keys": int(b.ambiguous_key.sum())},
    }
    json.dump(stats, open(os.path.join(out, "truth_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
