"""Truth table for HellaSwag (validation, 10,042 items) from the LLM annotations released with "What the
HellaSwag?" (Chizhov et al., 2025; annotator claude-3-5-sonnet-20241022, round 2 of PleIAs/hellaswag-annotations).

Flags: wrong_answer = the annotator's best option differs from the key (the paper's "Wrong answer" filter);
multiple_correct = the annotator names an equally good option. defective (primary) = wrong_answer; a broad
variant adds multiple_correct. This truth is model-judged, unlike MMLU-Redux and GSM8K-Platinum, and is used as
a secondary benchmark only.

Also computes the harness length normaliser per option (acc_norm divides the log-likelihood by the character
length of the preprocessed ending) and the text key used to verify row alignment with the leaderboard files.

  python scripts/s0b_extra_truth.py  ->  data/truth_hellaswag.parquet
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import ds_parquet_urls, get_bytes  # noqa: E402
from depstudy.textnorm import norm_key  # noqa: E402


def preprocess(text: str) -> str:
    """lm-evaluation-harness HellaSwag preprocessing (as used by the Open LLM Leaderboard v1)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def main():
    val = None
    for f in ds_parquet_urls("Rowan/hellaswag"):
        if f["split"] == "validation":
            part = pd.read_parquet(io.BytesIO(get_bytes(f["url"])))
            val = part if val is None else pd.concat([val, part])
    val = val.reset_index(drop=True)
    assert len(val) == 10042, len(val)
    ann = pd.read_parquet(io.BytesIO(get_bytes(
        "https://huggingface.co/datasets/PleIAs/hellaswag-annotations/resolve/main/annotations_round_2.parquet")))
    ann = ann.set_index("original_index")
    rows = []
    for i, r in val.iterrows():
        endings = list(r.endings)
        gold = int(r.label)
        q = preprocess(r.activity_label + ": " + r.ctx_a + " " + r.ctx_b.capitalize())
        a = ann.loc[i] if i in ann.index else None
        if a is not None:
            best = "ABCD".index(str(a.best_option).strip()[:1]) if str(a.best_option).strip()[:1] in "ABCD" else -1
            eq = str(a.equally_good_options).strip()
            wrong = best >= 0 and best != gold
            multi = eq not in ("None", "", "nan")
            impl = str(a.all_implausible).strip() == "Yes"
        else:
            wrong = multi = impl = None
        rows.append({"item_id": f"hellaswag__{i:05d}", "row": i, "gold": gold, "key": norm_key(q)[:80],
                     "len0": len(preprocess(endings[0])), "len1": len(preprocess(endings[1])),
                     "len2": len(preprocess(endings[2])), "len3": len(preprocess(endings[3])),
                     "annotated": a is not None, "wrong_answer": wrong, "multiple_correct": multi, "all_implausible": impl})
    t = pd.DataFrame(rows)
    t["excluded"] = ~t.annotated
    t["ambiguous_key"] = False
    t["defective"] = t.wrong_answer.fillna(False).astype(bool)
    t["defective_broad"] = (t.wrong_answer.fillna(False) | t.multiple_correct.fillna(False)).astype(bool)
    t.to_parquet(os.path.join(ROOT, "data", "truth_hellaswag.parquet"))
    st = {"items": int(len(t)), "annotated": int(t.annotated.sum()), "wrong_answer": int(t.defective.sum()),
          "multiple_correct": int(t.multiple_correct.fillna(False).sum()), "broad": int(t.defective_broad.sum())}
    stats = json.load(open(os.path.join(ROOT, "data", "truth_stats.json")))
    stats["hellaswag"] = st
    json.dump(stats, open(os.path.join(ROOT, "data", "truth_stats.json"), "w"), indent=1)
    print(st)


if __name__ == "__main__":
    main()
