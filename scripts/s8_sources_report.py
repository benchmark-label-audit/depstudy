"""Summaries of the cross-source checkpoints (exploratory): mean AP and P@k per condition and method, and AP per
MMLU-Redux error type (each type scored against clean items only), from the stored per-item scores.

  python scripts/s8_sources_report.py  ->  printed tables + results/sources_summary.json
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402

rows = []
for f in glob.glob(os.path.join(ROOT, "results", "ckpt_sources", "*.pkl")):
    rows += pickle.load(open(f, "rb"))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
order = ["helm_all", "v1_strong_k", "v1_random_k", "helm_plus_v1_strong100", "helm_plus_v1_all", "v1_all",
         "helm_vendor_concentrated", "helm_vendor_diverse"]
out = {}
for metric in ("ap", "pk"):
    tab = df.groupby(["bench", "method", "condition"])[metric].mean().unstack("condition")
    tab = tab[[c for c in order if c in tab.columns]]
    print(f"\n==== mean {metric} ====")
    print(tab.round(3).to_string())
    out[metric] = tab.round(4).reset_index().to_dict(orient="records")

# AP per MMLU-Redux error type: positives = items of that type, negatives = clean items
from s6_sources import load  # noqa: E402
t, *_ = load("mmlu")
et = t.error_type.values
types = [x for x in pd.Series(et).value_counts().index if x not in ("ok", "expert")]
res = []
for r in df[(df.bench == "mmlu") & (df.rep == 0)].itertuples():
    s = np.frombuffer(r.scores, dtype=np.float32)
    for ty in types:
        keep = (et == ty) | (et == "ok")
        res.append({"condition": r.condition, "method": r.method, "type": ty, "n_pos": int((et == ty).sum()),
                    "ap": Mt.average_precision(s[keep], (et[keep] == ty).astype(int)), "base": float((et[keep] == ty).mean())})
er = pd.DataFrame(res)
print("\n==== MMLU AP by error type (rep 0; base = prevalence among clean+type items) ====")
piv = er.pivot_table(index=["type", "method"], columns="condition", values="ap")
piv = piv[[c for c in order if c in piv.columns]]
print(er.groupby("type")[["n_pos", "base"]].first().round(3).to_string())
print(piv.round(3).to_string())
out["mmlu_by_type"] = er.round(4).to_dict(orient="records")
json.dump(out, open(os.path.join(ROOT, "results", "sources_summary.json"), "w"), indent=1)
