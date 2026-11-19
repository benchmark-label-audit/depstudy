"""Gate contrasts from the cell checkpoints (criteria 1 and 2 of the decision rules): dependence and lineage
effects on P@k per method with bootstrap CIs over panels, and the best method per cell.

  python scripts/s7_cell_report.py [ckpt_dir]  ->  printed table + results/<ckpt>_cells.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ck = sys.argv[1] if len(sys.argv) > 1 else "ckpt_gate"
rows = [r for f in glob.glob(os.path.join(ROOT, "results", ck, "*.json")) for r in json.load(open(f))]
df = pd.DataFrame(rows)
cells = df[df.kind == "cell"]
rng = np.random.default_rng(7)
HH, HL, LH, LL = "lineage_high__dep_high", "lineage_high__dep_low", "lineage_low__dep_high", "lineage_low__dep_low"
out = {}
for metric in ("pk", "ap"):
    for b in sorted(cells.bench.unique()):
        for m in sorted(cells.method.unique()):
            x = cells[(cells.bench == b) & (cells.method == m)]
            v = {c: x[x.condition == c][metric].values for c in (HH, HL, LH, LL, "random")}
            if any(len(v[c]) == 0 for c in (HH, HL, LH, LL)):
                continue
            dep = lambda s: (s[HH].mean() + s[LH].mean() - s[HL].mean() - s[LL].mean()) / 2
            lin = lambda s: (s[HH].mean() + s[HL].mean() - s[LH].mean() - s[LL].mean()) / 2
            bs_d, bs_l = [], []
            for _ in range(2000):
                s = {c: rng.choice(v[c], len(v[c])) for c in (HH, HL, LH, LL)}
                bs_d.append(dep(s)); bs_l.append(lin(s))
            out[f"{metric}|{b}|{m}"] = {
                "means": {c: round(float(v[c].mean()), 4) for c in v if len(v[c])},
                "dep_effect": round(float(dep(v)), 4), "dep_ci": [round(float(np.percentile(bs_d, 2.5)), 4), round(float(np.percentile(bs_d, 97.5)), 4)],
                "lin_effect": round(float(lin(v)), 4), "lin_ci": [round(float(np.percentile(bs_l, 2.5)), 4), round(float(np.percentile(bs_l, 97.5)), 4)]}
json.dump(out, open(os.path.join(ROOT, "results", f"{ck}_cells.json"), "w"), indent=1)
pd.set_option("display.width", 250)
for metric in ("pk", "ap"):
    print(f"\n==== {metric} (mean over 20 panels per cell; effects in absolute points of {metric}) ====")
    tab = []
    for k, r in out.items():
        mt, b, m = k.split("|")
        if mt != metric:
            continue
        mm = r["means"]
        tab.append([b, m, mm[HH], mm[HL], mm[LH], mm[LL], mm.get("random"), r["dep_effect"], str(r["dep_ci"]), r["lin_effect"], str(r["lin_ci"])])
    print(pd.DataFrame(tab, columns=["bench", "method", "HiLin-HiDep", "HiLin-LoDep", "LoLin-HiDep", "LoLin-LoDep", "random",
                                     "dep_eff", "dep_95CI", "lin_eff", "lin_95CI"]).to_string(index=False))
best = cells.groupby(["bench", "condition", "method"]).pk.mean().reset_index()
print("\n==== best method per cell (P@k) ====")
print(best.loc[best.groupby(["bench", "condition"]).pk.idxmax()].to_string(index=False))
