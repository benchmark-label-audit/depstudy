"""Dependence of the single panels, including the off-benchmark strong panel: mean residual correlation on the design
benchmark (whole-population 2PL fit, as for the cells) and on the audited benchmark, and the same-wrong rate.

  python scripts/s40_panel_dependence.py  ->  results/panel_dependence.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RES = os.path.join(ROOT, "results")


def main():
    import s19_logodds as S19
    import s10_size_capability as S10
    from s29_dose import DESIGN, same_wrong, panel_mean
    from depstudy.panels import residual_corr
    common, cap = S10.capability()
    rows = []
    for E, D in DESIGN.items():
        tE, mE, XE, gE, YE, ME = S19.data(E)
        tD, mD, XD, gD, YD, MD = S19.data(D)
        iD = {m: i for i, m in enumerate(mD)}
        com = [m for m in mE if m in iD]
        iE = {m: i for i, m in enumerate(mE)}
        cE = np.array([iE[m] for m in com]); cD = np.array([iD[m] for m in com])
        R_D = residual_corr(YD[:, cD], MD[:, cD]); R_E = residual_corr(YE[:, cE], ME[:, cE])
        SW_E, _ = same_wrong(XE[:, cE], gE)
        ci = {m: k for k, m in enumerate(com)}
        P0 = json.load(open(os.path.join(RES, "panels_hellaswag.json" if E == "hellaswag" else "panels.json")))["benches"][E]["singles"]
        panels = {k: v["models"] for k, v in P0.items() if k != "full"}
        panels["offbench_strong"] = list(cap[E].sort_values(ascending=False).index[:100])
        accE = (YE[:, cE] * ME[:, cE]).sum(0) / np.maximum(ME[:, cE].sum(0), 1)
        for name, ms in panels.items():
            idx = np.array([ci[m] for m in ms if m in ci])
            rows.append({"bench": E, "panel": name, "n": len(idx), "R_design": panel_mean(R_D, idx), "R_eval": panel_mean(R_E, idx),
                         "samewrong_eval": panel_mean(SW_E, idx), "acc_eval": float(accE[idx].mean())})
        print(E, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "panel_dependence.csv"), index=False)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
