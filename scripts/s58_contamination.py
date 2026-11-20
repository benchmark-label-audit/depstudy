"""Contamination diagnostic: agreement with the wrong keys on verified key errors for the strong, capable, and top-400
panels; models whose wrong-key agreement exceeds the linear trend on clean accuracy by more than 2 SD are flagged, and
DS-1 is rerun on the strong panel after replacing them with the next most accurate unflagged models.

  python scripts/s58_contamination.py > results/contamination.txt
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
from depstudy import methods as Mt  # noqa: E402
import s19_logodds as S19  # noqa: E402
from s10_size_capability import key_mask, capability  # noqa: E402

common, cap = capability()
NF = {"mmlu": None, "gsm8k": 10}
K = {"mmlu": 4, "gsm8k": None}


def ds_scores(X, g, bench):
    if bench == "mmlu":
        return Mt.dawid_skene_onecoin(X, g, K=4, logodds=True)[0]
    return Mt.dawid_skene_onecoin(X, g, logodds=True, n_false=10)[0]


for bench in ("mmlu", "gsm8k"):
    t, mods, X, g, Y, M = S19.data(bench)
    mods = np.array(mods)
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    y_all = t.defective.values.astype(int)
    acc = (Y * M).sum(0) / np.maximum(M.sum(0), 1)
    # wrong-key agreement: Y is correctness under shipped keys, so on key-error items Y = agreement with the wrong key
    wk = (Y[kpos] * M[kpos]).sum(0) / np.maximum(M[kpos].sum(0), 1)
    acc_clean = (Y[clean] * M[clean]).sum(0) / np.maximum(M[clean].sum(0), 1)
    idx = {m: i for i, m in enumerate(mods)}
    order = np.argsort(-acc)
    strong = order[:100]
    capv = cap[bench].reindex(mods).values
    capable = np.argsort(-np.nan_to_num(capv, nan=-1))[:100]
    top400 = order[:400]
    print(f"\n== {bench}: key errors {kpos.sum()}")
    for name, P in (("strong", strong), ("capable", capable), ("top400", top400), ("full", np.arange(len(mods)))):
        print(f"  {name:8s} mean acc(clean) {acc_clean[P].mean():.3f}  wrong-key agreement {wk[P].mean():.3f}")
    # outlier test within top 400: regress wk on clean accuracy
    a, b = np.polyfit(acc_clean[top400], wk[top400], 1)
    res = wk[top400] - (a * acc_clean[top400] + b)
    thr = 2 * res.std()
    flagged = top400[res > thr]
    print(f"  slope of wrong-key agreement on clean accuracy (top 400): {a:+.3f}; flagged {len(flagged)} models (> 2 SD)")
    print(f"  flagged in strong: {np.isin(flagged, strong).sum()}, in capable: {np.isin(flagged, capable).sum()}")
    # replace flagged models in the strong panel with next most accurate unflagged
    fl = set(flagged.tolist())
    strong_clean = [i for i in order if i not in fl][:100]
    for name, P in (("strong", strong), ("strong minus flagged", np.array(strong_clean)), ("capable", capable)):
        s = ds_scores(X[:, P], g, bench)
        fin = s[np.isfinite(s)]
        s = np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))
        ap = Mt.average_precision_expected(s, y_all)
        apk = Mt.average_precision_expected(s[keep], kpos[keep].astype(int))
        print(f"  DS-1 {name:22s} AP {ap:.3f}  key-error AP {apk:.3f}  wrong-key agr {wk[P].mean():.3f}")
