"""Mechanism check for IRT-4PL on GSM8K: unparseable answers counted as incorrect (default, observed in M) versus
treated as missing (M = X >= 0), on the full v1 population and the strong panel.

  python scripts/s23_irt_missing.py  ->  results/irt_missing_check.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.data import matrices  # noqa: E402


def main():
    from s4_panels import eligible_models
    from s10_size_capability import key_mask
    models = eligible_models(None)
    t, mods, X, g, Y, M = matrices(models, "gsm8k")
    y = t.defective.values.astype(int)
    kpos, clean = key_mask("gsm8k", t); keep = kpos | clean
    P = json.load(open(os.path.join(ROOT, "results", "panels.json")))["benches"]["gsm8k"]["singles"]
    idx = {m: i for i, m in enumerate(mods)}
    out = {}
    for cond in ("full", "small_strong"):
        c = np.array([idx[m] for m in P[cond]["models"]])
        Xp, Yp, Mp = X[:, c], Y[:, c], M[:, c]
        for variant, MM in (("unparseable_incorrect", Mp), ("unparseable_missing", Xp >= 0)):
            s = Mt.irt_ceiling_lr(Xp, g, Yp * MM, MM)
            out[f"{cond}/{variant}"] = {"ap": Mt.average_precision(s, y), "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int)),
                                        "share_unparseable": float(1 - (Xp >= 0).sum() / Mp.sum())}
            print(cond, variant, out[f"{cond}/{variant}"], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "irt_missing_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
