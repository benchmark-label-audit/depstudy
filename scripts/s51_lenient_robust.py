"""GSM8K central contrasts under strict and lenient answer extraction.

Writes the numbers of the bottom block of Table "robust" with tie-aware expected AP:
CRH strong - full (AP), rank average on the strong panel (AP), IRT-4PL on the full population (AP),
CRH HELM - strongest v1 of the same size (key AP), DS-1 HELM + all v1 - HELM (key AP).

  python scripts/s51_lenient_robust.py  ->  results/lenient_robust.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
DS = "ds_onecoin_nf10_lo"


def load(b, tag, m):
    if m == "ensemble":
        parts = [load(b, tag, x) for x in ("crh", DS, "irt4pl")]
        return np.mean([pd.Series(p).rank(method="average").values / len(p) for p in parts], axis=0)
    s = np.load(os.path.join(CACHE, f"{b}__{tag}__{m}.npy"))
    fin = s[np.isfinite(s)]
    return np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))


def main():
    from s10_size_capability import key_mask
    import s19_logodds as S19
    import s31_gsm8k_flex as S31
    out = {}
    for rule, b, single, source in (("strict", "gsm8k", lambda: S19.data("gsm8k"), lambda: S19.data("src_gsm8k")),
                                    ("lenient", "gsm8kflex", S31.single_data, S31.source_data)):
        t = single()[0]
        y = t.defective.values.astype(int)
        ts = source()[0]
        kp, cl = key_mask("gsm8k", ts); keep = kp | cl
        ys = kp[keep].astype(int)
        ap = lambda tag, m: Mt.average_precision_expected(load(b, tag, m), y)
        kap = lambda tag, m: Mt.average_precision_expected(load(b, tag, m)[keep], ys)
        out[rule] = {
            "crh_strong_minus_full_ap": ap("single_small_strong", "crh") - ap("single_full", "crh"),
            "ensemble_strong_ap": ap("single_small_strong", "ensemble"),
            "irt_full_ap": ap("single_full", "irt4pl"),
            "crh_helm_minus_v1strongk_keyap": kap("src_helm_all_0", "crh") - kap("src_v1_strong_k_0", "crh"),
            "ds_helm_plus_v1all_minus_helm_keyap": kap("src_helm_plus_v1_all_0", DS) - kap("src_helm_all_0", DS),
        }
        print(rule, {k: round(v, 3) for k, v in out[rule].items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "lenient_robust.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
