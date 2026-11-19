"""CrowdFM on the cross-source panels drawn with exactly the s6 generator (all 20 random v1 panels and the 20 + 20
vendor panels per benchmark), so every method is compared on identical panels. Replaces the kind == "source" rows
of results/crowdfm_scores.parquet (s15 drew its GSM8K random panels from a different point of the generator)
and caches item scores as $DEPSTUDY_DATA/item_scores/<bench>__src_<condition>_<rep>__crowdfm[_block100].npy.

  <crowdfm-venv>/python -I scripts/s15c_crowdfm_sources.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402
from crowdfm_adapter import crowdfm_scores  # noqa: E402

OUT = os.path.join(ROOT, "results", "crowdfm_scores.parquet")
CACHE = os.path.join(DATA_ROOT, "item_scores")
PK = {"mmlu": 200, "gsm8k": 50}


def main():
    import s19_logodds as S19
    from s10_size_capability import key_mask
    import s6_sources as S6
    specs = [t for t in S19.tasks() if t[0] == "sources"]
    prev = pd.read_parquet(OUT)
    rows = prev[prev.kind != "source"].to_dict(orient="records")
    D = {b: S6.load(b) for b in ("mmlu", "gsm8k")}
    for i, (family, bench, kind, cond, rep, tag, models, _) in enumerate(specs):
        t, mods, X, g, Y, M = D[bench]
        idx = {m: j for j, m in enumerate(mods)}
        c = np.array([idx[m] for m in models if m in idx])
        y = t.defective.values.astype(int)
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        K = 4 if bench == "mmlu" else None
        for variant, wb in (("crowdfm", None), ("crowdfm_block100", 100)):
            if wb and len(c) <= 100:
                continue
            p = os.path.join(CACHE, f"{bench}__{tag}__{variant}.npy")
            t0 = time.time()
            if os.path.exists(p):
                s = np.load(p); sec = float("nan")
            else:
                s = crowdfm_scores(X[:, c], g, K, worker_block=wb); sec = time.time() - t0
                np.save(p, s)
            rows.append({"bench": bench, "kind": "source", "condition": cond, "rep": rep, "n_models": int(len(c)), "method": variant,
                         "sec": sec, "ap": Mt.average_precision(s, y), "pk": Mt.precision_at_k(s, y, PK[bench]),
                         "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int))})
        if i % 10 == 0:
            print(i + 1, "/", len(specs), bench, cond, rep, flush=True)
            pd.DataFrame(rows).to_parquet(OUT)
    pd.DataFrame(rows).to_parquet(OUT)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
