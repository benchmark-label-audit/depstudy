"""CrowdFM on the added panels: the off-benchmark strong panel of each benchmark and the GSM8K
single panels under the lenient extraction ('gsm8kflex', s31). Probability score cached as 'crowdfm', log-odds score
as 'crowdfm_lo'; panels above 100 models also get the 100-model block variant ('crowdfm_block100').

  <crowdfm-venv>/python -I scripts/s34_crowdfm_new.py [--flex]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.data import matrices, n_options  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402
from crowdfm_adapter import crowdfm_scores_both  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")


def save(bench, tag, name, s):
    p = os.path.join(CACHE, f"{bench}__{tag}__{name}.npy")
    np.save(p + ".tmp.npy", np.asarray(s, np.float64)); os.replace(p + ".tmp.npy", p)


def run(D, bench, tag, models, K):
    t, mods, X, g, Y, M = D
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    if not os.path.exists(os.path.join(CACHE, f"{bench}__{tag}__crowdfm_lo.npy")):
        p, l = crowdfm_scores_both(X[:, c], g, K)
        save(bench, tag, "crowdfm", p); save(bench, tag, "crowdfm_lo", l)
    if len(c) > 100 and not os.path.exists(os.path.join(CACHE, f"{bench}__{tag}__crowdfm_block100_lo.npy")):
        p, l = crowdfm_scores_both(X[:, c], g, K, worker_block=100)
        save(bench, tag, "crowdfm_block100", p); save(bench, tag, "crowdfm_block100_lo", l)
    print(bench, tag, len(c), flush=True)


def main():
    from s4_panels import eligible_models
    import s10_size_capability as S10
    models = eligible_models(None)
    common, cap = S10.capability()
    if "--flex" in sys.argv:
        D = matrices(models, "gsm8kflex")
        t, mods, X, g, Y, M = D
        acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
        P = json.load(open(os.path.join(ROOT, "results", "panels.json")))["benches"]["gsm8k"]["singles"]
        for name, s in P.items():
            run(D, "gsm8kflex", f"single_{name}", s["models"], None)
        run(D, "gsm8kflex", "single_small_strong_flex", list(acc.sort_values(ascending=False).index[:100]), None)
        run(D, "gsm8kflex", "single_offbench_strong", list(cap["gsm8k"].sort_values(ascending=False).index[:100]), None)
        return
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        D = matrices(models, bench)
        K = n_options(bench) if bench in ("mmlu", "hellaswag") else None
        run(D, bench, "single_offbench_strong", list(cap[bench].sort_values(ascending=False).index[:100]), K)


if __name__ == "__main__":
    main()
