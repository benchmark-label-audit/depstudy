"""Engineering dry run on whatever has been fetched so far: builds matrices, residual dependence, and times
every method on one random panel. Defect labels are SHUFFLED, so no real detection metric is computed."""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy import methods as Mt  # noqa: E402
from depstudy.data import matrices  # noqa: E402
from depstudy.panels import residual_corr  # noqa: E402

from depstudy.data import fetch_records
models = sorted(fetch_records())
print("models available", len(models))
rng = np.random.default_rng(0)
for bench, nf in (("mmlu", 3), ("gsm8k", 10)):
    t = time.time(); tt, mods, X, g, Y, M = matrices(models, bench); print(bench, "matrices", X.shape, round(time.time() - t, 1), "s")
    print("  answered share", round(float((X >= 0).mean()), 3), "observed(IRT) share", round(float(M.mean()), 3), "mean acc", round(float((Y * M).sum() / M.sum()), 3))
    t = time.time(); R = residual_corr(Y, M); print("  residual corr", R.shape, "mean", round(float(R[np.triu_indices(len(R), 1)].mean()), 4), round(time.time() - t, 1), "s")
    n = min(100, X.shape[1]); cols = rng.choice(X.shape[1], n, replace=False)
    y = rng.permutation(tt.defective.values.astype(int))  # shuffled labels: timing only
    Xp, Yp, Mp = X[:, cols], Y[:, cols], M[:, cols]
    for name, f in [("mv", lambda: Mt.majority_vote(Xp, g)), ("ds1", lambda: Mt.dawid_skene_onecoin(Xp, g)[0]),
                    ("crh", lambda: Mt.crh(Xp, g)[0]), ("irt", lambda: Mt.irt_ceiling_lr(Xp, g, Yp, Mp)),
                    ("accu_nodep", lambda: Mt.accu_depen(Xp, g, nf, use_dependence=False)[0]),
                    ("accu_depen", lambda: Mt.accu_depen(Xp, g, nf)[0])] + ([("dsf", lambda: Mt.dawid_skene_full(Xp, g, 4)[0])] if bench == "mmlu" else []):
        t = time.time(); s = f(); dt = time.time() - t
        print(f"  {name:10s} {dt:6.1f}s  finite={np.isfinite(s).all()}  (shuffled-label AP {Mt.average_precision(s, y):.3f})")
