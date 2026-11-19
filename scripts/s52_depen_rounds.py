"""Round-by-round diagnostics of DEPEN (Dong et al. 2009 settings) on the strong panels and on their simulated
independent counterparts: share of source pairs with copying probability above 0.5 after each round, the mean
estimated source accuracy, and the share of items whose selected truth differs from ACCU's. Writes
results/depen_rounds.csv."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
import s19_logodds as S19  # noqa: E402
from s45_copy_validation import simulate_independent  # noqa: E402

RES = os.path.join(ROOT, "results")


def main():
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    rows = []
    for bench, nf, K in (("mmlu", 3, 4), ("gsm8k", 10, None)):
        t, mods, X, g, Y, M = S19.data(bench)
        idx = {m: i for i, m in enumerate(mods)}
        c = np.array([idx[m] for m in P[bench]["singles"]["small_strong"]["models"]])
        Xr = X[:, c]
        rng = np.random.default_rng(4500)  # same seed as s45 rep 0
        Xs, modal, _ = simulate_independent(Xr, rng, None)
        iu = np.triu_indices(Xr.shape[1], 1)
        for data, Xv in (("real", Xr), ("sim_indep", Xs)):
            _, post0, acc0, _ = Mt.accu_depen(Xv, g, nf, K=K, logodds=True, use_dependence=False)
            t0 = post0.argmax(1)
            for it in range(1, 6):
                _, post, acc, D = Mt.accu_depen(Xv, g, nf, K=K, logodds=True, iters=it)
                d = D[iu]
                rows.append(dict(bench=bench, data=data, rounds=it, flag=float((d > 0.5).mean()), meanD=float(d.mean()),
                                 mean_acc=float(acc.mean()), mean_acc_accu=float(acc0.mean()),
                                 truth_differs_from_accu=float((post.argmax(1) != t0).mean()),
                                 mean_max_post=float(post.max(1).mean()), mean_max_post_accu=float(post0.max(1).mean())))
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "depen_rounds.csv"), index=False)


if __name__ == "__main__":
    main()
