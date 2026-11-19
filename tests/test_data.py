"""Vectorised matrix builder vs. a plain-loop reference on a few real models."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depstudy import data as D  # noqa: E402


def reference(models, bench):
    t = D.truth(bench)
    lt = D.long_table(models, bench)
    lt = lt[lt.item_id.isin(set(t.item_id))]
    mods = [m for m in models if m in set(lt.model)]
    ans = {(r.item_id, r.model): r.answer for r in lt.itertuples()}
    X = np.full((len(t), len(mods)), -1, np.int32)
    for i, (iid, gold) in enumerate(zip(t.item_id, t.gold)):
        vals = {gold: 0} if bench == "gsm8k" else {"A": 0, "B": 1, "C": 2, "D": 3}
        for j, m in enumerate(mods):
            a = ans.get((iid, m))
            if a is None or (isinstance(a, float) and np.isnan(a)):
                continue
            if bench == "gsm8k" and a not in vals:
                vals[a] = len(vals)
            X[i, j] = vals[a]
    return t, mods, X


def test_equivalence():
    from depstudy.data import fetch_records
    models = sorted(fetch_records())[:6]
    for bench in ("mmlu", "gsm8k"):
        t1, m1, X1, g1, Y1, M1 = D._build(models, bench, 0.0)
        t2, m2, X2 = reference(models, bench)
        assert m1 == m2 and list(t1.item_id) == list(t2.item_id)
        # codes may differ only by relabelling of non-gold values; compare equality structure per item
        same = all(np.array_equal(X1[i][:, None] == X1[i][None, :], X2[i][:, None] == X2[i][None, :]) and
                   np.array_equal(X1[i] == 0, X2[i] == 0) if bench == "gsm8k" else np.array_equal(X1[i], X2[i])
                   for i in range(len(t1)))
        assert same, bench
        print(bench, "equivalent", X1.shape)


if __name__ == "__main__":
    test_equivalence()
