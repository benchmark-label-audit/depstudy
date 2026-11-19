"""Sanity tests on synthetic data with known label errors and known copying."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depstudy import methods as M  # noqa: E402


def synth(n_items=800, n_ind=30, n_copy=0, K=4, err_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, K, n_items)
    gold = truth.copy()
    bad = rng.random(n_items) < err_rate
    gold[bad] = (truth[bad] + rng.integers(1, K, bad.sum())) % K
    diff = rng.normal(0, 1, n_items)
    abil = rng.normal(0.5, 1, n_ind)
    X = np.zeros((n_items, n_ind + n_copy), int)
    for j in range(n_ind):
        p = 1 / (1 + np.exp(-(abil[j] - diff) * 1.5))
        right = rng.random(n_items) < p
        X[:, j] = np.where(right, truth, (truth + rng.integers(1, K, n_items)) % K)
    for j in range(n_copy):  # copiers of source 0 with 90% copy rate
        src = X[:, 0]
        own = (truth + rng.integers(0, K, n_items)) % K
        X[:, n_ind + j] = np.where(rng.random(n_items) < 0.9, src, own)
    return X, gold, bad


def test_methods_beat_random():
    X, g, bad = synth()
    base = bad.mean()
    for name, s in [("mv", M.majority_vote(X, g)), ("ds1", M.dawid_skene_onecoin(X, g)[0]),
                    ("dsf", M.dawid_skene_full(X, g, 4)[0]), ("crh", M.crh(X, g)[0]),
                    ("accu", M.accu_depen(X, g, n_false=3)[0]), ("irt", M.irt_ceiling_lr(X, g))]:
        ap = M.average_precision(s, bad)
        assert ap > base + 0.2, (name, ap, base)


def test_dependence_detects_copiers():
    X, g, bad = synth(n_ind=20, n_copy=10)
    _, post, acc, D = M.accu_depen(X, g, n_false=3)
    copy_dep = D[0, 20:].mean()
    ind_dep = D[1:20, 1:20][np.triu_indices(19, 1)].mean()
    assert copy_dep > 0.8 and ind_dep < 0.2, (copy_dep, ind_dep)


def test_metrics():
    s = np.array([0.9, 0.8, 0.1, 0.05]); y = np.array([1, 0, 1, 0])
    assert M.precision_at_k(s, y, 2) == 0.5
    assert abs(M.average_precision(s, y) - (1 + 2 / 3) / 2) < 1e-9
