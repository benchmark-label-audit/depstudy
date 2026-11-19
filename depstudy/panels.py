"""Residual dependence and matched panel construction for the Phase 1 gate."""
from __future__ import annotations

import numpy as np

from depstudy.methods import _sig

# MMLU subject groups (Hendrycks et al. categories)
STEM = {"abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry", "college_computer_science",
        "college_mathematics", "college_physics", "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning"}
HUM = {"formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history",
       "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
       "prehistory", "professional_law", "world_religions"}
SOC = {"econometrics", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics",
       "high_school_microeconomics", "high_school_psychology", "human_sexuality", "professional_psychology",
       "public_relations", "security_studies", "sociology", "us_foreign_policy"}


def subject_group(s: str) -> int:
    return 0 if s in STEM else 1 if s in HUM else 2 if s in SOC else 3


def residual_corr(Y: np.ndarray, M: np.ndarray, iters: int = 300, lr: float = 0.05) -> np.ndarray:
    """Pairwise correlation of correctness residuals after a 2PL fit (ability x item difficulty/discrimination).
    Y, M: (n_items, n_models). Missing entries contribute zero residual."""
    n_i, n_m = Y.shape
    Yf = np.where(M, Y, 0.0)
    acc = Yf.sum(0) / np.maximum(M.sum(0), 1)
    th = (acc - acc.mean()) / (acc.std() + 1e-9)
    a = np.ones(n_i); b = np.zeros(n_i)
    for _ in range(iters):
        p = _sig(a[:, None] * (th[None, :] - b[:, None]))
        e = (Yf - p) * M
        th += lr * (e * a[:, None]).sum(0) / np.maximum(M.sum(0), 1)
        th = (th - th.mean()) / (th.std() + 1e-9)
        a = np.clip(a + lr * (e * (th[None, :] - b[:, None])).sum(1) / np.maximum(M.sum(1), 1), 1 / 15, 15)
        b = np.clip(b - lr * (e * a[:, None]).sum(1) / np.maximum(M.sum(1), 1), -4, 4)
    p = _sig(a[:, None] * (th[None, :] - b[:, None]))
    E = (Yf - p) * M
    E = E - E.sum(0) / np.maximum(M.sum(0), 1) * M
    sd = np.sqrt((E ** 2).sum(0) / np.maximum(M.sum(0), 1)) + 1e-9
    Z = E / sd
    C = (Z.T @ Z) / np.maximum(M.astype(float).T @ M.astype(float), 1)
    np.fill_diagonal(C, 0.0)
    return C


def hhi(families: list[str]) -> float:
    _, c = np.unique(families, return_counts=True)
    s = c / c.sum()
    return float((s ** 2).sum())


def anneal_panel(n, allowed, fam, cap, R, feat, target, tol, sign, rng, steps=6000, lam=5.0):
    """Choose n models from `allowed` (indices) maximising sign * mean pairwise R subject to a per-family cap
    and to matching panel feature means/SDs to target within tol (soft penalty)."""
    allowed = np.array(allowed)
    def feasible_init():
        order = rng.permutation(allowed)
        P, cnt = [], {}
        for i in order:
            f = fam[i]
            if cnt.get(f, 0) < cap:
                P.append(i); cnt[f] = cnt.get(f, 0) + 1
            if len(P) == n:
                break
        return P, cnt
    P, cnt = feasible_init()
    if len(P) < n:
        return None
    P = list(P)
    inP = np.zeros(R.shape[0], bool); inP[P] = True

    def pen(Pidx):
        F = feat[Pidx]
        dm = np.abs(F.mean(0) - target["mean"]) / tol["mean"]
        ds = np.abs(F[:, 0].std() - target["sd"]) / tol["sd"]
        return float(np.maximum(dm - 1, 0).sum() + max(ds - 1, 0))

    S = R[np.ix_(P, P)].sum() / 2
    npairs = n * (n - 1) / 2
    cur = sign * S / npairs - lam * pen(P)
    T0 = 0.02
    for t in range(steps):
        T = T0 * (1 - t / steps) + 1e-5
        k = rng.integers(n); a = P[k]
        cand = allowed[~inP[allowed]]
        if len(cand) == 0:
            break
        b = cand[rng.integers(len(cand))]
        if fam[b] != fam[a] and cnt.get(fam[b], 0) >= cap:
            continue
        others = [x for x in P if x != a]
        dS = R[b, others].sum() - R[a, others].sum()
        P2 = others + [b]
        new = sign * (S + dS) / npairs - lam * pen(P2)
        if new >= cur or rng.random() < np.exp((new - cur) / T):
            P = P2; S += dS; cur = new
            inP[a] = False; inP[b] = True
            cnt[fam[a]] -= 1; cnt[fam[b]] = cnt.get(fam[b], 0) + 1
    return {"models": [int(x) for x in P], "mean_R": float(S / npairs), "penalty": pen(P)}
