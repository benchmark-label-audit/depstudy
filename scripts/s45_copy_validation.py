"""Validation of copy detection.

(a) Reference example: the motivating example of Dong, Berti-Equille, Srivastava (PVLDB 2009, Table 1): five
    sources, five researchers, S4 and S5 copy S3. The paper reports that voting errs on three researchers, that the
    dependence-aware method returns all true values, and dependence probabilities .87 (S1-S2) and .99 (S3/S4/S5) in
    round 1, .18 for S1-S2 after refinement.
(b) Synthetic sources: independent sources with an attractive wrong option on hard items (shared difficulty), with
    and without copier groups, and a set of wrong keys. ACCU, DEPEN (uniform false values) and DEPEN-NonUni (empirical
    same-false-value rate) are compared on key-error AP and on the inferred copying probability of copier pairs and
    of independent pairs, with the copying test fed the inferred truth or the oracle truth.
(c) Real panels: DEPEN-NonUni on the single panels and the dependence ladders (scored by s46 into the cache).

  python scripts/s45_copy_validation.py ref|synth  ->  results/copy_reference.json, results/copy_synthetic.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy import methods as Mt  # noqa: E402

RES = os.path.join(ROOT, "results")


def reference():
    objs = ["Stonebraker", "Dewitt", "Bernstein", "Carey", "Halevy"]
    table = {"S1": ["MIT", "MSR", "MSR", "UCI", "Google"],
             "S2": ["Berkeley", "MSR", "MSR", "AT&T", "Google"],
             "S3": ["MIT", "UWisc", "MSR", "BEA", "UW"],
             "S4": ["MIT", "UWisc", "MSR", "BEA", "UW"],
             "S5": ["MS", "UWisc", "MSR", "BEA", "UW"]}
    truth = table["S1"]
    srcs = list(table)
    X = np.full((5, 5), -1, np.int32); codes = []
    for i in range(5):
        vals = sorted({table[s][i] for s in srcs}); codes.append(vals)
        for j, s in enumerate(srcs):
            X[i, j] = vals.index(table[s][i])
    g = np.zeros(5, np.int32)
    out = {}
    for name, kw in (("vote", None), ("accu", dict(use_dependence=False)), ("depen", dict(use_dependence=True)),
                     ("depen_nonuni", dict(use_dependence=True, nonuni=True))):
        if kw is None:
            C = Mt.onehot_counts(X, g); pick = C.argmax(1); D = None
        else:
            _, post, acc, D = Mt.accu_depen(X, g, 100, iters=20, alpha=0.2, c=0.8, **kw)
            pick = post.argmax(1)
        vals = [codes[i][pick[i]] for i in range(5)]
        row = {"values": dict(zip(objs, vals)), "correct": int(sum(v == t for v, t in zip(vals, truth)))}
        if D is not None:
            row["P_dep"] = {"S1-S2": round(float(D[0, 1]), 3), "S3-S4": round(float(D[2, 3]), 3), "S3-S5": round(float(D[2, 4]), 3),
                            "S4-S5": round(float(D[3, 4]), 3), "S1-S3": round(float(D[0, 2]), 3)}
        out[name] = row
        print(name, row, flush=True)
    json.dump(out, open(os.path.join(RES, "copy_reference.json"), "w"), indent=1)


def simulate(rng, n_items=2000, n_indep=60, n_groups=0, group_size=5, k=4, acc_lo=0.55, acc_hi=0.9, hard_frac=0.4,
             attract=0.0, copy_rate=0.8, key_err_rate=0.04):
    """Answer matrix of independent sources (accuracy uniform in [acc_lo, acc_hi]) and copier groups (each copier copies
    its group's parent with probability copy_rate, otherwise answers independently). On hard items, an erring
    independent source picks one attractive wrong option with probability `attract`, else a uniform wrong option."""
    t = rng.integers(0, k, n_items)
    hard = rng.random(n_items) < hard_frac
    att = (t + rng.integers(1, k, n_items)) % k
    def answer(acc):
        a = np.where(rng.random(n_items) < np.where(hard, acc - 0.25, acc), t, -1)
        wrong = a < 0
        u = (t + rng.integers(1, k, n_items)) % k
        pick_att = hard & (rng.random(n_items) < attract)
        a[wrong] = np.where(pick_att[wrong], att[wrong], u[wrong])
        return a
    cols, kind = [], []
    for _ in range(n_indep):
        cols.append(answer(rng.uniform(acc_lo, acc_hi))); kind.append(("indep", -1))
    for gi in range(n_groups):
        parent = answer(rng.uniform(acc_lo, acc_hi)); cols.append(parent); kind.append(("parent", gi))
        for _ in range(group_size - 1):
            own = answer(rng.uniform(acc_lo, acc_hi))
            cols.append(np.where(rng.random(n_items) < copy_rate, parent, own)); kind.append(("copier", gi))
    X = np.stack(cols, 1).astype(np.int32)
    g = t.copy()
    ke = rng.random(n_items) < key_err_rate
    g[ke] = (t[ke] + rng.integers(1, k, ke.sum())) % k
    return X, g, t, ke, kind


def synthetic():
    rows = []
    scen = [("uniform errors, copiers", dict(attract=0.0, n_groups=8)),
            ("shared difficulty, no copiers", dict(attract=0.8, n_groups=0)),
            ("shared difficulty, copiers", dict(attract=0.8, n_groups=8))]
    for rep in range(5):
        for name, kw in scen:
            rng = np.random.default_rng(1000 + rep)
            X, g, t, ke, kind = simulate(rng, **kw)
            m = X.shape[1]
            same_group = np.array([[kind[a][1] >= 0 and kind[a][1] == kind[b][1] for b in range(m)] for a in range(m)])
            indep = np.array([[kind[a][0] == "indep" and kind[b][0] == "indep" for b in range(m)] for a in range(m)])
            np.fill_diagonal(same_group, False); np.fill_diagonal(indep, False)
            res = {"scenario": name, "rep": rep, "n_sources": m}
            for label, kwm in (("accu", dict(use_dependence=False)), ("depen", dict()), ("depen_nonuni", dict(nonuni=True)),
                               ("depen_oracle", dict(truth_override=t)), ("depen_nonuni_oracle", dict(nonuni=True, truth_override=t))):
                s, post, acc, D = Mt.accu_depen(X, g, 3, K=4, logodds=True, **kwm)
                res[f"keyAP_{label}"] = Mt.average_precision_expected(s, ke.astype(int))
                if label != "accu":
                    res[f"Pdep_copiers_{label}"] = float(D[same_group].mean()) if same_group.any() else np.nan
                    res[f"Pdep_indep_{label}"] = float(D[indep].mean())
            res["same_false_rate"] = Mt.same_false_rate(X, t)
            rows.append(res)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "copy_synthetic.csv"), index=False)
    pd.set_option("display.width", 250)
    print(d.groupby("scenario").mean(numeric_only=True).round(3).T.to_string())


def fit_2pl(Z, O, iters=600, lr=0.1):
    """2PL fit of P(Z_ij = 1) = sigmoid(a_i (theta_j - b_i)) on observed entries O (gradient ascent, theta standardized)."""
    n_i, n_m = Z.shape
    rate = (Z * O).sum(0) / np.maximum(O.sum(0), 1)
    th = (rate - rate.mean()) / (rate.std() + 1e-9)
    p_i = np.clip((Z * O).sum(1) / np.maximum(O.sum(1), 1), 0.01, 0.99)
    b = -np.log(p_i / (1 - p_i)); a = np.ones(n_i)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a[:, None] * (th[None, :] - b[:, None])))
        e = (Z - p) * O
        th += lr * (e * a[:, None]).sum(0) / np.maximum(O.sum(0), 1)
        th = (th - th.mean()) / (th.std() + 1e-9)
        a += lr * (e * (th[None, :] - b[:, None])).sum(1) / np.maximum(O.sum(1), 1)
        b -= lr * (e * a[:, None]).sum(1) / np.maximum(O.sum(1), 1)
        a = np.clip(a, 0.05, 8); b = np.clip(b, -8, 8)
    return th, a, b


def simulate_independent(X, rng, fit=None, K=None):
    """Conditionally independent sources with the panel's item difficulty and wrong-answer distribution:
    source j gives item i's modal value with the fitted 2PL probability, otherwise a value drawn from the item's
    empirical non-modal answers. When the panel gives no non-modal answer, each erring source draws independently:
    uniformly among the K - 1 other options (multiple choice) or among 10 values no source gave (free form).
    Missingness is kept."""
    n_i, n_m = X.shape
    O = X >= 0
    V = int(X.max()) + 1
    C = np.zeros((n_i, V + 1))
    for v in range(V):
        C[:, v] = ((X == v) & O).sum(1)
    modal = C[:, :V].argmax(1)
    Z = (X == modal[:, None]).astype(float)
    th, a, b = fit if fit is not None else fit_2pl(Z, O.astype(float))
    P = 1 / (1 + np.exp(-a[:, None] * (th[None, :] - b[:, None])))
    nm = C[:, :V].copy(); nm[np.arange(n_i), modal] = 0
    Xs = np.full_like(X, -1)
    hit = rng.random((n_i, n_m)) < P
    for i in range(n_i):
        o = np.nonzero(O[i])[0]
        miss = o[~hit[i, o]]
        Xs[i, o[hit[i, o]]] = modal[i]
        if len(miss):
            w = nm[i]
            if w.sum() > 0:
                Xs[i, miss] = rng.choice(V, len(miss), p=w / w.sum())
            elif K is not None:
                Xs[i, miss] = rng.choice([k for k in range(K) if k != modal[i]], len(miss))
            else:
                Xs[i, miss] = V + rng.integers(0, 10, len(miss))  # values no source gave
    return Xs, modal, (th, a, b)


def semi():
    """Semi-synthetic check of the copy test on capable model panels: simulate conditionally independent sources
    from each real panel's fitted item model (no copying by construction), optionally inject copiers, and measure
    the copying probabilities DEPEN assigns, with the inferred or the generating truth."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import s19_logodds as S19
    from s10_size_capability import key_mask
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    rows = []
    for bench, nf, K in (("mmlu", 3, 4), ("gsm8k", 10, None)):
        t, mods, X, g, Y, M = S19.data(bench)
        idx = {m: i for i, m in enumerate(mods)}
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        for pname in ("small_strong", "small_diverse"):
            c = np.array([idx[m] for m in P[bench]["singles"][pname]["models"]])
            Xr = X[:, c]
            fit = None
            for rep in range(3):
                rng = np.random.default_rng(4500 + rep)
                Xs, modal, fit = simulate_independent(Xr, rng, fit, K)
                variants = [("real", Xr, None), ("sim_indep", Xs, None)]
                # inject copiers: 10 sources each replace their answers by another source's with probability 0.8
                Xc = Xs.copy(); n_m = Xs.shape[1]
                cop = rng.choice(n_m, 20, replace=False); pairs = list(zip(cop[:10], cop[10:]))
                for child, parent in pairs:
                    msk = (rng.random(Xs.shape[0]) < 0.8) & (Xs[:, parent] >= 0) & (Xs[:, child] >= 0)
                    Xc[msk, child] = Xs[msk, parent]
                variants.append(("sim_copiers", Xc, pairs))
                for vname, Xv, inj in variants:
                    if vname == "real" and rep > 0:
                        continue
                    res = {"bench": bench, "panel": pname, "rep": rep, "data": vname, "uniform_rate": 1.0 / nf,
                           "same_false_rate_modal": Mt.same_false_rate(Xv, modal if vname != "real" else Mt.onehot_counts(Xv, g).argmax(1))}
                    iu = np.triu_indices(Xv.shape[1], 1)
                    injm = np.zeros((Xv.shape[1], Xv.shape[1]), bool)
                    if inj:
                        for a_, b_ in inj:
                            injm[a_, b_] = injm[b_, a_] = True
                    for label, kw in (("accu", dict(use_dependence=False)), ("depen", dict()), ("depen_nu", dict(nonuni=True)),
                                      ("depen_oracle", dict(truth_override=modal)), ("depen_nu_oracle", dict(nonuni=True, truth_override=modal))):
                        if vname == "real" and "oracle" in label:
                            continue
                        s, post, acc, D = Mt.accu_depen(Xv, g, nf, K=K, logodds=True, **kw)
                        res[f"keyAP_{label}"] = Mt.average_precision_expected(s[keep], kpos[keep].astype(int))
                        if label != "accu":
                            d = D[iu]; ij = injm[iu]
                            res[f"flag_{label}"] = float((d[~ij] > 0.5).mean())
                            res[f"meanD_{label}"] = float(d[~ij].mean())
                            if inj:
                                res[f"flag_injected_{label}"] = float((d[ij] > 0.5).mean())
                    rows.append(res)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "copy_semisynthetic.csv"), index=False)
    pd.set_option("display.width", 250)
    print(d.groupby(["bench", "panel", "data"]).mean(numeric_only=True).round(3).T.to_string())


def _roots(lin, m):
    """Declared root models of m (list); empty when m declares no base model."""
    import ast
    if m not in lin.index:
        return []
    r = lin.roots.get(m)
    if isinstance(r, str):
        r = ast.literal_eval(r) if r.strip() else []
    return list(r) if r is not None else []


def realoracle():
    """Copy test on the real single panels with the truth given as the shipped key. Reports the share of source pairs flagged as copiers
    (P(dep) > .5) under inferred and given truth, for uniform and NonUni false values, and within vs across
    declared families."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import s19_logodds as S19
    from s10_size_capability import key_mask
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    P.update({"hellaswag": json.load(open(os.path.join(RES, "panels_hellaswag.json")))["benches"]["hellaswag"]})
    lin = pd.read_parquet(os.path.join(ROOT, "data", "lineage.parquet")).set_index("model_dir")
    rows = []
    for bench, nf, K in (("mmlu", 3, 4), ("gsm8k", 10, None), ("hellaswag", 3, 4)):
        t, mods, X, g, Y, M = S19.data(bench)
        idx = {m: i for i, m in enumerate(mods)}
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        truth = g.copy()  # the shipped key; it is wrong on 1.9% / 0.8% / 3.7% of items, too few to move pair counts
        for pname in ("small_strong", "small_diverse", "corr_dedup"):
            ms = P[bench]["singles"][pname]["models"]
            if len(ms) > 400:
                continue
            c = np.array([idx[m] for m in ms])
            Xp = X[:, c]
            fam = np.array([str(lin.family.get(m)).lower() if m in lin.index else "none" for m in ms])
            iu = np.triu_indices(len(c), 1)
            samef = (fam[iu[0]] == fam[iu[1]]) & ~np.isin(fam[iu[0]], ["none", "nan"])
            # known different lineage: both models trace to exactly one declared root, and the roots differ
            rt = [_roots(lin, m) for m in ms]
            r1 = np.array([r[0] if len(r) == 1 else "" for r in rt])
            diffr = (r1[iu[0]] != "") & (r1[iu[1]] != "") & (r1[iu[0]] != r1[iu[1]])
            res = {"bench": bench, "panel": pname, "n": len(c), "n_samefam_pairs": int(samef.sum()),
                   "n_knowndiff_pairs": int(diffr.sum()), "uniform_rate": 1 / nf,
                   "same_false_rate_key": Mt.same_false_rate(Xp, truth)}
            for label, kw in (("depen", dict()), ("depen_nu", dict(nonuni=True)), ("depen_given", dict(truth_override=truth)),
                              ("depen_nu_given", dict(nonuni=True, truth_override=truth))):
                s, post, acc, D = Mt.accu_depen(Xp, g, nf, K=K, logodds=True, **kw)
                d = D[iu]
                res[f"flag_{label}"] = float((d > 0.5).mean())
                res[f"flag_samefam_{label}"] = float((d[samef] > 0.5).mean()) if samef.any() else np.nan
                res[f"flag_crossfam_{label}"] = float((d[~samef] > 0.5).mean())
                res[f"flag_knowndiff_{label}"] = float((d[diffr] > 0.5).mean()) if diffr.any() else np.nan
                res[f"keyAP_{label}"] = Mt.average_precision_expected(s[keep], kpos[keep].astype(int))
            rows.append(res)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "copy_real_given_truth.csv"), index=False)


if __name__ == "__main__":
    {"ref": reference, "synth": synthetic, "semi": semi, "realoracle": realoracle}[sys.argv[1]]()
