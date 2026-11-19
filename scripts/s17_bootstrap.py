"""Paired item-bootstrap confidence intervals and Holm-adjusted p-values for the main contrasts.

Families (each Holm-adjusted separately per metric):
  panels   : per benchmark x method, single-panel contrasts (small_strong - full, small_strong - corr_dedup,
             small_diverse - full, corr_dedup - full)
  methods  : per benchmark x panel (full, small_strong), method contrasts (crh/mv vs ds_onecoin, irt4pl, accu_nodep)
  sources  : per benchmark x method, cross-source contrasts (HELM vs v1 random/strong of equal size; adding v1 to HELM)
  sizecap  : per benchmark x size, top-capability panel - mean of 10 random panels (crh, mv, ds_onecoin)

Item scores of the single and size panels are recomputed (gate checkpoints keep only metrics) and cached on D:;
recomputed AP is checked against the stored AP. Cross-source scores come from ckpt_sources. Bootstrap: B
resamples of items (same indices for every score vector of a benchmark, so contrasts are paired); 95% percentile
CI; p from the bootstrap standard error (normal approximation), then Holm within each family.

  python scripts/s17_bootstrap.py [--B 2000]  ->  results/bootstrap_contrasts.csv, results/bootstrap_summary.json
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.data import matrices, n_options  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CACHE = os.path.join(DATA_ROOT, "item_scores")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
METHODS = ["mv", "crh", "ds_onecoin", "accu_nodep", "irt4pl"]
SINGLES = ["full", "corr_dedup", "small_strong", "small_diverse"]
ANALYSIS = ["mv", "crh", "ds_onecoin", "ds_onecoin_lo", "accu_nodep_lo", "ds_full_lo", "accu_depen_lo", "irt4pl",
            "crowdfm", "crowdfm_block100"]
METHOD_PAIRS = [("crh", "mv"), ("crh", "ds_onecoin_lo"), ("mv", "ds_onecoin_lo"), ("ds_onecoin_lo", "irt4pl"), ("crh", "irt4pl"),
                ("ds_onecoin_lo", "ds_onecoin"), ("ds_onecoin_lo", "ds_full_lo"), ("accu_depen_lo", "accu_nodep_lo"),
                ("ds_onecoin_lo", "crowdfm"), ("crh", "crowdfm"), ("crowdfm_block100", "crowdfm")]
SIZE_METHODS = ["crh", "mv", "ds_onecoin_lo"]
SOURCE_METHODS = ["mv", "crh", "ds_onecoin_lo", "accu_nodep_lo", "irt4pl", "crowdfm"]


def loadc(bench, tag, m):
    p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
    return np.load(p) if os.path.exists(p) else None


def method_scores(name, X, g, Y, M, bench):
    K = n_options(bench) if bench in ("mmlu", "hellaswag") else None
    if name == "mv":
        return Mt.majority_vote(X, g)
    if name == "crh":
        return Mt.crh(X, g)[0]
    if name == "ds_onecoin":
        return Mt.dawid_skene_onecoin(X, g, K=K)[0]
    if name == "accu_nodep":
        return Mt.accu_depen(X, g, NF[bench], use_dependence=False, K=K)[0]
    if name == "irt4pl":
        return Mt.irt_ceiling_lr(X, g, Y, M)
    raise KeyError(name)


def cached(key, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, key + ".npy")
    if os.path.exists(p):
        return np.load(p)
    s = np.asarray(fn(), np.float64)
    np.save(p + ".tmp.npy", s)
    os.replace(p + ".tmp.npy", p)
    return s


def panel_scores(D, bench, tag, models, which):
    t, mods, X, g, Y, M = D
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    out = {}
    for m in which:
        out[m] = cached(f"{bench}__{tag}__{m}", lambda m=m: method_scores(m, X[:, c], g, Y[:, c], M[:, c], bench))
    return out


_WD = {}


def _precompute(task):
    """Worker: one (benchmark, panel, method) item-score vector into the cache; matrices loaded once per worker."""
    bench, tag, models, method = task
    if bench not in _WD:
        from s4_panels import eligible_models
        _WD[bench] = matrices(eligible_models(None), bench)
    panel_scores(_WD[bench], bench, tag, models, [method])
    return task[:2] + (method,)


def ap_batch(s, y, I, chunk=400):
    """AP of score vector s against labels y on each bootstrap resample (rows of I)."""
    out = np.empty(len(I))
    for a in range(0, len(I), chunk):
        J = I[a:a + chunk]
        S, Yb = s[J], y[J].astype(np.float32)
        o = np.argsort(-S, axis=1, kind="stable")
        Ys = np.take_along_axis(Yb, o, 1)
        cum = np.cumsum(Ys, 1)
        prec = cum / np.arange(1, Ys.shape[1] + 1, dtype=np.float32)
        out[a:a + chunk] = (prec * Ys).sum(1) / np.maximum(Ys.sum(1), 1)
    return out


class Boot:
    """Bootstrap AP distributions for one benchmark, cached per score vector; two metrics (all defects, key errors)."""

    def __init__(self, y_all, kpos, keep, B, seed):
        rng = np.random.default_rng(seed)
        n = len(y_all)
        self.y = {"ap": y_all.astype(int), "ap_key": kpos.astype(int)}
        self.sub = {"ap": np.arange(n), "ap_key": np.flatnonzero(keep)}
        self.I = {k: rng.integers(0, len(v), size=(B, len(v))) for k, v in self.sub.items()}
        self.memo = {}

    def point(self, s, metric):
        sub = self.sub[metric]
        return Mt.average_precision(s[sub], self.y[metric][sub])

    def dist(self, key, s, metric):
        if (key, metric) not in self.memo:
            sub = self.sub[metric]
            self.memo[(key, metric)] = ap_batch(s[sub], self.y[metric][sub], self.I[metric])
        return self.memo[(key, metric)]


def contrast(bt, metric, a, b):
    """a, b: lists of (key, scores); the contrast is mean AP over a minus mean AP over b."""
    pa = np.mean([bt.point(s, metric) for _, s in a]); pb = np.mean([bt.point(s, metric) for _, s in b])
    da = np.mean([bt.dist(k, s, metric) for k, s in a], axis=0); db = np.mean([bt.dist(k, s, metric) for k, s in b], axis=0)
    d = da - db
    se = d.std(ddof=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(2 * norm.sf(abs(pa - pb) / se)) if se > 0 else (0.0 if pa != pb else 1.0)
    return {"estimate": pa - pb, "a": pa, "b": pb, "ci_lo": lo, "ci_hi": hi, "se": se, "p": p}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from s4_panels import eligible_models
    from s10_size_capability import key_mask
    import s10_size_capability as S10
    models = eligible_models(None)
    rows = []
    gate = pd.concat([pd.read_parquet(os.path.join(RES, "gate_scores.parquet")), pd.read_parquet(os.path.join(RES, "hellaswag_scores.parquet"))])
    common, cap = S10.capability()
    rng_size = np.random.default_rng(20261117)  # same generator as s10 / s12
    size_specs = {}
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)
        sp = []
        for n in S10.SIZES:
            sp.append(("top_capability", n, 0, order[:n]))
            for r in range(10):
                sp.append(("random", n, r, list(rng_size.choice(common, n, replace=False))))
        size_specs[b] = sp
    if a.workers > 1:  # fill the item-score cache in parallel; the bootstrap below then only reads it
        tasks = []
        for bench, pfile in (("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json")):
            P = json.load(open(os.path.join(RES, pfile)))["benches"][bench]["singles"]
            tasks += [(bench, f"single_{c}", P[c]["models"], m) for c in SINGLES for m in METHODS]
            tasks += [(bench, f"size_{k}_{n}_{r}", ms, m) for k, n, r, ms in size_specs[bench] for m in ("crh", "mv")]
        tasks = [x for x in tasks if not os.path.exists(os.path.join(CACHE, f"{x[0]}__{x[1]}__{x[3]}.npy"))]
        w = {"irt4pl": 20, "ds_onecoin": 3, "accu_nodep": 1, "crh": 1, "mv": 0.1}
        tasks.sort(key=lambda x: -len(x[2]) * w[x[3]] * (3 if x[0] == "gsm8k" and x[3] == "ds_onecoin" else 1))
        print("precompute tasks", len(tasks), flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_precompute, x) for x in tasks]
            for i, f in enumerate(as_completed(futs)):
                f.result()
                if i % 20 == 0:
                    print("precomputed", i + 1, "/", len(tasks), flush=True)
    checks = []
    for bi, (bench, pfile) in enumerate((("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json"))):
        t0 = time.time()
        D = matrices(models, bench)
        t = D[0]
        kpos, clean = key_mask(bench, t)
        bt = Boot(t.defective.values, kpos, kpos | clean, a.B, 1000 + bi)
        P = json.load(open(os.path.join(RES, pfile)))["benches"][bench]["singles"]
        S = {}
        for cond in SINGLES:
            panel_scores(D, bench, f"single_{cond}", P[cond]["models"], METHODS)  # fills the cache if needed
            S[cond] = {m: v for m in ANALYSIS if (v := loadc(bench, f"single_{cond}", m)) is not None}
            for m in METHODS:
                st = gate[(gate.bench == bench) & (gate.kind == "single") & (gate.condition == cond) & (gate.method == m)].ap
                if len(st):
                    checks.append({"bench": bench, "panel": cond, "method": m, "stored_ap": float(st.iloc[0]),
                                   "recomputed_ap": Mt.average_precision(loadc(bench, f"single_{cond}", m), t.defective.values.astype(int))})
        print(bench, "single-panel scores ready", round(time.time() - t0), "s", flush=True)
        for metric in ("ap", "ap_key"):
            for m in ANALYSIS:
                for ca, cb in (("small_strong", "full"), ("small_strong", "corr_dedup"), ("small_diverse", "full"), ("corr_dedup", "full")):
                    if m in S[ca] and m in S[cb]:
                        r = contrast(bt, metric, [(f"{ca}|{m}", S[ca][m])], [(f"{cb}|{m}", S[cb][m])])
                        rows.append({"family": "panels", "bench": bench, "metric": metric, "method": m, "contrast": f"{ca} - {cb}", **r})
            for panel in ("full", "corr_dedup", "small_strong", "small_diverse"):
                for ma, mb in METHOD_PAIRS:
                    if ma in S[panel] and mb in S[panel]:
                        r = contrast(bt, metric, [(f"{panel}|{ma}", S[panel][ma])], [(f"{panel}|{mb}", S[panel][mb])])
                        rows.append({"family": "methods", "bench": bench, "metric": metric, "method": f"{ma} vs {mb}",
                                     "contrast": f"{ma} - {mb} on {panel}", **r})
        # size x capability
        Z = {}
        for kind, n, rep, ms in size_specs[bench]:
            panel_scores(D, bench, f"size_{kind}_{n}_{rep}", ms, ["crh", "mv"])
            Z[(kind, n, rep)] = {m: v for m in SIZE_METHODS if (v := loadc(bench, f"size_{kind}_{n}_{rep}", m)) is not None}
        for metric in ("ap", "ap_key"):
            for m in SIZE_METHODS:
                for n in S10.SIZES:
                    rand = [(f"rand{n}_{k}|{m}", Z[("random", n, k)][m]) for k in range(10) if m in Z[("random", n, k)]]
                    if m not in Z[("top_capability", n, 0)] or len(rand) < 10:
                        continue
                    r = contrast(bt, metric, [(f"top{n}|{m}", Z[("top_capability", n, 0)][m])], rand)
                    rows.append({"family": "sizecap", "bench": bench, "metric": metric, "method": m, "contrast": f"top_capability - random, n={n}", **r})
        print(bench, "bootstrap done", round(time.time() - t0), "s", flush=True)
        del D, bt
    # cross-source contrasts: stored scores (ckpt_sources) plus cached log-odds / CrowdFM scores
    import s6_sources as S6
    for bi, bench in enumerate(("mmlu", "gsm8k")):
        t = S6.load(bench)[0]
        kpos, clean = key_mask(bench, t)
        bt = Boot(t.defective.values, kpos, kpos | clean, a.B, 2000 + bi)
        sc = {}
        for f in glob.glob(os.path.join(RES, "ckpt_sources", f"{bench}__*.pkl")):
            for r in pickle.load(open(f, "rb")):
                sc[(r["condition"], r["rep"], r["method"])] = np.frombuffer(r["scores"], np.float32).astype(np.float64)
        for f in glob.glob(os.path.join(CACHE, f"{bench}__src_*.npy")):
            stem = os.path.basename(f)[:-4]
            tag, m = stem.split("__")[1], stem.split("__")[2]
            cond, rep = tag[4:].rsplit("_", 1)
            sc[(cond, int(rep), m)] = np.load(f)
        for metric in ("ap", "ap_key"):
            for m in SOURCE_METHODS:
                def g1(c, rep=0):
                    return [(f"{c}{rep}|{m}", sc[(c, rep, m)])] if (c, rep, m) in sc else None
                rand = [(f"v1_random_k{k}|{m}", sc[("v1_random_k", k, m)]) for k in range(20) if ("v1_random_k", k, m) in sc]
                for lab, A, Bv in (("helm_all - v1_random_k (mean of 20)", g1("helm_all"), rand or None),
                                   ("helm_all - v1_strong_k", g1("helm_all"), g1("v1_strong_k")),
                                   ("helm_plus_v1_all - helm_all", g1("helm_plus_v1_all"), g1("helm_all")),
                                   ("helm_plus_v1_strong100 - helm_all", g1("helm_plus_v1_strong100"), g1("helm_all")),
                                   ("helm_all - v1_all", g1("helm_all"), g1("v1_all"))):
                    if A is None or Bv is None:
                        continue
                    r = contrast(bt, metric, A, Bv)
                    rows.append({"family": "sources", "bench": bench, "metric": metric, "method": m, "contrast": lab, **r})
        print("sources", bench, "done", flush=True)
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for (fam, metric), grp in df.groupby(["family", "metric"]):
        df.loc[grp.index, "p_holm"] = holm(grp.p.values)
    df.to_csv(os.path.join(RES, "bootstrap_contrasts.csv"), index=False)
    ck = pd.DataFrame(checks)
    summ = {"B": a.B, "n_contrasts": int(len(df)),
            "recompute_check_max_abs_ap_diff": float((ck.stored_ap - ck.recomputed_ap).abs().max()) if len(ck) else None,
            "significant_holm_0.05": df[df.p_holm < 0.05].groupby(["family", "metric"]).size().to_dict()}
    summ = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in summ.items()}
    json.dump(summ, open(os.path.join(RES, "bootstrap_summary.json"), "w"), indent=1)
    ck.to_csv(os.path.join(RES, "bootstrap_recompute_check.csv"), index=False)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
