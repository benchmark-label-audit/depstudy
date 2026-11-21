"""Correlated-worker aggregation and copy-detection checks (review items S4 and S11).

Methods (item scores go to the shared cache, $DEPSTUDY_DATA/item_scores/<bench>__<tag>__<method>.npy)
  ebcc_lo             EBCC (Li, Rubinstein, Cohn, ICML 2019), authors' defaults, log-odds ranking. On GSM8K, whose
                      free-form answers have no global class set, EBCC's subtype model with one-coin confusion and
                      n_f = 10 (methods.ebcc_onecoin), as one-coin DS uses there.
  ebcc1_lo            the one-coin subtype model on MMLU and HellaSwag as well (robustness)
  accu_depen_warm_lo  AccuCopy (Dong et al.'s settings) warm-started from the converged truth and accuracies of
                      AccuNoDep, so the copying test never sees the soft first round

Panels: the single panels (s5/s5b, plus the cross-benchmark top 100 of s28) and the dependence ladders (s29).

  python scripts/s59_ebcc.py run --workers 6        -> results/ckpt_ebcc/*.json and cached item scores
  python scripts/s59_ebcc.py copyrates              -> results/copy_rates.csv (injected copiers at copy rates 0.3, 0.5, 0.8)
  python scripts/s59_ebcc.py stats                  -> results/ebcc_singles.csv, results/ebcc_contrasts.csv, results/ebcc_dose.csv

The interval of the share of accuracy error removed by verification is computed by s60_budget_ci.py.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CK = os.path.join(RES, "ckpt_ebcc")
CACHE = os.path.join(DATA_ROOT, "item_scores")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
DS = {"mmlu": "ds_onecoin_lo", "gsm8k": "ds_onecoin_nf10_lo", "hellaswag": "ds_onecoin_lo"}


def _K(bench):
    return 4 if bench in ("mmlu", "hellaswag") else None


def score(method, X, g, bench):
    from depstudy import methods as Mt
    K = _K(bench)
    with np.errstate(divide="ignore"):
        if method == "ebcc_lo":
            if K:
                return Mt.ebcc(X, g, K)[0], None
            return Mt.ebcc_onecoin(X, g, K=None, n_false=NF[bench])[0], None
        if method == "ebcc1_lo":
            return Mt.ebcc_onecoin(X, g, K=K, n_false=None if K else NF[bench])[0], None
        if method == "accu_depen_warm_lo":
            _, post, acc, _ = Mt.accu_depen(X, g, NF[bench], use_dependence=False, K=K, logodds=True)
            s, _, _, D = Mt.accu_depen(X, g, NF[bench], K=K, logodds=True, init=(post.argmax(1), acc))
            iu = np.triu_indices(D.shape[0], 1)
            return s, float((D[iu] > 0.5).mean())
    raise KeyError(method)


def tasks():
    import s10_size_capability as S10
    out = []
    for bench, pfile in (("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json")):
        P = json.load(open(os.path.join(RES, pfile)))["benches"][bench]
        for name, s in P["singles"].items():
            ms = ["ebcc_lo"] + (["ebcc1_lo"] if bench != "gsm8k" else [])
            if name == "small_strong":
                ms.append("accu_depen_warm_lo")
            out.append(("single", bench, name, 0, f"single_{name}", s["models"], ms))
    common, cap = S10.capability()
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)[:100]
        out.append(("single", b, "offbench_strong", 0, "single_offbench_strong", order, ["ebcc_lo"]))
    DP = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"]
    for E, B in DP.items():
        for lv, L in B["dose"].items():
            for r, p in enumerate(L["reps"]):
                ms = ["ebcc_lo"] + (["accu_depen_warm_lo"] if E == "gsm8k" else [])
                out.append(("dose", E, lv, r, f"dose_{lv}_{r}", p["models"], ms))
    return out


def job(task):
    import s19_logodds as S19
    family, bench, cond, rep, tag, models, methods = task
    t, mods, X, g, Y, M = S19.data(bench)
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    rows = []
    for m in methods:
        p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
        flag = None
        t0 = time.time()
        if os.path.exists(p) and m != "accu_depen_warm_lo":
            sec = float("nan")
        else:
            s, flag = score(m, X[:, c], g, bench)
            sec = time.time() - t0
            np.save(p + ".tmp.npy", np.asarray(s, np.float64)); os.replace(p + ".tmp.npy", p)
        rows.append({"family": family, "bench": bench, "condition": cond, "rep": rep, "n_models": int(len(c)), "method": m,
                     "flag_share": flag, "sec": sec})
    key = f"{family}__{bench}__{cond}__{rep}"
    with open(os.path.join(CK, key + ".json.tmp"), "w") as fh:
        json.dump(rows, fh)
    os.replace(os.path.join(CK, key + ".json.tmp"), os.path.join(CK, key + ".json"))
    return key, len(c)


def run(workers, heavy_workers):
    os.makedirs(CK, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    T = [t for t in tasks() if not os.path.exists(os.path.join(CK, f"{t[0]}__{t[1]}__{t[2]}__{t[3]}.json"))]
    light = [t for t in T if len(t[5]) < 1000]
    heavy = sorted([t for t in T if len(t[5]) >= 1000], key=lambda t: -len(t[5]))
    print("todo light", len(light), "heavy", len(heavy), flush=True)
    for group, w in ((light, workers), (heavy, heavy_workers)):
        if not group:
            continue
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = [ex.submit(job, t) for t in group]
            for i, f in enumerate(as_completed(futs)):
                k, n = f.result()
                if i % 20 == 0 or n >= 1000:
                    print(i + 1, "/", len(group), k, n, time.strftime("%H:%M:%S"), flush=True)


def copyrates():
    """Injected copiers at copy rates 0.3, 0.5 and 0.8 on simulated independent strong panels (s45.semi design):
    share of injected pairs that AccuCopy flags, given the generating truth and when it infers the truth."""
    import s19_logodds as S19
    import s45_copy_validation as S45
    from depstudy import methods as Mt
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    rows = []
    for bench, nf, K in (("mmlu", 3, 4), ("gsm8k", 10, None)):
        t, mods, X, g, Y, M = S19.data(bench)
        idx = {m: i for i, m in enumerate(mods)}
        c = np.array([idx[m] for m in P[bench]["singles"]["small_strong"]["models"]])
        Xr = X[:, c]
        fit = None
        for rep in range(3):
            rng = np.random.default_rng(4500 + rep)
            Xs, modal, fit = S45.simulate_independent(Xr, rng, fit, K)
            n_m = Xs.shape[1]
            for rate in (0.3, 0.5, 0.8):
                rr = np.random.default_rng(5900 + rep)
                Xc = Xs.copy()
                cop = rr.choice(n_m, 20, replace=False); pairs = list(zip(cop[:10], cop[10:]))
                for child, parent in pairs:
                    msk = (rr.random(Xs.shape[0]) < rate) & (Xs[:, parent] >= 0) & (Xs[:, child] >= 0)
                    Xc[msk, child] = Xs[msk, parent]
                injm = np.zeros((n_m, n_m), bool)
                for a_, b_ in pairs:
                    injm[a_, b_] = injm[b_, a_] = True
                iu = np.triu_indices(n_m, 1)
                for label, kw in (("given", dict(truth_override=modal)), ("inferred", dict())):
                    _, _, _, D = Mt.accu_depen(Xc, g, nf, K=K, logodds=True, **kw)
                    d, ij = D[iu], injm[iu]
                    rows.append({"bench": bench, "rep": rep, "copy_rate": rate, "truth": label,
                                 "flag_injected": float((d[ij] > 0.5).mean()), "flag_independent": float((d[~ij] > 0.5).mean())})
                    print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "copy_rates.csv"), index=False)
    print(d.groupby(["bench", "copy_rate", "truth"])[["flag_injected", "flag_independent"]].mean().round(3).to_string())


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run_ = 0.0; m = len(p)
    for k, i in enumerate(order):
        run_ = max(run_, min(1.0, (m - k) * p[i])); adj[i] = run_
    return adj


def stats():
    import s39_stats_v2 as S39
    import s35_dep_analysis as S35
    import s19_logodds as S19
    from s10_size_capability import key_mask
    from depstudy import methods as Mt
    single_rows, crow, drow = [], [], []
    ck = pd.DataFrame([r for f in glob.glob(os.path.join(CK, "*.json")) for r in json.load(open(f))])
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        t = S19.data(bench)[0]
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        bt = S39.Boot(t, kpos, keep, 20261124 + len(bench), None)
        ds = DS[bench]
        for panel in ("full", "corr_dedup", "small_strong", "small_diverse", "offbench_strong"):
            for m in ("ebcc_lo", "ebcc1_lo", ds, "accu_nodep_lo", "accu_depen_lo", "accu_depen_warm_lo", "mv", "glad_lo"):
                s = S39.load(bench, f"single_{panel}", m)
                if s is None:
                    continue
                fl = ck[(ck.family == "single") & (ck.bench == bench) & (ck.condition == panel) & (ck.method == m)].flag_share
                single_rows.append({"bench": bench, "panel": panel, "method": m, "ap": bt.point(s, "ap"), "ap_key": bt.point(s, "ap_key"),
                                    "flag_share": float(fl.iloc[0]) if len(fl) and fl.iloc[0] is not None else np.nan})
        pairs = [("ebcc_lo", ds, p) for p in ("full", "small_strong", "offbench_strong")]
        pairs += [("accu_depen_warm_lo", "accu_nodep_lo", "small_strong"), ("accu_depen_warm_lo", "accu_depen_lo", "small_strong")]
        for metric in ("ap", "ap_key"):
            for a, b, panel in pairs:
                sa, sb = S39.load(bench, f"single_{panel}", a), S39.load(bench, f"single_{panel}", b)
                if sa is None or sb is None:
                    continue
                d = bt.dist(f"{panel}|{a}", sa, metric) - bt.dist(f"{panel}|{b}", sb, metric)
                crow.append({"family": f"S4-{bench}-{metric}", "bench": bench, "metric": metric, "contrast": f"{panel}: {a} - {b}",
                             "a": bt.point(sa, metric), "b": bt.point(sb, metric),
                             **S39.summarize(d, bt.point(sa, metric) - bt.point(sb, metric))})
        # ladders: two-way bootstrap (items stratified, panels within level), as s35
        DP = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"][bench]
        levels = sorted(DP["dose"].keys())
        items = S35.Items(bench, 500, 20261121 + len(bench))
        rng = np.random.default_rng(5959)
        methods = ["ebcc_lo", ds] + (["accu_depen_warm_lo", "accu_nodep_lo", "accu_depen_lo"] if bench == "gsm8k" else [])
        for metric in ("ap", "ap_key"):
            per = {}
            for m in methods:
                vecs, lv = [], []
                for li, L in enumerate(levels):
                    for r in range(len(DP["dose"][L]["reps"])):
                        s = S35.load(bench, f"dose_{L}_{r}", m)
                        if s is not None:
                            vecs.append(s); lv.append(li)
                if len(set(lv)) < len(levels):
                    continue
                lv = np.array(lv)
                pt = np.array([items.point(s, metric) for s in vecs])
                bb = items.boot(np.stack(vecs), metric)
                per[m] = (pt, bb, lv)
            top = len(levels) - 1
            J = [np.concatenate([rng.choice(np.flatnonzero(per[methods[0]][2] == k), (per[methods[0]][2] == k).sum())
                                 for k in range(len(levels))]) for _ in range(500)]
            for m, (pt, bb, lv) in per.items():
                for k, L in enumerate(levels):
                    drow.append({"bench": bench, "metric": metric, "method": m, "effect": f"mean_{L}", "estimate": float(pt[lv == k].mean())})
                bd = np.array([bb[j[lv[j] == top], b].mean() - bb[j[lv[j] == 0], b].mean() for b, j in enumerate(J)])
                drow.append({"bench": bench, "metric": metric, "method": m, "family": f"S4-dose-{bench}-{metric}", "effect": "top_minus_bottom_level",
                             **{k: v for k, v in S35.summarize("x", pt[lv == top].mean() - pt[lv == 0].mean(), bd).items() if k != "effect"}})
            if "ebcc_lo" in per and ds in per:
                pa, ba, la = per["ebcc_lo"]; pb, bb2, lb = per[ds]
                for k in (0, top):
                    bd = np.array([ba[j[la[j] == k], b].mean() - bb2[j[lb[j] == k], b].mean() for b, j in enumerate(J)])
                    drow.append({"bench": bench, "metric": metric, "method": f"ebcc_lo - {ds}", "family": f"S4-dose-{bench}-{metric}",
                                 "effect": f"diff_{levels[k]}",
                                 **{kk: v for kk, v in S35.summarize("x", pa[la == k].mean() - pb[lb == k].mean(), bd).items() if kk != "effect"}})
    S = pd.DataFrame(single_rows); C = pd.DataFrame(crow); Dd = pd.DataFrame(drow)
    if len(C):
        C["p_holm"] = C.groupby("family").p_boot.transform(lambda p: holm(p.values))
    if "family" in Dd:
        m = Dd.family.notna()
        Dd.loc[m, "p_boot"] = Dd.loc[m, "p_boot"].clip(lower=1.0 / 500)   # B = 500: the smallest attainable tail share
        Dd.loc[m, "p_holm"] = Dd[m].groupby("family").p_boot.transform(lambda p: holm(p.values))
    S.to_csv(os.path.join(RES, "ebcc_singles.csv"), index=False)
    C.to_csv(os.path.join(RES, "ebcc_contrasts.csv"), index=False)
    Dd.to_csv(os.path.join(RES, "ebcc_dose.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string()); print(C.round(4).to_string()); print(Dd.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "copyrates", "stats"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--heavy-workers", dest="heavy_workers", type=int, default=2)
    a = ap.parse_args()
    {"run": lambda: run(a.workers, a.heavy_workers), "copyrates": copyrates, "stats": stats}[a.cmd]()


if __name__ == "__main__":
    main()
