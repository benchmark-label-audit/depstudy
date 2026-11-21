"""Correlation-aware fusion and the vote order of copy detection (review items S4, S9).

Methods (item scores go to the shared cache)
  precreccorr_lo           PrecRecCorr (Pochampally et al., SIGMOD 2014), exact within clusters of at most 8 correlated
                           sources, clusters independent, truth estimated iteratively (methods.precreccorr)
  precreccorr4_lo          the same with clusters of at most 4 sources (sensitivity, single panels)
  accu_depen_dongorder_lo  AccuCopy with Dong et al.'s greedy dependence order of the sources that share a value, instead
                           of the accuracy order (methods.dong_order)

Panels: the single panels (s5/s5b, plus the cross-benchmark top 100 of s28) and the dependence ladders (s29).

  python scripts/s61_precreccorr.py run --workers 6   -> results/ckpt_precreccorr/*.json and cached item scores
  python scripts/s61_precreccorr.py stats              -> results/precreccorr_singles.csv, precreccorr_contrasts.csv, precreccorr_dose.csv
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
CK = os.path.join(RES, "ckpt_precreccorr")
CACHE = os.path.join(DATA_ROOT, "item_scores")
NF = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
DS = {"mmlu": "ds_onecoin_lo", "gsm8k": "ds_onecoin_nf10_lo", "hellaswag": "ds_onecoin_lo"}


def _K(bench):
    return 4 if bench in ("mmlu", "hellaswag") else None


def score(method, X, g, bench):
    from depstudy import methods as Mt
    K = _K(bench)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "precreccorr_lo":
            s, _, cl = Mt.precreccorr(X, g, K=K, max_cluster=8)
            return s, {"n_clusters": len(cl), "max_cluster": int(max(len(c) for c in cl))}
        if method == "precreccorr4_lo":
            s, _, cl = Mt.precreccorr(X, g, K=K, max_cluster=4)
            return s, {"n_clusters": len(cl), "max_cluster": int(max(len(c) for c in cl))}
        if method == "accu_depen_dongorder_lo":
            s, _, _, D = Mt.accu_depen(X, g, NF[bench], K=K, logodds=True, vote_order="dependence")
            iu = np.triu_indices(D.shape[0], 1)
            return s, {"flag_share": float((D[iu] > 0.5).mean())}
    raise KeyError(method)


def tasks():
    import s10_size_capability as S10
    out = []
    for bench, pfile in (("mmlu", "panels.json"), ("gsm8k", "panels.json"), ("hellaswag", "panels_hellaswag.json")):
        P = json.load(open(os.path.join(RES, pfile)))["benches"][bench]
        for name, s in P["singles"].items():
            ms = ["precreccorr_lo", "precreccorr4_lo"]
            if name == "small_strong":
                ms.append("accu_depen_dongorder_lo")
            out.append(("single", bench, name, 0, f"single_{name}", s["models"], ms))
    common, cap = S10.capability()
    for b in ("mmlu", "gsm8k", "hellaswag"):
        order = list(cap[b].sort_values(ascending=False).index)[:100]
        out.append(("single", b, "offbench_strong", 0, "single_offbench_strong", order, ["precreccorr_lo"]))
    DP = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"]
    for E, B in DP.items():
        for lv, L in B["dose"].items():
            for r, p in enumerate(L["reps"]):
                ms = ["precreccorr_lo"] + (["accu_depen_dongorder_lo"] if E == "gsm8k" else [])
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
        t0 = time.time()
        s, info = score(m, X[:, c], g, bench)
        sec = time.time() - t0
        np.save(p + ".tmp.npy", np.asarray(s, np.float64)); os.replace(p + ".tmp.npy", p)
        rows.append({"family": family, "bench": bench, "condition": cond, "rep": rep, "n_models": int(len(c)), "method": m,
                     "sec": sec, **info})
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


def stats():
    import s39_stats_v2 as S39
    import s35_dep_analysis as S35
    import s19_logodds as S19
    import s59_ebcc as S59
    from s10_size_capability import key_mask
    ck = pd.DataFrame([r for f in glob.glob(os.path.join(CK, "*.json")) for r in json.load(open(f))])
    single_rows, crow, drow = [], [], []
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        t = S19.data(bench)[0]
        kpos, clean = key_mask(bench, t)
        keep = kpos | clean
        bt = S39.Boot(t, kpos, keep, 20261124 + len(bench), None)
        ds = DS[bench]
        for panel in ("full", "corr_dedup", "small_strong", "small_diverse", "offbench_strong"):
            for m in ("precreccorr_lo", "precreccorr4_lo", ds, "accu_depen_lo", "accu_depen_dongorder_lo", "ebcc_lo"):
                s = S39.load(bench, f"single_{panel}", m)
                if s is None:
                    continue
                info = ck[(ck.family == "single") & (ck.bench == bench) & (ck.condition == panel) & (ck.method == m)]
                single_rows.append({"bench": bench, "panel": panel, "method": m, "ap": bt.point(s, "ap"), "ap_key": bt.point(s, "ap_key"),
                                    **({k: info.iloc[0][k] for k in ("flag_share", "n_clusters", "max_cluster") if k in info and len(info)}
                                       if len(info) else {})})
        pairs = [("precreccorr_lo", ds, p) for p in ("full", "small_strong", "offbench_strong")]
        pairs += [("accu_depen_dongorder_lo", "accu_depen_lo", "small_strong")]
        for metric in ("ap", "ap_key"):
            for a, b, panel in pairs:
                sa, sb = S39.load(bench, f"single_{panel}", a), S39.load(bench, f"single_{panel}", b)
                if sa is None or sb is None:
                    continue
                d = bt.dist(f"{panel}|{a}", sa, metric) - bt.dist(f"{panel}|{b}", sb, metric)
                crow.append({"family": f"S4b-{bench}-{metric}", "bench": bench, "metric": metric, "contrast": f"{panel}: {a} - {b}",
                             "a": bt.point(sa, metric), "b": bt.point(sb, metric),
                             **S39.summarize(d, bt.point(sa, metric) - bt.point(sb, metric))})
        DP = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"][bench]
        levels = sorted(DP["dose"].keys())
        items = S35.Items(bench, 500, 20261121 + len(bench))
        rng = np.random.default_rng(6161)
        methods = ["precreccorr_lo", ds] + (["accu_depen_dongorder_lo", "accu_depen_lo"] if bench == "gsm8k" else [])
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
                per[m] = (np.array([items.point(s, metric) for s in vecs]), items.boot(np.stack(vecs), metric), lv)
            top = len(levels) - 1
            lv0 = per[methods[0]][2]
            J = [np.concatenate([rng.choice(np.flatnonzero(lv0 == k), (lv0 == k).sum()) for k in range(len(levels))]) for _ in range(500)]
            for m, (pt, bb, lv) in per.items():
                for k, L in enumerate(levels):
                    drow.append({"bench": bench, "metric": metric, "method": m, "effect": f"mean_{L}", "estimate": float(pt[lv == k].mean())})
                bd = np.array([bb[j[lv[j] == top], b].mean() - bb[j[lv[j] == 0], b].mean() for b, j in enumerate(J)])
                drow.append({"bench": bench, "metric": metric, "method": m, "family": f"S4b-dose-{bench}-{metric}", "effect": "top_minus_bottom_level",
                             **{k: v for k, v in S35.summarize("x", pt[lv == top].mean() - pt[lv == 0].mean(), bd).items() if k != "effect"}})
            for a, b in (("precreccorr_lo", ds), ("accu_depen_dongorder_lo", "accu_depen_lo")):
                if a not in per or b not in per:
                    continue
                pa, ba, la = per[a]; pb, bb2, lb = per[b]
                for k in (0, top):
                    bd = np.array([ba[j[la[j] == k], bb_].mean() - bb2[j[lb[j] == k], bb_].mean() for bb_, j in enumerate(J)])
                    drow.append({"bench": bench, "metric": metric, "method": f"{a} - {b}", "family": f"S4b-dose-{bench}-{metric}",
                                 "effect": f"diff_{levels[k]}",
                                 **{kk: v for kk, v in S35.summarize("x", pa[la == k].mean() - pb[lb == k].mean(), bd).items() if kk != "effect"}})
    S = pd.DataFrame(single_rows); C = pd.DataFrame(crow); Dd = pd.DataFrame(drow)
    if len(C):
        C["p_holm"] = C.groupby("family").p_boot.transform(lambda p: S59.holm(p.values))
    m = Dd.family.notna()
    Dd.loc[m, "p_boot"] = Dd.loc[m, "p_boot"].clip(lower=1.0 / 500)
    Dd.loc[m, "p_holm"] = Dd[m].groupby("family").p_boot.transform(lambda p: S59.holm(p.values))
    S.to_csv(os.path.join(RES, "precreccorr_singles.csv"), index=False)
    C.to_csv(os.path.join(RES, "precreccorr_contrasts.csv"), index=False)
    Dd.to_csv(os.path.join(RES, "precreccorr_dose.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string()); print(C.round(4).to_string()); print(Dd.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "stats"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--heavy-workers", dest="heavy_workers", type=int, default=3)
    a = ap.parse_args()
    {"run": lambda: run(a.workers, a.heavy_workers), "stats": stats}[a.cmd]()


if __name__ == "__main__":
    main()
