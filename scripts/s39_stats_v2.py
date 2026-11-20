"""Statistics for the paper.

* Every AP and P@k is the expected value under random ordering of tied scores (methods.average_precision_expected).
* Item bootstrap stratified by defect status, B = 2,000, paired across methods and panels; random-panel means add panel resampling (two-way).
  On MMLU a subject-cluster bootstrap (57 subjects) is reported next to the item bootstrap.
* p-values are two-sided bootstrap tail shares; Holm correction within each family = research question x benchmark x metric x bootstrap
  scheme (AP and key AP test different defect sets). The diagnostic contrasts (log odds versus probability ranking of DS-1, with the same
  n_false on both sides) form one family per metric and scheme that pools the three benchmarks (m = 3 for the item bootstrap).
* Rows of the shipped tables that cannot be recomputed (a missing item-score file) stop the script before anything is written.
* Equivalence: a contrast is declared equivalent when its 90% interval lies within +-0.02 AP (TOST).
* The rank average of CRH, DS-1 and IRT-4PL is computed here.

  python scripts/s39_stats_v2.py  ->  results/singles_v2.parquet, results/contrasts_v2.csv
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
B = 2000
PK = {"mmlu": 200, "gsm8k": 100, "hellaswag": 200}
DS = {"mmlu": "ds_onecoin_lo", "gsm8k": "ds_onecoin_nf10_lo", "hellaswag": "ds_onecoin_lo"}
PANELS = ["full", "corr_dedup", "small_strong", "small_diverse", "offbench_strong"]
SINGLE_METHODS = ["mv", "crh", "catd", "ds_onecoin_lo", "ds_onecoin_nf10_lo", "ds_onecoin", "ds_onecoin_nf10", "ds_full_lo", "accu_nodep_lo",
                  "accu_depen_lo", "accu_depen_nu_lo", "accu_depen_key_lo", "accu_depen_nu_key_lo", "glad_lo", "irt4pl", "irt4pl_miss", "crowdfm", "crowdfm_block100", "entropy", "ensemble"]
SIZES = [10, 30, 100, 300, 1000]


def load(bench, tag, m):
    if m == "ensemble":
        parts = [load(bench, tag, x) for x in ("crh", DS[bench], "irt4pl")]
        if any(p is None for p in parts):
            return None
        return np.mean([pd.Series(p).rank(method="average").values / len(p) for p in parts], axis=0)
    p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
    if not os.path.exists(p):
        return None
    s = np.load(p)
    fin = s[np.isfinite(s)]
    return np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))


class Boot:
    def __init__(self, t, kpos, keep, seed, subjects=None):
        rng = np.random.default_rng(seed)
        n = len(t)
        self.y = {"ap": t.defective.values.astype(int), "ap_key": kpos.astype(int)}
        self.sub = {"ap": np.arange(n), "ap_key": np.flatnonzero(keep)}
        # stratified: positives and negatives are resampled separately, so every resample keeps the defect count
        # (GSM8K has 10 key errors; an unstratified resample can contain none)
        self.I = {}
        for k, v in self.sub.items():
            yy = self.y[k][v]
            pos, neg = v[yy == 1], v[yy == 0]
            self.I[k] = np.concatenate([pos[rng.integers(0, len(pos), size=(B, len(pos)))],
                                        neg[rng.integers(0, len(neg), size=(B, len(neg)))]], axis=1)
        self.C = None
        if subjects is not None:                      # subject-cluster resamples (concatenated item lists)
            subj = np.asarray(subjects); names = np.unique(subj)
            idx = {s: np.flatnonzero(subj == s) for s in names}
            self.C = {}
            for k, v in self.sub.items():
                vs = set(v.tolist())
                rows = []
                for b in range(B):
                    pick = rng.choice(names, len(names))
                    rows.append(np.concatenate([[i for i in idx[s] if i in vs] for s in pick]).astype(int))
                self.C[k] = rows
        self.memo = {}

    def point(self, s, metric):
        sub = self.sub[metric]
        return Mt.average_precision_expected(s[sub], self.y[metric][sub])

    def dist(self, key, s, metric, scheme="item"):
        mk = (key, metric, scheme)
        if mk not in self.memo:
            if scheme == "item":
                I = self.I[metric]
                out = np.concatenate([Mt.average_precision_expected_batch(s[I[a:a + 250]], self.y[metric][I[a:a + 250]])
                                      for a in range(0, B, 250)])
            else:
                out = np.array([Mt.average_precision_expected(s[r], self.y[metric][r]) for r in self.C[metric]])
            self.memo[mk] = out
        return self.memo[mk]


def summarize(d, est):
    lo, hi = np.percentile(d, [2.5, 97.5]); l90, h90 = np.percentile(d, [5, 95])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"estimate": float(est), "lo": float(lo), "hi": float(hi), "p_boot": max(p, 1.0 / B),
            "equiv_0.02": bool(l90 > -0.02 and h90 < 0.02)}


def bench_job(bench):
    import s19_logodds as S19
    from s10_size_capability import key_mask
    t = S19.data(bench)[0]
    kpos, clean = key_mask(bench, t)
    keep = kpos | clean
    bt = Boot(t, kpos, keep, 20261124 + len(bench), t.subject.values if bench == "mmlu" else None)
    rng = np.random.default_rng(7)
    y = t.defective.values.astype(int)
    rows, crow = [], []
    schemes = ["item", "cluster"] if bench == "mmlu" else ["item"]
    # ------------------------------------------------------------- point table (singles)
    for panel in PANELS:
        for m in SINGLE_METHODS:
            s = load(bench, f"single_{panel}", m)
            if s is None:
                continue
            rows.append({"bench": bench, "panel": panel, "method": m, "ap": bt.point(s, "ap"),
                         "pk": Mt.precision_at_k_expected(s, y, PK[bench]), "ap_key": bt.point(s, "ap_key"),
                         "auroc_key": float(pd.Series(np.r_[s[kpos], s[clean]]).rank().values[:kpos.sum()].sum()
                                            - kpos.sum() * (kpos.sum() + 1) / 2) / (kpos.sum() * clean.sum())})

    def contrast(fam, name, a, b, metric):
        """a, b: lists of (tag, method); a list of several entries is a random-panel mean (panels resampled)."""
        for scheme in schemes:
            def dist(spec):
                vecs = [(f"{tag}|{m}", load(bench, tag, m)) for tag, m in spec]
                vecs = [(k, v) for k, v in vecs if v is not None]
                if not vecs:
                    return None, None
                D = np.stack([bt.dist(k, v, metric, scheme) for k, v in vecs])
                P = np.array([bt.point(v, metric) for k, v in vecs])
                if len(vecs) == 1:
                    return D[0], P[0]
                J = rng.integers(0, len(vecs), size=(B, len(vecs)))
                return D[J, np.arange(B)[:, None]].mean(1), P.mean()
            da, pa = dist(a); db, pb = dist(b)
            if da is None or db is None:
                return
            crow.append({"family": fam, "bench": bench, "contrast": name, "metric": metric, "scheme": scheme,
                         "a": pa, "b": pb, **summarize(da - db, pa - pb)})

    ds = DS[bench]
    for metric in ("ap", "ap_key"):
        # RQ1: capability, size
        for m in ("mv", "crh", ds, "glad_lo", "irt4pl"):
            contrast(f"RQ1-{bench}", f"{m}: strong - full", [("single_small_strong", m)], [("single_full", m)], metric)
            contrast(f"RQ1-{bench}", f"{m}: offbench - full", [("single_offbench_strong", m)], [("single_full", m)], metric)
            contrast(f"RQ1-{bench}", f"{m}: offbench - strong", [("single_offbench_strong", m)], [("single_small_strong", m)], metric)
        for m in ("crh", ds, "glad_lo"):
            mm = m
            for n in SIZES:
                contrast(f"RQ1-{bench}", f"{mm}: top{n} - random{n}", [(f"size_top_capability_{n}_0", mm)],
                         [(f"size_random_{n}_{r}", mm) for r in range(10)], metric)
            contrast(f"RQ1-{bench}", f"{mm}: random300 - all", [(f"size_random_300_{r}", mm) for r in range(10)],
                     [("size_all", mm)], metric)
        # RQ3: methods
        for panel in ("full", "small_strong", "offbench_strong"):
            tg = f"single_{panel}"
            pairs = [(ds, "crh"), (ds, "irt4pl"), (ds, "mv"), ("glad_lo", ds), ("catd", "crh"), ("crh", "mv"), ("entropy", "mv"),
                     ("crowdfm", ds), ("ensemble", ds), ("ensemble", "crh"), ("ensemble", "irt4pl")]
            pairs += [("accu_depen_lo", "accu_nodep_lo"), ("accu_depen_nu_lo", "accu_nodep_lo"),
                      ("accu_depen_key_lo", "accu_nodep_lo"), ("accu_depen_nu_key_lo", "accu_nodep_lo")]
            if bench != "gsm8k":
                pairs.append(("ds_full_lo", ds))
            else:
                pairs += [("ds_onecoin_nf10_lo", "ds_onecoin_lo"), ("irt4pl_miss", "irt4pl")]
            for x, z in pairs:
                contrast(f"RQ3-{bench}", f"{panel}: {x} - {z}", [(tg, x)], [(tg, z)], metric)
        # same false-value count on both sides: n_false = 10 on GSM8K (s53 writes the probability scores)
        prob = "ds_onecoin_nf10" if bench == "gsm8k" else "ds_onecoin"
        contrast("diag", f"full: {ds} log-odds - probability", [("single_full", ds)], [("single_full", prob)], metric)
    return rows, crow


def sources_job(bench):
    import s6_sources as S6
    from s10_size_capability import key_mask
    t = S6.load(bench)[0]
    kpos, clean = key_mask(bench, t)
    bt = Boot(t, kpos, kpos | clean, 20261125 + len(bench))
    rng = np.random.default_rng(8)
    crow = []
    ds = DS[bench]
    for metric in ("ap", "ap_key"):
        for m in ("mv", "crh", ds, "glad_lo", "irt4pl"):
            def d(spec):
                vecs = [(f"{tag}|{m}", load(bench, tag, m)) for tag in spec]
                vecs = [(k, v) for k, v in vecs if v is not None]
                if not vecs:
                    return None, None
                D = np.stack([bt.dist(k, v, metric) for k, v in vecs]); P = np.array([bt.point(v, metric) for k, v in vecs])
                if len(vecs) == 1:
                    return D[0], P[0]
                J = rng.integers(0, len(vecs), size=(B, len(vecs)))
                return D[J, np.arange(B)[:, None]].mean(1), P.mean()
            specs = {"helm_all": ["src_helm_all_0"], "v1_strong_k": ["src_v1_strong_k_0"],
                     "v1_random_k": [f"src_v1_random_k_{r}" for r in range(20)],
                     "helm_plus_v1_strong100": ["src_helm_plus_v1_strong100_0"], "helm_plus_v1_all": ["src_helm_plus_v1_all_0"],
                     "vendor_concentrated": [f"src_helm_vendor_concentrated_{r}" for r in range(20)],
                     "vendor_diverse": [f"src_helm_vendor_diverse_{r}" for r in range(20)]}
            for a, b in (("helm_all", "v1_strong_k"), ("helm_all", "v1_random_k"), ("helm_plus_v1_all", "helm_all"),
                         ("helm_plus_v1_strong100", "helm_all"), ("vendor_concentrated", "vendor_diverse")):
                da, pa = d(specs[a]); db, pb = d(specs[b])
                if da is None or db is None:
                    continue
                crow.append({"family": f"RQ1-sources-{bench}", "bench": bench, "contrast": f"{m}: {a} - {b}", "metric": metric,
                             "scheme": "item", "a": pa, "b": pb, **summarize(da - db, pa - pb)})
    return [], crow


def holm(df):
    df = df.copy(); df["p_holm"] = np.nan
    for (fam, scheme, metric), g in df.groupby(["family", "scheme", "metric"]):  # family = RQ x benchmark x metric
        order = g.p_boot.sort_values().index
        m = len(order); prev = 0.0
        for k, i in enumerate(order):
            val = min(1.0, max(prev, (m - k) * df.at[i, "p_boot"])); df.at[i, "p_holm"] = val; prev = val
    return df


def check_complete(rows, c):
    """Stop when a row of the shipped tables cannot be recomputed, since a missing row also changes Holm family sizes."""
    miss = []
    ps, pc = os.path.join(RES, "singles_v2.parquet"), os.path.join(RES, "contrasts_v2.csv")
    if os.path.exists(ps):
        k = ["bench", "panel", "method"]
        old = set(map(tuple, pd.read_parquet(ps)[k].astype(str).values))
        miss += [("singles",) + x for x in sorted(old - set(map(tuple, pd.DataFrame(rows)[k].astype(str).values)))]
    if os.path.exists(pc):
        k = ["family", "bench", "contrast", "metric", "scheme"]
        old = set(map(tuple, pd.read_csv(pc)[k].astype(str).values))
        miss += [("contrasts",) + x for x in sorted(old - set(map(tuple, c[k].astype(str).values)))]
    if miss:
        for x in miss[:40]:
            print("missing:", x)
        sys.exit(f"{len(miss)} rows of the shipped tables could not be recomputed; check {CACHE}. Nothing was written.")


def main():
    jobs = [("b", b) for b in ("mmlu", "gsm8k", "hellaswag")] + [("s", b) for b in ("mmlu", "gsm8k")]
    rows, crow = [], []
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(bench_job if k == "b" else sources_job, b) for k, b in jobs]
        for f in futs:
            r, c = f.result(); rows += r; crow += c
    c = holm(pd.DataFrame(crow))
    check_complete(rows, c)
    pd.DataFrame(rows).to_parquet(os.path.join(RES, "singles_v2.parquet"))
    c.to_csv(os.path.join(RES, "contrasts_v2.csv"), index=False)
    pd.set_option("display.width", 250)
    d = pd.DataFrame(rows)
    print(d.pivot_table(index=["bench", "panel"], columns="method", values="ap").round(3).to_string())
    print(d.pivot_table(index=["bench", "panel"], columns="method", values="ap_key").round(3).to_string())
    print(c[c.scheme == "item"].round(4).to_string(max_rows=400))


if __name__ == "__main__":
    main()
