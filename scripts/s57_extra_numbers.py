"""Supplementary numbers: method x panel ranks of GSM8K item 1309 (Example 3.1), ranks and leave-one-out key-error AP
of the ten revised GSM8K keys, HELM subset key errors, and per-method defects in the top-k queue of the strong panels.
Also the numbers that the paper states in its text only: the answer counts of Example 3.1, the scale of the DS-1 log
posterior odds, the rank agreement of ACCU and DS-1, the weight shares of the weak MMLU sources, the same-false-value
rates of HELM and the most accurate v1 models, and the coincidence rates used by AccuCopy-NonUni.

  python scripts/s57_extra_numbers.py > results/extra_numbers.txt
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402
import s19_logodds as S19  # noqa: E402
from s10_size_capability import key_mask  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
DS = {"mmlu": "ds_onecoin_lo", "gsm8k": "ds_onecoin_nf10_lo", "hellaswag": "ds_onecoin_lo"}


def load(bench, tag, m):
    if m == "ensemble":
        parts = [load(bench, tag, x) for x in ("crh", DS[bench], "irt4pl")]
        return np.mean([pd.Series(p).rank(method="average").values / len(p) for p in parts], axis=0)
    s = np.load(os.path.join(CACHE, f"{bench}__{tag}__{m}.npy"))
    fin = s[np.isfinite(s)]
    return np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))


def exp_rank(s, i):
    """Expected rank (1 = top) of item i under random tie order."""
    hi = (s > s[i]).sum()
    tie = (s == s[i]).sum()
    return hi + (tie + 1) / 2


def exp_topk_hits(s, y, k):
    order = np.sort(np.unique(s))[::-1]
    hits, taken = 0.0, 0
    for v in order:
        idx = s == v
        n = idx.sum()
        if taken + n <= k:
            hits += y[idx].sum(); taken += n
        else:
            hits += y[idx].sum() * (k - taken) / n; taken = k
        if taken >= k:
            break
    return hits


# ---- Example 3.1: 2x2 (method x panel) for GSM8K item 1309
t, *_ = S19.data("gsm8k")
i = int(np.flatnonzero(t.item_id.values == "gsm8k__1309")[0])
print("item 1309 status:", t.status.values[i], "| n items", len(t))
for tag in ("single_full", "single_offbench_strong", "single_small_strong"):
    for m in ("mv", "crh", DS["gsm8k"]):
        print(f"  {tag:26s} {m:20s} rank {exp_rank(load('gsm8k', tag, m), i):7.1f}")

# ---- GSM8K revised keys: ranks and leave-one-out key-error AP
kpos, clean = key_mask("gsm8k", t)
keep = kpos | clean
print("revised keys:", kpos.sum(), "clean:", clean.sum())
for tag in ("single_small_strong", "single_offbench_strong", "single_full"):
    s = load("gsm8k", tag, DS["gsm8k"])
    ranks = sorted(exp_rank(s[keep], j) for j in np.flatnonzero(kpos[keep]))
    base = Mt.average_precision_expected(s[keep], kpos[keep].astype(int))
    loo = []
    for j in np.flatnonzero(kpos):
        kk = keep.copy(); kk[j] = False
        loo.append(Mt.average_precision_expected(s[kk], kpos[kk].astype(int)))
    print(f"  {tag}: key-error AP {base:.3f}; LOO range {min(loo):.3f}-{max(loo):.3f}; ranks among {keep.sum()} key-err+clean: "
          + ", ".join(f"{r:.0f}" for r in ranks))

# ---- HELM GSM8K subset: revised keys present
import s6_sources as S6  # noqa: E402
for b in ("mmlu", "gsm8k"):
    ts, mods, X, g, Y, M = S6.load(b)
    kp, cl = key_mask(b, ts)
    print(f"HELM {b}: items {len(ts)}, key errors {kp.sum()}, defects {ts.defective.sum()}, models {len(mods)}")
    acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
    print("   accuracy on shared items: per-population summary below")
    helm = [m for m in mods if str(m).startswith("helm")] or None
    print("   model id sample:", list(mods[:3]), "...", list(mods[-3:]))
    print("   acc describe:", acc.describe().round(3).to_dict())

# ---- budget: per-method top-k on strong panels
for b, k in (("mmlu", 200), ("gsm8k", 100)):
    tb, *_ = S19.data(b)
    kp, cl = key_mask(b, tb)
    y = tb.defective.values.astype(int)
    for m in ("mv", "crh", DS[b], "glad_lo", "irt4pl", "ensemble"):
        s = load(b, "single_small_strong", m)
        print(f"budget {b} strong {m:20s} defects@{k} {exp_topk_hits(s, y, k):6.1f}/{y.sum()}  key@{k} {exp_topk_hits(s, kp.astype(int), k):5.1f}/{kp.sum()}")


# ---- HELM versus the most accurate v1 models of the same panel size: accuracy and shared models (by name)
import re  # noqa: E402


def _norm(name):
    s = name.lower().split(":")[-1]
    s = s.split("__")[-1] if "__" in s else s.split("_", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", re.sub(r"-(hf|instruct|chat)$", "", s))


for b in ("mmlu", "gsm8k"):
    ts, mods, X, g, Y, M = S6.load(b)
    acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
    helm = [m for m in mods if str(m).startswith("helm:")]
    v1 = [m for m in mods if not str(m).startswith("helm:")]
    top = acc[v1].sort_values(ascending=False).index[:len(helm)]
    shared = sorted({_norm(m) for m in helm} & {_norm(m) for m in top})
    print(f"HELM vs v1 top same n ({b}): HELM mean acc {acc[helm].mean():.3f}, v1 top mean acc {acc[top].mean():.3f}, "
          f"shared models {len(shared)} {shared}")

# ---- CrowdFM: ties of 1 - pi on the full panels (log-odds ranking would only matter if scores tied)
for b in ("mmlu", "gsm8k", "hellaswag"):
    s = np.load(os.path.join(CACHE, f"{b}__single_full__crowdfm.npy"))
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    print(f"CrowdFM {b} full panel: {len(s)} items, {len(cnt)} distinct scores, tied share {(cnt[inv] > 1).mean():.3f}")


# ---- Example 3.1: the answers on item 1309, the items above it, and the off-benchmark panel (Section 3)
from depstudy.data import long_table  # noqa: E402
import s10_size_capability as S10  # noqa: E402

t, mods, X, g, Y, M = S19.data("gsm8k")
i = int(np.flatnonzero(t.item_id.values == "gsm8k__1309")[0])
a = long_table(list(mods), "gsm8k")
a = a[a.item_id == "gsm8k__1309"]
vc = a.answer.dropna().value_counts()
print(f"Example 3.1: item 1309, shipped key {t.gold.values[i]}, expert key {t.gold_platinum.values[i] if 'gold_platinum' in t else 'n/a'}; "
      f"{len(mods)} models, {int((X[i] >= 0).sum())} parseable answers; most frequent: "
      + ", ".join(f"{v} x {n}" for v, n in vc.head(3).items()))
mv = load("gsm8k", "single_full", "mv")
above = mv > mv[i]
obs = X >= 0
agree = (X == 0).sum(1) / np.maximum(obs.sum(1), 1)  # share of parseable answers that give the key
clean = ~t.defective.values.astype(bool)
print(f"  full panel, MV: {int(above.sum())} items above item 1309, {int((above & clean & (agree < 0.5)).sum())} of them clean items "
      "on which most parseable answers reject the key")
common, cap = S10.capability()
top100 = list(cap["gsm8k"].sort_values(ascending=False).index)[:100]  # the 100 models with the highest MMLU accuracy
vt = a[a.model.isin(set(top100))].answer.dropna().value_counts()
print(f"  off-benchmark top 100: answers {({k: int(v) for k, v in vt.head(3).items()})}, key given by {int(vt.get(t.gold.values[i], 0))}")
y = t.defective.values.astype(int)
print(f"  off-benchmark top 100, DS-1: expected defects among the top 7 items "
      f"{exp_topk_hits(load('gsm8k', 'single_offbench_strong', DS['gsm8k']), y, 7):.1f}")

# ---- Section 5.4: scale of the log posterior odds, rank agreement of ACCU and DS-1, weights of the weak MMLU sources
s = load("mmlu", "single_full", "ds_onecoin_lo")
print(f"DS-1 full MMLU: median |log posterior odds| {np.median(np.abs(s)):.0f} nats; 1 - pi rounds to 0 or 1 beyond "
      f"{-np.log(np.finfo(float).eps / 2):.1f} nats")
from scipy.stats import kendalltau  # noqa: E402
taus = []
for b in ("mmlu", "gsm8k", "hellaswag"):
    for p in ("full", "corr_dedup", "small_strong", "small_diverse"):
        taus.append((kendalltau(load(b, f"single_{p}", "accu_nodep_lo"), load(b, f"single_{p}", DS[b])).statistic, b, p))
print("Kendall tau of ACCU (no dependence step) and DS-1 on the 12 single panels: min %.6f (%s %s), max %.6f"
      % (min(taus)[0], min(taus)[1], min(taus)[2], max(taus)[0]))
t, mods, X, g, Y, M = S19.data("mmlu")
acc = (Y * M).sum(0) / np.maximum(M.sum(0), 1)
low = acc < 0.30
_, _, p = Mt.dawid_skene_onecoin(X, g, K=4, logodds=True)
w = np.log(3 * p / (1 - p))   # one-coin weight log((K-1) p / (1 - p)); negative below chance
wp = np.clip(w, 0, None)
_, wc = Mt.crh(X, g)
print(f"full MMLU: {int(low.sum())} of {len(acc)} models below 0.30 accuracy; their share of the positive DS-1 weight "
      f"{wp[low].sum() / wp.sum():.4f} (signed {w[low].sum() / w.sum():.4f}, {int((w < 0).sum())} models below chance); "
      f"their share of the CRH weight {wc[low].sum() / wc.sum():.4f} (log m = {np.log(len(acc)):.2f}, median CRH weight {np.median(wc):.2f})")

# ---- Section 5.2: rate at which two erring models give the same false value, HELM versus the most accurate v1 models
ts, mods, X, g, Y, M = S6.load("gsm8k")
acc = pd.Series((Y * M).sum(0) / np.maximum(M.sum(0), 1), index=mods)
helm = [m for m in mods if str(m).startswith("helm:")]
v1 = [m for m in mods if not str(m).startswith("helm:")]
ix = {m: k for k, m in enumerate(mods)}
for name, sel in (("HELM", helm), ("v1, most accurate", list(acc[v1].sort_values(ascending=False).index[:len(helm)]))):
    c = np.array([ix[m] for m in sel])
    print(f"GSM8K shared items, {name} ({len(c)} models): same false value rate against the shipped key "
          f"{Mt.same_false_rate(X[:, c], g):.4f}")

# ---- Section 5.5: coincidence rate of erring sources used by AccuCopy-NonUni (strong panels)
import json  # noqa: E402
for b in ("mmlu", "gsm8k"):
    dg = json.load(open(os.path.join(REPO, "results", "depen_diag", f"{b}__single_small_strong.json")))
    print(f"NonUni {b} strong panel: same false value rate under the final NonUni-inferred truth {dg['same_false_rate_nu']:.4f}, "
          f"under the default AccuCopy truth {dg['same_false_rate_uni']:.4f}, uniform 1/n_f {dg['uniform_rate']:.4f}")
# GSM8K dependence ladder (designed on MMLU): AccuCopy-NonUni flags and same-false-value rates per level, against the
# strong panel above (selected on GSM8K accuracy)
import glob  # noqa: E402
import re  # noqa: E402
lv = {}
for p in glob.glob(os.path.join(REPO, "results", "depen_diag", "gsm8k__dose_level*_*.json")):
    dg = json.load(open(p))
    lv.setdefault(int(re.search(r"dose_level(\d+)_", p).group(1)), []).append((dg["shareD50_nu"], dg["same_false_rate_nu"]))
for k in sorted(lv):
    f, s = np.array(lv[k]).T
    print(f"NonUni gsm8k ladder level {k} ({len(f)} panels): flagged pairs {f.mean():.3f}, same false value rate {s.mean():.3f}")
