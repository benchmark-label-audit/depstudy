"""Figures of the paper. All AP values are expected AP under random ordering of ties.

  fig_sizecap   key-error AP vs panel size, most capable (off-benchmark) vs random, CRH and DS-1
  fig_dose      key-error AP vs mean residual correlation along the dependence ladders, four ladders
  fig_types     AUROC of each defect type against clean items (strong panel) and non-key answer concentration
  fig_sources   key-error AP on the cross-source panels (column width)
  fig_budget    share of MMLU key errors found within the first k items of the verification queue (column width)

  python scripts/s42_figures_v2.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")
CACHE = os.path.join(DATA_ROOT, "item_scores")
COL = {"crh": "#2a78d6", "ds": "#eb6834", "irt4pl": "#1baf7a", "mv": "#eda100", "crowdfm": "#6250d6",
       "accu_depen_lo": "#e87ba4", "accu_depen_nu_lo": "#9c3d6b", "glad_lo": "#008300", "entropy": "#e34948"}
LAB = {"crh": "CRH", "ds": "DS-1", "irt4pl": "IRT-4PL", "mv": "MV", "crowdfm": "CrowdFM", "accu_depen_lo": "AccuCopy", "accu_depen_nu_lo": "AccuCopy-NonUni",
       "glad_lo": "GLAD", "entropy": "Entropy"}
DS = {"mmlu": "ds_onecoin_lo", "gsm8k": "ds_onecoin_nf10_lo", "hellaswag": "ds_onecoin_lo"}
BLAB = {"mmlu": "MMLU", "gsm8k": "GSM8K", "hellaswag": "HellaSwag"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.family": "sans-serif", "pdf.fonttype": 42, "lines.linewidth": 1.4})
_T = {}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def mname(bench, m):
    return DS[bench] if m == "ds" else m


def truth(bench, sources=False):
    k = (bench, sources)
    if k not in _T:
        from s10_size_capability import key_mask
        if sources:
            import s6_sources as S6
            t = S6.load(bench)[0]
        else:
            import s19_logodds as S19
            t = S19.data(bench)[0]
        kpos, clean = key_mask(bench, t)
        _T[k] = (t, kpos, clean)
    return _T[k]


def load(bench, tag, m):
    p = os.path.join(CACHE, f"{bench}__{tag}__{m}.npy")
    if not os.path.exists(p):
        return None
    s = np.load(p); fin = s[np.isfinite(s)]
    return np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))


def key_ap(bench, tag, m, sources=False):
    s = load(bench, tag, mname(bench, m))
    if s is None:
        return np.nan
    t, kpos, clean = truth(bench, sources)
    keep = kpos | clean
    return Mt.average_precision_expected(s[keep], kpos[keep].astype(int))


def sizecap():
    c = pd.read_csv(os.path.join(RES, "contrasts_v2.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    for ax, b in zip(axes, ("mmlu", "gsm8k", "hellaswag")):
        for m in ("crh", "ds"):
            n = [10, 30, 100, 300, 1000]
            top = [key_ap(b, f"size_top_capability_{k}_0", m) for k in n]
            rnd = [np.mean([key_ap(b, f"size_random_{k}_{r}", m) for r in range(10)]) for k in n]
            allv = key_ap(b, "size_all", m)
            mm = mname(b, m)
            sig = {}
            for k in n:
                row = c[(c.contrast == f"{mm}: top{k} - random{k}") & (c.metric == "ap_key") & (c.scheme == "item") & (c.bench == b)]
                sig[k] = bool(len(row)) and float(row.p_holm.iloc[0]) < 0.05 and float(row.estimate.iloc[0]) > 0
            ax.plot(n, top, color=COL[m], lw=1.4, label=f"{LAB[m]}, most capable")
            for k, v in zip(n, top):
                ax.plot(k, v, marker="o", ms=3.6, color=COL[m], mfc=COL[m] if sig[k] else "white", mew=1.1)
            ax.plot(n, rnd, color=COL[m], ls="--", lw=1.1, label=f"{LAB[m]}, random (mean of 10)")
            ax.axhline(allv, color=COL[m], ls=":", lw=0.9, label=f"{LAB[m]}, all 2,999")
        ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1000"])
        ax.minorticks_off(); ax.set_title(BLAB[b], color=INK); ax.set_xlabel("panel size (models)"); style(ax)
    axes[0].set_ylabel("key-error AP")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(os.path.join(RES, "fig_sizecap.pdf")); fig.savefig(os.path.join(RES, "fig_sizecap.png"), dpi=200)


def dose():
    """Four ladders: GSM8K and HellaSwag (design on MMLU), MMLU split-half (pool 200), MMLU (design on GSM8K)."""
    P = json.load(open(os.path.join(RES, "dose_panels.json")))["benches"]
    S2 = json.load(open(os.path.join(RES, "splitdose_panels_p200.json")))
    dep = pd.read_csv(os.path.join(RES, "panel_dependence.csv")) if os.path.exists(os.path.join(RES, "panel_dependence.csv")) else None
    import s19_logodds as S19
    from s10_size_capability import key_mask
    from s37_mmlu_splitdose import halves
    methods = ["crh", "ds", "accu_depen_lo", "accu_depen_nu_lo", "irt4pl"]  # GLAD coincides with DS-1 on every ladder
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85))
    specs = [("gsm8k", "GSM8K (design: MMLU)"), ("hellaswag", "HellaSwag (design: MMLU)"), ("split", "MMLU (design: other half)"),
             ("mmlu", "MMLU (design: GSM8K)")]
    strong_R = {"gsm8k": 0.230, "hellaswag": 0.565, "mmlu": 0.095, "split": float(np.mean([F["R_small_strong_design"] for F in S2["folds"]]))}
    for ax, (b, title) in zip(axes, specs):
        for m in methods:
            xs, ys, es = [], [], []
            if b == "split":
                t, mods, X, g, Y, M = S19.data("mmlu")
                H = halves(t)
                for li in range(6):
                    vals, rs = [], []
                    for f, F in enumerate(S2["folds"]):
                        tE = t[H[1 - f]].reset_index(drop=True)
                        kpos, clean = key_mask("mmlu", tE); keep = kpos | clean
                        L = F["dose"][f"level{li}"]
                        for r, p in enumerate(L["reps"]):
                            s = np.load(os.path.join(CACHE, f"mmluhalf{1 - f}__splitdose_p200_f{f}_level{li}_{r}__{mname('mmlu', m)}.npy"))
                            fin = s[np.isfinite(s)]; s = np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))
                            vals.append(Mt.average_precision_expected(s[keep], kpos[keep].astype(int))); rs.append(p["mean_R_design"])
                    xs.append(np.mean(rs)); ys.append(np.mean(vals)); es.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                for li in range(6):
                    L = P[b]["dose"][f"level{li}"]
                    vals = [key_ap(b, f"dose_level{li}_{r}", m) for r in range(len(L["reps"]))]
                    xs.append(np.mean([p["mean_R_design"] for p in L["reps"]])); ys.append(np.nanmean(vals))
                    es.append(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals))))
            xs, ys, es = map(np.array, (xs, ys, es))
            ax.plot(xs, ys, color=COL[m], marker="o", ms=2.8, lw=1.2, label=LAB[m], ls="--" if m == "accu_depen_nu_lo" else "-")
            ax.fill_between(xs, ys - 1.96 * es, ys + 1.96 * es, color=COL[m], alpha=0.15, lw=0)
        ax.axvline(strong_R[b], color=MUTED, ls=":", lw=0.9)
        ax.set_title(title, color=INK); ax.set_xlabel("mean residual correlation"); style(ax)
    axes[0].set_ylabel("key-error AP")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(RES, "fig_dose.pdf")); fig.savefig(os.path.join(RES, "fig_dose.png"), dpi=200)


def types():
    a = pd.read_parquet(os.path.join(RES, "errortypes_auroc.parquet"))
    c = pd.read_parquet(os.path.join(RES, "errortypes_concentration.parquet"))
    TY = {"mmlu": [("wrong_groundtruth", "key\nerror"), ("bad_question_clarity", "unclear\nq."), ("bad_options_clarity", "unclear\nopt."),
                   ("multiple_correct_answers", "several\ncorr."), ("no_correct_answer", "no\ncorr.")],
          "gsm8k": [("revised_key", "revised\nkey"), ("removed_question", "removed\nitem")],
          "hellaswag": [("wrong_answer", "key\nerror"), ("multiple_correct_only", "several\nendings")]}
    fig = plt.figure(figsize=(7.0, 1.85))
    gs = fig.add_gridspec(1, 4, width_ratios=[6.2, 2, 2, 3.0])
    for k, b in enumerate(("mmlu", "gsm8k", "hellaswag")):
        ax = fig.add_subplot(gs[0, k])
        for j, m in enumerate(("crh", "ds", "irt4pl")):
            mm = mname(b, m)
            x = a[(a.bench == b) & (a.panel == "small_strong") & (a.method == mm)].set_index("type")
            v = [x.loc[t, "auroc"] for t, _ in TY[b]]
            lo = [x.loc[t, "auroc"] - x.loc[t, "auroc_lo"] for t, _ in TY[b]]
            hi = [x.loc[t, "auroc_hi"] - x.loc[t, "auroc"] for t, _ in TY[b]]
            ax.errorbar(np.arange(len(v)) + (j - 1) * 0.22, v, yerr=[lo, hi], fmt="o", ms=3, color=COL[m], lw=0.9, capsize=0,
                        label=LAB[m])
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(len(TY[b]))); ax.set_xticklabels([l for _, l in TY[b]], fontsize=6)
        ax.set_ylim(0.35, 1.0); ax.set_title(BLAB[b], color=INK); style(ax)
        if k == 0:
            ax.set_ylabel("AUROC vs. clean items")
    ax = fig.add_subplot(gs[0, 3])
    x = c[(c.panel == "small_strong") & (c.bench == "mmlu")].set_index("type")
    order = ["wrong_groundtruth", "bad_question_clarity", "bad_options_clarity", "multiple_correct_answers", "no_correct_answer", "clean"]
    labs = ["key error", "unclear q.", "unclear opt.", "several corr.", "no correct", "clean"]
    ax.barh(range(len(order)), [x.loc[o, "top_of_nonkey_mean"] for o in order], color=["#2a78d6"] + ["#9fb8d9"] * 4 + ["#c9c8c4"], height=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("top option's share\nof non-key answers"); ax.set_title("MMLU concentration", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="upper center", bbox_to_anchor=(0.35, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(RES, "fig_types.pdf")); fig.savefig(os.path.join(RES, "fig_types.png"), dpi=200)


def sources():
    conds = [("helm_all", ["src_helm_all_0"]), ("v1_strong_k", ["src_v1_strong_k_0"]),
             ("v1_random_k", [f"src_v1_random_k_{r}" for r in range(20)]), ("helm_plus_v1_strong100", ["src_helm_plus_v1_strong100_0"]),
             ("helm_plus_v1_all", ["src_helm_plus_v1_all_0"])]
    clab = ["HELM", "v1 top\n(same n)", "v1 rand.\n(same n)", "HELM\n+100 v1", "HELM\n+all v1"]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.05), sharex=True)
    for ax, b in zip(axes, ("mmlu", "gsm8k")):
        for j, m in enumerate(("crh", "ds", "irt4pl")):
            v = [np.nanmean([key_ap(b, tg, m, sources=True) for tg in tags]) for _, tags in conds]
            ax.bar(np.arange(len(conds)) + (j - 1) * 0.26, v, width=0.24, color=COL[m], label=LAB[m])
        ax.set_ylabel(f"{BLAB[b]}\nkey-error AP"); style(ax)
    axes[1].set_xticks(range(len(conds))); axes[1].set_xticklabels(clab)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(RES, "fig_sources.pdf")); fig.savefig(os.path.join(RES, "fig_sources.png"), dpi=200)


def budget():
    """Expected share of the 104 MMLU key errors within the first k items (random order within ties)."""
    t, kpos, clean = truth("mmlu")
    fig, ax = plt.subplots(figsize=(3.35, 1.5))
    ks = np.arange(10, 1001, 10)
    specs = [("crh", "single_small_strong", "CRH, strong", "-"), ("ds", "single_small_strong", "DS-1, strong", "-"),
             ("ds", "single_full", "DS-1, all 2,999", "--"), ("crh", "single_full", "CRH, all 2,999", "--")]
    for m, tag, lab, ls in specs:
        s = load("mmlu", tag, mname("mmlu", m))
        y = [Mt.precision_at_k_expected(s, kpos.astype(int), k) * k / kpos.sum() for k in ks]
        ax.plot(ks, y, color=COL[m], ls=ls, lw=1.3, label=lab)
    ax.plot(ks, ks / len(t), color=MUTED, ls=":", lw=0.9, label="random order")
    ax.axvline(200, color=GRID, lw=0.8)
    ax.set_xlabel("items verified ($k$)"); ax.set_ylabel("key errors found"); ax.set_ylim(0, 1.18); ax.set_yticks([0, .2, .4, .6, .8, 1]); style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=6, ncol=2, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_budget.pdf")); fig.savefig(os.path.join(RES, "fig_budget.png"), dpi=200)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sizecap", "dose", "types", "sources", "budget"]
    for w in which:
        globals()[w](); print(w, "done", flush=True)


def rankshift():
    """Rank of each top-100 model under the shipped keys versus after correcting the verified defects."""
    import s33_rankshift as R
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75))
    for ax, b in zip(axes, ("mmlu", "gsm8k")):
        t, mods, C0, C1, fix = R.correctness(b)
        n = len(t); defect = np.array([f is not None for f in fix])
        s0 = R.scores(C0, np.ones(n, bool)); Cf, kf = R.apply(C0, C1, fix, np.flatnonzero(defect)); s1 = R.scores(Cf, kf)
        r0, r1 = R.rank(s0), R.rank(s1)
        top = np.argsort(-s0)[:100]
        moved = np.abs(r0[top] - r1[top]) >= 5
        ax.plot([1, 100], [1, 100], color=GRID, lw=0.8, zorder=0)
        ax.scatter(r0[top][~moved], r1[top][~moved], s=5, color="#9fb8d9", lw=0, label="moved < 5")
        ax.scatter(r0[top][moved], r1[top][moved], s=6, color="#2a78d6", lw=0, label="moved $\geq$ 5")
        ax.set_xlim(0, 101); ax.set_title(BLAB[b], color=INK); ax.set_xlabel("rank, shipped keys")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rank, corrected")
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper left", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_rankshift.pdf")); fig.savefig(os.path.join(RES, "fig_rankshift.png"), dpi=200)
