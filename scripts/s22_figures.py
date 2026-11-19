"""Paper figures from the result files: size x capability curves, lineage x dependence effects, error-type lifts,
and cross-source comparison. Colors follow one fixed method -> slot mapping (validated categorical palette).

  python scripts/s22_figures.py  ->  results/fig_sizecap.pdf, fig_cells.pdf, fig_errortypes.pdf, fig_sources.pdf
"""
import os
import pickle
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RES = os.path.join(ROOT, "results")
COL = {"crh": "#2a78d6", "ds_onecoin_lo": "#eb6834", "irt4pl": "#1baf7a", "mv": "#eda100", "crowdfm": "#6250d6",
       "accu_depen_lo": "#e87ba4", "ds_full_lo": "#008300", "accu_nodep_lo": "#e34948"}
LAB = {"crh": "CRH", "ds_onecoin_lo": "DS-1", "irt4pl": "IRT-4PL", "mv": "MV", "crowdfm": "CrowdFM", "accu_depen_lo": "Depen",
       "ds_full_lo": "DS", "accu_nodep_lo": "Accu"}
BLAB = {"mmlu": "MMLU", "gsm8k": "GSM8K", "hellaswag": "HellaSwag"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.family": "sans-serif", "pdf.fonttype": 42, "lines.linewidth": 1.4})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _boot_band(bench, tag, m, B=500, seed=0):
    """95% item-bootstrap interval of key-error AP for one cached score vector."""
    from depstudy.hf import DATA_ROOT
    from depstudy.data import matrices
    from s4_panels import eligible_models
    from s10_size_capability import key_mask
    from s17_bootstrap import ap_batch
    if bench not in _T:
        t = matrices(eligible_models(None), bench)[0]
        kpos, clean = key_mask(bench, t)
        _T[bench] = (kpos, kpos | clean)
    kpos, keep = _T[bench]
    p = os.path.join(DATA_ROOT, "item_scores", f"{bench}__{tag}__{m}.npy")
    if not os.path.exists(p):
        return np.nan, np.nan
    sub = np.flatnonzero(keep)
    s = np.load(p)[sub]; y = kpos[sub].astype(int)
    I = np.random.default_rng(seed).integers(0, len(sub), size=(B, len(sub)))
    a = ap_batch(s, y, I)
    return np.percentile(a, [2.5, 97.5])


_T = {}


def sizecap():
    s = pd.read_parquet(os.path.join(RES, "size_capability.parquet"))
    l = pd.read_parquet(os.path.join(RES, "logodds_scores.parquet"))
    l = l[l.family == "sizecap"].rename(columns={"condition": "n"}); l["n"] = l.n.astype(int)
    cols = ["bench", "kind", "n", "rep", "method", "ap_key"]
    d = pd.concat([s[cols], l[cols]])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
    for ax, b in zip(axes, ("mmlu", "gsm8k", "hellaswag")):
        for m in ("crh", "ds_onecoin_lo"):
            x = d[(d.bench == b) & (d.method == m)]
            top = x[x.kind == "top_capability"].groupby("n").ap_key.mean()
            r = x[x.kind == "random"].groupby("n").ap_key
            mu, se = r.mean(), r.std(ddof=1) / np.sqrt(r.count())
            bc = pd.read_csv(os.path.join(RES, "bootstrap_contrasts.csv"))
            bc = bc[(bc.family == "sizecap") & (bc.metric == "ap_key") & (bc.bench == b) & (bc.method == m)]
            sig = {int(c.split("n=")[1]): ph < 0.05 for c, ph in zip(bc.contrast, bc.p_holm)}
            ax.plot(top.index, top.values, color=COL[m], lw=1.4, label=f"{LAB[m]}, most capable")
            for n_, v_ in top.items():
                ax.plot(n_, v_, marker="o", ms=3.6, color=COL[m], mfc=COL[m] if sig.get(int(n_), False) else "white", mew=1.1)
            ax.plot(mu.index, mu.values, color=COL[m], ls="--", lw=1.1, label=f"{LAB[m]}, random (mean of 10)")
            a = x[x.kind == "all"].ap_key
            if len(a):
                ax.axhline(a.mean(), color=COL[m], ls=":", lw=0.9, label=f"{LAB[m]}, all 2,999")
        ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1000"])
        ax.minorticks_off()
        ax.set_title(BLAB[b], color=INK); ax.set_xlabel("panel size (models)")
        style(ax)
    axes[0].set_ylabel("key-error AP")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(os.path.join(RES, "fig_sizecap.pdf")); fig.savefig(os.path.join(RES, "fig_sizecap.png"), dpi=200)


def cells():
    e = pd.read_csv(os.path.join(RES, "cells_effects.csv"))
    e = e[(e.metric == "pk") & e.effect.isin(["dependence", "lineage"])]
    order = ["mv", "crh", "ds_onecoin_lo", "ds_full_lo", "accu_depen_lo", "irt4pl", "crowdfm"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, b in zip(axes, ("mmlu", "gsm8k", "hellaswag")):
        x = e[e.bench == b]
        ms = order
        for k, (eff, mk, c, off) in enumerate((("dependence", "o", "#2a78d6", -0.17), ("lineage", "s", "#eb6834", 0.17))):
            for i, m in enumerate(ms):
                r = x[(x.method == m) & (x.effect == eff)]
                if not len(r):
                    continue
                r = r.iloc[0]
                y = i + off
                ax.plot([r.ci_lo * 100, r.ci_hi * 100], [y, y], color=c, lw=1.2)
                ax.plot(r.estimate * 100, y, marker=mk, color=c, ms=4, mfc=c if r.p_holm < 0.05 else "white",
                        label=(f"{eff} effect" if i == 0 else None))
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB[m] for m in ms])
        ax.set_ylim(len(ms) - 0.5, -0.5)
        ax.set_title(BLAB[b], color=INK); ax.set_xlabel("effect on P@k (points)")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(RES, "fig_cells.pdf")); fig.savefig(os.path.join(RES, "fig_cells.png"), dpi=200)


def errortypes():
    d = pd.read_parquet(os.path.join(RES, "error_types_v2.parquet"))
    d = d[(d.panel == "small_strong") & d.method.isin(["crh", "ds_onecoin_lo", "irt4pl"])]
    names = {"wrong_groundtruth": "wrong\nkey", "bad_question_clarity": "unclear\nquestion", "bad_options_clarity": "unclear\noptions",
             "multiple_correct_answers": "several\ncorrect", "no_correct_answer": "no correct\noption", "revised_key": "revised\nkey",
             "removed_question": "removed\nitem", "wrong_answer": "wrong\nkey", "multiple_correct_only": "several\nendings"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [5, 2, 2]}, sharey=True)
    for ax, b in zip(axes, ("mmlu", "gsm8k", "hellaswag")):
        x = d[d.bench == b]
        types = list(dict.fromkeys(x.sort_values("lift", ascending=False).type))
        w = 0.26
        for k, m in enumerate(("crh", "ds_onecoin_lo", "irt4pl")):
            v = [x[(x.type == t) & (x.method == m)].lift.mean() for t in types]
            ax.bar(np.arange(len(types)) + (k - 1) * w, v, width=w - 0.03, color=COL[m], label=LAB[m])
        ax.axhline(1, color=MUTED, lw=0.7, ls="--")
        ax.set_yscale("log"); ax.set_xticks(range(len(types)))
        ax.set_xticklabels([f"{names[t]}\n(n={int(x[x.type == t].n_pos.iloc[0])})" for t in types])
        ax.set_title(BLAB[b], color=INK); style(ax)
    axes[0].set_ylabel("lift over chance (AP / prevalence)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(RES, "fig_errortypes.pdf")); fig.savefig(os.path.join(RES, "fig_errortypes.png"), dpi=200)


def sources_table():
    """Key-error AP and AP on the cross-source panels for every method (prob methods from stored scores)."""
    from depstudy import methods as Mt
    from s10_size_capability import key_mask
    import s6_sources as S6
    rows = []
    for b in ("mmlu", "gsm8k"):
        t = S6.load(b)[0]
        y = t.defective.values.astype(int)
        kpos, clean = key_mask(b, t); keep = kpos | clean
        for f in glob.glob(os.path.join(RES, "ckpt_sources", f"{b}__*.pkl")):
            for r in pickle.load(open(f, "rb")):
                if r["method"] not in ("mv", "crh", "irt4pl"):
                    continue
                s = np.frombuffer(r["scores"], np.float32).astype(np.float64)
                rows.append({"bench": b, "condition": r["condition"], "rep": r["rep"], "method": r["method"], "n_models": r["n_models"],
                             "ap": Mt.average_precision(s, y), "ap_key": Mt.average_precision(s[keep], kpos[keep].astype(int))})
    d = pd.DataFrame(rows)
    l = pd.read_parquet(os.path.join(RES, "logodds_scores.parquet")); l = l[l.family == "sources"]
    c = pd.read_parquet(os.path.join(RES, "crowdfm_scores.parquet")); c = c[c.kind == "source"]
    cols = ["bench", "condition", "rep", "method", "n_models", "ap", "ap_key"]
    d = pd.concat([d[cols], l[cols], c[cols]])
    d.to_parquet(os.path.join(RES, "sources_all.parquet"))
    return d


def sources(d):
    conds = ["helm_all", "v1_random_k", "v1_strong_k", "helm_plus_v1_strong100", "helm_plus_v1_all", "v1_all"]
    clab = ["HELM\nall", "v1\nrandom\n(same n)", "v1\nstrongest\n(same n)", "HELM +\n100 v1", "HELM +\nall v1", "v1\nall"]
    ms = ["crh", "ds_onecoin_lo", "irt4pl", "crowdfm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=False)
    for ax, b in zip(axes, ("mmlu", "gsm8k")):
        x = d[d.bench == b]
        w = 0.2
        for k, m in enumerate(ms):
            v = [x[(x.condition == c) & (x.method == m)].ap_key.mean() for c in conds]
            ax.bar(np.arange(len(conds)) + (k - 1.5) * w, v, width=w - 0.03, color=COL[m], label=LAB[m])
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(clab)
        ax.set_title(BLAB[b], color=INK); style(ax)
    axes[0].set_ylabel("key-error AP")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(RES, "fig_sources.pdf")); fig.savefig(os.path.join(RES, "fig_sources.png"), dpi=200)


if __name__ == "__main__":
    sizecap(); cells(); errortypes()
    d = sources_table(); sources(d)
    pd.set_option("display.width", 220)
    print(d.groupby(["bench", "condition", "method"])[["ap", "ap_key"]].mean().unstack("method").round(3).to_string())
