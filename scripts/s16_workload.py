"""Workload characterization of the answer logs (Section 4 of the paper): model accuracy distributions (v1 and
HELM), item agreement with the key, residual dependence (within vs across declared families), lineage coverage,
and the share removed by correlation deduplication.

  python scripts/s16_workload.py  ->  results/workload.json, results/fig_workload.pdf
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy.data import helm_models, matrices  # noqa: E402
from depstudy.panels import residual_corr  # noqa: E402
from s4_panels import eligible_models  # noqa: E402


def q(x):
    x = np.asarray(x, float)
    return {k: round(float(np.percentile(x, p)), 4) for k, p in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))} | {"mean": round(float(x.mean()), 4)}


def main():
    models = eligible_models(None)
    lin = pd.read_parquet(os.path.join(ROOT, "data", "lineage.parquet")).set_index("model_dir")
    out = {"population": len(models)}
    accs, agree = {}, {}
    mats = {}
    for b in ("mmlu", "gsm8k", "hellaswag"):
        t, mods, X, g, Y, M = matrices(models, b)
        mats[b] = (t, mods, Y, M)
        acc = (Y * M).sum(0) / np.maximum(M.sum(0), 1)
        ag = (Y * M).sum(1) / np.maximum(M.sum(1), 1)
        accs[b], agree[b] = acc, ag
        # GSM8K: M marks attempted answers, so unparseable ones count against the key in ag; the paper's share uses
        # parseable answers only (on MMLU and HellaSwag the two masks coincide)
        P = M & (X >= 0)
        agp = (Y * P).sum(1) / np.maximum(P.sum(1), 1)
        y = t.defective.values.astype(bool)
        out[b] = {"items": len(t), "models": len(mods), "answers": int(M.sum()), "defective": int(y.sum()),
                  "model_accuracy": q(acc), "item_agreement_clean": q(ag[~y]), "item_agreement_defective": q(ag[y]),
                  "share_items_majority_disagrees": round(float((ag < 0.5).mean()), 4),
                  "share_items_majority_disagrees_parseable": round(float((agp < 0.5).mean()), 4)}
    for hb, b in (("helm_mmlu", "mmlu"), ("helm_gsm8k", "gsm8k")):
        t, mods, X, g, Y, M = matrices(helm_models(hb), hb, min_coverage=0.0)
        out[hb] = {"models": len(mods), "model_accuracy": q((Y * M).sum(0) / np.maximum(M.sum(0), 1))}
    # residual dependence within vs across declared families (MMLU residuals; same 2PL as the panels)
    t, mods, Y, M = mats["mmlu"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(mods), min(1200, len(mods)), replace=False)
    R = residual_corr(Y[:, sel], M[:, sel])
    fam = np.array([str(lin.family.get(mods[i])).lower() if isinstance(lin.family.get(mods[i]), str) and not str(lin.family.get(mods[i])).startswith("MULTI:") else None for i in sel], dtype=object)
    iu = np.triu_indices(len(sel), 1)
    same = np.array([fam[i] is not None and fam[i] == fam[j] for i, j in zip(*iu)])
    both = np.array([fam[i] is not None and fam[j] is not None for i, j in zip(*iu)])
    r = R[iu]
    out["residual_dependence_mmlu"] = {"pairs": int(len(r)), "all": q(r), "same_family": q(r[same]) if same.any() else None,
                                       "different_declared_families": q(r[both & ~same]), "n_same_family_pairs": int(same.sum()),
                                       "share_pairs_above_0.2": round(float((r > 0.2).mean()), 4)}
    declared = lin.family.notna() & ~lin.family.fillna("").str.startswith("MULTI:")
    out["lineage"] = {"models": int(len(lin)), "api_ok": int((lin.api_status == "ok").sum()), "single_root": int(declared.sum()),
                      "multi_root": int(lin.family.fillna("").str.startswith("MULTI:").sum()),
                      "families": int(lin[declared].family.str.lower().nunique()),
                      "families_ge5": int((lin[declared].family.str.lower().value_counts() >= 5).sum())}
    pan = json.load(open(os.path.join(ROOT, "results", "panels.json")))["benches"]
    panh = json.load(open(os.path.join(ROOT, "results", "panels_hellaswag.json")))["benches"]
    out["dedup_kept"] = {"mmlu": pan["mmlu"]["singles"]["corr_dedup"]["size"], "gsm8k": pan["gsm8k"]["singles"]["corr_dedup"]["size"],
                         "hellaswag": panh["hellaswag"]["singles"]["corr_dedup"]["size"], "of": len(models)}
    json.dump(out, open(os.path.join(ROOT, "results", "workload.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:3000])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, no Type 3 fonts
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.6))
        for b, c in (("mmlu", "#2a78d6"), ("gsm8k", "#eb6834"), ("hellaswag", "#1baf7a")):
            ax[0].hist(accs[b], bins=40, histtype="step", color=c, label=b)
        ax[0].set_xlabel("model accuracy"); ax[0].set_ylabel("models"); ax[0].legend(fontsize=7, frameon=False)
        t, _, _, _ = mats["mmlu"]; y = t.defective.values.astype(bool)
        ax[1].hist(agree["mmlu"][~y], bins=40, density=True, histtype="step", color="#888780", label="clean")
        ax[1].hist(agree["mmlu"][y], bins=40, density=True, histtype="step", color="#e34948", label="defective")
        ax[1].set_xlabel("share of models agreeing with key (MMLU)"); ax[1].set_ylabel("density of items"); ax[1].legend(fontsize=7, frameon=False)
        ax[2].hist(r[both & ~same], bins=60, density=True, histtype="step", color="#888780", label="different families")
        if same.any():
            ax[2].hist(r[same], bins=60, density=True, histtype="step", color="#6250d6", label="same family")
        ax[2].set_xlabel("residual correlation (MMLU)"); ax[2].set_ylabel("density of model pairs"); ax[2].legend(fontsize=7, frameon=False)
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "fig_workload.pdf")); fig.savefig(os.path.join(ROOT, "results", "fig_workload.png"), dpi=160)
        print("figure saved")
    except Exception as e:
        print("figure skipped:", repr(e))


if __name__ == "__main__":
    main()
