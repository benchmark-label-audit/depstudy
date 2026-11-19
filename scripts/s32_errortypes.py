"""Error-type analysis with prevalence-free metrics.

T1  For every benchmark x single panel x method x defect type: AUROC of the type against clean items, with a
    stratified item bootstrap (B = 1,000), and AP at matched prevalence: the positives of each type are subsampled
    to the size of the benchmark's rarest type (R = 200 draws), against all clean items, so that AP no longer
    depends on how frequent the type is. AP is the expected AP under random ordering of tied scores.
T2  Answer concentration per item, by type: the key's share of the answers, the share of the most frequent non-key
    answer, and the normalized entropy of the non-key answers, on the full population and the small-strong panel.

  python scripts/s32_errortypes.py  ->  results/errortypes_auroc.parquet, results/errortypes_concentration.parquet
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "item_scores")
RES = os.path.join(ROOT, "results")
PANELS = ["full", "corr_dedup", "small_strong", "small_diverse", "offbench_strong"]
METHODS = ["mv", "crh", "catd", "ds_onecoin_lo", "ds_onecoin_nf10_lo", "ds_full_lo", "accu_nodep_lo", "accu_depen_lo",
           "glad_lo", "irt4pl", "irt4pl_miss", "crowdfm", "entropy"]
B, R = 1000, 200


def auroc(s, pos, neg):
    """Mann-Whitney AUROC with ties counted one half."""
    x = np.concatenate([s[pos], s[neg]])
    r = pd.Series(x).rank(method="average").values
    n1, n0 = pos.sum(), neg.sum()
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_boot(s, pos_idx, neg_idx, rng):
    out = np.empty(B)
    for b in range(B):
        p = rng.choice(pos_idx, len(pos_idx)); n = rng.choice(neg_idx, len(neg_idx))
        x = np.concatenate([s[p], s[n]])
        r = pd.Series(x).rank(method="average").values
        out[b] = (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
    return out


def matched_ap(s, pos_idx, neg_idx, k, rng):
    aps = []
    for _ in range(R):
        p = rng.choice(pos_idx, k, replace=False) if len(pos_idx) > k else pos_idx
        idx = np.concatenate([p, neg_idx])
        y = np.zeros(len(idx), int); y[:len(p)] = 1
        aps.append(Mt.average_precision_expected(s[idx], y))
    return float(np.mean(aps))


def concentration(X, g):
    C = Mt.onehot_counts(X, g)
    tot = C.sum(1)
    r = np.arange(len(g))
    key_share = C[r, g] / np.maximum(tot, 1)
    Cn = C.copy(); Cn[r, g] = 0
    ntot = Cn.sum(1)
    top_nonkey = Cn.max(1) / np.maximum(tot, 1)
    P = Cn / np.maximum(ntot, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(P > 0, P * np.log(P), 0.0).sum(1)
    k = (Cn > 0).sum(1)
    Hn = np.where(k > 1, H / np.log(np.maximum(k, 2)), 0.0)
    top_of_nonkey = Cn.max(1) / np.maximum(ntot, 1)
    return key_share, top_nonkey, top_of_nonkey, Hn


def main():
    import json
    import zlib
    import s19_logodds as S19
    from s9_error_types import types_of
    import s10_size_capability as S10
    rng = np.random.default_rng(20261119)
    rows, crow = [], []
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]
    PH = json.load(open(os.path.join(RES, "panels_hellaswag.json")))["benches"]
    for bench in ("mmlu", "gsm8k", "hellaswag"):
        t, mods, X, g, Y, M = S19.data(bench)
        tys, clean = types_of(bench, t)
        neg_idx = np.flatnonzero(clean)
        kmin = min(int(v.sum()) for v in tys.values())
        for panel in PANELS:
            for m in METHODS:
                p = os.path.join(CACHE, f"{bench}__single_{panel}__{m}.npy")
                if not os.path.exists(p):
                    continue
                s = np.load(p)
                # log-odds scores are -inf on items without any non-key answer (bottom of the ranking) and +inf
                # where the key is impossible; map them past the finite range in the same direction
                fin = s[np.isfinite(s)]
                s = np.where(s == np.inf, fin.max() + 1, np.where(s == -np.inf, fin.min() - 1, s))
                for ty, pos in tys.items():
                    pos_idx = np.flatnonzero(pos)
                    # one stream per row, so the draws do not depend on which other caches exist
                    rng = np.random.default_rng([20261119, zlib.crc32(f"{bench}|{panel}|{m}|{ty}".encode())])
                    a = auroc(s, pos, clean)
                    bd = auroc_boot(s, pos_idx, neg_idx, rng)
                    kk = pos | clean
                    ap = Mt.average_precision_expected(s[kk], pos[kk].astype(int))
                    rows.append({"bench": bench, "panel": panel, "method": m, "type": ty, "n_pos": int(pos.sum()),
                                 "n_clean": int(clean.sum()), "auroc": a, "auroc_lo": float(np.percentile(bd, 2.5)),
                                 "auroc_hi": float(np.percentile(bd, 97.5)), "ap": ap, "base": float(pos.sum() / kk.sum()),
                                 "ap_matched": matched_ap(s, pos_idx, neg_idx, kmin, rng), "k_matched": kmin,
                                 "base_matched": kmin / (kmin + len(neg_idx))})
            print(bench, panel, flush=True)
        # concentration on the full population and the small-strong panel
        idx = {m: i for i, m in enumerate(mods)}
        PP = (PH if bench == "hellaswag" else P)[bench]["singles"]
        common, cap = S10.capability()
        panels = {"full": PP["full"]["models"], "small_strong": PP["small_strong"]["models"],
                  "offbench_strong": list(cap[bench].sort_values(ascending=False).index[:100])}
        for pname, ms in panels.items():
            c = np.array([idx[m] for m in ms if m in idx])
            ks, tn, ton, hn = concentration(X[:, c], g)
            groups = dict(tys); groups["clean"] = clean
            for ty, mask in groups.items():
                crow.append({"bench": bench, "panel": pname, "type": ty, "n": int(mask.sum()),
                             "key_share_mean": float(ks[mask].mean()), "key_share_median": float(np.median(ks[mask])),
                             "top_nonkey_share_mean": float(tn[mask].mean()), "top_nonkey_share_median": float(np.median(tn[mask])),
                             "top_of_nonkey_mean": float(ton[mask].mean()), "nonkey_entropy_norm_mean": float(hn[mask].mean())})
    d = pd.DataFrame(rows); c = pd.DataFrame(crow)
    d.to_parquet(os.path.join(RES, "errortypes_auroc.parquet"))
    c.to_parquet(os.path.join(RES, "errortypes_concentration.parquet"))
    pd.set_option("display.width", 250)
    for met in ("auroc", "ap_matched"):
        print(met)
        print(d[d.panel.isin(["full", "small_strong", "offbench_strong"])].pivot_table(
            index=["bench", "type"], columns=["panel", "method"], values=met).round(3).T.to_string())
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
