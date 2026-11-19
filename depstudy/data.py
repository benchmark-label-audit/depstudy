"""Assemble item x model answer matrices from the per-model prediction files (vectorised, cached)."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from depstudy.hf import DATA_ROOT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(DATA_ROOT, "preds")
CACHE = os.path.join(DATA_ROOT, "matrix_cache")


def fetch_records() -> dict:
    """Latest successful fetch record per model across fetch_log*.jsonl (single run or shards)."""
    import glob
    import json
    recs = {}
    for f in sorted(glob.glob(os.path.join(DATA_ROOT, "fetch_log*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                recs[r["model_dir"]] = r
    return recs


def truth(bench: str) -> pd.DataFrame:
    t = pd.read_parquet(os.path.join(ROOT, "data", f"truth_{bench}.parquet"))
    t = t[~t.excluded & ~t.ambiguous_key]
    return t.reset_index(drop=True)


def _read_one(m: str, bench: str):
    p = os.path.join(PRED, m + ".parquet")
    if not os.path.exists(p):
        return None
    try:
        import pyarrow.parquet as pq
        d = pq.read_table(p, columns=["bench", "item_id", "answer"]).to_pandas()
    except Exception:
        return None
    d = d[d.bench == bench][["item_id", "answer"]]
    d["model"] = m
    return d


def long_table(models: list[str], bench: str) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as ex:
        parts = [d for d in ex.map(lambda m: _read_one(m, bench), models) if d is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["item_id", "answer", "model"])


HS_PRED = os.path.join(DATA_ROOT, "preds_hellaswag")
HELM = os.path.join(DATA_ROOT, "helm")
TRUTH_OF = {"mmlu": "mmlu", "gsm8k": "gsm8k", "hellaswag": "hellaswag", "helm_mmlu": "mmlu", "helm_gsm8k": "gsm8k",
            "gsm8kflex": "gsm8k"}
N_OPTIONS = {"mmlu": 4, "hellaswag": 4, "helm_mmlu": 4}


def n_options(bench: str):
    """Fixed number of answer options for multiple-choice benchmarks; None for free-form answers."""
    return N_OPTIONS.get(bench)


KIND = {"mmlu": "mc4", "hellaswag": "mc4", "helm_mmlu": "mc4", "gsm8k": "num", "helm_gsm8k": "num", "gsm8kflex": "num"}


def hellaswag_long(models: list[str]) -> pd.DataFrame:
    """v1 HellaSwag answers (acc_norm rule) mapped to validation items through the canonical row permutation.
    Models whose row order differs from the canonical one are skipped."""
    t = pd.read_parquet(os.path.join(ROOT, "data", "truth_hellaswag.parquet"))
    perm = np.load(os.path.join(ROOT, "data", "hellaswag_perm.npy"))
    L = t[["len0", "len1", "len2", "len3"]].values.astype(np.float64)[perm]
    item_ids = t.item_id.values[perm]
    canon = pd.read_parquet(os.path.join(HS_PRED, "stabilityai__stablelm-tuned-alpha-7b.parquet"), columns=["key"]).key.values

    def one(m):
        p = os.path.join(HS_PRED, m + ".parquet")
        if not os.path.exists(p):
            return None
        try:
            d = pd.read_parquet(p)
        except Exception:
            return None
        if len(d) != len(canon) or not (d.key.values == canon).all():
            return None
        P = d[["p0", "p1", "p2", "p3"]].values.astype(np.float64)
        ok = np.isfinite(P).all(1)
        ans = np.where(ok, np.argmax(np.where(np.isfinite(P), P, -1e30) / L, 1), -1)
        out = pd.DataFrame({"item_id": item_ids, "answer": [("ABCD"[a] if a >= 0 else None) for a in ans]})
        out["model"] = m
        return out
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as ex:
        parts = [d for d in ex.map(one, models) if d is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["item_id", "answer", "model"])


GSM_RAW = os.path.join(DATA_ROOT, "preds_gsm8k_raw")


def gsm8k_flex_long(models: list[str]) -> pd.DataFrame:
    """v1 GSM8K answers under the lenient 'flexible-extract' rule (last number in the generation), from the raw
    generations refetched by s27."""
    def one(m):
        p = os.path.join(GSM_RAW, m + ".parquet")
        if not os.path.exists(p):
            return None
        d = pd.read_parquet(p, columns=["item_id", "flexible"]).rename(columns={"flexible": "answer"})
        d["model"] = m
        return d
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as ex:
        parts = [d for d in ex.map(one, models) if d is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["item_id", "answer", "model"])


def helm_long(bench: str) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(HELM, f"{bench}.parquet"), columns=["model", "item_id", "answer"])
    d["model"] = "helm:" + d.model
    return d


def helm_models(bench: str) -> list[str]:
    return sorted(helm_long(bench).model.unique())


def _build(models: list[str], bench: str, min_coverage: float):
    t = truth(TRUTH_OF[bench])
    if bench == "hellaswag":
        lt = hellaswag_long(models)
    elif bench == "gsm8kflex":
        lt = gsm8k_flex_long(models)
    elif bench.startswith("helm_"):
        lt = helm_long(bench)
        lt = lt[lt.model.isin(set(models))]
    else:
        lt = long_table(models, bench)
    if bench == "hellaswag":
        t["gold"] = t.gold.map(lambda k: "ABCD"[int(k)])
    return assemble(t, lt, models, KIND[bench], min_coverage)


def assemble(t: pd.DataFrame, lt: pd.DataFrame, models: list[str], kind: str, min_coverage: float):
    """Item x model matrices from a long table (item_id, answer, model) and a truth table; kind 'mc4' or 'num'.
    Used for single sources and for panels that combine sources (answers are recoded jointly)."""
    lt = lt[lt.item_id.isin(set(t.item_id))].copy()
    have = set(lt.model.unique())
    present_models = [m for m in models if m in have]
    item_pos = pd.Series(np.arange(len(t)), index=t.item_id)
    model_pos = pd.Series(np.arange(len(present_models)), index=present_models)
    lt["i"] = item_pos.reindex(lt.item_id).values
    lt["j"] = model_pos.reindex(lt.model).values
    n_i, n_m = len(t), len(present_models)
    X = np.full((n_i, n_m), -1, dtype=np.int32)
    M = np.zeros((n_i, n_m), dtype=bool)
    M[lt.i.values, lt.j.values] = True          # row present = attempted
    if kind == "mc4":
        g = t.gold.map({"A": 0, "B": 1, "C": 2, "D": 3}).values.astype(np.int32)
        code = lt.answer.map({"A": 0, "B": 1, "C": 2, "D": 3})
        ok = code.notna().values
        X[lt.i.values[ok], lt.j.values[ok]] = code.values[ok].astype(np.int32)
    else:
        g = np.zeros(n_i, dtype=np.int32)
        gold = pd.Series(t.gold.values, index=np.arange(n_i))
        a = lt[lt.answer.notna()][["i", "j", "answer"]]
        is_gold = a.answer.values == gold.reindex(a.i).values
        other = a[~is_gold]
        # per item: gold = 0, other distinct values numbered 1.. in order of first appearance
        uniq = other.drop_duplicates(["i", "answer"]).copy()
        uniq["code"] = uniq.groupby("i").cumcount() + 1
        other = other.merge(uniq[["i", "answer", "code"]], on=["i", "answer"], how="left")
        X[a.i.values[is_gold], a.j.values[is_gold]] = 0
        X[other.i.values, other.j.values] = other.code.values.astype(np.int32)
    Y = (X == g[:, None]).astype(np.float32)
    if kind == "mc4":
        M = X >= 0
    cov = M.mean(1)
    keep = cov >= min_coverage
    return t[keep].reset_index(drop=True), present_models, X[keep], g[keep], Y[keep], M[keep]


def matrices(models: list[str], bench: str, min_coverage: float = 0.8):
    """Returns (items DataFrame, models list, X codes, gold codes, Y correctness, M observed-for-IRT).

    MMLU: codes 0..3. GSM8K: per-item value ids with gold = 0; unparseable answers are missing (-1) in X but
    observed and incorrect in (Y, M). Results are cached in $DEPSTUDY_DATA/matrix_cache, keyed by the model list."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(("|".join(sorted(models)) + bench + str(min_coverage)).encode()).hexdigest()[:16]
    p = os.path.join(CACHE, f"{bench}_{key}.npz")
    tp = os.path.join(CACHE, f"{bench}_{key}_items.parquet")
    if os.path.exists(p) and os.path.exists(tp):
        z = np.load(p, allow_pickle=True)
        return pd.read_parquet(tp), list(z["models"]), z["X"], z["g"], z["Y"], z["M"]
    t, mods, X, g, Y, M = _build(sorted(models), bench, min_coverage)
    np.savez(p, models=np.array(mods, dtype=object), X=X, g=g, Y=Y, M=M)
    t.to_parquet(tp)
    return t, mods, X, g, Y, M
