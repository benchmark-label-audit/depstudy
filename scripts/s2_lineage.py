"""Step 2: declared lineage for the models in the manifest.

Sources (declared metadata only; nothing is inferred from responses here):
  1. Hugging Face model API: cardData.base_model and base_model:* tags; config architecture; parameter count.
  2. Open LLM Leaderboard v2 contents: Base Model, Merged, Type, Architecture, #Params.
The declared root of a model is found by following base_model links until a model declares no base.
Models that declare no base and are not themselves pretrained releases are 'undeclared'.

  python scripts/s2_lineage.py [--models file.txt]  ->  data/lineage.parquet, data/lineage_stats.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import DATA_ROOT, ds_parquet_urls, get_bytes  # noqa: E402

CACHE = os.path.join(DATA_ROOT, "model_api")
_lock = threading.Lock()
_last = [0.0]


def throttle(min_gap=0.62):
    with _lock:
        wait = _last[0] + min_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def model_info(repo_id: str) -> dict:
    p = os.path.join(CACHE, repo_id.replace("/", "__") + ".json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    throttle()
    b = get_bytes("https://huggingface.co/api/models/" + urllib.parse.quote(repo_id, safe="/"), tries=6)
    info = {"_status": "missing"} if b is None else {"_status": "ok", **json.loads(b.decode("utf-8"))}
    keep = {k: info.get(k) for k in ("_status", "id", "cardData", "tags", "config", "safetensors", "pipeline_tag")}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(keep, f)
    return keep


def declared_bases(info: dict) -> tuple[list[str], str]:
    """Base models declared by a model, and the relation (finetune/merge/adapter/quantized)."""
    bases, rel = [], ""
    for t in info.get("tags") or []:
        if isinstance(t, str) and t.startswith("base_model:") and t.count(":") >= 2:
            _, r, b = t.split(":", 2)
            if r in ("finetune", "merge", "adapter", "quantized"):
                bases.append(b); rel = rel or r
    cd = info.get("cardData") or {}
    bm = cd.get("base_model")
    if isinstance(bm, str):
        bases.append(bm)
    elif isinstance(bm, list):
        bases += [x for x in bm if isinstance(x, str)]
    seen, out = set(), []
    for b in bases:
        b = b.strip()
        if b and b not in seen and "/" in b:
            seen.add(b); out.append(b)
    return out, rel


def roots(repo_id: str, depth: int = 0, path: tuple = ()) -> set[str]:
    if depth > 6 or repo_id in path:
        return {repo_id}
    info = model_info(repo_id)
    bases, _ = declared_bases(info) if info.get("_status") == "ok" else ([], "")
    bases = [b for b in bases if b != repo_id]
    if not bases:
        return {repo_id}
    out = set()
    for b in bases:
        out |= roots(b, depth + 1, path + (repo_id,))
    return out


def v2_contents() -> pd.DataFrame:
    for f in ds_parquet_urls("open-llm-leaderboard/contents"):
        return pd.read_parquet(io.BytesIO(get_bytes(f["url"])))
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.models:
        dirs = [l.strip() for l in open(a.models, encoding="utf-8") if l.strip()]
    else:
        man = pd.read_parquet(os.path.join(ROOT, "data", "manifest.parquet"))
        man = man[man.readme.fillna(False) & man.n_mmlu_subjects.fillna(0).eq(57) & man.has_gsm8k.fillna(False)]
        dirs = list(man.model_dir)
    v2 = v2_contents()
    v2 = v2.set_index("fullname") if len(v2) else v2

    def one(d):
        repo = d.replace("__", "/", 1)
        info = model_info(repo)
        bases, rel = declared_bases(info) if info.get("_status") == "ok" else ([], "")
        rt = sorted(roots(repo)) if bases else []
        v2b = v2t = None; v2m = None
        if len(v2) and repo in v2.index:
            row = v2.loc[repo]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            v2b = row.get("Base Model"); v2t = row.get("Type"); v2m = bool(row.get("Merged"))
        cfg = info.get("config") or {}
        arch = (cfg.get("architectures") or [None])[0] if isinstance(cfg.get("architectures"), list) else None
        params = (info.get("safetensors") or {}).get("total")
        return {"model_dir": d, "repo": repo, "api_status": info.get("_status"), "bases": json.dumps(bases), "relation": rel,
                "roots": json.dumps(rt), "architecture": arch, "model_type": cfg.get("model_type"), "params": params,
                "v2_base": v2b, "v2_type": v2t, "v2_merged": v2m}

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(one, dirs)):
            rows.append(r)
            if i % 250 == 0:
                print(i, flush=True)
    df = pd.DataFrame(rows)
    # family = single declared root; v2 base used when the API declares none
    def fam(r):
        rt = json.loads(r.roots)
        if len(rt) == 1:
            return rt[0]
        if len(rt) > 1:
            return "MULTI:" + "|".join(rt)
        if isinstance(r.v2_base, str) and r.v2_base not in ("Removed", r.repo):
            return r.v2_base
        return None
    df["family"] = df.apply(fam, axis=1)
    # a model that declares no base is its own family root if another model declares it as a root,
    # or if the v2 leaderboard lists it as a pretrained release
    root_set = set()
    for rt in df.roots:
        root_set |= set(json.loads(rt))
    selfroot = df.family.isna() & (df.repo.isin(root_set) | df.v2_type.fillna("").str.contains("pretrained"))
    df.loc[selfroot, "family"] = df.loc[selfroot, "repo"]
    df["is_root"] = selfroot | df.repo.isin(root_set)
    df["declared"] = df.family.notna()
    df.to_parquet(os.path.join(ROOT, "data", "lineage.parquet"))
    fams = df[df.declared & ~df.family.str.startswith("MULTI:")].family.value_counts()
    stats = {"models": int(len(df)), "api_ok": int((df.api_status == "ok").sum()), "declared": int(df.declared.sum()),
             "multi_root": int(df.family.fillna("").str.startswith("MULTI:").sum()),
             "families_with_ge5": int((fams >= 5).sum()), "top_families": fams.head(20).to_dict()}
    json.dump(stats, open(os.path.join(ROOT, "data", "lineage_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
