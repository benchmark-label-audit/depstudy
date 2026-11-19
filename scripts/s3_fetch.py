"""Step 3: stream per-sample predictions for MMLU (57 subjects, 5-shot) and GSM8K (5-shot).

Two download routes share the work, each kept under its own Hugging Face rate limit:
  * LFS route: one tree API call per run directory + one LFS batch call, then the files come from the CDN
    (counts ~2-3 calls against the 'api' quota of 500 / 5 min per model);
  * resolve route: one /resolve/ request per file (58 per model against the 'resolvers' quota of 3000 / 5 min).
Each parquet is parsed in memory into (bench, item_id, answer, lb_acc) rows matched to the truth tables and
discarded; raw files are never written to disk.

  python scripts/s3_fetch.py [--limit N] [--models file.txt] [--workers 12]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depstudy.hf import DATA_ROOT, UA, details_base, file_path_from_name, get_bytes  # noqa: E402
from depstudy.textnorm import gsm8k_pred, norm_key  # noqa: E402

OUT = os.path.join(DATA_ROOT, "preds")
LOG = os.path.join(DATA_ROOT, "fetch_log.jsonl")
REPO = "open-llm-leaderboard-old/details_{m}"


class Budget:
    """Sliding-window token bucket: at most `limit` calls in any `window` seconds."""

    def __init__(self, limit, window=300.0):
        self.limit, self.window, self.q, self.lock = limit, window, deque(), threading.Lock()

    def _trim(self, now):
        while self.q and now - self.q[0] > self.window:
            self.q.popleft()

    def try_take(self, k=1):
        with self.lock:
            now = time.time(); self._trim(now)
            if len(self.q) + k <= self.limit:
                self.q.extend([now] * k); return True
            return False

    def take(self, k=1):
        while not self.try_take(k):
            time.sleep(0.5)

    def room(self, k=1):
        with self.lock:
            self._trim(time.time())
            return len(self.q) + k <= self.limit


API = Budget(320)        # leave ~180 / 5 min of the 500 'api' quota for the lineage script
RES = Budget(2800)       # of the 3000 'resolvers' quota


def load_truth():
    tm = pd.read_parquet(os.path.join(ROOT, "data", "truth_mmlu.parquet"))
    tg = pd.read_parquet(os.path.join(ROOT, "data", "truth_gsm8k.parquet"))
    mm = {k: (i, g) for k, i, g, a in zip(tm.match_key, tm.item_id, tm.gold, tm.ambiguous_key) if isinstance(k, str) and not a}
    gg = {k: (i, g) for k, i, g, a in zip(tg.match_key, tg.item_id, tg.gold, tg.ambiguous_key) if not a}
    size = tm.groupby("subject").size().to_dict()
    return mm, gg, size


def _acc_values(df):
    if "metrics" in df.columns:
        return [bool(m.get("acc")) if isinstance(m, dict) and "acc" in m else None for m in df["metrics"]]
    if "acc" in df.columns:
        return [None if v is None else str(v).strip() in ("True", "1", "1.0", "true") for v in df["acc"]]
    return [None] * len(df)


def _as_list(preds):
    """Predictions are stored as a list/array in most files but as its string repr in some older ones
    (deviation D2: those were previously dropped)."""
    if preds is None:
        return []
    if isinstance(preds, str):
        import ast
        try:
            v = ast.literal_eval(preds)
            return list(v) if isinstance(v, (list, tuple)) else [v]
        except Exception:
            return [preds]
    return list(preds)


def parse_mmlu(df, subject, mm):
    col = "query" if "query" in df.columns else "example"
    rows = []
    for ex, preds, acc in zip(df[col], df["predictions"], _acc_values(df)):
        q = re.sub(r"\s*Answer:\s*$", "", str(ex))
        hit = mm.get(subject + "||" + norm_key(q))
        p = _as_list(preds)
        if hit is None or len(p) != 4:
            continue
        rows.append(("mmlu", hit[0], "ABCD"[int(np.argmax(p))], acc, hit[1]))
    return rows


def parse_gsm8k(df, gg):
    col = "example" if "example" in df.columns else "question"
    rows = []
    for ex, preds, acc in zip(df[col], df["predictions"], _acc_values(df)):
        hit = gg.get(norm_key(ex))
        if hit is None:
            continue
        pl = _as_list(preds)
        gen = pl[0] if pl else ""
        rows.append(("gsm8k", hit[0], gsm8k_pred(gen), acc, hit[1]))
    return rows


def _cols(names):
    text = "query" if "query" in names else ("example" if "example" in names else "question")
    return [c for c in (text, "predictions", "metrics", "acc") if c in names]


def _read(b):
    """Parse only the needed columns (converting the token-id arrays to pandas dominates CPU otherwise)."""
    import pyarrow.parquet as pq
    try:
        f = pq.ParquetFile(io.BytesIO(b))
        return f.read(columns=_cols(f.schema_arrow.names)).to_pandas()
    except Exception:
        return None


_tls = threading.local()


def _session():
    import requests
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.trust_env = False  # direct connection; the local proxy is ~30x slower for HF (see depstudy/hf.py)
        s.headers.update(UA)
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=3))
        _tls.s = s
    return s


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range on a keep-alive session; the tail (footer) is
    fetched once and cached so that the parquet metadata costs a single request."""

    TAIL = 16384

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        start = max(0, size - self.TAIL)
        self.tail_start, self.tail = start, self._get(start, size - 1)

    def _get(self, a, b):
        for k in range(4):
            try:
                r = _session().get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=90)
                if r.status_code in (200, 206):
                    return r.content if r.status_code == 206 else r.content[a:b + 1]
            except Exception:
                pass
            time.sleep(2 * (k + 1))
        raise IOError("range request failed")

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if self.pos >= self.tail_start:
            b = self.tail[self.pos - self.tail_start:self.pos - self.tail_start + n]
        else:
            b = self._get(self.pos, self.pos + n - 1)
        self.pos += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf)); buf[:len(b)] = b; return len(b)


SMALL = 300_000


def _range(url, a, b):
    for k in range(4):
        try:
            r = _session().get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=90)
            if r.status_code == 206:
                return r.content
            if r.status_code == 200:
                return r.content[a:b + 1]
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    raise IOError("range request failed")


def projected_read(url, size, gap=65536):
    """Fetch the footer, compute the byte ranges of the needed column chunks from the parquet metadata,
    merge nearby ranges, fetch them in a few requests into a sparse in-memory copy, and parse that copy.
    (pyarrow's own remote reads issue hundreds of tiny requests and cost seconds of CPU per file.)"""
    import pyarrow.parquet as pq
    tail = _range(url, max(0, size - 16384), size - 1)
    flen = int.from_bytes(tail[-8:-4], "little")
    if flen + 8 > len(tail):
        tail = _range(url, max(0, size - flen - 8), size - 1)
    buf = bytearray(size)
    buf[size - len(tail):] = tail
    meta = pq.ParquetFile(io.BytesIO(bytes(buf))).metadata
    names = pq.ParquetFile(io.BytesIO(bytes(buf))).schema_arrow.names
    cols = _cols(names)
    spans = []
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            if cc.path_in_schema.split(".")[0] not in cols:
                continue
            start = cc.dictionary_page_offset if cc.has_dictionary_page and cc.dictionary_page_offset else cc.data_page_offset
            spans.append((start, start + cc.total_compressed_size))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        buf[a:b] = _range(url, a, b - 1)
    return pq.ParquetFile(io.BytesIO(bytes(buf))).read(columns=cols).to_pandas()


def cdn_table(url, size):
    """Read only the needed columns: small files in one GET, larger ones by column-projected Range reads."""
    try:
        if size <= SMALL:
            r = _session().get(url, timeout=90)
            if r.status_code != 200:
                return None
            return _read(r.content)
        return projected_read(url, size)
    except Exception:
        return None


def resolve_table(model_dir, candidates, min_rows, skip_latest=False):
    """Newest file with >= min_rows rows via /resolve/ (older runs tried when the newest is a smoke test)."""
    cands = list(reversed(candidates))
    if skip_latest:
        cands = cands[1:]
    for entry in cands:
        path = file_path_from_name(entry.split("\t", 1)[1])
        if path is None:
            continue
        RES.take(1)
        b = get_bytes(details_base(model_dir) + urllib.parse.quote(path))
        df = None if b is None else _read(b)
        if df is not None and len(df) >= min_rows:
            return df
    return None


def resolve_projected(model_dir, wanted):
    """Resolve route without downloading through /resolve/: take the 302 Location (signed CDN URL) and the
    linked size, then read only the needed columns from the CDN. One 'resolvers' call per file."""
    def one(item):
        k, p = item
        RES.take(1)
        try:
            r = _session().get(details_base(model_dir) + urllib.parse.quote(p), allow_redirects=False, timeout=60)
        except Exception:
            return k, None
        loc = r.headers.get("Location")
        size = r.headers.get("X-Linked-Size") or r.headers.get("x-linked-size")
        if r.status_code not in (301, 302, 307) or not loc or not size:
            return k, None
        if loc.startswith("/"):
            loc = "https://huggingface.co" + loc
        return k, cdn_table(loc, int(size))
    with ThreadPoolExecutor(max_workers=6) as ex:
        return {k: df for k, df in ex.map(one, wanted.items()) if df is not None}


def lfs_tables(model_dir, wanted):
    """wanted: key -> latest path. Returns key -> DataFrame (missing keys omitted). Uses tree + LFS batch + CDN."""
    repo = REPO.format(m=model_dir)
    by_dir = {}
    for k, p in wanted.items():
        by_dir.setdefault(p.split("/", 1)[0], []).append(k)
    oid = {}
    for d in by_dir:
        API.take(1)
        tree = get_bytes(f"https://huggingface.co/api/datasets/{repo}/tree/main/{urllib.parse.quote(d)}")
        if tree is None:
            continue
        for e in json.loads(tree):
            if e.get("lfs"):
                oid[e["path"]] = (e["lfs"]["oid"], e["lfs"]["size"])
    objs = {oid[p][0]: oid[p][1] for p in wanted.values() if p in oid}
    if not objs:
        return {}
    API.take(1)
    body = json.dumps({"operation": "download", "transfers": ["basic"],
                       "objects": [{"oid": o, "size": s} for o, s in objs.items()]}).encode()
    try:
        r = _session().post(f"https://huggingface.co/datasets/{repo}.git/info/lfs/objects/batch", data=body, timeout=60,
                            headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
        if r.status_code != 200:
            return {}
        batch = r.json()
    except Exception:
        return {}
    href = {o["oid"]: o.get("actions", {}).get("download", {}).get("href") for o in batch.get("objects", [])}
    def dl(item):
        k, p = item
        if p not in oid or not href.get(oid[p][0]):
            return k, None
        return k, cdn_table(href[oid[p][0]], oid[p][1])
    with ThreadPoolExecutor(max_workers=6) as ex:
        return {k: df for k, df in ex.map(dl, wanted.items()) if df is not None}


def fetch_model(model_dir, files, mm, gg, size):
    dst = os.path.join(OUT, model_dir + ".parquet")
    if os.path.exists(dst):
        return {"model_dir": model_dir, "status": "cached"}
    t0 = time.time()
    try:
        minrows = {k: (1300 if k == "gsm8k" else int(0.95 * size[k.split(":", 1)[1]])) for k in files}
        latest = {k: file_path_from_name(c[-1].split("\t", 1)[1]) for k, c in files.items() if c}
        route = None
        while route is None:  # prefer the cheap LFS route; use /resolve/ when the API quota is busy
            if API.room(3):
                route = "lfs"
            elif RES.room(len(files)):
                route = "resolve"
            else:
                time.sleep(1.0)
        want = {k: p for k, p in latest.items() if p}
        tables = lfs_tables(model_dir, want) if route == "lfs" else resolve_projected(model_dir, want)
        rows, missing = [], []
        for k, cand in files.items():
            df = tables.get(k)
            if df is not None and len(df) < minrows[k]:
                df = resolve_table(model_dir, cand, minrows[k], skip_latest=True)
            elif df is None:
                df = resolve_table(model_dir, cand, minrows[k])
            if df is None:
                missing.append(k); continue
            rows += parse_gsm8k(df, gg) if k == "gsm8k" else parse_mmlu(df, k.split(":", 1)[1], mm)
        rec = {"model_dir": model_dir, "route": route, "seconds": round(time.time() - t0, 1), "missing": missing}
        if not rows:
            rec["status"] = "empty"; return rec
        out = pd.DataFrame(rows, columns=["bench", "item_id", "answer", "lb_acc", "gold"]).drop_duplicates(["bench", "item_id"])
        chk = out.dropna(subset=["lb_acc"])
        agree = (chk.lb_acc.astype(bool) == (chk.answer == chk.gold)).mean() if len(chk) else float("nan")
        rec.update(status="ok", n_mmlu=int((out.bench == "mmlu").sum()), n_gsm8k=int((out.bench == "gsm8k").sum()),
                   lb_consistency=round(float(agree), 4))
        out.drop(columns=["gold"]).to_parquet(dst)
        return rec
    except Exception as e:
        return {"model_dir": model_dir, "status": "error", "error": repr(e)[:300], "trace": traceback.format_exc()[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--logname", default="")
    a = ap.parse_args()
    global API, RES, LOG
    API = Budget(320 // a.nshards)   # the quotas are per IP, so shards split them
    RES = Budget(2800 // a.nshards)
    if a.nshards > 1:
        LOG = os.path.join(DATA_ROOT, f"fetch_log_shard{a.shard}.jsonl")
    if a.logname:  # later-sorting log name so that refetch records supersede earlier ones
        LOG = os.path.join(DATA_ROOT, a.logname)
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_parquet(os.path.join(ROOT, "data", "manifest.parquet"))
    man = man[man.readme.fillna(False) & man.n_mmlu_subjects.fillna(0).eq(57) & man.has_gsm8k.fillna(False)]
    man = man.sample(frac=1.0, random_state=20261115)  # random fetch order
    if a.models:
        want = set(l.strip() for l in open(a.models, encoding="utf-8") if l.strip())
        man = man[man.model_dir.isin(want)]
    if a.limit:
        man = man.head(a.limit)
    man = man.iloc[a.shard::a.nshards]
    mm, gg, size = load_truth()
    jobs = [(r.model_dir, json.loads(r.files)) for r in man.itertuples()]
    print("models", len(jobs), flush=True)
    t0 = time.time()
    lock = threading.Lock()
    def run(j):
        rec = fetch_model(j[0], j[1], mm, gg, size)
        with lock, open(LOG, "a", encoding="utf-8") as log:
            log.write(json.dumps(rec) + "\n")
        return rec
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(run, jobs)):
            if i % 25 == 0:
                print(i, round(time.time() - t0), {k: rec.get(k) for k in ("model_dir", "status", "route", "seconds", "lb_consistency")}, flush=True)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
