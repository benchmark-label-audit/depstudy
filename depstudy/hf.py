"""Minimal Hugging Face access helpers (public data only, no token).

File downloads go through /resolve/ (generous rate limit); API calls through /api/ (tight limit for
anonymous clients), so callers should prefer resolve-based access and keep API calls to a minimum.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "research-script"}
# Answer logs and caches live outside the repository (see README, "Data").
DATA_ROOT = os.environ.get("DEPSTUDY_DATA",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw"))
# Downloads bypass any local HTTP proxy by default; set DEPSTUDY_USE_PROXY=1 to use the system proxy.
_OPENER = (urllib.request.build_opener() if os.environ.get("DEPSTUDY_USE_PROXY")
           else urllib.request.build_opener(urllib.request.ProxyHandler({})))


def get_bytes(url: str, tries: int = 6, timeout: int = 120) -> bytes | None:
    """GET with exponential backoff; honours HTTP 429 Retry-After. Returns None on 404."""
    delay = 2.0
    for _ in range(tries):
        try:
            with _OPENER.open(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):  # HF answers 401 for deleted/private repos to anonymous clients
                return None
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else max(delay, 30.0))
            else:
                time.sleep(delay)
        except Exception:
            time.sleep(delay)
        delay = min(delay * 2, 120.0)
    return None


def get_json(url: str, **kw):
    b = get_bytes(url, **kw)
    return None if b is None else json.loads(b.decode("utf-8"))


def ds_parquet_urls(dataset: str) -> list[dict]:
    """Parquet export of a public dataset via the datasets-server API."""
    d = get_json(f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(dataset)}")
    return [] if not d else d.get("parquet_files", [])


def details_base(model_dir: str) -> str:
    return f"https://huggingface.co/datasets/open-llm-leaderboard-old/details_{model_dir}/resolve/main/"


def readme_config_splits(readme: str) -> dict[str, list[str]]:
    """Map config_name -> list of data-file names (oldest..newest, 'latest' last) from a details README."""
    out: dict[str, list[str]] = {}
    for blk in re.split(r"\n- config_name: ", readme)[1:]:
        name = blk.split("\n", 1)[0].strip()
        files = re.findall(r"split: (\S+)\s+path:\s+- '\*\*/([^']+)'", blk)
        out[name] = [f"{s}\t{p}" for s, p in files]
    return out


def file_path_from_name(fname: str) -> str | None:
    """details_harness|task|k_<ts>.parquet lives in directory <ts>/."""
    m = re.search(r"_(\d{4}-\d\d-\d\dT[\d:\-\.]+)\.parquet$", fname)
    return None if not m else f"{m.group(1)}/{fname}"
