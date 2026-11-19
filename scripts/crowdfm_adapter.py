"""CrowdFM baseline adapter.

CrowdFM: Liu et al., "Towards a Foundation Model for Crowdsourced Label Aggregation", ICLR 2026;
code https://github.com/liiuhaao/CrowdFM, reviewed commit b08f4e70b258f332a4c8e11195a3082f390adc55,
unpacked at $DEPSTUDY_DATA/third_party/CrowdFM (override with env CROWDFM_DIR).

Interface (same as depstudy.methods): crowdfm_scores(X, g, K) -> 1 - posterior(gold), shape (n_items,).
X (n_items, n_models) integer codes, -1 = missing; g (n_items,) gold codes; K = number of options for
multiple-choice benchmarks (codes 0..K-1), None for free-form benchmarks (per-item value ids, gold = 0).
CrowdFM terms: worker = model (column), task = item (row), option = answer value.

How it runs
* Weights: checkpoint.pt shipped in the repo (plain state_dict, 168 float32 tensors, 831,873 parameters),
  SHA-256 pinned to the reviewed file and loaded with torch.load(weights_only=True). The scoring path imports
  no repo code.
* Forward pass: an operation-by-operation re-implementation of src/cfm/model/CFMEncoder.py and
  CFM.forward (dim 32, 10 layers, 4 heads, config/default.yaml). Edges are processed in chunks, so memory is
  about 120 B per answer plus one chunk, against about 4 KB per answer for the reference code; this is what
  makes 5,600 x 3,000 matrices fit on a 12 GB GPU. `--selftest --reference` compares the logits with the
  repo's own CFM class (needs torch-geometric).
* Option embeddings are N(0, I) draws (CrowdData.reset_parameters). evaluate.py draws them once per seed
  42..46; this adapter draws them identically and averages the posteriors over those seeds.
* Multiple choice (K given): option nodes 0..K-1 are shared across items, the encoding of the repo's own
  datasets. Free-form (K None): every (item, value) pair is its own option node and the readout is a softmax
  over the item's candidates 0..max(X[i], g[i]); per-item ids carry no meaning across items (value 0 is the
  gold by construction), so a shared node would tie together unrelated answers and carry the gold identity.
* worker_block (optional, off by default): CrowdFM was pretrained on simulated crowds of 20-100 workers with
  2-10 answers per task; worker_block=B splits the models into random disjoint groups of <= B, runs CrowdFM
  on each group and averages the gold posterior over groups. Use it as a sensitivity check for wide matrices.

CLI (run with python -I):
  python -I crowdfm_adapter.py --selftest [--reference] [--datasets Bird RTE ...]   # repo datasets, accuracy
  python -I crowdfm_adapter.py --bench 5600 3000 --K 4                              # synthetic time/memory
  python -I crowdfm_adapter.py --npz in.npz --out scores.npy                        # arrays X, g[, K]
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

CROWDFM_DIR = Path(os.environ.get("CROWDFM_DIR", os.path.join(
    os.environ.get("DEPSTUDY_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")),
    "third_party", "CrowdFM")))
CKPT_SHA256 = "9bdf79734d1a70017025a301227aa0cb33d14da5246c94547df1e7a4a08de75d"
DIM, LAYERS, HEADS = 32, 10, 4          # config/default.yaml -> model
SEEDS = (42, 43, 44, 45, 46)            # evaluate.py default seeds
CHUNK = 1 << 20                         # edges per chunk

_WEIGHTS: dict = {}


def _default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _expected_shapes() -> dict:
    D, H = DIM, HEADS
    s = {"encoder.x_worker": (D,), "encoder.x_task": (D,),
         "encoder.layer_norm.weight": (D,), "encoder.layer_norm.bias": (D,),
         "task_option_classifactor.0.weight": (2 * D, 2 * D), "task_option_classifactor.0.bias": (2 * D,),
         "task_option_classifactor.2.weight": (1, 2 * D), "task_option_classifactor.2.bias": (1,)}
    for side in ("worker", "task"):
        for ly in range(LAYERS):
            for p in "qkv":
                s[f"encoder.{p}_{side}.{ly}.weight"] = (H * D, 3 * D)
                s[f"encoder.{p}_{side}.{ly}.bias"] = (H * D,)
            s[f"encoder.out_{side}.{ly}.weight"] = (D, H * D)
            s[f"encoder.out_{side}.{ly}.bias"] = (D,)
    return s


def load_weights(device, ckpt: str | Path | None = None) -> dict:
    """SHA-256-checked, weights_only load of the pretrained state_dict; strict key/shape check."""
    import torch
    ckpt = Path(ckpt) if ckpt else CROWDFM_DIR / "checkpoint.pt"
    key = (str(ckpt), str(device))
    if key not in _WEIGHTS:
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if digest != CKPT_SHA256:
            raise RuntimeError(f"{ckpt}: sha256 {digest} is not the reviewed checkpoint {CKPT_SHA256}")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        exp, got = _expected_shapes(), {k: tuple(v.shape) for k, v in sd.items()}
        if got != exp:
            bad = sorted(k for k in set(exp) | set(got) if exp.get(k) != got.get(k))
            raise RuntimeError(f"checkpoint keys/shapes differ from CFM(dim={DIM}, layer={LAYERS}, head={HEADS}): {bad[:10]}")
        _WEIGHTS[key] = {k: v.to(device=device, dtype=torch.float32) for k, v in sd.items()}
    return _WEIGHTS[key]


def _seg_softmax(src, index, n):
    """torch_geometric.utils.softmax(src, index, dim=0, num_nodes=n) for 2-D src (same max shift, +1e-16)."""
    idx = index.unsqueeze(-1).expand_as(src)
    mx = src.new_zeros((n, src.shape[1])).scatter_reduce_(0, idx, src, reduce="amax", include_self=False)
    out = (src - mx.index_select(0, index)).exp_()
    tot = src.new_zeros((n, src.shape[1])).index_add_(0, index, out) + 1e-16
    return out.div_(tot.index_select(0, index))


def _encode(W, wid, tid, oid, n_w, n_t, zo, chunk=CHUNK):
    """CFMEncoder.forward, chunked over edges. Returns task embeddings z_t (n_t, DIM)."""
    import torch
    F = torch.nn.functional
    D, H, N = DIM, HEADS, wid.numel()
    ln = (W["encoder.layer_norm.weight"], W["encoder.layer_norm.bias"])
    zw = W["encoder.x_worker"].expand(n_w, D)
    zt = W["encoder.x_task"].expand(n_t, D)
    spans = [(a, min(a + chunk, N)) for a in range(0, N, chunk)]
    for ly in range(LAYERS):
        def P(name):
            return W[f"encoder.{name}.{ly}.weight"], W[f"encoder.{name}.{ly}.bias"]

        def feat(a, b):
            return torch.cat([zw.index_select(0, wid[a:b]), zt.index_select(0, tid[a:b]),
                              zo.index_select(0, oid[a:b])], dim=-1)

        sw = zo.new_empty((N, H))
        st = zo.new_empty((N, H))
        for a, b in spans:
            z = feat(a, b)
            sw[a:b] = (F.linear(z, *P("q_worker")).view(-1, H, D) * F.linear(z, *P("k_worker")).view(-1, H, D)).sum(-1) / math.sqrt(D)
            st[a:b] = (F.linear(z, *P("q_task")).view(-1, H, D) * F.linear(z, *P("k_task")).view(-1, H, D)).sum(-1) / math.sqrt(D)
        aw = _seg_softmax(sw, wid, n_w)
        del sw
        at = _seg_softmax(st, tid, n_t)
        del st
        agg_w = zo.new_zeros((n_w, D))
        agg_t = zo.new_zeros((n_t, D))
        for a, b in spans:
            z = feat(a, b)
            mw = (aw[a:b].unsqueeze(-1) * F.linear(z, *P("v_worker")).view(-1, H, D)).reshape(-1, H * D)
            agg_w.index_add_(0, wid[a:b], F.linear(mw, *P("out_worker")))
            mt = (at[a:b].unsqueeze(-1) * F.linear(z, *P("v_task")).view(-1, H, D)).reshape(-1, H * D)
            agg_t.index_add_(0, tid[a:b], F.linear(mt, *P("out_task")))
        del aw, at
        zw = F.layer_norm(zw + agg_w, (D,), *ln)
        zt = F.layer_norm(zt + agg_t, (D,), *ln)
    return zt


def _readout(W, zt, zo, cand_t, cand_o):
    """CFM.task_option_classifactor on (task, option) candidate pairs -> logits."""
    import torch
    F = torch.nn.functional
    h = torch.cat([zt.index_select(0, cand_t), zo.index_select(0, cand_o)], dim=-1)
    h = F.leaky_relu(F.linear(h, W["task_option_classifactor.0.weight"], W["task_option_classifactor.0.bias"]))
    return F.linear(h, W["task_option_classifactor.2.weight"], W["task_option_classifactor.2.bias"]).squeeze(-1)


def _option_x(n_opt, seed, device):
    """Same draw as evaluate.py: set_seed(seed) then CrowdData.reset_parameters -> torch.randn((n_opt, dim))."""
    import torch
    return torch.randn((n_opt, DIM), generator=torch.Generator().manual_seed(int(seed))).to(device)


def _run(graph, seeds, device, chunk, return_logits=False):
    """Per-seed candidate posteriors (softmax within task) as float64 numpy arrays."""
    import torch
    wid, tid, oid, n_w, n_t, n_opt, cand_t, cand_o = graph
    W = load_weights(device)
    lt = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=device)
    wid, tid, oid, ct, co = map(lt, (wid, tid, oid, cand_t, cand_o))
    out = []
    with torch.inference_mode():
        for s in seeds:
            zo = _option_x(n_opt, s, device)
            logit = _readout(W, _encode(W, wid, tid, oid, n_w, n_t, zo, chunk), zo, ct, co)
            if return_logits:
                out.append(logit.double().cpu().numpy())
            else:
                out.append(_seg_softmax(logit.unsqueeze(-1), ct, n_t).squeeze(-1).double().cpu().numpy())
    return out


def _graph_from_matrix(X, g, K):
    X = np.asarray(X)
    g = np.asarray(g).astype(np.int64)
    n, m = X.shape
    ti, wj = np.nonzero(X >= 0)
    v = X[ti, wj].astype(np.int64)
    if g.min() < 0:
        raise ValueError("gold codes must be >= 0")
    if K is not None:
        K = int(K)
        if (v.size and v.max() >= K) or g.max() >= K:
            raise ValueError(f"codes must lie in 0..{K - 1} when K={K}")
        n_opt, oid = K, v
        cand_t, cand_o = np.repeat(np.arange(n), K), np.tile(np.arange(K), n)
        gold = np.arange(n) * K + g
    else:
        # candidates per item: the values this panel gave plus gold. Free-form codes are numbered over the whole
        # population, so they are renumbered per item to drop values no panel source gave.
        V = int(max(X.max(), g.max())) + 1
        P = np.zeros((n, V), bool)
        P[ti, v] = True
        P[np.arange(n), g] = True
        rank = np.cumsum(P, axis=1) - 1
        c = P.sum(1)
        off = np.concatenate([[0], np.cumsum(c)[:-1]])
        n_opt = int(c.sum())
        oid = off[ti] + rank[ti, v]
        cand_t, cand_o = np.repeat(np.arange(n), c), np.arange(n_opt)
        gold = off + rank[np.arange(n), g]
    return (wj, ti, oid, m, n, n_opt, cand_t, cand_o), gold


def crowdfm_gold_posterior(X, g, K=None, seeds=SEEDS, device=None, chunk=CHUNK):
    """CrowdFM posterior of the gold value per item, averaged over option-embedding seeds."""
    device = device or _default_device()
    graph, gold = _graph_from_matrix(X, g, K)
    return np.mean([p[gold] for p in _run(graph, seeds, device, chunk)], axis=0)


def crowdfm_scores(X, g, K=None, seeds=SEEDS, device=None, chunk=CHUNK, worker_block=None, block_seed=0):
    """1 - CrowdFM posterior(gold): higher = shipped gold label more likely wrong."""
    X = np.asarray(X)
    n, m = X.shape
    if not worker_block or m <= worker_block:
        return 1.0 - crowdfm_gold_posterior(X, g, K, seeds, device, chunk)
    groups = np.array_split(np.random.default_rng(block_seed).permutation(m), math.ceil(m / worker_block))
    acc, cnt, acc_all = np.zeros(n), np.zeros(n), np.zeros(n)
    for cols in groups:
        Xs = X[:, np.sort(cols)]
        has = (Xs >= 0).any(axis=1)
        pg = crowdfm_gold_posterior(Xs, g, K, seeds, device, chunk)
        acc[has] += pg[has]
        cnt[has] += 1
        acc_all += pg
    return 1.0 - np.where(cnt > 0, acc / np.maximum(cnt, 1), acc_all / len(groups))



def _logodds_from_logits(logit, cand_t, gold, n):
    """Per item: log sum_{v != gold} exp(logit_v) - logit_gold, over the item's candidates (segment by cand_t)."""
    logit = np.asarray(logit, np.float64).reshape(-1)
    key = logit[gold]
    other = logit.copy()
    other[gold] = -np.inf
    m = np.full(n, -np.inf)
    np.maximum.at(m, cand_t, other)
    fin = np.isfinite(m)
    m0 = np.where(fin, m, 0.0)
    ssum = np.zeros(n)
    np.add.at(ssum, cand_t, np.exp(other - m0[cand_t]))
    lse = np.where(fin, m0 + np.log(np.maximum(ssum, 1e-300)), -np.inf)
    return lse - key


def crowdfm_both(X, g, K=None, seeds=SEEDS, device=None, chunk=CHUNK):
    """One forward pass per seed; returns (1 - mean posterior(gold), mean log posterior odds against gold)."""
    graph, gold = _graph_from_matrix(X, g, K)
    n = np.asarray(X).shape[0]
    cand_t = np.asarray(graph[6])
    probs, los = [], []
    for logit in _run(graph, seeds, device or _default_device(), chunk, return_logits=True):
        lo = _logodds_from_logits(logit, cand_t, gold, n)
        los.append(lo)
        probs.append(1.0 / (1.0 + np.exp(np.clip(lo, -700, 700))))  # posterior(gold) = 1 / (1 + odds)
    return 1.0 - np.mean(probs, axis=0), np.mean(los, axis=0)


def crowdfm_scores_both(X, g, K=None, seeds=SEEDS, device=None, chunk=CHUNK, worker_block=None, block_seed=0):
    """(probability score, log-odds score); with worker_block, both are averaged over the blocks that answer the item."""
    X = np.asarray(X)
    n, m = X.shape
    if not worker_block or m <= worker_block:
        return crowdfm_both(X, g, K, seeds, device, chunk)
    groups = np.array_split(np.random.default_rng(block_seed).permutation(m), math.ceil(m / worker_block))
    sp, sl, cnt, ap_, al_ = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for cols in groups:
        Xs = X[:, np.sort(cols)]
        has = (Xs >= 0).any(axis=1)
        p, l = crowdfm_both(Xs, g, K, seeds, device, chunk)
        sp[has] += p[has]; sl[has] += l[has]; cnt[has] += 1
        ap_ += p; al_ += l
    k = len(groups)
    return (np.where(cnt > 0, sp / np.maximum(cnt, 1), ap_ / k), np.where(cnt > 0, sl / np.maximum(cnt, 1), al_ / k))


# ------------------------------------------------------------------------------------------- checks
def _load_repo_dataset(d: Path):
    """Replicates src/cfm/data/load_data.py:load_single_data (first-appearance ids for workers/tasks/options)."""
    import pandas as pd
    lab, tru = pd.read_csv(d / "label.csv"), pd.read_csv(d / "truth.csv")
    w2, t2, o2 = {}, {}, {}
    wid = np.array([w2.setdefault(w, len(w2)) for w in lab["worker"]])
    tid = np.array([t2.setdefault(t, len(t2)) for t in lab["task"]])
    oid = np.array([o2.setdefault(o, len(o2)) for o in lab["answer"]])
    y = [-1] * len(t2)
    for t, tr in zip(tru["task"], tru["truth"]):
        o = o2.setdefault(tr, len(o2))
        if t not in t2:
            t2[t] = len(t2)
            y.append(o)
        else:
            y[t2[t]] = o
    n_t, K = len(t2), len(o2)
    graph = (wid, tid, oid, len(w2), n_t, K, np.repeat(np.arange(n_t), K), np.tile(np.arange(K), n_t))
    return graph, np.array(y)


def selftest(datasets=None, reference=False, device=None, chunk=CHUNK):
    import torch
    device = device or _default_device()
    root = CROWDFM_DIR / "data"
    names = datasets or sorted(p.name for p in root.iterdir() if p.is_dir())
    accs = []
    for name in names:
        graph, y = _load_repo_dataset(root / name)
        n_t, K = graph[4], graph[5]
        t0 = time.time()
        post = _run(graph, SEEDS, device, chunk)
        dt = time.time() - t0
        ok = y != -1
        acc = float(np.mean([(p.reshape(n_t, K).argmax(1)[ok] == y[ok]).mean() for p in post]))
        accs.append(acc)
        line = f"{name:8s} tasks={n_t:6d} workers={graph[3]:5d} K={K:2d} edges={len(graph[0]):7d} acc(mean of seeds)={acc:.4f} time={dt:.1f}s"
        if reference:
            line += f" max|logit-ref|={_reference_diff(graph, device):.2e}"
        print(line, flush=True)
    print(f"Average acc over {len(accs)} datasets: {np.mean(accs):.4f}")
    if str(device).startswith("cuda"):
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def _reference_diff(graph, device, seed=SEEDS[0]):
    """Max abs difference between this implementation's logits and the repo's CFM class (torch-geometric)."""
    import torch
    src = str(CROWDFM_DIR / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from cfm.data.crowd_data import CrowdData
    from cfm.model.CFM import CFM
    wid, tid, oid, n_w, n_t, n_opt, _, _ = graph
    model = CFM(dim=DIM, layer=LAYERS, head=HEADS, dropout=0.1, device=device).to(device)
    load_weights(device)  # sha check
    model.load_state_dict(torch.load(CROWDFM_DIR / "checkpoint.pt", map_location=device, weights_only=True), strict=True)
    model.eval()
    data = CrowdData(dim=DIM)
    data.triple = torch.as_tensor(np.stack([wid, oid, tid]), dtype=torch.long)
    data.num_worker, data.num_task, data.num_option = n_w, n_t, n_opt
    data.option_x = _option_x(n_opt, seed, "cpu")
    data.to(device)
    with torch.inference_mode():
        ref = model(data)["hat_task_option"].double().cpu().numpy().reshape(-1)
    ours = _run(graph, [seed], device, CHUNK, return_logits=True)[0]
    return float(np.abs(ours - ref).max())


def bench(n_items, n_models, K=4, device=None, chunk=CHUNK, seeds=(42,)):
    import torch
    device = device or _default_device()
    rng = np.random.default_rng(0)
    g = rng.integers(0, K, n_items)
    acc = rng.uniform(0.3, 0.9, n_models)
    X = np.where(rng.random((n_items, n_models)) < acc, g[:, None], rng.integers(0, K, (n_items, n_models)))
    t0 = time.time()
    s = crowdfm_scores(X, g, K, seeds=seeds, device=device, chunk=chunk)
    print(f"{n_items}x{n_models} K={K} seeds={len(seeds)} device={device}: {time.time() - t0:.1f}s, "
          f"mean posterior(gold)={1 - s.mean():.3f}")
    if str(device).startswith("cuda"):
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--bench", nargs=2, type=int, metavar=("N_ITEMS", "N_MODELS"))
    ap.add_argument("--npz")
    ap.add_argument("--out")
    ap.add_argument("--K", type=int)
    ap.add_argument("--device")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--worker-block", type=int)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.datasets, a.reference, a.device, a.chunk)
    elif a.bench:
        bench(*a.bench, K=a.K or 4, device=a.device, chunk=a.chunk)
    elif a.npz:
        z = np.load(a.npz)
        K = a.K if a.K is not None else (int(z["K"]) if "K" in z and int(z["K"]) > 0 else None)
        s = crowdfm_scores(z["X"], z["g"], K, device=a.device, chunk=a.chunk, worker_block=a.worker_block)
        np.save(a.out, s)
        print(f"wrote {a.out}: {s.shape}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
