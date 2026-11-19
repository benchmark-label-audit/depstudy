"""Independent re-annotation of a stratified HellaSwag sample by a second model family.

The HellaSwag defect labels of Chizhov et al. (2025) come from claude-3-5-sonnet. A stratified sample (150 items
labelled wrong-key, 150 labelled several-defensible-endings only, 200 clean; seeded) is re-annotated with the same
protocol by gpt-6.1-sol: for each item, the most plausible ending and the endings that are equally good. The script
reports agreement (Cohen's kappa, positive agreement) on 'key is wrong' and 'several endings are defensible', and the
audit AP of each method on the sample against (a) the original labels and (b) the items on which both annotators
agree.

  python scripts/s38_hellaswag_relabel.py fetch|annotate|analyze
"""
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from depstudy import methods as Mt  # noqa: E402
from depstudy.hf import DATA_ROOT, get_bytes  # noqa: E402

OUT = os.path.join(DATA_ROOT, "hellaswag_relabel")
RES = os.path.join(ROOT, "results")
SEED, BATCH, MODEL = 20261123, 20, "gpt-6.1-sol"
URL = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
# Shell command that reads the prompt on stdin and prints the model reply; "{model}" is replaced by MODEL.
ANNOTATOR_CMD = os.environ.get("ANNOTATOR_CMD", "")
INSTR = """You are annotating items of the HellaSwag commonsense benchmark. Each item gives a context and four candidate
endings A-D. For every item, read the context and the endings and decide (1) which ending is the most plausible
continuation, and (2) which other endings are equally plausible, i.e. a careful annotator could not prefer the best
one over them. Judge the text only; the dataset's key is not shown. Return one JSON object per line and nothing else:
{"id": <id>, "best_option": "A"|"B"|"C"|"D", "equally_good_options": [letters other than best_option, possibly empty]}"""


def preprocess(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def fetch():
    os.makedirs(OUT, exist_ok=True)
    val = pd.read_parquet(io.BytesIO(get_bytes(URL)))
    assert len(val) == 10042, len(val)
    t = pd.read_parquet(os.path.join(ROOT, "data", "truth_hellaswag.parquet"))
    t = t[t.annotated].copy()
    w = t.wrong_answer.fillna(False).astype(bool); mc = t.multiple_correct.fillna(False).astype(bool)
    rng = np.random.default_rng(SEED)
    pick = np.concatenate([rng.choice(t.row[w].values, 150, replace=False), rng.choice(t.row[mc & ~w].values, 150, replace=False),
                           rng.choice(t.row[~w & ~mc].values, 200, replace=False)])
    rng.shuffle(pick)
    rows = []
    for i in pick:
        r = val.iloc[int(i)]
        ctx = preprocess(r.activity_label + ": " + r.ctx_a + " " + r.ctx_b.capitalize())
        rows.append({"row": int(i), "context": ctx, "endings": [preprocess(e) for e in r.endings], "gold": int(r.label)})
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, "sample.parquet"))
    print("sample", len(rows))


def call(batch_id, items):
    p = os.path.join(OUT, f"out_{batch_id:03d}.jsonl")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    body = INSTR + "\n\n" + "\n\n".join(
        f"id: {it['row']}\ncontext: {it['context']}\n" + "\n".join(f"{l}. {e}" for l, e in zip("ABCD", it["endings"])) for it in items)
    for attempt in range(4):
        if not ANNOTATOR_CMD:
            raise SystemExit("Set ANNOTATOR_CMD; the released annotations are in hellaswag_relabel/ (see README).")
        r = subprocess.run(ANNOTATOR_CMD.format(model=MODEL), shell=True,
                           input=body.encode("utf-8"), capture_output=True, timeout=900)
        txt = r.stdout.decode("utf-8", "replace")
        lines = [l for l in txt.splitlines() if l.strip().startswith("{")]
        if len(lines) >= len(items) * 0.9:
            open(p, "w", encoding="utf-8").write("\n".join(lines))
            return p
        time.sleep(10 * (attempt + 1))
    open(p + ".fail", "w", encoding="utf-8").write(txt + "\n---\n" + r.stderr.decode("utf-8", "replace")[-2000:])
    return None


def annotate():
    s = pd.read_parquet(os.path.join(OUT, "sample.parquet"))
    items = s.to_dict(orient="records")
    batches = [items[k:k + BATCH] for k in range(0, len(items), BATCH)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda b: call(*b), enumerate(batches)))
    print("batches", len(batches), "ok", sum(r is not None for r in res))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def analyze():
    import glob
    s = pd.read_parquet(os.path.join(OUT, "sample.parquet")).set_index("row")
    ann = {}
    for f in glob.glob(os.path.join(OUT, "out_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            try:
                x = json.loads(l)
                ann[int(x["id"])] = x
            except Exception:
                continue
    t = pd.read_parquet(os.path.join(ROOT, "data", "truth_hellaswag.parquet")).set_index("row")
    rows = []
    for i, r in s.iterrows():
        if i not in ann:
            continue
        x = ann[i]
        best = str(x.get("best_option", "")).strip()[:1]
        eq = [str(e).strip()[:1] for e in (x.get("equally_good_options") or [])]
        if best not in "ABCD" or not best:
            continue
        gold = "ABCD"[r.gold]
        rows.append({"row": i, "orig_wrong": bool(t.loc[i, "wrong_answer"]), "orig_multi": bool(t.loc[i, "multiple_correct"]),
                     "new_wrong": best != gold and gold not in eq, "new_multi": len(eq) > 0,
                     "new_best_is_key": best == gold})
    d = pd.DataFrame(rows)
    out = {"annotated": int(len(d)),
           "kappa_wrong": kappa(d.orig_wrong, d.new_wrong), "kappa_multi": kappa(d.orig_multi, d.new_multi),
           "agree_wrong": float((d.orig_wrong == d.new_wrong).mean()),
           "new_wrong_given_orig_wrong": float(d.new_wrong[d.orig_wrong].mean()),
           "new_wrong_given_orig_clean": float(d.new_wrong[~d.orig_wrong & ~d.orig_multi].mean()),
           "new_multi_given_orig_multi_only": float(d.new_multi[d.orig_multi & ~d.orig_wrong].mean()),
           "new_multi_given_orig_clean": float(d.new_multi[~d.orig_wrong & ~d.orig_multi].mean())}
    # audit AP on the sample: original labels vs items where both annotators agree on wrong-key status
    import s19_logodds as S19
    tt = S19.data("hellaswag")[0]
    pos = pd.Series(np.arange(len(tt)), index=tt.item_id)
    d["i"] = pos.reindex([f"hellaswag__{r:05d}" for r in d.row]).values
    d = d.dropna(subset=["i"]); d["i"] = d.i.astype(int)
    agree = d[d.orig_wrong == d.new_wrong]
    ap = {}
    for panel in ("full", "small_strong", "offbench_strong"):
        for m in ("mv", "crh", "ds_onecoin_lo", "glad_lo", "irt4pl", "entropy"):
            p = os.path.join(DATA_ROOT, "item_scores", f"hellaswag__single_{panel}__{m}.npy")
            if not os.path.exists(p):
                continue
            sc = np.load(p)
            ap[f"{panel}/{m}"] = {
                "orig": Mt.average_precision_expected(sc[d.i.values], d.orig_wrong.values.astype(int)),
                "new": Mt.average_precision_expected(sc[d.i.values], d.new_wrong.values.astype(int)),
                "agreed": Mt.average_precision_expected(sc[agree.i.values], agree.orig_wrong.values.astype(int)),
                "base_orig": float(d.orig_wrong.mean()), "base_agreed": float(agree.orig_wrong.mean())}
    out["ap_on_sample"] = ap
    out["n_agreed"] = int(len(agree)); out["n_agreed_wrong"] = int(agree.orig_wrong.sum())
    json.dump(out, open(os.path.join(RES, "hellaswag_relabel.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"fetch": fetch, "annotate": annotate, "analyze": analyze}[sys.argv[1]]()
