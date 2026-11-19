"""Exploratory: is the 'wrong key vs ill-posed item' split visible on all three v1 benchmarks?

Positives per error type are scored against clean items only; reported as AP and lift = AP / prevalence.
MMLU-Redux: wrong_groundtruth vs bad_question_clarity / multiple_correct_answers / no_correct_answer /
bad_options_clarity. GSM8K-Platinum: revised (key fixed, 10 items) vs removed (question dropped as ambiguous or
inconsistent, 110). HellaSwag (LLM-judged): wrong_answer vs multiple_correct only.
Panels: small_strong, small_diverse, five random panels; full population with the fast methods only.

  python scripts/s9_error_types.py --workers 6  ->  results/error_types.parquet + printed summary
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
NFALSE = {"mmlu": 3, "gsm8k": 10, "hellaswag": 3}
PANELS = {"mmlu": "panels.json", "gsm8k": "panels.json", "hellaswag": "panels_hellaswag.json"}
_D = {}


def _K(bench):
    from depstudy.data import n_options
    return n_options(bench)


def types_of(bench, t):
    if bench == "mmlu":
        et = t.error_type.values
        clean = et == "ok"
        return {k: et == k for k in ("wrong_groundtruth", "bad_question_clarity", "multiple_correct_answers",
                                     "no_correct_answer", "bad_options_clarity")}, clean
    if bench == "gsm8k":
        st = t.status.values
        return {"revised_key": st == "revised", "removed_question": st == "removed"}, np.isin(st, ["consensus", "verified"])
    w = t.wrong_answer.fillna(False).values.astype(bool)
    mc = t.multiple_correct.fillna(False).values.astype(bool)
    return {"wrong_answer": w, "multiple_correct_only": mc & ~w}, ~w & ~mc


def _init():
    from depstudy.data import matrices
    from s4_panels import eligible_models
    models = eligible_models(None)
    for b in ("mmlu", "gsm8k", "hellaswag"):
        _D[b] = matrices(models, b)


def job(spec):
    from depstudy import methods as Mt
    bench, panel, models, fast_only = spec
    t, mods, X, g, Y, M = _D[bench]
    idx = {m: i for i, m in enumerate(mods)}
    c = np.array([idx[m] for m in models if m in idx])
    Xp, Yp, Mp = X[:, c], Y[:, c], M[:, c]
    meths = [("mv", lambda: Mt.majority_vote(Xp, g)), ("crh", lambda: Mt.crh(Xp, g)[0])]
    if not fast_only:
        meths += [("ds_onecoin", lambda: Mt.dawid_skene_onecoin(Xp, g, K=_K(bench))[0]), ("irt4pl", lambda: Mt.irt_ceiling_lr(Xp, g, Yp, Mp)),
                  ("accu_nodep", lambda: Mt.accu_depen(Xp, g, NFALSE[bench], use_dependence=False, K=_K(bench))[0])]
    tys, clean = types_of(bench, t)
    rows = []
    for name, f in meths:
        s = f()
        for ty, pos in tys.items():
            keep = pos | clean
            y = pos[keep].astype(int)
            if y.sum() == 0:
                continue
            ap = Mt.average_precision(s[keep], y)
            rows.append({"bench": bench, "panel": panel, "n_models": len(c), "method": name, "type": ty,
                         "n_pos": int(y.sum()), "base": float(y.mean()), "ap": ap, "lift": ap / float(y.mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    specs = []
    for b in ("mmlu", "gsm8k", "hellaswag"):
        P = json.load(open(os.path.join(ROOT, "results", PANELS[b])))["benches"][b]
        specs.append((b, "small_strong", P["singles"]["small_strong"]["models"], False))
        specs.append((b, "small_diverse", P["singles"]["small_diverse"]["models"], False))
        for r, p in enumerate(P["cells"]["random"][:5]):
            specs.append((b, f"random_{r}", p["models"], False))
        specs.append((b, "full", P["singles"]["full"]["models"], True))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for r in ex.map(job, specs):
            rows += r
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "results", "error_types.parquet"))
    df["panel_group"] = df.panel.str.replace(r"random_\d", "random", regex=True)
    pd.set_option("display.width", 250)
    tab = df.groupby(["bench", "type", "method", "panel_group"]).lift.mean().unstack("panel_group")
    print(df.groupby(["bench", "type"]).agg(n_pos=("n_pos", "first"), base=("base", "first")).round(4).to_string())
    print(tab.round(1).to_string())


if __name__ == "__main__":
    main()
