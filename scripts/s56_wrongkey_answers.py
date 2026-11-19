"""Answers of the strong MMLU panel on the wrong-key items that MMLU-Redux corrects (Section 5.3): share of models
that give the expert answer, share that give the shipped key, and share of items whose most frequent non-key answer
is the expert answer. Writes results/wrongkey_answers.json."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import s19_logodds as S19  # noqa: E402
import s33_rankshift as S33  # noqa: E402

RES = os.path.join(ROOT, "results")
LUT = {"A": 0, "B": 1, "C": 2, "D": 3, "0": 0, "1": 1, "2": 2, "3": 3}


def main():
    t, mods, X, g, Y, M = S19.data("mmlu")
    red = pd.read_parquet(os.path.join(S33.PREV, "mmlu_redux_all.parquet"))
    red["k"] = red.groupby("subject").cumcount()
    red["item_id"] = red.subject + "__" + red.k.map(lambda x: f"{x:03d}")
    ca = red.set_index("item_id").correct_answer.reindex(t.item_id)
    nk = np.array([LUT.get(str(v).strip(), -1) if isinstance(v, (str, int, np.integer)) else -1 for v in ca.values])
    P = json.load(open(os.path.join(RES, "panels.json")))["benches"]["mmlu"]["singles"]["small_strong"]["models"]
    idx = {m: i for i, m in enumerate(mods)}
    Xs = X[:, np.array([idx[m] for m in P])]
    wk = np.flatnonzero((t.error_type.values == "wrong_groundtruth") & (nk >= 0) & (nk != g))
    expert = [float((Xs[i] == nk[i]).mean()) for i in wk]
    key = [float((Xs[i] == g[i]).mean()) for i in wk]
    modal_is_expert = []
    for i in wk:
        v = Xs[i][(Xs[i] >= 0) & (Xs[i] != g[i])]
        modal_is_expert.append(bool(len(v)) and int(np.bincount(v, minlength=4).argmax()) == int(nk[i]))
    out = {"n_items": int(len(wk)), "share_models_expert_answer": float(np.mean(expert)),
           "share_models_key": float(np.mean(key)), "share_items_modal_nonkey_is_expert": float(np.mean(modal_is_expert))}
    print(out)
    json.dump(out, open(os.path.join(RES, "wrongkey_answers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
