"""Text keys and answer normalisation shared by truth building and prediction parsing."""
from __future__ import annotations

import math
import re


def norm_key(s: str) -> str:
    return re.sub(r"\W+", " ", str(s)).strip().lower()


def mmlu_key(subject: str, question: str, choices) -> str:
    return subject + "||" + norm_key(str(question) + " " + " ".join(f"{l}. {c}" for l, c in zip("ABCD", list(choices))))


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def canon_number(s: str | None) -> str | None:
    """Canonical string for a numeric answer: commas removed, trailing .0 dropped."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "")
    m = _NUM.search(s)
    if not m:
        return None
    t = m.group(0).replace(",", "")
    if len(t) > 30:  # degenerate digit runs from small models; keep as a literal value
        return t
    try:
        v = float(t)
    except ValueError:
        return None
    if not math.isfinite(v):
        return t
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def gsm8k_gold(answer_field: str) -> str | None:
    """Reference answer after '####' in the GSM8K answer field."""
    if "####" not in str(answer_field):
        return canon_number(answer_field)
    return canon_number(str(answer_field).split("####")[-1])


_HARNESS_ANS = re.compile(r"#### (\-?[0-9\.\,]+)")


def gsm8k_pred(generation: str) -> str | None:
    """Model answer as the v1 leaderboard harness extracts it: the first match of '#### <number>'; otherwise
    no answer (the harness scores it as invalid)."""
    m = _HARNESS_ANS.search(str(generation))
    if not m:
        return None
    return canon_number(m.group(1).strip().rstrip("."))
