"""Local scorer — replicate the two grading modes for self-eval on the samples.

exact_match: SQuAD-style normalized equality, with numeric tolerance.
llm_judge:   an LLM decides same-idea AND same-language (mirrors the real rubric).
"""
import re

from ..solve.formatter import squad_normalize

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _num(s):
    m = _NUM.search(str(s).replace(",", ""))
    return float(m.group(0)) if m else None


def score_exact(pred, gold) -> float:
    if squad_normalize(pred) == squad_normalize(gold):
        return 1.0
    p, g = _num(pred), _num(gold)
    if p is not None and g is not None and abs(p - g) < 1e-6:
        return 1.0
    return 0.0


def score_judge(question, pred, gold, role: str = "verify") -> float:
    from .. import clients
    sys = ("You grade a QA answer. Given the QUESTION, a REFERENCE answer, and a CANDIDATE "
           "answer, reply with ONE word: YES if the candidate conveys the same core idea as "
           "the reference AND is written in the same language as the question; else NO.")
    u = f"QUESTION: {question}\nREFERENCE: {gold}\nCANDIDATE: {pred}\nYES or NO?"
    try:
        out = clients.chat([{"role": "system", "content": sys}, {"role": "user", "content": u}],
                           role=role, max_tokens=3, temperature=0)
        return 1.0 if out.strip().upper().startswith("Y") else 0.0
    except Exception:
        return 0.0


def score(question, pred, gold, answer_type) -> float:
    if answer_type == "exact_match":
        return score_exact(pred, gold)
    return score_judge(question, pred, gold)
