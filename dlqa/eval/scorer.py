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
    sys = ("You grade a candidate answer against a reference answer for a QA task. Reply with ONE "
           "word: YES or NO.\n"
           "Say YES if the candidate conveys the same core facts/meaning as the reference — minor "
           "differences in wording, length, or extra detail are fine, as long as the key fact(s) "
           "match — AND it is written in the same language as the question.\n"
           "Say NO only if it contradicts the reference, misses or gets the key fact wrong, or is "
           "in a different language than the question.")
    u = f"QUESTION: {question}\n\nREFERENCE: {gold}\n\nCANDIDATE: {pred}\n\nSame core meaning and same language? YES or NO."
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
