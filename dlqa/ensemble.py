"""Stabilization layer over the agent — kills output variance without adding exploration noise.

  exact_match -> run N times at temp 0 and MAJORITY-VOTE the normalized answer (fixes glitches
                 like an occasional empty count or a mis-rounded average).
  llm_judge   -> single run + enforce the answer is in the question's language (a scoring rule).

Deliberately NOT the diverse-temperature ensemble (that injected errors); temp stays 0 so the
votes agree except on genuine glitches, which the majority absorbs.
"""
from collections import Counter

from . import clients
from .agent import solve_agent
from .solve.formatter import squad_normalize
from .solve.router import detect_language

_LANG = {"vi": "Vietnamese", "zh": "Chinese", "en": "English"}


def _enforce_language(question, answer):
    if not answer or "not enough data" in answer.lower():
        return answer
    ql = detect_language(question)
    if detect_language(answer) == ql:
        return answer
    try:
        return clients.chat([{"role": "user", "content":
            f"Rewrite this answer in {_LANG.get(ql, 'the question language')}, preserving the meaning "
            f"and any proper nouns/numbers exactly, with no preamble:\n{answer}"}],
            role="synth", temperature=0, max_tokens=400).strip()
    except Exception:
        return answer


def solve_ensemble(question, answer_type, asr=None, retriever=None, n=1):
    if answer_type == "exact_match":
        cands = [solve_agent(question, answer_type, asr=asr, retriever=retriever, temperature=0.0)
                 for _ in range(n)]
        norms = [squad_normalize(a) for a, _ in cands]
        win, _ = Counter(norms).most_common(1)[0]
        for (a, e), nz in zip(cands, norms):
            if nz == win:
                return a, e
        return cands[0]

    ans, ev = solve_agent(question, answer_type, asr=asr, retriever=retriever, temperature=0.0)
    return _enforce_language(question, ans), ev
