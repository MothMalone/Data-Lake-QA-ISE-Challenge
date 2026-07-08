"""SOTA orchestration: self-consistency over diverse tool-agent rollouts.

Per question we run N independent agent rollouts (varied temperature / model), then
aggregate answer-type-aware:
  - exact_match -> normalized majority vote (kills variance like Q4's '-3', Q13's '7.5')
  - llm_judge   -> an LLM selects the best-grounded candidate; abstain if the majority abstain

This is the proven self-consistency + best-of-N-with-judge recipe, which cost lets us run freely.
"""
import re
from collections import Counter

from . import clients, config
from .agent import solve_agent
from .solve.formatter import ABSTAIN, squad_normalize


def _rollout_configs(n):
    base = config.MODELS["agent"]
    pool = [(base, 0.0), (base, 0.5), (base, 0.7), (base, 0.3), (base, 0.6)]
    return (pool * ((n // len(pool)) + 1))[:n]


def _is_abstain(a):
    return (not a) or squad_normalize(str(a)) == squad_normalize(ABSTAIN)


def solve_ensemble(question, answer_type, asr=None, n=3, verbose=False):
    cands = []
    for model, temp in _rollout_configs(n):
        try:
            ans, ev = solve_agent(question, answer_type, asr=asr, model=model, temperature=temp)
        except Exception as e:
            if verbose:
                print("   rollout error:", str(e)[:100])
            ans, ev = ABSTAIN, []
        cands.append({"answer": ans, "ev": ev})
        if verbose:
            print(f"   rollout(t={temp}) -> {str(ans)[:70]!r}")

    if answer_type == "exact_match":
        return _vote_exact(cands)
    return _select_llm(question, cands)


def _vote_exact(cands):
    norms = [squad_normalize(c["answer"]) for c in cands]
    win, _ = Counter(norms).most_common(1)[0]
    for c, nz in zip(cands, norms):
        if nz == win:
            return c["answer"], c["ev"]
    return cands[0]["answer"], cands[0]["ev"]


def _select_llm(question, cands):
    if sum(1 for c in cands if _is_abstain(c["answer"])) > len(cands) / 2:
        return ABSTAIN, []
    real = [c for c in cands if not _is_abstain(c["answer"])]
    if not real:
        return ABSTAIN, []
    if len(real) == 1:
        return real[0]["answer"], real[0]["ev"]
    listing = "\n\n".join(f"[{i}] {c['answer']}" for i, c in enumerate(real))
    prompt = (f"QUESTION:\n{question}\n\nCandidate answers:\n{listing}\n\n"
              "Choose the SINGLE best candidate — most accurate and complete, grounded in the data, "
              "and in the same language as the question. Reply with ONLY its index number.")
    try:
        out = clients.chat([{"role": "user", "content": prompt}], role="verify", max_tokens=5, temperature=0)
    except Exception:
        out = "0"
    m = re.search(r"\d+", out or "")
    idx = max(0, min(int(m.group()) if m else 0, len(real) - 1))
    return real[idx]["answer"], real[idx]["ev"]
