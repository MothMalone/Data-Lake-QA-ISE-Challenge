"""Conservative abstention gate — catches false-premise / ungrounded answers (e.g. Q9).

Returns False (abstain) ONLY when the answer clearly isn't supported by the lake's
evidence — a false/unverifiable premise, missing required data, or a world-knowledge
guess. Defaults to KEEP so it doesn't nuke good answers.
"""
from .. import clients

_SYS = (
    "You audit whether a proposed answer is grounded in the given evidence from a data lake. "
    "Reply with ONE word: KEEP or ABSTAIN.\n"
    "Reply ABSTAIN only if ANY holds: (a) the question rests on a false or unverifiable premise; "
    "(b) the evidence files cannot actually determine the answer (they lack the required "
    "data/attribute); (c) the answer clearly comes from outside world knowledge rather than the "
    "evidence. Otherwise reply KEEP. When unsure, reply KEEP."
)


def verify(question, answer, evidence_files, evidence_hits=None, lane=None) -> bool:
    ev = ", ".join(evidence_files) if evidence_files else "(none)"
    snippets = ""
    if evidence_hits:
        snippets = "\n".join((h.get("text", "") or "")[:300] for h in evidence_hits[:3])
    u = (f"QUESTION: {question}\nPROPOSED ANSWER: {answer}\nEVIDENCE FILES: {ev}\n"
         f"EVIDENCE SNIPPETS:\n{snippets or '(none)'}\n\nKEEP or ABSTAIN?")
    try:
        out = clients.chat([{"role": "system", "content": _SYS}, {"role": "user", "content": u}],
                           role="verify", max_tokens=3, temperature=0)
        return not out.strip().upper().startswith("ABSTAIN")
    except Exception:
        return True
