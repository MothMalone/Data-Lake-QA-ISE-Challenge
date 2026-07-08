"""solve(question) -> (answer, evidences): route -> lane -> verify -> format.

Retriever + manifest + optional asr callable are injected so the same code runs locally
(BM25, no audio) and on Kaggle (bge-m3 hybrid + faster-whisper).
"""
import re
from pathlib import Path

from rapidfuzz import fuzz

from .. import config
from .formatter import format_answer
from .lane_compute import solve_compute
from .lane_extract import solve_extract, synthesize
from .lane_perceive import digit_profile_folder, vlm_answer
from .router import route, IMG_EXT, TAB_EXT, AUD_EXT
from .verifier import verify


def _abs(f):
    return config.DATA_LAKE_ROOT / f


def _perceive_count(question, folder):
    prof = digit_profile_folder(config.DATA_LAKE_ROOT / folder)
    if re.search(r"blue|xanh", question, re.I):
        val = sum(1 for r in prof if r["blue"])
    else:
        val = sum(1 for r in prof if r["num_digits"] == 1)
    return {"value": val, "evidences": [folder]}


def _best_audio(question, manifest):
    """Resolve the actual audio file (audio isn't in the text index, so match by name)."""
    auds = [f for f in manifest if Path(f).suffix.lower() in AUD_EXT]
    if not auds:
        return None
    ql = re.sub(r"[-_]", " ", question.lower())
    return max(auds, key=lambda f: fuzz.partial_ratio(re.sub(r"[-_]", " ", Path(f).stem.lower()), ql))


def solve(question, answer_type, manifest, retriever, asr=None):
    plan = route(question, answer_type, manifest, retriever)
    lane, lang = plan["lane"], plan["language"]
    hits = None

    if lane == "compute":
        tabs = [_abs(f) for f in plan["files"] if Path(f).suffix.lower() in TAB_EXT][:4]
        res = solve_compute(question, tabs) if tabs else {"value": None, "evidences": []}
        tr = res.get("transcripts") or []
        if tr:                                         # give the verifier the actual computation
            hits = [{"text": (str(tr[-1].get("code", "")) + "\n" + str(tr[-1].get("stdout", "")))[:900]}]

    elif lane == "perceive_count_folder":
        res = _perceive_count(question, plan["folder"])

    elif lane == "perceive_image":
        imgs = [f for f in plan["files"] if Path(f).suffix.lower() in IMG_EXT][:8]
        ans = vlm_answer(question, [_abs(f) for f in imgs]) if imgs else None
        res = {"value": ans, "evidences": imgs}

    elif lane == "perceive_audio":
        aud = _best_audio(question, manifest)
        transcript = asr(_abs(aud)) if (asr and aud) else ""
        if transcript.strip():
            ans = synthesize(question, [{"source_relative_path": aud, "text": transcript}], lang)
            res = {"value": None if ans.strip().upper().startswith("NOT_ENOUGH") else ans,
                   "evidences": [aud]}
        else:                                              # no transcript -> abstain, don't invent
            res = {"value": None, "evidences": []}

    elif lane == "extract":
        res = solve_extract(question, retriever, lang)
        hits = res.get("hits")

    else:
        res = {"value": None, "evidences": []}

    value = res.get("value")
    evidences = res.get("evidences", [])
    # conservative verification: abstain only if clearly unsupported / false-premise (Q9)
    if value is not None and lane != "perceive_count_folder":
        if not verify(question, value, evidences, hits, lane):
            value, evidences = None, []

    answer = format_answer(question, value, answer_type)
    return answer, evidences, plan
