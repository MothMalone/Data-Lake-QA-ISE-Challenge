"""solve(question) -> (answer, evidences): route -> lane -> format.

Retriever + manifest + optional asr callable are injected so the same code runs locally
(BM25, no audio) and on Kaggle (bge-m3 hybrid + faster-whisper).
"""
import re
from pathlib import Path

from .. import config
from .formatter import format_answer
from .lane_compute import solve_compute
from .lane_extract import solve_extract, synthesize
from .lane_perceive import digit_profile_folder, vlm_answer
from .router import route, IMG_EXT, TAB_EXT


def _abs(f):
    return config.DATA_LAKE_ROOT / f


def _perceive_count(question, folder):
    prof = digit_profile_folder(config.DATA_LAKE_ROOT / folder)
    if re.search(r"blue|xanh", question, re.I):
        val = sum(1 for r in prof if r["blue"])
    else:
        val = sum(1 for r in prof if r["num_digits"] == 1)
    return {"value": val, "evidences": [folder]}


def solve(question, answer_type, manifest, retriever, asr=None) -> tuple:
    plan = route(question, answer_type, manifest, retriever)
    lane, lang = plan["lane"], plan["language"]

    if lane == "compute":
        tabs = [_abs(f) for f in plan["files"] if Path(f).suffix.lower() in TAB_EXT][:4]
        res = solve_compute(question, tabs) if tabs else {"value": None, "evidences": []}

    elif lane == "perceive_count_folder":
        res = _perceive_count(question, plan["folder"])

    elif lane == "perceive_image":
        imgs = [f for f in plan["files"] if Path(f).suffix.lower() in IMG_EXT][:8]
        ans = vlm_answer(question, [_abs(f) for f in imgs]) if imgs else None
        res = {"value": ans, "evidences": imgs}

    elif lane == "perceive_audio":
        if asr and plan["files"]:
            transcript = asr(_abs(plan["files"][0]))
            ctx = [{"source_relative_path": plan["files"][0], "text": transcript}]
            ans = synthesize(question, ctx, lang)
            res = {"value": None if ans.strip().upper().startswith("NOT_ENOUGH") else ans,
                   "evidences": plan["files"][:1]}
        else:
            res = {"value": None, "evidences": []}   # no local ASR -> abstain

    elif lane == "extract":
        res = solve_extract(question, retriever, lang)

    else:  # abstain
        res = {"value": None, "evidences": []}

    answer = format_answer(question, res.get("value"), answer_type)
    return answer, res.get("evidences", []), plan
