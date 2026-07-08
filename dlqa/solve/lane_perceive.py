"""Perceive lane: VLM over images (+ ASR for audio).

count-over-folder (Q4/Q5): ask the VLM a strict-JSON micro-question PER image, then
aggregate deterministically in Python — never ask the VLM for the total.
single-image reason (Q6/Q8), multi-page synthesis (Q15). Audio ASR (Q14) is done in
the Kaggle notebook (faster-whisper); a hook is left here.
"""
import json
import re
from pathlib import Path

from .. import clients, config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def _vlm_json(prompt, images, **kw):
    txt = clients.vlm(prompt, images, **kw)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(config.DATA_LAKE_ROOT))
    except Exception:
        return str(p)


_DIGIT_PROMPT = (
    "Look at this image and answer with STRICT JSON only.\n"
    "Count how many numeric digit characters (0-9) are shown as the main displayed content. "
    "Count only actual digit glyphs — NOT letters and NOT spelled-out number words "
    "(e.g. the written word 'ONE' does not count as a digit). If the same single number is "
    "shown once, that is one digit; a strip like 0 1 2 3 4 5 6 7 8 9 is many digits.\n"
    "Also decide whether a shown digit is presented in BLUE: true if the digit glyph itself is "
    "blue/cyan, OR the digit sits on a predominantly blue badge/disc/background.\n"
    'Return exactly: {"num_digits": <int>, "digits": "<digits shown>", "blue": <true|false>}'
)


def digit_profile_folder(folder) -> list:
    """Per-image {file, num_digits, blue, digits} for every image in a folder."""
    folder = Path(folder)
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    out = []
    for p in imgs:
        r = _vlm_json(_DIGIT_PROMPT, p, temperature=0) or {}
        out.append({
            "file": _rel(p),
            "num_digits": r.get("num_digits"),
            "blue": bool(r.get("blue")),
            "digits": r.get("digits"),
        })
    return out


def count_exactly_one_digit(folder):
    prof = digit_profile_folder(folder)
    n = sum(1 for r in prof if r["num_digits"] == 1)
    return {"value": n, "evidences": [_rel(Path(folder))], "detail": prof}


def count_blue_digit(folder):
    prof = digit_profile_folder(folder)
    n = sum(1 for r in prof if r["blue"])
    return {"value": n, "evidences": [_rel(Path(folder))], "detail": prof}


def vlm_answer(question, images, json_schema_hint="") -> str:
    """Free-form single/multi-image VLM answer (Q6/Q8/Q15)."""
    prompt = question + (f"\n{json_schema_hint}" if json_schema_hint else "")
    return clients.vlm(prompt, images, temperature=0, max_tokens=800)
