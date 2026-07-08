"""Router: language, file resolution, and lane selection.

Rules-first (cheap, deterministic); resolves evidence by filename/folder + content
retrieval (never trusts a single cited path — the real test won't cite sources).
Degrades gracefully on unknown formats.
"""
import re
from pathlib import Path

from rapidfuzz import fuzz

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
AUD_EXT = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"}
TAB_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".sql"}

_VI = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộ"
          "ờớởỡợùúủũụừứửữựỳýỷỹỵ")
_NUM_INTENT = re.compile(
    r"correlation|average|\bmean\b|how many|total number|number of|significant|"
    r"bao nhiêu|trung bình|đếm|\bcount\b|\bsum\b|\bmax\b|\bmin\b|median|std|"
    r"highest|lowest|most|nhiều nhất|lớn nhất|cao nhất", re.I)
_COUNT_INTENT = re.compile(r"how many|number of|bao nhiêu|count|đếm", re.I)


def detect_language(q: str) -> str:
    if re.search(r"[一-鿿]", q):
        return "zh"
    if any(c in _VI for c in q.lower()):
        return "vi"
    return "en"


def _norm(s: str) -> str:
    return re.sub(r"[-_]", " ", str(s).lower())


def resolve_files(question, manifest, retriever, topn=4):
    ql = _norm(question)
    scored = []
    for f in manifest:
        p = Path(f)
        s = max(fuzz.partial_ratio(_norm(p.stem), ql), fuzz.token_set_ratio(_norm(p.stem), ql))
        for part in p.parts[:-1]:                       # folder-name mention
            if len(part) > 3 and _norm(part) in ql:
                s += 45
        scored.append((s, f))
    scored.sort(reverse=True)
    fname = [f for s, f in scored if s >= 72][:topn]
    ret = []
    if retriever:
        ret = list(dict.fromkeys(h["source_relative_path"] for h in retriever.search(question, k=8)))[:topn]
    return list(dict.fromkeys(fname + ret)), fname


def _folder_mentioned(question, manifest):
    ql = _norm(question)
    dirs = {str(Path(f).parent) for f in manifest if str(Path(f).parent) != "."}
    for d in dirs:
        if len(Path(d).name) > 3 and _norm(Path(d).name) in ql:
            return d
    return None


def route(question, answer_type, manifest, retriever) -> dict:
    lang = detect_language(question)
    q = question or ""
    files, fname = resolve_files(question, manifest, retriever, topn=6)
    exts = [Path(f).suffix.lower() for f in files[:6]]
    top_ext = exts[0] if exts else ""
    folder = _folder_mentioned(question, manifest)

    if folder and _COUNT_INTENT.search(q) and re.search(r"image|images|ảnh|hình|picture", q, re.I):
        lane = "perceive_count_folder"
    elif top_ext in AUD_EXT or re.search(r"\baudio\b|\.m4a|recording|meeting summary|workshop.*audio", q, re.I):
        lane = "perceive_audio"
    elif top_ext in IMG_EXT:
        lane = "perceive_image"
    elif any(e in TAB_EXT for e in exts) and _NUM_INTENT.search(q):
        lane = "compute"
    elif not files:
        lane = "abstain"
    else:
        lane = "extract"

    return {"language": lang, "lane": lane, "files": files, "folder": folder, "fname_hits": fname}
