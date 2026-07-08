"""Formatter — innovation #1: obey the answer-format contract embedded in the question.

The organizers' generator reuses a small set of scaffolds ("Yêu cầu [chỉ] trả về…",
"rounded to N decimal places", inline MCQ options), so parsing the directive out of the
question and applying it deterministically generalizes to the unseen 100. Then apply
exact_match normalization (SQuAD-style, with a CJK carve-out).
"""
import re
import string

_PUNCT = str.maketrans("", "", string.punctuation + "，。、；：！？（）")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def has_cjk(s) -> bool:
    return any("一" <= c <= "鿿" for c in str(s))


def squad_normalize(s, strip_articles: bool = True) -> str:
    """SQuAD normalize_answer, but don't strip English articles from CJK text."""
    s = str(s).lower().strip().translate(_PUNCT)
    if strip_articles and not has_cjk(s):
        s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _mcq_options(q: str) -> dict:
    opts = {}
    for m in re.finditer(r"(?:^|[\s(（])([A-E])[.)．]\s*([^\nA-E]*?)(?=(?:\s+[A-E][.)．])|\n|$)", q):
        t = m.group(2).strip()
        if t:
            opts[m.group(1).upper()] = t
    return opts


def _wants_mcq(q: str) -> bool:
    return bool(re.search(r"lựa chọn|chọn .*đáp án|đáp án đúng|choose|which of the", q, re.I))


def _round_nd(q: str):
    m = (re.search(r"round(?:ed)?\s+to\s+(\w+)\s+decimal", q, re.I)
         or re.search(r"làm tròn.*?(\d+)\s*chữ số", q, re.I))
    if not m:
        return None
    w = m.group(1).lower()
    return int(w) if w.isdigit() else _WORDNUM.get(w)


def _map_option(value, opts: dict):
    v = str(value).strip()
    m = _NUM.search(v)
    if m:
        fv = float(m.group(0))
        for L, t in opts.items():
            tn = _NUM.search(t)
            if tn and abs(float(tn.group(0)) - fv) < 1e-6:
                return L
    vn = squad_normalize(v)
    for L, t in opts.items():
        tn = squad_normalize(t)
        if tn and (tn == vn or vn in tn or tn in vn):
            return L
    return None


ABSTAIN = "Not enough data to answer."


def format_answer(question: str, value, answer_type: str = "exact_match") -> str:
    if value is None:
        return ABSTAIN
    q = question or ""

    # MCQ -> single letter
    opts = _mcq_options(q)
    if opts and _wants_mcq(q):
        L = _map_option(value, opts)
        if L:
            return L

    # explicit rounding directive
    nd = _round_nd(q)
    if nd is not None:
        m = _NUM.search(str(value))
        if m:
            return f"{float(m.group(0)):.{nd}f}"

    # count / "how many" questions -> extract the bare number from a possibly-verbose answer
    if answer_type == "exact_match" and re.search(
            r"how many|total number|number of|bao nhiêu|đếm|\bcount\b", q, re.I):
        nums = _NUM.findall(str(value))
        if nums:
            n = float(nums[-1])
            return str(int(n)) if n.is_integer() else nums[-1]

    s = str(value).strip()
    # uppercase directive (VI "chữ hoa" / EN "uppercase")
    if re.search(r"chữ hoa|in hoa|uppercase|viết hoa", q, re.I):
        s = s.upper()
    # bare integer: drop trailing .0
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s
