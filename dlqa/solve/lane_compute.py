"""Compute lane: code-as-action (Program-of-Thoughts).

schema-card -> LLM emits pandas/sqlite script -> sandbox exec -> error-feedback
repair loop -> self-consistency majority vote. Deterministic answers for the
numeric/tabular archetypes (Pearson, avg, count, group-by, superlatives).
"""
import json
import re
from collections import Counter

from .. import clients, sandbox
from ..schema_card import card_for

_SYS = (
    "You are a meticulous data analyst. Write ONE self-contained Python 3 script "
    "using pandas and the standard library (sqlite3 is available for .sql dumps). "
    "It must read the given file(s) from their ABSOLUTE paths and compute the answer "
    "to the QUESTION. The FILE(S) section shows RAW cells (header=None); decide whether "
    "a sheet actually has a header from the data and the README, and count list-style "
    "sheets with header=None so you do not lose the first row. Watch for traps: pick the "
    "exact sheet/column/table the question means (use the README to map terms to sheets). "
    'Print ONLY a single-line JSON object to stdout: '
    '{"value": <the answer>, "unit": <string or null>, "explanation": <one short sentence>}. '
    "No prose, no extra prints, no plots. Do not use the network, subprocess, or write files."
)


def _extract_code(txt: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
    code = m.group(1) if m else txt
    # drop any stray fence lines (unclosed block / multiple blocks) so ast.parse is clean
    code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("```"))
    return code.strip()


def _parse_json(stdout: str):
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    m = re.search(r"\{.*\}", stdout, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _norm(v):
    """Normalize a value for voting: numbers compared numerically, else lowercased str."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    s = str(v).strip()
    try:
        return round(float(s), 6)
    except Exception:
        return s.lower()


def solve_compute(question: str, files, k: int = 3, max_repairs: int = 2,
                  temperature: float = 0.4, verbose: bool = False) -> dict:
    """Return {value, evidences, votes}. value is None if all K samples failed."""
    cards = "\n\n".join(card_for(f) for f in files)
    user = (f"QUESTION:\n{question}\n\nFILE(S) (use these absolute paths):\n{cards}\n\n"
            "Return the script in a ```python code block.")
    results, transcripts = [], []
    for i in range(k):
        messages = [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
        value = None
        for attempt in range(max_repairs + 1):
            temp = 0.0 if (i == 0 and attempt == 0) else temperature
            txt = clients.chat(messages, role="coder", temperature=temp, max_tokens=2500)
            code = _extract_code(txt)
            try:
                res = sandbox.run(code)
            except ValueError as e:
                messages += [{"role": "assistant", "content": txt},
                             {"role": "user", "content": f"The sandbox rejected the code: {e}. Rewrite without it."}]
                continue
            if res["ok"]:
                parsed = _parse_json(res["stdout"])
                if parsed is not None and "value" in parsed:
                    value = parsed["value"]
                    transcripts.append({"code": code, "stdout": res["stdout"].strip()})
                    break
                messages += [{"role": "assistant", "content": txt},
                             {"role": "user", "content":
                              f"Your script printed:\n{res['stdout'][:500]}\n"
                              "It must print ONLY the JSON object with a 'value' key. Fix it."}]
            else:
                messages += [{"role": "assistant", "content": txt},
                             {"role": "user", "content":
                              f"The script failed:\n{res['stderr'][:800]}\nFix the error and return the corrected script."}]
        if value is not None:
            results.append(value)
        if verbose:
            print(f"  sample {i+1}: {value!r}")

    evidences = [str(f) for f in files]
    if not results:
        return {"value": None, "evidences": evidences, "votes": [], "transcripts": transcripts}
    norms = [_norm(v) for v in results]
    winner, _count = Counter(norms).most_common(1)[0]
    value = next(v for v, nz in zip(results, norms) if nz == winner)
    return {"value": value, "evidences": evidences, "votes": results,
            "agreement": _count / len(results), "transcripts": transcripts}
