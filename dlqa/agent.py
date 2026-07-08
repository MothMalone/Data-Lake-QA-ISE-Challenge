"""Tool-using agent: one strong model reasons over the data lake with tools.

Replaces the brittle router/resolution/verifier stack. The model itself decides which
file(s) are relevant, reads/inspects them, writes+runs code (including multi-file joins),
looks at images, transcribes audio, and returns a grounded answer + evidence.
"""
import json
from pathlib import Path

from . import clients, config, sandbox
from .solve.formatter import format_answer

TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".sql"}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in the data lake as relative paths. Optional case-insensitive substring filter.",
        "parameters": {"type": "object", "properties": {
            "contains": {"type": "string", "description": "only files whose path contains this"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file. Tables (csv/xlsx/sql) return schema + sheet names + a preview and the absolute path for code. For an xlsx, pass 'sheet' to dump that specific sheet's rows in full. Text/pdf/ppt/docx/html/md return extracted text.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "sheet": {"type": "string", "description": "for xlsx: name of one sheet to read in full"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python (pandas, numpy, sqlite3, math, json, re available). A variable DATA_LAKE holds the lake's absolute root path; build paths with os.path.join(DATA_LAKE, '<relative path>'). print() the result. Use this for ALL counting/averages/correlations/joins across files.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Semantic search over the lake's text content — returns the most relevant chunks with their source file. Use it to LOCATE where information lives (especially inside large documents, or across many files) before reading in full.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "count_matching_images",
        "description": "For 'how many images ...' questions: asks a YES/NO question of EACH image in a folder and returns the count of YES plus per-image results. Reliable — use this instead of eyeballing.",
        "parameters": {"type": "object", "properties": {
            "folder": {"type": "string"},
            "per_image_question": {"type": "string", "description": "a YES/NO question about ONE image, e.g. 'Does this image show exactly one digit (0-9)?'"}}, "required": ["folder", "per_image_question"]}}},
    {"type": "function", "function": {
        "name": "view_image",
        "description": "Look at ONE image and answer a question about it (OCR text, read a table, detect colors).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "question": {"type": "string"}}, "required": ["path", "question"]}}},
    {"type": "function", "function": {
        "name": "transcribe_audio",
        "description": "Transcribe an audio file (.m4a/.mp3/.wav) to text.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Submit the final answer and the exact evidence filenames used.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "evidences": {"type": "array", "items": {"type": "string"}}}, "required": ["answer", "evidences"]}}},
]

_SYS = """You answer ONE question about a heterogeneous, multilingual (English/Vietnamese/Chinese) data lake of mixed-format files. Work step by step with the tools.

Method:
1. list_files first. Reason about which file(s) truly match the question — by folder, filename, AND content. The lake contains DECOYS (similarly-named or same-topic files that are wrong); pick the file that genuinely answers the question, not a look-alike.
2. Inspect: use search(query) to LOCATE relevant content across the lake (especially a passage inside a large document); read_file for text/tables (it gives you the absolute path for code); view_image for one image; count_matching_images for "how many images ..." questions; transcribe_audio for audio.
3. If the question refers to SEVERAL items or documents (a comparison, or "the common point of X, Y and Z"), find and read ALL of them before answering — never answer from just one.
4. For ANY numeric/tabular answer — counts, averages, correlations, max/min, or joins across files — WRITE CODE with run_python and compute it EXACTLY (don't round early). Never eyeball numbers. To count over a set of images/files, inspect EACH one and tally in code. A variable DATA_LAKE holds the lake root; join multiple files in one script when needed.
5. Ground every claim in the files. If the lake genuinely lacks the data to answer, the answer is EXACTLY: Not enough data to answer. Never fall back on outside/world knowledge.
6. final_answer(answer, evidences) — evidences = the relative filenames you actually used.

Answer format:
- exact_match: obey any explicit instruction in the question ("return only the name in uppercase without the country", "rounded to two decimal places", multiple-choice A/B/C/D → return just the letter). Return the bare value, nothing else.
- llm_judge: give the concise factual answer/definition ITSELF, in the SAME LANGUAGE as the question, phrased as a reference answer key would (1-2 sentences, lead with the key fact). Do NOT narrate your process or describe what an image looks like — state the answer.
- "show me / where is / find me X" questions: answer by NAMING the file that contains X (e.g. 'Ảnh ... nằm trong file "X".'). Do not refuse or second-guess whether the content truly matches — if a file clearly holds it, state that file.

Be efficient (usually 3-8 tool calls) and always finish with final_answer."""


def _extract_exact(question, answer):
    """For exact_match: distil a verbose answer down to the bare value the question asks for."""
    prompt = (f"QUESTION: {question}\n\nAn answer was given:\n{answer}\n\nExtract ONLY the exact final "
              "value the question asks for, in the minimal form required — just the number, the single "
              "letter, or the single name — obeying any format instruction in the question (uppercase, "
              "'only the ordinal', etc.). Output only that value, nothing else.")
    try:
        return clients.chat([{"role": "user", "content": prompt}], role="synth",
                            max_tokens=30, temperature=0).strip()
    except Exception:
        return answer


def _manifest():
    return sorted(str(p.relative_to(config.DATA_LAKE_ROOT))
                  for p in config.DATA_LAKE_ROOT.rglob("*") if p.is_file())


def _resolve(path):
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    cand = config.DATA_LAKE_ROOT / path
    if cand.exists():
        return cand
    base = Path(path).name
    for f in config.DATA_LAKE_ROOT.rglob(base):
        return f
    return None


def _relpath(e):
    e = str(e)
    root = str(config.DATA_LAKE_ROOT)
    if e.startswith(root):
        return e[len(root):].lstrip("/")
    for marker in ("/Data-Lake/", "/data_lake/", "Data-Lake/", "data_lake/"):
        if marker in e:
            return e.split(marker, 1)[1]
    return e


def _tool_list_files(contains=None):
    m = _manifest()
    if contains:
        m = [f for f in m if str(contains).lower() in f.lower()]
    return "\n".join(m) if m else "(no matching files)"


def _tool_read_file(path, sheet=None):
    p = _resolve(path)
    if not p:
        return f"File not found: {path}"
    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"} and sheet:
        import pandas as pd
        try:
            df = pd.read_excel(p, sheet_name=sheet, header=None, nrows=80)
            return f"absolute_path = {p}\nsheet '{sheet}' (header=None, first 80 rows):\n{df.to_string(max_rows=80)}"
        except Exception as e:
            return f"could not read sheet '{sheet}' of {path}: {e}"
    if ext in TABLE_EXTS:
        from .schema_card import card_for
        return f"absolute_path = {p}\n" + card_for(p)
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}:
        return f"{p} is an image — use view_image to read it."
    if ext in {".m4a", ".mp3", ".wav", ".flac", ".ogg"}:
        return f"{p} is audio — use transcribe_audio."
    from .ingest import extract_units
    text = "\n\n".join(t for _, t in extract_units(p))
    return f"absolute_path = {p}\n{text[:7000]}"


def _tool_run_python(code):
    preamble = f'DATA_LAKE = r"{config.DATA_LAKE_ROOT}"\nimport os\n'
    try:
        res = sandbox.run(preamble + code, timeout=40)
    except ValueError as e:
        return f"Rejected by sandbox: {e}"
    if res["ok"]:
        return (res["stdout"][:3500] or "(ran, but printed nothing — remember to print the result)")
    return "ERROR:\n" + res["stderr"][:2000]


def _tool_view_image(path, question):
    p = _resolve(path)
    if not p:
        return f"Image not found: {path}"
    try:
        return clients.vlm(question, [p], temperature=0, max_tokens=900)
    except Exception as e:
        return f"view_image failed: {e}"


def _tool_search(query, retriever):
    if retriever is None:
        return "(search unavailable here — use list_files/read_file instead)"
    try:
        hits = retriever.search(query, k=6)
    except Exception as e:
        return f"search failed: {e}"
    if not hits:
        return "(no results)"
    return "\n\n".join(f"[{h['source_relative_path']}]\n{(h.get('text') or '')[:400]}" for h in hits)


def _tool_count_images(folder, per_image_question):
    p = config.DATA_LAKE_ROOT / folder
    if not (p.exists() and p.is_dir()):
        rp = _resolve(folder)
        p = rp if (rp and rp.is_dir()) else p
    if not (p.exists() and p.is_dir()):
        return f"Folder not found: {folder}"
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
    imgs = sorted(x for x in p.iterdir() if x.suffix.lower() in exts)
    if not imgs:
        return f"No images in {folder}"
    q = per_image_question.strip() + " Answer with ONLY 'YES' or 'NO'."
    lines, count = [], 0
    for img in imgs:
        try:
            a = clients.vlm(q, [img], temperature=0, max_tokens=4)
        except Exception:
            a = "NO"
        yes = a.strip().upper().startswith("Y")
        count += int(yes)
        lines.append(f"{img.name}: {'YES' if yes else 'NO'}")
    return f"count(YES) = {count} out of {len(imgs)}\n" + "\n".join(lines)


def _tool_transcribe(path, asr):
    p = _resolve(path)
    if not p:
        return f"Audio not found: {path}"
    if asr is None:
        return "(no transcription available in this environment)"
    try:
        t = asr(p)
        return t[:5000] if t else "(empty transcript)"
    except Exception as e:
        return f"transcription failed: {e}"


def _assistant_dict(msg):
    d = {"role": "assistant", "content": msg.content or ""}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in tcs]
    return d


def solve_agent(question, answer_type, asr=None, retriever=None, model=None, temperature=0.0, max_steps=18, verbose=False):
    manifest = _manifest()
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": f"answer_type = {answer_type}\n\nFILES ({len(manifest)}):\n"
                                     + "\n".join(manifest) + f"\n\nQUESTION:\n{question}"},
    ]
    final, evidences, touched, seen = None, [], [], set()
    for _ in range(max_steps):
        try:
            msg = clients.chat_tools(messages, TOOLS, model=model, role="agent")
        except Exception as e:
            if verbose:
                print("   [agent chat error]", str(e)[:160])
            break
        messages.append(_assistant_dict(msg))
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            final = (msg.content or "").strip()
            break
        for tc in tcs:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            sig = name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)
            if name in ("list_files", "read_file") and sig in seen:
                result = ("(you already ran this exact call — do NOT repeat it; write run_python "
                          "to load and inspect/compute over the data directly, then final_answer)")
            elif name == "final_answer":
                final = str(args.get("answer", "")).strip()
                evidences = args.get("evidences", []) or []
                result = "ok"
            elif name == "list_files":
                result = _tool_list_files(args.get("contains"))
            elif name == "read_file":
                result = _tool_read_file(args.get("path", ""), args.get("sheet"))
                if "not found" not in result[:20].lower():
                    touched.append(_relpath(args.get("path", "")))
            elif name == "run_python":
                result = _tool_run_python(args.get("code", ""))
            elif name == "search":
                result = _tool_search(args.get("query", ""), retriever)
            elif name == "count_matching_images":
                result = _tool_count_images(args.get("folder", ""), args.get("per_image_question", question))
                touched.append(args.get("folder", ""))
            elif name == "view_image":
                result = _tool_view_image(args.get("path", ""), args.get("question", question))
                if "not found" not in result[:20].lower():
                    touched.append(_relpath(args.get("path", "")))
            elif name == "transcribe_audio":
                result = _tool_transcribe(args.get("path", ""), asr)
                touched.append(_relpath(args.get("path", "")))
            else:
                result = f"unknown tool {name}"
            seen.add(sig)
            if verbose:
                print(f"   -> {name}({str(args)[:80]}) => {str(result)[:100].strip()!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)[:6500]})
        if final is not None:
            break

    if not final:
        return "Not enough data to answer.", []
    evidences = [_relpath(e) for e in evidences] or list(dict.fromkeys(t for t in touched if t))
    evidences = list(dict.fromkeys(evidences))[:5]
    # exact_match must be a bare token — distil a verbose agent answer (e.g. "Project 5: ..." -> "5")
    if answer_type == "exact_match" and len(final.split()) > 3 and "not enough data" not in final.lower():
        final = _extract_exact(question, final)
    answer = format_answer(question, final, answer_type)
    return answer, evidences
