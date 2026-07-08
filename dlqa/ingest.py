"""Extract a text/table corpus from the lake — shared by the local BM25 retriever and
the Kaggle bge-m3 index. Images/audio are left to the perceive lane.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from . import config

TEXT_EXTS = {".txt", ".md", ".sql", ".csv", ".tsv", ".xlsx", ".xls",
             ".pdf", ".docx", ".ppt", ".pptx", ".doc", ".html", ".htm"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def _soffice():
    for c in ("soffice", "libreoffice",
              "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if shutil.which(c) or Path(c).exists():
            return c
    return None


def chunks(text, size=1200, overlap=150):
    text = re.sub(r"[ \t]+", " ", (text or "")).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def _table_text(df, max_rows=40):
    return f"columns: {list(df.columns)}\n" + df.head(max_rows).to_string(index=False)


def extract_units(path: Path):
    """Return list of (kind, text) for a file (before chunking)."""
    ext = path.suffix.lower()
    recs = []
    try:
        if ext in (".txt", ".md", ".sql"):
            recs.append(("text", path.read_text(errors="ignore")))
        elif ext in (".csv", ".tsv"):
            df = pd.read_csv(path, nrows=2000)
            recs.append(("table", _table_text(df)))
        elif ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(path)
            for s in xl.sheet_names:
                try:
                    df = xl.parse(s, nrows=800)
                except Exception:
                    continue
                recs.append(("table", f"sheet {s} " + _table_text(df)))
        elif ext == ".pdf":
            import fitz
            for i, pg in enumerate(fitz.open(path)):
                t = pg.get_text().strip()
                if t:
                    recs.append((f"page{i+1}", t))
        elif ext == ".docx":
            import docx
            d = docx.Document(str(path))
            recs.append(("text", "\n".join(p.text for p in d.paragraphs)))
        elif ext in (".ppt", ".pptx", ".doc"):
            so = _soffice()
            if so:
                with tempfile.TemporaryDirectory() as td:
                    subprocess.run([so, "--headless", "--convert-to", "pdf", "--outdir", td, str(path)],
                                   timeout=240, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    pdfs = list(Path(td).glob("*.pdf"))
                    if pdfs:
                        import fitz
                        for i, pg in enumerate(fitz.open(pdfs[0])):
                            t = pg.get_text().strip()
                            if t:
                                recs.append((f"slide{i+1}", t))
        elif ext in (".html", ".htm"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
            recs.append(("text", soup.get_text(" ", strip=True)))
    except Exception as e:
        print(f"  [ingest] {path.name}: {e}")
    return recs


def _ocr_image(path: Path) -> str:
    """VLM OCR so images are retrievable by content (scholarship poster, aviation scans)."""
    from . import clients
    try:
        return clients.vlm(
            "Transcribe ALL text visible in this image (OCR): headings, table cells, names, "
            "numbers, labels, captions. Output only the transcribed text, no commentary.",
            [path], temperature=0, max_tokens=800)
    except Exception as e:
        print(f"  [ocr] {path.name}: {e}")
        return ""


def build_corpus(root=None, ocr_images=False):
    root = Path(root or config.DATA_LAKE_ROOT)
    records = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rel = str(p.relative_to(root))
        if ext in TEXT_EXTS:
            for kind, text in extract_units(p):
                for j, ch in enumerate(chunks(text)):
                    records.append({"source_relative_path": rel, "kind": kind, "chunk": j, "text": ch})
        elif ocr_images and ext in IMG_EXTS:
            for j, ch in enumerate(chunks(_ocr_image(p))):
                records.append({"source_relative_path": rel, "kind": "image_ocr", "chunk": j, "text": ch})
    return records
