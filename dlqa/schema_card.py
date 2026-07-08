"""Compact schema cards for the compute lane — describe a file WITHOUT dumping it.

Keeps tokens down and guards the 70MB xlsx: send columns/dtypes/sheets/README/DDL
plus a tiny head, never the full table. The LLM then writes code that reads the
real file from disk.
"""
import re
from pathlib import Path

import pandas as pd

_BIG = 5 * 1024 * 1024  # above this, skip full-sheet shape probes
_README_HINTS = ("readme", "legend", "description", "dictionary", "info", "note", "key")


def _csv_card(path: Path) -> str:
    df = pd.read_csv(path, nrows=200)
    try:
        with open(path, "rb") as f:
            nrows = sum(1 for _ in f) - 1
    except Exception:
        nrows = "?"
    cols = ", ".join(f"{c}:{t}" for c, t in zip(df.columns, df.dtypes.astype(str)))
    head = df.head(3).to_csv(index=False).strip()
    return f"[CSV] {path}\nrows≈{nrows}; columns: {cols}\nhead(3):\n{head}"


def _xlsx_card(path: Path) -> str:
    """Show RAW cells (header=None) so header-vs-headerless is decidable by eye.

    The baseline trap (Q1): a list-style sheet with no header row — reading it with a
    default header undercounts by one. Showing raw cells + the raw row count exposes it.
    """
    xl = pd.ExcelFile(path)
    small = path.stat().st_size < _BIG
    parts = [f"[XLSX] {path}\nsheets: {xl.sheet_names}"]
    for s in xl.sheet_names:
        low = s.lower()
        try:
            if any(h in low for h in _README_HINTS):
                full = xl.parse(s, header=None).astype(str)
                parts.append(f"  sheet '{s}' (README/legend, full):\n{full.to_string(index=False, header=False)[:2500]}")
                continue
            raw = xl.parse(s, header=None, nrows=6).astype(object)
            nrows = ""
            if small:
                try:
                    nrows = f", rows(raw, header=None)={xl.parse(s, header=None).shape[0]}"
                except Exception:
                    pass
            body = raw.to_string(index=False, header=False)[:800]
            parts.append(f"  sheet '{s}'{nrows}, first cells (raw, header=None):\n{body}")
        except Exception as e:
            parts.append(f"  sheet '{s}': <unreadable: {e}>")
    return "\n".join(parts)


def _sql_card(path: Path) -> str:
    txt = path.read_text(errors="ignore")
    creates = re.findall(r"CREATE TABLE[\s\S]*?\);", txt, re.I)
    n_insert = len(re.findall(r"INSERT\s+INTO", txt, re.I))
    ddl = "\n".join(creates)[:2000] or "(no CREATE TABLE found)"
    return f"[SQL] {path}\n{n_insert} INSERT statement(s). DDL:\n{ddl}"


def card_for(path) -> str:
    path = Path(path)
    if not path.exists():
        return f"[MISSING] {path}"
    ext = path.suffix.lower()
    try:
        if ext in (".csv", ".tsv"):
            return _csv_card(path)
        if ext in (".xlsx", ".xls"):
            return _xlsx_card(path)
        if ext == ".sql":
            return _sql_card(path)
        return f"[{ext or 'file'}] {path}\n" + path.read_text(errors="ignore")[:1500]
    except Exception as e:
        return f"[{ext or 'file'}] {path} (could not profile: {e}; size={path.stat().st_size})"
