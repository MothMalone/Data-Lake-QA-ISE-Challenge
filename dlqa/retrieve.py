"""Local BM25 retriever over the text corpus.

Runs anywhere (pure Python, no torch) so the pipeline is testable off-Kaggle; on Kaggle the
bge-m3 + Qdrant hybrid index augments/replaces this. CJK is tokenized per-character so
Chinese queries (Q3) retrieve properly.
"""
import json
import re

from rank_bm25 import BM25Okapi

from . import config, ingest

_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")


def _tok(s):
    return _TOKEN.findall(str(s).lower())


class BM25Retriever:
    def __init__(self, records):
        self.records = records
        self.bm25 = BM25Okapi([_tok(r["text"]) for r in records]) if records else None

    def search(self, query, k=8, source_filter=None):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            r = self.records[i]
            if source_filter and source_filter not in r["source_relative_path"]:
                continue
            out.append({**r, "score": float(scores[i])})
            if len(out) >= k:
                break
        return out


_cache = None


def get_retriever(rebuild=False) -> BM25Retriever:
    """Build (and disk-cache) the corpus once, return a retriever."""
    global _cache
    if _cache is not None and not rebuild:
        return _cache
    path = config.WORK_DIR / "corpus.json"
    if path.exists() and not rebuild:
        records = json.loads(path.read_text())
    else:
        records = ingest.build_corpus()
        path.write_text(json.dumps(records, ensure_ascii=False))
    _cache = BM25Retriever(records)
    return _cache
