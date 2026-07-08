"""Kaggle orchestration: build the bge-m3 hybrid index, wire ASR, run the pipeline.

Heavy imports are lazy (inside functions) so `import dlqa` stays light off-Kaggle.
The notebook just does:  from dlqa.kaggle import run_all; run_all()
"""
import glob
import os
import subprocess
import tempfile
from pathlib import Path


def find_data_root() -> str:
    """Locate the data-lake root under /kaggle/input by a landmark file."""
    hits = glob.glob("/kaggle/input/**/class_grades.sql", recursive=True)
    if hits:
        return str(Path(hits[0]).parent)
    # fallback: the input subdir with the most files
    dirs = {}
    for p in glob.glob("/kaggle/input/*/**/", recursive=True):
        dirs[p] = len(list(Path(p).glob("*")))
    return max(dirs, key=dirs.get) if dirs else "/kaggle/input"


def find_questions() -> str:
    """Pick the xlsx that actually has a 'question' column (not a data-lake table)."""
    import pandas as pd
    for p in glob.glob("/kaggle/input/**/*.xlsx", recursive=True):
        try:
            cols = [str(c).lower() for c in pd.read_excel(p, nrows=0).columns]
            if any("question" in c for c in cols):
                return p
        except Exception:
            continue
    return ""


def build_hybrid_retriever(records, device="cuda"):
    """BGE-M3 dense + Qdrant/bm25 sparse, RRF fusion. Returns an object with
    .search(query, k, source_filter) — the same interface the lanes expect."""
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient, models

    texts = [r["text"] for r in records]
    emb = SentenceTransformer("BAAI/bge-m3", device=device)
    dense = emb.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse = list(bm25.embed(texts))

    client = QdrantClient(location=":memory:")
    coll = "dlqa"
    client.create_collection(
        coll,
        vectors_config={"dense": models.VectorParams(size=int(dense.shape[1]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    pts = []
    for i, r in enumerate(records):
        sv = sparse[i]
        pts.append(models.PointStruct(id=i, payload=r, vector={
            "dense": dense[i].tolist(),
            "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())}))
    for i in range(0, len(pts), 64):
        client.upsert(coll, pts[i:i + 64])

    class HybridRetriever:
        def search(self, query, k=8, source_filter=None):
            dq = emb.encode([query], normalize_embeddings=True)[0].tolist()
            sq = list(bm25.query_embed([query]))[0]
            res = client.query_points(coll, prefetch=[
                models.Prefetch(query=models.SparseVector(indices=sq.indices.tolist(), values=sq.values.tolist()),
                                using="bm25", limit=40),
                models.Prefetch(query=dq, using="dense", limit=40)],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=max(k * 3, k), with_payload=True)
            out = []
            for p in res.points:
                pl = p.payload
                if source_filter and source_filter not in pl.get("source_relative_path", ""):
                    continue
                out.append({**pl, "score": float(p.score)})
                if len(out) >= k:
                    break
            return out

    return HybridRetriever()


def _semantic_chunks(sents, embs, max_sent=8, min_sim=0.45):
    """Group consecutive sentences while they stay semantically similar (semantic chunking)."""
    if len(sents) <= 1:
        return sents
    chunks, cur = [], [sents[0]]
    for i in range(1, len(sents)):
        sim = float((embs[i] * embs[i - 1]).sum())
        if sim < min_sim or len(cur) >= max_sent:
            chunks.append(" ".join(cur))
            cur = [sents[i]]
        else:
            cur.append(sents[i])
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_agentic_retriever(ocr_images=True, device="cuda"):
    """Semantic-chunked BGE-M3 dense + BM25 sparse hybrid retriever (the agent's search tool).

    Each chunk is prefixed with its source file (a cheap contextual-retrieval signal), split at
    semantic boundaries, and images are OCR'd in so the agent can search visual content too.
    Exposes .search(query, k, source_filter) -> [{source_relative_path, text, score}].
    """
    import re
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient, models
    from . import config, ingest

    emb = SentenceTransformer("BAAI/bge-m3", device=device)
    root = config.DATA_LAKE_ROOT
    records = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext, rel = p.suffix.lower(), str(p.relative_to(root))
        blocks = []
        if ext in ingest.TEXT_EXTS:
            blocks = [t for _, t in ingest.extract_units(p)]
        elif ocr_images and ext in ingest.IMG_EXTS:
            blocks = [ingest._ocr_image(p)]
        for t in blocks:
            sents = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", t or "") if s.strip()]
            if not sents:
                continue
            sembs = emb.encode(sents, normalize_embeddings=True)
            for ch in _semantic_chunks(sents, sembs):
                records.append({"source_relative_path": rel, "text": f"File: {rel}\n{ch}"})
    print("semantic chunks:", len(records), "from",
          len({r["source_relative_path"] for r in records}), "files")

    texts = [r["text"] for r in records]
    dense = emb.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse = list(bm25.embed(texts))
    client = QdrantClient(location=":memory:")
    coll = "dlqa"
    client.create_collection(
        coll,
        vectors_config={"dense": models.VectorParams(size=int(dense.shape[1]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    pts = []
    for i, r in enumerate(records):
        sv = sparse[i]
        pts.append(models.PointStruct(id=i, payload=r, vector={
            "dense": dense[i].tolist(),
            "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())}))
    for i in range(0, len(pts), 64):
        client.upsert(coll, pts[i:i + 64])

    class _Retriever:
        def search(self, query, k=6, source_filter=None):
            dq = emb.encode([query], normalize_embeddings=True)[0].tolist()
            sq = list(bm25.query_embed([query]))[0]
            res = client.query_points(coll, prefetch=[
                models.Prefetch(query=models.SparseVector(indices=sq.indices.tolist(), values=sq.values.tolist()),
                                using="bm25", limit=40),
                models.Prefetch(query=dq, using="dense", limit=40)],
                query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True)
            out = []
            for pt in res.points:
                pl = pt.payload
                if source_filter and source_filter not in pl.get("source_relative_path", ""):
                    continue
                out.append({**pl, "score": float(pt.score)})
            return out

    return _Retriever()


def make_asr(model_size="large-v3", device="cuda"):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type="float16")

    def asr(path):
        try:  # ffmpeg-decode first (robust for odd .m4a/AAC), then transcribe a WAV
            with tempfile.TemporaryDirectory() as td:
                wav = os.path.join(td, "a.wav")
                subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", wav],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                segs, _ = model.transcribe(wav, vad_filter=True)
                return " ".join(s.text for s in segs).strip()
        except Exception as e:
            print("  [asr] failed:", e)
            return ""

    return asr


def _col(cols, *names):
    for n in names:
        for k, v in cols.items():
            if n in k:
                return v
    return None


def run_all(out="/kaggle/working/submission.csv", data_root=None, questions=None, use_asr=True, ocr_images=True):
    """Agentic RAG: build a semantic-chunked bge-m3 retriever, then let one reliable tool-agent
    (search/read/code/vision/audio) answer each question -> submission.csv (+ scorecard)."""
    import csv
    import json
    import pandas as pd

    os.environ.setdefault("DLQA_PROJECT_ROOT", "/kaggle/working")
    os.environ["DLQA_DATA_LAKE_ROOT"] = data_root or find_data_root()

    from . import config
    from .ensemble import solve_ensemble
    from .eval import scorer

    print("DATA_LAKE_ROOT =", config.DATA_LAKE_ROOT)
    print("building semantic bge-m3 retriever (the agent's search tool) ...")
    retriever = build_agentic_retriever(ocr_images=ocr_images)
    asr = make_asr() if use_asr else None

    qpath = questions or find_questions()
    assert qpath, "No questions xlsx (with a 'question' column) found under /kaggle/input"
    qdf = pd.read_excel(qpath)
    cols = {str(c).lower().strip(): c for c in qdf.columns}
    id_col = _col(cols, "stt", "id")
    q_col = _col(cols, "question")
    at_col = _col(cols, "answer type", "answer_type", "answertype")
    gt_col = _col(cols, "groundtruth", "ground truth")
    print(f"questions: {qpath} | {len(qdf)} rows")

    rows, report = [], []
    for i, r in qdf.iterrows():
        sid = r[id_col] if id_col else i + 1
        q = str(r[q_col])
        at = str(r[at_col]).strip() if at_col else "llm_judge"
        try:
            answer, evidences = solve_ensemble(q, at, asr=asr, retriever=retriever)
        except Exception as e:
            print(f"  Q{sid} error -> abstain ({type(e).__name__}: {e})")
            answer, evidences = "Not enough data to answer.", []
        rows.append({"id": sid, "answer": answer, "evidences": json.dumps(evidences, ensure_ascii=False)})
        print(f"  Q{sid} -> {str(answer)[:70]!r}", flush=True)
        if gt_col is not None:
            report.append((sid, at, scorer.score(q, answer, str(r[gt_col]), at), answer))

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidences"])
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out, "with", len(rows), "rows")
    if report:
        for sid, at, s, ans in report:
            print(f"Q{sid} [{at}] {'PASS' if s >= 1 else 'FAIL'}  {str(ans)[:55]!r}")
        print(f"\nSCORE: {sum(1 for it in report if it[2] >= 1)}/{len(report)}")
    return out
