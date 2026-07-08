# dlqa — organizer-aware data-lake QA (iSE challenge)

A router-gated hybrid solver over a heterogeneous, multilingual (EN/VI/ZH) data lake.
Each question is routed to the cheapest sufficient lane and formatted to the answer
contract embedded in the question:

- **compute** — code-as-action (LLM writes pandas/SQL, runs it in a sandbox, self-consistency vote) for tables / spreadsheets / SQL.
- **extract** — BGE-M3 dense + BM25 sparse hybrid retrieval (RRF) → mirroring synthesis in the question's language.
- **perceive** — VLM per-image (count-over-folder, single/multi-image OCR/reason) and Whisper ASR for audio.
- **abstain** — emits the canonical "not enough data" answer when nothing grounds.

## Run on Kaggle (T4×2)

1. New Notebook → Accelerator = **GPU T4 ×2**, Internet = **On**.
2. **Add Input:** the data-lake dataset (the uploaded zip) and the questions `.xlsx`.
3. Cell 1 — install + clone:

```python
!pip -q install sentence-transformers qdrant-client fastembed faster-whisper
!apt-get -qq install -y libreoffice >/dev/null 2>&1
!rm -rf /kaggle/working/dlqa-repo && git clone -q https://github.com/MothMalone/dlqa.git /kaggle/working/dlqa-repo
!pip -q install -e /kaggle/working/dlqa-repo
```

4. Cell 2 — key + run:

```python
import os; os.environ["OPENROUTER_API_KEY"] = "sk-or-...your key..."
from dlqa.kaggle import run_all
run_all()          # -> /kaggle/working/submission.csv (+ scorecard if groundtruth present)
```

`run_all()` auto-locates the lake (landmark file) and the questions sheet (the xlsx with a
`question` column), builds the index, solves every question, and writes `submission.csv`
(`id, answer, evidences`). To iterate: I push a fix, you re-run Cell 1 (re-clone) + Cell 2.

Local dev (no GPU) uses the pure-Python BM25 retriever in `dlqa.retrieve`.
