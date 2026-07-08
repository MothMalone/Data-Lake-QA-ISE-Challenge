"""Run the tool-using agent over the 15 samples -> scorecard + submission_agent.csv.

Local run has no Whisper, so Q14 (audio) abstains here; everything else is testable.
Run:  .venv/bin/python -m dlqa.eval.run_agent
"""
import csv
import json
import sys

import pandas as pd

from .. import config
from ..ensemble import solve_ensemble
from ..retrieve import get_retriever
from . import scorer


def main():
    df = pd.read_excel(config.PROJECT_ROOT / "0.Sample_Data.xlsx")
    ret = get_retriever()   # local BM25 stands in for the Kaggle semantic bge-m3 retriever
    rows, passed = [], 0
    for _, r in df.iterrows():
        sid, q, at, gt = int(r["STT"]), str(r["Question"]), str(r["Answer Type"]), str(r["Groundtruth"])
        try:
            ans, ev = solve_ensemble(q, at, retriever=ret)
        except Exception as e:
            ans, ev = "Not enough data to answer.", []
            print(f"Q{sid} ERROR: {type(e).__name__}: {e}")
        s = scorer.score(q, ans, gt, at)
        passed += int(s >= 1)
        rows.append({"id": sid, "answer": ans, "evidences": json.dumps(ev, ensure_ascii=False)})
        print(f"Q{sid:>2} [{at:^11}] {'PASS' if s >= 1 else 'FAIL'}  {str(ans)[:60]!r}", flush=True)

    with open(config.PROJECT_ROOT / "submission_agent.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidences"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nAGENT SCORE: {passed}/{len(df)}  (Q14 audio needs Kaggle Whisper)", flush=True)


if __name__ == "__main__":
    main()
