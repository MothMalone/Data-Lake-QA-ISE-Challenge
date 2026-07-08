"""Run the SOTA ensemble (self-consistency over agent rollouts) over the 15 samples.

Run:  .venv/bin/python -m dlqa.eval.run_ensemble
"""
import csv
import json

import pandas as pd

from .. import config
from ..ensemble import solve_ensemble
from . import scorer

N = int(__import__("os").getenv("DLQA_N", "3"))


def main():
    df = pd.read_excel(config.PROJECT_ROOT / "0.Sample_Data.xlsx")
    rows, passed = [], 0
    for _, r in df.iterrows():
        sid, q, at, gt = int(r["STT"]), str(r["Question"]), str(r["Answer Type"]), str(r["Groundtruth"])
        try:
            ans, ev = solve_ensemble(q, at, n=N)
        except Exception as e:
            ans, ev = "Not enough data to answer.", []
            print(f"Q{sid} ERROR: {type(e).__name__}: {e}", flush=True)
        s = scorer.score(q, ans, gt, at)
        passed += int(s >= 1)
        rows.append({"id": sid, "answer": ans, "evidences": json.dumps(ev, ensure_ascii=False)})
        print(f"Q{sid:>2} [{at:^11}] {'PASS' if s >= 1 else 'FAIL'}  {str(ans)[:62]!r}", flush=True)

    with open(config.PROJECT_ROOT / "submission_ensemble.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidences"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nENSEMBLE SCORE (n={N}): {passed}/{len(df)}  (Q14 audio needs Kaggle Whisper)", flush=True)


if __name__ == "__main__":
    main()
