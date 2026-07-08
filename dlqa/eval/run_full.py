"""Full pipeline over the 15 samples -> submission.csv + scorecard.

Local run: BM25 retriever, no ASR (Q14 abstains). On Kaggle the notebook injects the
bge-m3 hybrid retriever + faster-whisper. Run:  .venv/bin/python -m dlqa.eval.run_full
"""
import csv
import json

import pandas as pd

from .. import config
from ..retrieve import get_retriever
from ..solve.solve import solve
from . import scorer


def manifest():
    return [str(p.relative_to(config.DATA_LAKE_ROOT))
            for p in config.DATA_LAKE_ROOT.rglob("*") if p.is_file()]


def main():
    df = pd.read_excel(config.PROJECT_ROOT / "0.Sample_Data.xlsx")
    man = manifest()
    ret = get_retriever()
    rows, report = [], []
    for _, r in df.iterrows():
        sid, q, at, gt = int(r["STT"]), str(r["Question"]), str(r["Answer Type"]), str(r["Groundtruth"])
        answer, evidences, plan = solve(q, at, man, ret)
        rows.append({"id": sid, "answer": answer, "evidences": json.dumps(evidences, ensure_ascii=False)})
        s = scorer.score(q, answer, gt, at)
        report.append((sid, at, plan["lane"], answer, gt, s))

    out = config.PROJECT_ROOT / "submission.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidences"])
        w.writeheader()
        w.writerows(rows)

    passed = sum(1 for *_, s in report if s >= 1)
    print(f"wrote {out}\n")
    for sid, at, lane, ans, gt, s in report:
        tag = "PASS" if s >= 1 else "FAIL"
        print(f"Q{sid:>2} [{at:^11}] {lane:<22} {tag}  ans={ans[:48]!r}  gt={gt[:32]!r}")
    print(f"\nLOCAL SCORE: {passed}/{len(report)}  "
          f"(bge-m3 retrieval + ASR on Kaggle lift the extract/audio rows)")


if __name__ == "__main__":
    main()
