"""Produce submission.csv over the 15 samples + a scorecard.

Milestone status: the compute lane (Q1/Q7/Q13) is wired through formatter + scorer
end to end; the other questions are abstention placeholders until the extract/perceive
lanes and the file-resolver land (they need the full 83-file lake + the Colab index).
File-resolution is stubbed here with the cited sources — the general router/resolver is M4/M8.

Run from project root:  .venv/bin/python -m dlqa.eval.run_samples
"""
import csv
import json

import pandas as pd

from .. import config
from ..solve.formatter import format_answer, ABSTAIN
from ..solve.lane_compute import solve_compute
from . import scorer

# Stubbed routing for the compute lane (id -> relative source files). Temporary.
COMPUTE = {
    1: ["biomedical/1-s2.0-S0092867420301070-mmc3.xlsx"],
    7: ["da-dev-tables/Credit.csv"],
    13: ["class_grades.sql"],
}


def main():
    df = pd.read_excel(config.PROJECT_ROOT / "0.Sample_Data.xlsx")
    rows, report = [], []
    for _, r in df.iterrows():
        sid = int(r["STT"])
        q, at, gt = str(r["Question"]), str(r["Answer Type"]), str(r["Groundtruth"])
        if sid in COMPUTE:
            files = [config.DATA_LAKE_ROOT / f for f in COMPUTE[sid]]
            res = solve_compute(q, files, k=3)
            answer = format_answer(q, res["value"], at)
            evidences = COMPUTE[sid]
            s = scorer.score_exact(answer, gt) if at == "exact_match" else None
            report.append((sid, at, answer, gt, s, res.get("agreement")))
        else:
            answer, evidences = ABSTAIN, []
            report.append((sid, at, answer, gt, None, None))
        rows.append({"id": sid, "answer": answer,
                     "evidences": json.dumps(evidences, ensure_ascii=False)})

    out = config.PROJECT_ROOT / "submission.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer", "evidences"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}\n")
    attempted = 0
    passed = 0
    for sid, at, ans, gt, s, agr in report:
        if s is None:
            print(f"Q{sid:>2} [{at:^11}] answer={ans!r}  (lane not yet implemented)")
        else:
            attempted += 1
            passed += int(s >= 1)
            tag = "PASS" if s >= 1 else "FAIL"
            ag = f"  vote-agreement={agr:.0%}" if agr is not None else ""
            print(f"Q{sid:>2} [{at:^11}] answer={ans!r}  gt={gt!r}  {tag}{ag}")
    print(f"\ncompute-lane score: {passed}/{attempted} attempted correct "
          f"(the other {len(report) - attempted} await extract/perceive lanes)")


if __name__ == "__main__":
    main()
