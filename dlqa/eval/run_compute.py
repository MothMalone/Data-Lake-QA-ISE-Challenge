"""Smoke-eval the compute lane on the three verified compute-lane sample questions.

Run from the project root:  .venv/bin/python -m dlqa.eval.run_compute
Q13's compute value is the raw mean (7.55); the MCQ->letter (C) mapping is the
formatter's job (milestone M4), so we check the numeric value here.
"""
from .. import config
from ..solve.lane_compute import solve_compute

CASES = [
    ("Q1", "How many are the significant genes by acetylproteomics?",
     ["biomedical/1-s2.0-S0092867420301070-mmc3.xlsx"], "16"),
    ("Q7", 'Determine the correlation coefficient between the "Limit" and "Balance" '
           'columns in the Credit.csv file where "correlation_value" is the calculated '
           'Pearson correlation coefficient between "Limit" and "Balance", rounded to '
           'two decimal places.',
     ["da-dev-tables/Credit.csv"], "0.86"),
    ("Q13", "Điểm trung bình môn Toán của lớp 10A1 là bao nhiêu? "
            "(A. 7.45  B. 7.50  C. 7.55  D. 7.60)",
     ["class_grades.sql"], "7.55 (=C)"),
]


def main():
    print(f"data lake: {config.DATA_LAKE_ROOT}")
    print(f"coder model: {config.MODELS['coder']}\n")
    for qid, q, files, exp in CASES:
        paths = [config.DATA_LAKE_ROOT / f for f in files]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"{qid}: SKIP (missing files: {missing})")
            continue
        r = solve_compute(q, paths, k=3, verbose=True)
        agree = r.get("agreement")
        print(f"{qid}: value={r['value']!r}  votes={r['votes']}  "
              f"agreement={agree:.0%}" if agree is not None else f"{qid}: value={r['value']!r}")
        print(f"      expected≈{exp}\n")


if __name__ == "__main__":
    main()
