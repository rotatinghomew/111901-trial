"""Command-line entry point.

    python -m trial [--data DIR] [--follow-up DAYS] [--audit PATH]

Prints the reviewer-facing report.  ``--audit`` additionally writes the
internal resident-level trace used by the project team to compare waves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_study
from .models import default_follow_up_days
from .report import assess_batch, audit_trace, render_audit_csv, render_reviewer_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--follow-up", type=int, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args(argv)

    study = load_study(args.data)
    days = args.follow_up or default_follow_up_days(study)
    recommendation = assess_batch(study, days)
    print(render_reviewer_report(study, recommendation, days))

    if args.audit is not None:
        rows = audit_trace(study, days)
        args.audit.write_text(render_audit_csv(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
