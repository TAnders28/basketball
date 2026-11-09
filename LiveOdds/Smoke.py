"""Read-only smoke command against the public sources. Not a unit test.

A passing unit test uses labeled fixtures and cannot establish live readiness.
This command prints whatever the sources return at process time, including a
blocked report when a source is down or a book fails validation. It does not
place orders or write secrets. It does not save a dataset.

    python -m LiveOdds.Smoke
    python -m LiveOdds.Smoke --date 2026-11-08
"""

import argparse
import json
from datetime import datetime

from LiveOdds.Report import BuildReport, FormatReport
from LiveOdds.Sources import LoadBundle
from LiveOdds.Time import EASTERN


def Main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Observe public NBA odds once. Read-only; not a unit test.")
    parser.add_argument("--date", default=datetime.now(EASTERN).strftime("%Y-%m-%d"))
    parser.add_argument("--table", action="store_true", help="Print the markdown table before the JSON")
    args = parser.parse_args(argv)
    datetime.strptime(args.date, "%Y-%m-%d")
    report = BuildReport(LoadBundle(args.date, live_sample=True))
    if args.table:
        print(FormatReport(report))
        print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    Main()
