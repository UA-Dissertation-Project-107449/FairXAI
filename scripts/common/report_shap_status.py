#!/usr/bin/env python3
"""Print one report over every ``shap_status.json`` a run wrote.

``--strict`` exits 1 when any model fell back or failed.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from fairxai.explainability.tabular import (  # noqa: E402
    collect_shap_status,
    format_shap_status_report,
    has_problems,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, help="Run output directory to scan")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any model has fallback or failed SHAP",
    )
    args = parser.parse_args(argv)

    entries = collect_shap_status(Path(args.run_root))
    print(format_shap_status_report(entries))
    if args.strict and has_problems(entries):
        print("STRICT_SHAP: stopping because SHAP fell back or failed (see above).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
