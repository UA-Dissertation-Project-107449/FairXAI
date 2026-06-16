#!/usr/bin/env python3
"""Dermatology phase runner: baseline model comparison (stage 9).

Collates the per-model metrics and the stage-8 fairness report for the current
run into one canonical table (CSV + Markdown) under ``baseline/comparison/``.
No model load, no retraining, no experiment manifest.

Invoked by the pipeline with ``RUN_ID`` exported; can also be run standalone:

    RUN_ID=<run_id> python3 scripts/dermatology/compare.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from fairxai.comparison.dermatology import compare_run  # noqa: E402

PIPELINE = "dermatology"


def _resolve_run_id() -> str:
    run_id = os.getenv("RUN_ID")
    if run_id:
        return run_id
    base = ROOT_DIR / "output" / PIPELINE
    latest_txt = base / "latest_run.txt"
    if latest_txt.exists():
        return latest_txt.read_text().strip()
    link = base / "latest_run"
    if link.is_symlink():
        return link.resolve().name
    raise SystemExit("RUN_ID not set and no latest dermatology run found.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Restrict to these datasets.")
    parser.add_argument("--model-types", nargs="*", help="Restrict to these model types.")
    parser.add_argument("-v", action="store_const", const=1, dest="verbose", default=0)
    parser.add_argument("-vv", action="store_const", const=2, dest="verbose")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_id = _resolve_run_id()
    run_root = ROOT_DIR / "output" / PIPELINE / "runs" / run_id

    print(f"[PHASE 9] Comparing baseline models for run {run_id}")
    rows = compare_run(run_root, datasets=args.datasets, model_types=args.model_types)

    if not rows:
        print("  No baseline models found to compare.")
        return
    for r in sorted(rows, key=lambda x: (x["auc"] is None, -(x["auc"] or 0.0))):
        auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
        f1 = "n/a" if r["f1"] is None else f"{r['f1']:.3f}"
        print(f"  {r['model']}: acc {r['accuracy']:.3f} · f1 {f1} · auc {auc}")
    print(f"  Report: {run_root / 'baseline' / 'comparison'}")


if __name__ == "__main__":
    main()
