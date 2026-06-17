#!/usr/bin/env python3
"""Dermatology phase runner: post-prediction fairness assessment (stage 8).

Reads the baseline test-prediction CSVs for the current run and writes a
subgroup fairness report. No model load, no retraining. Sensitive attributes and
min-group support come from ``configs/pipelines/dermatology.yaml``.

Invoked by the pipeline with ``RUN_ID`` exported; can also be run standalone:

    RUN_ID=<run_id> python3 scripts/dermatology/assess_predictions.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from fairxai.cli.runner_base import setup_phase_logging  # noqa: E402
from fairxai.fairness.image_assessment import (  # noqa: E402
    DEFAULT_MIN_GROUP_SAMPLES,
    assess_run,
)

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
    parser.add_argument("--min-group-samples", type=int, default=None)
    parser.add_argument("-v", action="store_const", const=1, dest="verbose", default=0)
    parser.add_argument("-vv", action="store_const", const=2, dest="verbose")
    args = parser.parse_args()

    run_id = _resolve_run_id()
    run_root = ROOT_DIR / "output" / PIPELINE / "runs" / run_id
    setup_phase_logging(
        ROOT_DIR,
        "assess.log",
        verbose=args.verbose,
        log_subdir=PIPELINE,
        run_id=run_id,
        stage_name="assess",
    )

    cfg_path = ROOT_DIR / "configs" / "pipelines" / f"{PIPELINE}.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    fairness_cfg = cfg.get("fairness", {})
    sensitive_attrs = fairness_cfg.get("sensitive_attributes", ["age_group", "sex"])
    min_group = args.min_group_samples
    if min_group is None:
        min_group = fairness_cfg.get("min_group_samples", DEFAULT_MIN_GROUP_SAMPLES)

    logging.info(
        "[PHASE] Assessing dermatology post-prediction fairness run_id=%s min_group_samples=%s",
        run_id,
        min_group,
    )
    print(f"[PHASE 8] Assessing post-prediction fairness for run {run_id}")
    reports = assess_run(
        run_root,
        sensitive_attrs,
        min_group_samples=min_group,
        datasets=args.datasets,
        model_types=args.model_types,
    )

    if not reports:
        logging.warning("No prediction CSVs assessed for run %s", run_id)
        print("  No prediction CSVs assessed.")
        return
    for key, report in sorted(reports.items()):
        op = report["overall_performance"]
        auc = "n/a" if op["auc"] is None else f"{op['auc']:.3f}"
        print(f"  {key}: acc {op['accuracy']:.3f} · f1 {op['f1']:.3f} · auc {auc}")
    out_dir = run_root / "baseline" / "prediction_fairness"
    logging.info("[SUCCESS] Assessed %d dermatology prediction file(s): %s", len(reports), out_dir)
    print(f"  Report: {out_dir}")


if __name__ == "__main__":
    main()
