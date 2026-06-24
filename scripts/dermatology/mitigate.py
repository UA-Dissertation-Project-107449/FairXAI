#!/usr/bin/env python3
"""Dermatology phase runner: post-processing fairness mitigation (stage 11).

Reuses the baseline train/test prediction CSVs for the current run and writes a
before/after mitigation report. No model load, no retraining: group-wise decision
thresholds are learned per sensitive attribute (in isolation) for every configured
fairlearn constraint, fit on train predictions and applied to test predictions.

Invoked by the pipeline with ``RUN_ID`` exported; can also be run standalone:

    RUN_ID=<run_id> python3 scripts/dermatology/mitigate.py
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
from fairxai.fairness.image_mitigation import (  # noqa: E402
    DEFAULT_CONSTRAINTS,
    DEFAULT_MIN_GROUP_SAMPLES,
    DEFAULT_OBJECTIVE,
    mitigate_run,
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
    parser.add_argument(
        "--figures",
        dest="figures",
        action="store_true",
        default=None,
        help="Render before/after PNGs (overrides config).",
    )
    parser.add_argument(
        "--no-figures",
        dest="figures",
        action="store_false",
        help="Skip before/after PNGs (overrides config).",
    )
    parser.add_argument("-v", action="store_const", const=1, dest="verbose", default=0)
    parser.add_argument("-vv", action="store_const", const=2, dest="verbose")
    args = parser.parse_args()

    run_id = _resolve_run_id()
    run_root = ROOT_DIR / "output" / PIPELINE / "runs" / run_id
    setup_phase_logging(
        ROOT_DIR,
        "mitigate.log",
        verbose=args.verbose,
        log_subdir=PIPELINE,
        run_id=run_id,
        stage_name="mitigate",
    )

    cfg_path = ROOT_DIR / "configs" / "pipelines" / f"{PIPELINE}.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    fairness_cfg = cfg.get("fairness", {})
    mitigation_cfg = cfg.get("mitigation", {})

    if not mitigation_cfg.get("enabled", True):
        print("[PHASE 11] Mitigation disabled in config; skipping.")
        return

    sensitive_attrs = mitigation_cfg.get(
        "sensitive_attributes",
        fairness_cfg.get("sensitive_attributes", ["age_group", "sex", "fitzpatrick_group"]),
    )
    constraints = mitigation_cfg.get("constraints", DEFAULT_CONSTRAINTS)
    objective = mitigation_cfg.get("objective", DEFAULT_OBJECTIVE)
    min_group = args.min_group_samples
    if min_group is None:
        min_group = mitigation_cfg.get(
            "min_group_samples", fairness_cfg.get("min_group_samples", DEFAULT_MIN_GROUP_SAMPLES)
        )
    write_figures = (
        bool(args.figures)
        if args.figures is not None
        else bool(mitigation_cfg.get("figures", False))
    )

    logging.info(
        "[PHASE] Mitigating dermatology predictions run_id=%s constraints=%s objective=%s "
        "figures=%s",
        run_id,
        constraints,
        objective,
        write_figures,
    )
    print(f"[PHASE 11] Post-processing mitigation for run {run_id}")
    reports = mitigate_run(
        run_root,
        sensitive_attrs,
        constraints=constraints,
        objective=objective,
        min_group_samples=min_group,
        datasets=args.datasets,
        model_types=args.model_types,
        write_figures=write_figures,
    )

    if not reports:
        logging.warning("No prediction pairs mitigated for run %s", run_id)
        print("  No prediction pairs mitigated.")
        return
    for key in sorted(reports):
        n_attrs = len(reports[key].get("sensitive_attributes", {}))
        print(f"  {key}: mitigated {n_attrs} attribute(s) x {len(constraints)} constraint(s)")
    out_dir = run_root / "baseline" / "mitigation"
    logging.info("[SUCCESS] Mitigated %d model(s): %s", len(reports), out_dir)
    print(f"  Report: {out_dir}")
    if write_figures:
        print(f"  Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
