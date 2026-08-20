#!/usr/bin/env python3
"""Dermatology phase runner: fairness mitigation (stage 11).

Two parts, both writing before/after reports for the current run:

1. **Post-processing** — reuses the baseline train/test prediction CSVs. No model
   load, no retraining: group-wise decision thresholds are learned per sensitive
   attribute (in isolation) for every configured fairlearn constraint, fit on
   train predictions and applied to test predictions.
2. **Feature space** (``mitigation.feature_space``) — rebuilds each model's
   frozen-backbone feature matrix with one eval-mode forward pass and runs
   cardiac's pre/in-processing catalog over it through the shared
   ``MitigationEngine``. Skipped with ``--no-feature-space``.

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
from fairxai.fairness.image_feature_mitigation import (  # noqa: E402
    DEFAULT_FEATURE_TECHNIQUES,
    mitigate_run_features,
)
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
    parser.add_argument(
        "--feature-space",
        dest="feature_space",
        action="store_true",
        default=None,
        help="Run pre/in-processing mitigation on frozen features (overrides config).",
    )
    parser.add_argument(
        "--no-feature-space",
        dest="feature_space",
        action="store_false",
        help="Skip feature-space mitigation; post-processing only (overrides config).",
    )
    parser.add_argument(
        "--techniques",
        nargs="*",
        default=None,
        help=f"Feature-space techniques (default: {' '.join(DEFAULT_FEATURE_TECHNIQUES)}).",
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
        # Not fatal for the stage: the feature-space pass reads checkpoints, not
        # prediction CSVs, so it can still produce a report.
        logging.warning("No prediction pairs mitigated for run %s", run_id)
        print("  No prediction pairs mitigated.")
    else:
        for key in sorted(reports):
            n_attrs = len(reports[key].get("sensitive_attributes", {}))
            print(f"  {key}: mitigated {n_attrs} attribute(s) x {len(constraints)} constraint(s)")
        out_dir = run_root / "baseline" / "mitigation"
        logging.info("[SUCCESS] Mitigated %d model(s): %s", len(reports), out_dir)
        print(f"  Report: {out_dir}")
        if write_figures:
            print(f"  Figures: {out_dir / 'figures'}")

    _run_feature_space(cfg, args, run_root, sensitive_attrs, min_group)


def _run_feature_space(cfg, args, run_root: Path, sensitive_attrs, min_group: int) -> None:
    """Stage 11 part 2: pre/in-processing over the frozen-backbone features.

    Unlike the post-processing pass this loads each checkpoint and runs a forward
    pass, so it is opt-out (``--no-feature-space``) and never fails the stage: a
    torch/checkpoint problem is logged and the post-processing report still stands.
    """
    fs_cfg = (cfg.get("mitigation", {}) or {}).get("feature_space", {}) or {}
    enabled = (
        bool(args.feature_space) if args.feature_space is not None else bool(fs_cfg.get("enabled"))
    )
    if not enabled:
        print("[PHASE 11] Feature-space mitigation disabled; skipping.")
        return

    training_cfg = cfg.get("training", {}) or {}
    image_cfg = training_cfg.get("image", {}) or {}
    processed_dir = ROOT_DIR / cfg.get("paths", {}).get(
        "processed_dir", "data/processed/dermatology"
    )
    techniques = args.techniques or fs_cfg.get("techniques", DEFAULT_FEATURE_TECHNIQUES)

    from fairxai.utils.gpu import detect_accelerator

    requested = fs_cfg.get("device", image_cfg.get("device", "auto"))
    resolved = detect_accelerator(requested)
    # Only cuda and cpu are torch device strings the extraction path can hand to
    # torch.device; anything else (rocm on an unsupported build) falls back.
    device = resolved if resolved in {"cuda", "cpu"} else "cpu"

    logging.info(
        "[PHASE] Feature-space mitigation run_root=%s techniques=%s device=%s",
        run_root,
        techniques,
        device,
    )
    print(f"[PHASE 11] Feature-space mitigation ({len(techniques)} technique(s), device={device})")
    try:
        reports = mitigate_run_features(
            run_root,
            sensitive_attrs,
            processed_dir=processed_dir,
            image_col=image_cfg.get("image_column", "image_path"),
            target_col=training_cfg.get("target", "skin_cancer"),
            techniques=techniques,
            model_type=fs_cfg.get("model_type", "logistic_regression"),
            min_group_samples=min_group,
            datasets=args.datasets,
            model_types=args.model_types,
            device=device,
            batch_size=int(fs_cfg.get("batch_size", image_cfg.get("batch_size", 32))),
            num_workers=int(fs_cfg.get("num_workers", image_cfg.get("num_workers", 0))),
        )
    except Exception as exc:  # noqa: BLE001 - post-processing results must survive
        logging.warning("Feature-space mitigation failed: %s", exc)
        print(f"  Feature-space mitigation failed: {exc}")
        return

    if not reports:
        print("  No checkpoints available for feature-space mitigation.")
        return
    for key in sorted(reports):
        n_attrs = len(reports[key].get("sensitive_attributes", {}))
        print(f"  {key}: {n_attrs} attribute(s) x {len(techniques)} technique(s)")
    fs_dir = run_root / "baseline" / "mitigation" / "feature_space"
    logging.info("[SUCCESS] Feature-space mitigated %d model(s): %s", len(reports), fs_dir)
    print(f"  Report: {fs_dir}")


if __name__ == "__main__":
    main()
