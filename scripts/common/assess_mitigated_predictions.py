#!/usr/bin/env python3
"""Bootstrap confidence intervals for the mitigation comparison's arms.

Stage 8 puts every baseline model's fairness numbers under an evidence layer:
each group metric and parity gap carries a bootstrap interval, and each pair of
groups carries a signed difference with a multiplicity-adjusted p-value. The
mitigation comparison reported point estimates only, so a technique that moved
a parity gap by a few points could not be said to have moved it at all.

This pass closes that gap. It reads the per-arm predictions the mitigation
stage already writes, and puts every arm -- the unmitigated baseline included --
through the same bootstrap the baseline assessment uses, with the same
configuration. Nothing is retrained and no prediction changes; the arms are
scored, not produced, here.

Two kinds of output land under ``prediction_fairness/`` beside the arms:

* per-arm ``*_ci.csv`` / ``*_pairwise_ci.csv``, in the shape stage 8 writes;
* ``arms_ci.csv`` / ``arms_pairwise_ci.csv``, the same rows concatenated with
  the arm's dataset, family, technique, and constrained attribute prefixed, so
  a before/after comparison is one table read rather than a directory walk.

Usage:
    RUN_ID=<run_id> python3 scripts/common/assess_mitigated_predictions.py
    python3 scripts/common/assess_mitigated_predictions.py --run-id <id> \
        --datasets cleveland_uci --model-types logistic_regression
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from assess_predictions import (  # noqa: E402
    _DEFAULT_SENSITIVE,
    _write_uncertainty,
    decode_sensitive_attributes,
    resolve_sensitive_columns,
)

from fairxai.cli.runner_base import load_pipeline_config, setup_phase_logging  # noqa: E402
from fairxai.cli.runner_utils import get_run_root, resolve_latest_run_dir  # noqa: E402

_ARM_KEYS = ["dataset", "model_type", "technique", "constraint_attr"]


def _load_index(predictions_dir: Path) -> List[Dict]:
    """Read the arm index the mitigation stage writes beside its predictions.

    The index is required rather than reconstructed: a filename cannot be
    parsed back into its four components, because dataset names, model
    families, technique names, and attribute names all contain underscores.
    """
    index_path = predictions_dir / "index.json"
    if not index_path.exists():
        logging.error(
            "No arm index at %s. Run the mitigation stage first; it writes the "
            "predictions and the index together.",
            index_path,
        )
        return []

    with open(index_path) as handle:
        entries = json.load(handle)

    if not isinstance(entries, list):
        logging.error("Arm index at %s is not a list of entries", index_path)
        return []
    return entries


def _select_arms(
    entries: List[Dict],
    datasets: Optional[List[str]],
    model_types: Optional[List[str]],
    techniques: Optional[List[str]],
) -> List[Dict]:
    """Filter the index down to the arms the caller asked for."""
    selected = entries
    if datasets:
        selected = [e for e in selected if e.get("dataset") in datasets]
    if model_types:
        selected = [e for e in selected if e.get("model_type") in model_types]
    if techniques:
        selected = [e for e in selected if e.get("technique") in techniques]
    return selected


def _prefix_arm(frame: pd.DataFrame, entry: Dict) -> pd.DataFrame:
    """Put the arm's identity in front of a result table's own columns."""
    tagged = frame.copy()
    for key in reversed(_ARM_KEYS):
        tagged.insert(0, key, entry.get(key, ""))
    return tagged


def assess_arm(
    entry: Dict,
    predictions_dir: Path,
    output_dir: Path,
    configured_sensitive: List[str],
    uncertainty_cfg: Dict,
) -> Dict:
    """Bootstrap one arm and write its two interval tables.

    Returns the arm's metadata plus the two tables, or an empty dict when the
    arm produced nothing. A failure on one arm is logged and skipped: the
    intervals are an addition to a comparison that already stands on its point
    estimates, and one unusable arm must not cost the rest.
    """
    pred_file = predictions_dir / entry["file"]
    if not pred_file.exists():
        logging.warning("Arm %s: predictions file missing (%s)", entry.get("file"), pred_file)
        return {}

    df = pd.read_csv(pred_file)
    df = decode_sensitive_attributes(df)

    resolved = resolve_sensitive_columns(df, configured_sensitive or _DEFAULT_SENSITIVE)
    if not resolved:
        logging.warning(
            "Arm %s: none of the configured sensitive attributes resolved; skipped",
            entry.get("file"),
        )
        return {}

    file_stem = pred_file.stem
    logging.info(
        "[ARM] %s dataset=%s model=%s technique=%s constraint=%s rows=%d sensitive=%s",
        file_stem,
        entry.get("dataset"),
        entry.get("model_type"),
        entry.get("technique"),
        entry.get("constraint_attr"),
        len(df),
        resolved,
    )

    metadata = _write_uncertainty(df, resolved, output_dir, file_stem, uncertainty_cfg)
    if not metadata:
        return {}

    ci_file = output_dir / f"{file_stem}_ci.csv"
    pairwise_file = output_dir / f"{file_stem}_pairwise_ci.csv"
    return {
        "metadata": metadata,
        "table": pd.read_csv(ci_file),
        "pairwise": pd.read_csv(pairwise_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap fairness intervals for every mitigation arm"
    )
    parser.add_argument("--pipeline", default="cardiac", help="Pipeline name (default: cardiac)")
    parser.add_argument("--run-id", default=None, help="Run ID (default: RUN_ID env, then latest)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names (CLI filter)")
    parser.add_argument("--model-types", nargs="+", default=None, help="Model families (filter)")
    parser.add_argument("--techniques", nargs="+", default=None, help="Technique names (filter)")
    parser.add_argument(
        "--mitigation-dir",
        default=None,
        help="Mitigation output directory (default: <run>/experiments/mitigation)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = args.pipeline

    base_results = project_root / "output" / pipeline
    run_id = args.run_id or os.getenv("RUN_ID")
    run_root = (
        get_run_root(base_results, run_id) if run_id else resolve_latest_run_dir(base_results)
    )
    if not run_root:
        logging.error("No pipeline run found under %s. Pass --run-id or set RUN_ID.", base_results)
        sys.exit(1)

    setup_phase_logging(
        project_root,
        "mitigation_fairness_intervals.log",
        verbose=args.verbose,
        run_id=run_root.name,
        stage_name="mitigate",
        sub_stage="intervals",
    )

    mitigation_dir = (
        Path(args.mitigation_dir)
        if args.mitigation_dir
        else run_root / "experiments" / "mitigation"
    )
    predictions_dir = mitigation_dir / "predictions"
    output_dir = mitigation_dir / "prediction_fairness"

    logging.info("[PHASE] Mitigation fairness intervals started")
    logging.info(
        "Run context: pipeline=%s run=%s predictions_dir=%s output_dir=%s",
        pipeline,
        run_root.name,
        predictions_dir,
        output_dir,
    )

    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    fairness_cfg = pipeline_cfg.get("fairness", {}) or {}
    configured_sensitive = fairness_cfg.get("sensitive_attributes") or _DEFAULT_SENSITIVE
    # The same bootstrap configuration the baseline assessment runs under.
    # Reading it from one place is what makes a baseline interval and a
    # mitigated interval comparable at all.
    uncertainty_cfg = fairness_cfg.get("uncertainty", {}) or {}
    if not uncertainty_cfg.get("enabled", True):
        logging.info("Bootstrap intervals disabled by configuration; nothing to do")
        return

    entries = _load_index(predictions_dir)
    if not entries:
        sys.exit(1)

    selected = _select_arms(entries, args.datasets, args.model_types, args.techniques)
    if not selected:
        logging.error("No arms matched the requested filters (%d in the index)", len(entries))
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Assessing %d of %d arms", len(selected), len(entries))

    tables, pairwise_tables, summary_rows = [], [], []
    for entry in selected:
        assessed = assess_arm(
            entry, predictions_dir, output_dir, configured_sensitive, uncertainty_cfg
        )
        if not assessed:
            continue
        tables.append(_prefix_arm(assessed["table"], entry))
        pairwise_tables.append(_prefix_arm(assessed["pairwise"], entry))
        summary_rows.append({**{k: entry.get(k, "") for k in _ARM_KEYS}, **assessed["metadata"]})

    if not tables:
        logging.error("No arm produced intervals; nothing written")
        sys.exit(1)

    combined_ci = output_dir / "arms_ci.csv"
    combined_pairwise = output_dir / "arms_pairwise_ci.csv"
    summary_file = output_dir / "arms_summary.csv"

    pd.concat(tables, ignore_index=True).to_csv(combined_ci, index=False)
    pd.concat(pairwise_tables, ignore_index=True).to_csv(combined_pairwise, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    logging.info("[SUCCESS] Combined intervals: %s", combined_ci)
    logging.info("[SUCCESS] Combined pairwise differences: %s", combined_pairwise)
    logging.info("[SUCCESS] Per-arm summary: %s", summary_file)
    logging.info(
        "[SUCCESS] Mitigation fairness intervals complete: %d of %d arms assessed",
        len(tables),
        len(selected),
    )


if __name__ == "__main__":
    main()
