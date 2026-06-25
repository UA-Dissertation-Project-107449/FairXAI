"""Profiling-sensitivity study - how dataset profiling responds to controlled knobs.

Generates a grid of synthetic datasets (two tiers, varied missingness / class
imbalance / cardinality-type-mix / size & difficulty), runs FairXAI profiling on
each, and scores the observed semantic types against the generator's ground
truth. The grid doubles as the test bed for the categorical-vs-continuous
type-inference fix.

Outputs land under::

    output/<pipeline>/studies/profiling_sensitivity/<study_id>/
        datasets/    generated CSVs (with NaNs) + <id>.meta.json
        profiles/    raw characterize_dataset JSON per dataset
        figures/     (written by generate_profiling_sensitivity_plots.py)
        dataset_results.csv  column_results.csv  type_confusion.csv
        knob_response_summary.csv  study_summary.json  study_manifest.json

Usage
-----
python scripts/studies/run_profiling_sensitivity_study.py --grid-size smoke -v
python scripts/studies/run_profiling_sensitivity_study.py --pipeline synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, setup_study_logging
from fairxai.cli.runner_utils import (
    resolve_run_id,
    update_output_study_pointer,
    update_study_pointer,
)
from fairxai.data.synthetic import (
    SyntheticConfig,
    build_grid,
    build_smoke_grid,
    generate,
    write_dataset,
    write_grid_manifest,
)
from fairxai.profiling import domain_characterization as dc

logger = logging.getLogger(__name__)

STUDY_TYPE = "profiling_sensitivity"
SENSITIVE_COLUMNS = ["sex", "age_group", "race"]


def _knob_value(cfg: SyntheticConfig) -> Any:
    """The parameter varied for this config's sweep (for knob-response plots)."""
    return {
        "missingness": cfg.missing_pct,
        "imbalance": cfg.minority_ratio,
        "separability": cfg.class_sep,
        "size": cfg.n_samples,
        "cardinality": cfg.lowcard_levels,
    }.get(cfg.label, "baseline")


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median (numeric) / mode (other) imputation so complexity metrics run."""
    out = df.copy()
    for column in out.columns:
        if not out[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].fillna(out[column].median())
        else:
            mode = out[column].mode(dropna=True)
            out[column] = out[column].fillna(mode.iloc[0] if not mode.empty else "missing")
    return out


def _merge_raw_missingness(result: dict, raw_csv: Path) -> dict:
    """Overlay column profiles + missing fields from the raw (NaN) data.

    Complexity/EBM are computed on the imputed copy, but the reported semantic
    types and missing percentages must reflect the original dataset.
    """
    raw = dc.profile_dataset(str(raw_csv))
    result["column_profiles"] = raw["column_profiles"]
    result["feature_distributions"] = raw["feature_distributions"]
    miss = {p["name"]: p["missing_pct"] for p in raw["column_profiles"]}
    result["missing_percentages"] = miss
    top = max(miss, key=miss.get) if miss else None
    result["top_missing_column"] = top
    result["top_missing_pct"] = miss.get(top) if top else None
    return result


def _characterize_safe(
    df: pd.DataFrame, csv_path: Path, profiles_dir: Path, target_column: str
) -> tuple[dict, str]:
    """Characterize one dataset, handling missing values and absent EBM.

    Returns ``(result, status)``:
      * ``"ok"`` - complete dataset profiled directly.
      * ``"imputed_for_metrics"`` - had NaNs; complexity/EBM computed on an
        imputed copy, missingness/types overlaid from the raw data.
      * ``"ebm_unavailable"`` - EBM/interpret missing; column profiles only.
    """
    has_missing = bool(df.isna().any().any())
    target_csv = csv_path
    status = "ok"

    if has_missing:
        imputed_dir = profiles_dir / ".imputed"
        imputed_dir.mkdir(parents=True, exist_ok=True)
        target_csv = imputed_dir / csv_path.name
        _impute(df).to_csv(target_csv, index=False)
        status = "imputed_for_metrics"

    try:
        result = dc.characterize_dataset(
            filename=str(target_csv),
            output_dir=profiles_dir,
            target_column=target_column,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "ebm" in message or "interpret" in message:
            logger.warning(
                "[WARNING] EBM difficulty unavailable for %s (%s); profiling only.",
                csv_path.name,
                exc,
            )
            result = dc.profile_dataset(str(csv_path))
            (profiles_dir / f"{csv_path.stem}.json").write_text(
                json.dumps(result, indent=2, default=str)
            )
            return result, "ebm_unavailable"
        raise

    if has_missing:
        result = _merge_raw_missingness(result, csv_path)
    return result, status


def _score_columns(
    result: dict, ground_truth: list[dict], cfg: SyntheticConfig
) -> tuple[list[dict], int]:
    """Join profiler output to ground truth; return per-column rows + miss count."""
    profiles = {p["name"]: p for p in result.get("column_profiles", [])}
    rows: list[dict] = []
    misclassified = 0
    for truth in ground_truth:
        observed = profiles.get(truth["name"], {})
        obs_semantic = observed.get("semantic_type")
        obs_inferred = observed.get("inferred_type")
        semantic_ok = obs_semantic == truth["expected_semantic_type"]
        if not semantic_ok:
            misclassified += 1
        rows.append(
            {
                "dataset_id": cfg.dataset_id(),
                "tier": cfg.tier,
                "label": cfg.label,
                "knob_value": _knob_value(cfg),
                "name": truth["name"],
                "role": truth["role"],
                "expected_semantic_type": truth["expected_semantic_type"],
                "observed_semantic_type": obs_semantic,
                "semantic_type_correct": semantic_ok,
                "expected_inferred_type": truth["expected_inferred_type"],
                "observed_inferred_type": obs_inferred,
                "inferred_type_correct": obs_inferred == truth["expected_inferred_type"],
                "n_unique": observed.get("n_unique"),
                "distinct_ratio": observed.get("distinct_ratio"),
                "missing_pct_observed": observed.get("missing_pct"),
                "missing_pct_design": truth["missing_pct_design"],
                "missing_mechanism": truth["missing_mechanism"],
            }
        )
    return rows, misclassified


def _dataset_row(
    result: dict, cfg: SyntheticConfig, status: str, accuracy: float, misclassified: int
) -> dict:
    metrics = result.get("metrics", {})
    row = {**asdict(cfg)}
    row.update(
        {
            "dataset_id": cfg.dataset_id(),
            "knob_value": _knob_value(cfg),
            "status": status,
            "semantic_type_accuracy": round(accuracy, 4),
            "n_columns_misclassified": misclassified,
            "nSamples": metrics.get("nSamples"),
            "nFeatures": metrics.get("nFeatures"),
            "nClasses": metrics.get("nClasses"),
            "ebmDifficulty": metrics.get("ebmDifficulty"),
            "class_balance_label": result.get("class_balance_label"),
            "class_balance_delta": result.get("class_balance_delta"),
            "top_missing_pct": result.get("top_missing_pct"),
            "target_missing_pct": result.get("target_missing_pct"),
        }
    )
    # Flatten complexity metrics (F2Imbalance..BayesImbalance, etc.).
    for key, value in metrics.items():
        if key not in row:
            row[key] = value
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="synthetic", help="Output namespace.")
    parser.add_argument("--grid-size", choices=["smoke", "default"], default="default")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N datasets.")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    project_root = get_project_root(Path(__file__))
    study_id = resolve_run_id()
    setup_study_logging(
        project_root, STUDY_TYPE, study_id, "study.log", args.verbose, log_subdir=args.pipeline
    )
    update_study_pointer(project_root / "logs" / args.pipeline, STUDY_TYPE, study_id, logger)

    study_root = project_root / "output" / args.pipeline / "studies" / STUDY_TYPE / study_id
    datasets_dir = study_root / "datasets"
    profiles_dir = study_root / "profiles"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    configs = build_smoke_grid(args.seed) if args.grid_size == "smoke" else build_grid(args.seed)
    if args.limit:
        configs = configs[: args.limit]
    logger.info("[PHASE] %d datasets (%s grid)", len(configs), args.grid_size)

    manifest_records: list[dict] = []
    column_rows: list[dict] = []
    dataset_rows: list[dict] = []
    confusion: dict[tuple[str, str], int] = {}
    status_counts: dict[str, int] = {}
    n_failed = 0

    for idx, cfg in enumerate(configs, 1):
        dataset_id = cfg.dataset_id()
        logger.info("[RUN] %d/%d %s", idx, len(configs), dataset_id)
        df, ground_truth = generate(cfg)
        csv_path, meta_path = write_dataset(df, cfg, ground_truth, datasets_dir)
        gt_dicts = [asdict(col) for col in ground_truth]
        target_col = "heart_disease" if cfg.tier == "healthcare" else "target"

        try:
            result, status = _characterize_safe(df, csv_path, profiles_dir, target_col)
        except Exception as exc:  # noqa: BLE001 - record and continue
            n_failed += 1
            logger.error("[ERROR] profiling failed for %s: %s", dataset_id, exc)
            manifest_records.append(
                {"dataset_id": dataset_id, "csv": str(csv_path), "status": "failed"}
            )
            continue

        rows, misclassified = _score_columns(result, gt_dicts, cfg)
        column_rows.extend(rows)
        scored = len(rows) or 1
        accuracy = (scored - misclassified) / scored
        dataset_rows.append(_dataset_row(result, cfg, status, accuracy, misclassified))

        for row in rows:
            key = (str(row["expected_semantic_type"]), str(row["observed_semantic_type"]))
            confusion[key] = confusion.get(key, 0) + 1

        status_counts[status] = status_counts.get(status, 0) + 1
        manifest_records.append(
            {
                "dataset_id": dataset_id,
                "csv": str(csv_path),
                "meta": str(meta_path),
                "profile": str(profiles_dir / f"{csv_path.stem}.json"),
                "status": status,
                "semantic_type_accuracy": round(accuracy, 4),
            }
        )

    # Tables
    _write_csv(study_root / "column_results.csv", column_rows)
    _write_csv(study_root / "dataset_results.csv", dataset_rows)
    confusion_rows = [
        {"expected": expected, "observed": observed, "count": count}
        for (expected, observed), count in sorted(confusion.items())
    ]
    _write_csv(study_root / "type_confusion.csv", confusion_rows)
    knob_rows = _build_knob_response(dataset_rows)
    _write_csv(study_root / "knob_response_summary.csv", knob_rows)

    write_grid_manifest(configs, manifest_records, study_root / "grid_manifest.json")

    mean_accuracy = (
        sum(r["semantic_type_accuracy"] for r in dataset_rows) / len(dataset_rows)
        if dataset_rows
        else 0.0
    )
    summary = {
        "study_id": study_id,
        "pipeline": args.pipeline,
        "grid_size": args.grid_size,
        "seed": args.seed,
        "n_datasets": len(configs),
        "status_counts": status_counts,
        "n_failed": n_failed,
        "mean_semantic_type_accuracy": round(mean_accuracy, 4),
    }
    (study_root / "study_summary.json").write_text(json.dumps(summary, indent=2))
    (study_root / "study_manifest.json").write_text(
        json.dumps({"study_id": study_id, "study_root": str(study_root), **summary}, indent=2)
    )
    update_output_study_pointer(project_root / "output" / args.pipeline, STUDY_TYPE, study_id)

    logger.info("[SUCCESS] %s", json.dumps(summary))
    print(json.dumps(summary, indent=2))
    return 1 if n_failed else 0


def _build_knob_response(dataset_rows: list[dict]) -> list[dict]:
    """Tidy long-form rows: tier, knob, knob_value, metric, value."""
    metrics = [
        "semantic_type_accuracy",
        "ebmDifficulty",
        "top_missing_pct",
        "class_balance_delta",
    ]
    rows: list[dict] = []
    for record in dataset_rows:
        for metric in metrics:
            value = record.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "tier": record["tier"],
                    "knob": record["label"],
                    "knob_value": record["knob_value"],
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
