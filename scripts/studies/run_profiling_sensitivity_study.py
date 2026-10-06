"""Profiling-sensitivity study - how dataset profiling responds to controlled knobs.

Generates a grid of synthetic datasets (two tiers, varied missingness / class
imbalance / cardinality-type-mix / size & difficulty), runs FairXAI profiling on
each, and scores the observed semantic types against the generator's ground
truth. The grid doubles as the test bed for the categorical-vs-continuous
type-inference fix.

Seeding is paired and repeated: replicate ``r`` builds the whole grid with seed
``--seed + r``, so within a replicate every condition shares its baseline's
random draws and the knob is the only difference. Each knob's response is then
reported as the paired delta against the same replicate's baseline, summarised
over replicates (mean, sd, 95% t-interval).

Datasets are profiled exactly as the product profiles an upload: the raw CSV
(NaNs included) goes to ``characterize_dataset``, which imputes internally for
the complexity metrics only.

Outputs land under::

    output/<pipeline>/studies/profiling_sensitivity/<study_id>/
        datasets/    generated CSVs (with NaNs) + <id>.meta.json (replicate 0)
        profiles/    raw characterize_dataset JSON per dataset (replicate 0)
        figures/     (written by generate_profiling_sensitivity_plots.py)
        dataset_results.csv  column_results.csv  type_confusion.csv
        paired_deltas.csv  knob_response_summary.csv
        study_summary.json  study_manifest.json  grid_manifest.json

Every replicate is regenerable from its seed, so only replicate 0 keeps its
CSVs and profile JSONs unless ``--keep-all-artifacts`` is given.

Usage
-----
python scripts/studies/run_profiling_sensitivity_study.py --grid-size smoke -v
python scripts/studies/run_profiling_sensitivity_study.py --pipeline synthetic
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

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
BASELINE_CONDITION = "base"

# Every per-dataset quantity whose response to the knobs is reported.
RESPONSE_METRICS = [
    "semantic_type_accuracy",
    "ebmDifficulty",
    *dc.EBM_FEATURE_ORDER,
    "top_missing_pct",
    "class_balance_delta",
    "duplicate_pct_observed",
]

# Distributions whose versions can move a profiling number.
VERSIONED_PACKAGES = ["fairxai", "numpy", "pandas", "scikit-learn", "scipy", "interpret"]


def _knob_value(cfg: SyntheticConfig) -> Any:
    """The parameter varied for this config's sweep (for knob-response plots)."""
    return {
        "missingness": cfg.missing_pct,
        "imbalance": cfg.minority_ratio,
        "separability": cfg.class_sep,
        "size": cfg.n_samples,
        "cardinality": cfg.lowcard_levels,
        "duplicates": cfg.duplicate_pct,
    }.get(cfg.label, "baseline")


def _condition(cfg: SyntheticConfig) -> str:
    """Key unique to one condition within a tier (keeps the missingness mechanism)."""
    value = _knob_value(cfg)
    if value == "baseline":
        return BASELINE_CONDITION
    if cfg.label == "missingness":
        return f"missingness_{cfg.missing_mechanism}_{value:g}"
    return f"{cfg.label}_{value:g}"


def _characterize_safe(csv_path: Path, profiles_dir: Path, target_column: str) -> tuple[dict, str]:
    """Characterize one raw dataset through the product path, handling absent EBM.

    Returns ``(result, status)``:
      * ``"ok"`` - full characterization (complexity metrics + EBM difficulty).
      * ``"ebm_unavailable"`` - EBM/interpret missing; column profiles only.
    """
    try:
        result = dc.characterize_dataset(
            filename=str(csv_path),
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
    return result, "ok"


def _score_columns(
    result: dict, ground_truth: list[dict], cfg: SyntheticConfig, replicate: int
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
                "replicate": replicate,
                "dataset_id": cfg.dataset_id(),
                "tier": cfg.tier,
                "label": cfg.label,
                "condition": _condition(cfg),
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
    result: dict,
    cfg: SyntheticConfig,
    replicate: int,
    status: str,
    accuracy: float,
    misclassified: int,
    df: pd.DataFrame,
) -> dict:
    metrics = result.get("metrics", {})
    row_count = result.get("row_count") or len(df)
    duplicate_count = result.get("duplicate_count")
    row = {"replicate": replicate, **asdict(cfg)}  # includes duplicate_pct as the design value
    row.update(
        {
            "dataset_id": cfg.dataset_id(),
            "condition": _condition(cfg),
            "knob_value": _knob_value(cfg),
            "status": status,
            "has_missing": bool(df.isna().any().any()),
            "semantic_type_accuracy": round(accuracy, 4),
            "n_columns_misclassified": misclassified,
            # Share of rows that are copies: d / (1 + d) for d rows copied per original.
            "duplicate_pct_expected": round(cfg.duplicate_pct / (1.0 + cfg.duplicate_pct), 4),
            "duplicate_count": duplicate_count,
            "duplicate_pct_observed": (
                round(duplicate_count / row_count, 4) if duplicate_count is not None else None
            ),
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


def _build_paired_deltas(dataset_rows: list[dict]) -> list[dict]:
    """Long-form rows: each metric of each dataset minus its replicate's baseline."""
    baselines = {
        (r["replicate"], r["tier"]): r for r in dataset_rows if r["condition"] == BASELINE_CONDITION
    }
    rows: list[dict] = []
    for record in dataset_rows:
        baseline = baselines.get((record["replicate"], record["tier"]))
        for metric in RESPONSE_METRICS:
            value = record.get(metric)
            if value is None:
                continue
            base_value = baseline.get(metric) if baseline else None
            rows.append(
                {
                    "replicate": record["replicate"],
                    "seed": record["seed"],
                    "tier": record["tier"],
                    "knob": record["label"],
                    "condition": record["condition"],
                    "knob_value": record["knob_value"],
                    "metric": metric,
                    "value": value,
                    "baseline_value": base_value,
                    "delta": value - base_value if base_value is not None else None,
                }
            )
    return rows


def _mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    """Mean, sd and two-sided 95% t-interval of ``values`` (NaN when undefined)."""
    values = values.dropna().astype(float)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 4
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / n**0.5
    return mean, sd, mean - half, mean + half


def _build_knob_response(delta_rows: list[dict]) -> list[dict]:
    """One row per (tier, condition, metric): level and paired delta over replicates."""
    if not delta_rows:
        return []
    frame = pd.DataFrame(delta_rows)
    keys = ["tier", "knob", "condition", "metric"]
    rows: list[dict] = []
    for key, group in frame.groupby(keys, sort=False):
        mean, sd, low, high = _mean_ci(group["value"])
        d_mean, d_sd, d_low, d_high = _mean_ci(group["delta"])
        rows.append(
            {
                **dict(zip(keys, key)),
                "knob_value": group["knob_value"].iloc[0],
                "n_replicates": int(group["value"].notna().sum()),
                "mean": mean,
                "sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "delta_mean": d_mean,
                "delta_sd": d_sd,
                "delta_ci95_low": d_low,
                "delta_ci95_high": d_high,
            }
        )
    return rows


def _type_accuracy_by_tier(confusion: dict[tuple[str, str, str], int]) -> dict[str, dict]:
    """Scored columns, correct columns and accuracy per tier, over all replicates."""
    out: dict[str, dict] = {}
    for (tier, expected, observed), count in confusion.items():
        entry = out.setdefault(tier, {"columns": 0, "correct": 0})
        entry["columns"] += count
        entry["correct"] += count if expected == observed else 0
    for entry in out.values():
        entry["accuracy"] = round(entry["correct"] / entry["columns"], 4)
    return out


def _git_state(repo: Path) -> dict[str, Any]:
    """Commit, branch and dirty flag of ``repo`` (all ``None`` outside git)."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": bool(_git("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "branch": None, "is_dirty": None}


def _environment(project_root: Path) -> dict[str, Any]:
    """Everything a rerun needs to reproduce the numbers: code, packages, EBM model."""
    packages: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    ebm: dict[str, str | None] = {"path": None, "sha256": None}
    try:
        model_path = dc._resolve_ebm_model_path()
        ebm = {
            "path": str(model_path),
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
    except FileNotFoundError:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git": _git_state(project_root),
        "ebm_model": ebm,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="synthetic", help="Output namespace.")
    parser.add_argument("--grid-size", choices=["smoke", "default"], default="default")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N datasets.")
    parser.add_argument("--seed", type=int, default=20260625, help="Seed of replicate 0.")
    parser.add_argument(
        "--replicates",
        type=int,
        default=20,
        help="Grids to run; replicate r uses seed --seed + r for every condition.",
    )
    parser.add_argument(
        "--keep-all-artifacts",
        action="store_true",
        help="Keep CSVs and profile JSONs of every replicate, not only replicate 0.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")

    project_root = get_project_root(Path(__file__))
    study_id = resolve_run_id()
    setup_study_logging(
        project_root, STUDY_TYPE, study_id, "study.log", args.verbose, log_subdir=args.pipeline
    )
    update_study_pointer(project_root / "logs" / args.pipeline, STUDY_TYPE, study_id, logger)
    # Captured before anything is written, so the dirty flag reflects the code only.
    environment = _environment(project_root)

    study_root = project_root / "output" / args.pipeline / "studies" / STUDY_TYPE / study_id
    scratch_root = study_root / ".scratch"
    study_root.mkdir(parents=True, exist_ok=True)

    builder = build_smoke_grid if args.grid_size == "smoke" else build_grid
    seeds = [args.seed + replicate for replicate in range(args.replicates)]

    all_configs: list[SyntheticConfig] = []
    manifest_records: list[dict] = []
    column_rows: list[dict] = []
    dataset_rows: list[dict] = []
    confusion: dict[tuple[str, str, str], int] = {}
    status_counts: dict[str, int] = {}
    n_failed = 0

    for replicate, seed in enumerate(seeds):
        configs = builder(seed)
        if args.limit:
            configs = configs[: args.limit]
        all_configs.extend(configs)

        keep = replicate == 0 or args.keep_all_artifacts
        artifact_root = study_root if keep else scratch_root
        datasets_dir = artifact_root / "datasets"
        profiles_dir = artifact_root / "profiles"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[PHASE] replicate %d/%d seed=%d: %d datasets (%s grid)",
            replicate + 1,
            len(seeds),
            seed,
            len(configs),
            args.grid_size,
        )

        for idx, cfg in enumerate(configs, 1):
            dataset_id = cfg.dataset_id()
            logger.info("[RUN] %d/%d %s", idx, len(configs), dataset_id)
            df, ground_truth = generate(cfg)
            csv_path, meta_path = write_dataset(df, cfg, ground_truth, datasets_dir)
            gt_dicts = [asdict(col) for col in ground_truth]
            target_col = "heart_disease" if cfg.tier == "healthcare" else "target"

            try:
                result, status = _characterize_safe(csv_path, profiles_dir, target_col)
            except Exception as exc:  # noqa: BLE001 - record and continue
                n_failed += 1
                logger.error("[ERROR] profiling failed for %s: %s", dataset_id, exc)
                manifest_records.append(
                    {
                        "replicate": replicate,
                        "dataset_id": dataset_id,
                        "csv": str(csv_path) if keep else None,
                        "status": "failed",
                    }
                )
                continue

            rows, misclassified = _score_columns(result, gt_dicts, cfg, replicate)
            column_rows.extend(rows)
            scored = len(rows) or 1
            accuracy = (scored - misclassified) / scored
            dataset_rows.append(
                _dataset_row(result, cfg, replicate, status, accuracy, misclassified, df)
            )

            for row in rows:
                key = (
                    cfg.tier,
                    str(row["expected_semantic_type"]),
                    str(row["observed_semantic_type"]),
                )
                confusion[key] = confusion.get(key, 0) + 1

            status_counts[status] = status_counts.get(status, 0) + 1
            manifest_records.append(
                {
                    "replicate": replicate,
                    "dataset_id": dataset_id,
                    "csv": str(csv_path) if keep else None,
                    "meta": str(meta_path) if keep else None,
                    "profile": str(profiles_dir / f"{csv_path.stem}.json") if keep else None,
                    "status": status,
                    "semantic_type_accuracy": round(accuracy, 4),
                }
            )

        if not keep:
            shutil.rmtree(scratch_root, ignore_errors=True)

    # Tables
    _write_csv(study_root / "column_results.csv", column_rows)
    _write_csv(study_root / "dataset_results.csv", dataset_rows)
    confusion_rows = [
        {"tier": tier, "expected": expected, "observed": observed, "count": count}
        for (tier, expected, observed), count in sorted(confusion.items())
    ]
    _write_csv(study_root / "type_confusion.csv", confusion_rows)
    delta_rows = _build_paired_deltas(dataset_rows)
    _write_csv(study_root / "paired_deltas.csv", delta_rows)
    _write_csv(study_root / "knob_response_summary.csv", _build_knob_response(delta_rows))

    write_grid_manifest(all_configs, manifest_records, study_root / "grid_manifest.json")

    accuracies = [r["semantic_type_accuracy"] for r in dataset_rows]
    summary = {
        "study_id": study_id,
        "pipeline": args.pipeline,
        "grid_size": args.grid_size,
        "seeding": "paired",
        "seed": args.seed,
        "replicates": args.replicates,
        "n_datasets": len(all_configs),
        "status_counts": status_counts,
        "n_failed": n_failed,
        "mean_semantic_type_accuracy": (
            round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0
        ),
        "min_semantic_type_accuracy": min(accuracies) if accuracies else 0.0,
        "type_accuracy_by_tier": _type_accuracy_by_tier(confusion),
    }
    (study_root / "study_summary.json").write_text(json.dumps(summary, indent=2))
    manifest = {
        "study_id": study_id,
        "study_root": str(study_root),
        **summary,
        "seeds": seeds,
        "argv": sys.argv if argv is None else ["run_profiling_sensitivity_study.py", *argv],
        "environment": environment,
    }
    (study_root / "study_manifest.json").write_text(json.dumps(manifest, indent=2))
    update_output_study_pointer(project_root / "output" / args.pipeline, STUDY_TYPE, study_id)

    logger.info("[SUCCESS] %s", json.dumps(summary))
    print(json.dumps(summary, indent=2))
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
