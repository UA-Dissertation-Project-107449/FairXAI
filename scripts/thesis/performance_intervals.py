"""Bootstrap intervals on the performance numbers Chapter 6 quotes, from saved predictions.

Two tables, both post hoc from the runs in runs.yaml (no retrain):

- baseline_performance.csv: F1, accuracy, precision, recall and AUC of every
  stage-7 baseline on its held-out test split, each with a percentile interval
  (bootstrap_performance_metrics).
- ablation_paired.csv: every feature-selection mode against exclude_sensitive,
  paired over the shared test rows (paired_arm_differences). The fits differ,
  so the interval covers test-sample noise only, not refit variance.

Replicates follow adaptive_bootstrap_replicates unless --n-boot is given.

    python3 scripts/thesis/performance_intervals.py [--out output/thesis] [--n-boot 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from thesis_runs import (
    CARD,
    DEFAULT_OUT,
    RUN_C70,
    RUN_C70_SUB,
    RUN_CARDIAC,
    UCI,
    UCI_MODELS,
    study_dir,
)

from fairxai.fairness.uncertainty import (
    adaptive_bootstrap_replicates,
    bootstrap_performance_metrics,
    paired_arm_differences,
)

SENSITIVE = ["age_group", "sex"]
REFERENCE = "exclude_sensitive"
SOURCES = ((RUN_CARDIAC, UCI), (RUN_C70_SUB, ["cardio70k"]), (RUN_C70, ["cardio70k"]))


def baseline_table(n_boot: int | None, n_jobs: int) -> pd.DataFrame:
    """Single-arm intervals for every baseline model on its test split."""
    parts = []
    for run, cohorts in SOURCES:
        pred = CARD / "runs" / run / "baseline" / "results" / "predictions"
        for cohort in cohorts:
            for model in UCI_MODELS:
                path = pred / f"{cohort}_{model}_test.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                res = bootstrap_performance_metrics(df, SENSITIVE, n_boot=n_boot, n_jobs=n_jobs)
                parts.append(res.table.assign(run=run, cohort=cohort, model=model))
                print(f"baseline {run[-6:]} {cohort} {model}: n={len(df)}")
    return pd.concat(parts, ignore_index=True)


def ablation_table(n_boot: int | None, n_jobs: int) -> pd.DataFrame:
    """Paired change of every ablation mode against the reference mode."""
    parts = []
    for run, _ in SOURCES:
        runs = study_dir("feature_selection", run) / "runs"
        for ref_dir in sorted(runs.glob(f"fs_{REFERENCE}__*")):
            model = ref_dir.name.split("__")[1]
            pred = Path("baseline") / "results" / "predictions"
            for ref_path in sorted((ref_dir / pred).glob("*_test.csv")):
                cohort = ref_path.name.replace(f"_{model}_test.csv", "")
                ref = pd.read_csv(ref_path)
                boots = n_boot or adaptive_bootstrap_replicates(len(ref))
                for arm_dir in sorted(runs.glob(f"fs_*__{model}")):
                    mode = arm_dir.name[3:].split("__")[0]
                    arm_path = arm_dir / pred / ref_path.name
                    if mode == REFERENCE or not arm_path.exists():
                        continue
                    res = paired_arm_differences(
                        ref,
                        pd.read_csv(arm_path),
                        SENSITIVE,
                        n_boot=boots,
                        n_jobs=n_jobs,
                        baseline_label=REFERENCE,
                        arm_label=mode,
                    )
                    parts.append(res.table.assign(run=run, cohort=cohort, model=model, mode=mode))
                    print(f"ablation {run[-6:]} {cohort} {model} {mode}: n={len(ref)} B={boots}")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=None, help="override the adaptive count")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("baseline_performance", baseline_table),
        ("ablation_paired", ablation_table),
    ):
        path = args.out / f"{name}.csv"
        build(args.n_boot, args.n_jobs).to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
