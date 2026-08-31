#!/usr/bin/env python3
"""Null-calibration study for the fairness bootstrap.

Answers a question the unit tests cannot: when a cohort contains *no* disparity
at all, how often does the assessment nevertheless report one? Predictions are
generated independently of every sensitive attribute, so every true group
difference is exactly zero and every reported finding is a false one.

The study exists because the obvious reading of a bootstrap result — "the
interval on the parity gap excludes zero, therefore there is a disparity" — is
invalid for the ``max(rate) - min(rate)`` statistic that most parity metrics
report. That statistic is bounded below by zero and biased upward under
resampling. This script measures how badly, and confirms that the pairwise
difference table does not inherit the problem.

It also compares the three resampling schemes, so the default is a measured
choice rather than an assumed one.

Usage:
    python3 scripts/studies/run_bootstrap_calibration.py
    python3 scripts/studies/run_bootstrap_calibration.py --seeds 50 --n-boot 2000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.fairness.uncertainty import (  # noqa: E402
    STRATIFY_GROUP,
    STRATIFY_GROUP_OUTCOME,
    STRATIFY_NONE,
    bootstrap_fairness_metrics,
)

SCHEMES = (STRATIFY_GROUP_OUTCOME, STRATIFY_GROUP, STRATIFY_NONE)
SENSITIVE = ["sex", "age_group"]


def null_cohort(n: int, seed: int) -> pd.DataFrame:
    """A cohort whose predictions are independent of every sensitive attribute.

    Scores are drawn without reference to sex or age band, so the population
    value of every group difference is zero by construction. Any finding the
    assessment reports on this frame is a false positive.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "sex": rng.choice(["f", "m"], n),
            "age_group": rng.choice(["<50", "50-60", ">60"], n),
        }
    )
    df["y_proba"] = rng.random(n)
    df["y_pred"] = (df["y_proba"] > 0.5).astype(int)
    return df


def evaluate_scheme(scheme: str, seeds: int, n_rows: int, n_boot: int, alpha: float) -> dict:
    """Run the null cohort through one resampling scheme and count false findings."""
    unadjusted = adjusted = comparisons = 0
    runs_with_a_finding = 0
    gap_excluding_zero = gap_total = 0

    for seed in range(seeds):
        result = bootstrap_fairness_metrics(
            null_cohort(n_rows, seed),
            SENSITIVE,
            n_boot=n_boot,
            alpha=alpha,
            stratify=scheme,
            random_state=seed,
        )
        pairwise = result.pairwise
        unadjusted += int((pairwise["p_value"] < alpha).sum())
        adjusted += int(pairwise["significant"].sum())
        comparisons += len(pairwise)
        runs_with_a_finding += int(pairwise["significant"].any())

        gaps = result.table[result.table["quantity"].str.contains("difference")]
        gap_excluding_zero += int((~gaps["includes_zero"]).sum())
        gap_total += len(gaps)

    return {
        "stratify": scheme,
        "seeds": seeds,
        "n_rows": n_rows,
        "n_boot": n_boot,
        "alpha": alpha,
        "comparisons": comparisons,
        "false_rate_unadjusted": unadjusted / comparisons if comparisons else np.nan,
        "false_rate_adjusted": adjusted / comparisons if comparisons else np.nan,
        "runs_with_a_false_finding": runs_with_a_finding,
        "max_gap_ci_excludes_zero": gap_excluding_zero / gap_total if gap_total else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Null-calibration study for the fairness bootstrap"
    )
    parser.add_argument("--seeds", type=int, default=20, help="Independent null cohorts to draw")
    parser.add_argument("--n-rows", type=int, default=400, help="Rows per cohort")
    parser.add_argument("--n-boot", type=int, default=1200, help="Bootstrap replicates per cohort")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided level")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path for the results table",
    )
    args = parser.parse_args()

    # The module warns per run about small strata and p-value resolution; on a
    # sweep those are expected and would drown the results.
    logging.disable(logging.WARNING)

    rows = [
        evaluate_scheme(scheme, args.seeds, args.n_rows, args.n_boot, args.alpha)
        for scheme in SCHEMES
    ]
    table = pd.DataFrame(rows)

    print(
        table.to_string(
            index=False,
            formatters={
                "false_rate_unadjusted": "{:.2%}".format,
                "false_rate_adjusted": "{:.2%}".format,
                "max_gap_ci_excludes_zero": "{:.1%}".format,
            },
        )
    )
    print(
        f"\nNominal per-comparison rate is {args.alpha:.0%}. A max-gap interval near 100% is the "
        "expected failure this study documents: that statistic cannot be read as a test."
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
