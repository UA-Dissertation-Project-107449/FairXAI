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

Three tables come out of it. The summary carries one row per resampling scheme
and replicate count, pooled over metrics, and is what Section 5.5.6 quotes. The
per-metric table splits the same counters by metric, because a false-finding
rate pooled across metrics hides which metric produced the findings. The third
runs the paired arm test on two exchangeable arms, which is the null for
"this technique changed something".

Usage:
    python3 scripts/studies/run_bootstrap_calibration.py
    python3 scripts/studies/run_bootstrap_calibration.py --seeds 50 --n-boot 2000
    python3 scripts/studies/run_bootstrap_calibration.py --n-boot 1000 2000 4000
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.fairness.uncertainty import (  # noqa: E402
    STRATIFY_GROUP,
    STRATIFY_GROUP_OUTCOME,
    STRATIFY_NONE,
    bootstrap_fairness_metrics,
    paired_arm_differences,
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


def evaluate_scheme(
    scheme: str, seeds: int, n_rows: int, n_boot: int, alpha: float
) -> Tuple[dict, List[dict]]:
    """Run the null cohort through one resampling scheme and count false findings.

    Returns the pooled summary row and one row per metric. Both carry the same
    counters; the per-metric rows exist because pooling hides which metric a
    false finding came from, and Appendix C reports that breakdown.
    """
    unadjusted = adjusted = comparisons = untested = 0
    runs_with_a_finding = 0
    gap_excluding_zero = gap_total = 0
    per_metric: Dict[str, Counter] = defaultdict(Counter)

    for seed in range(seeds):
        result = bootstrap_fairness_metrics(
            null_cohort(n_rows, seed),
            SENSITIVE,
            n_boot=n_boot,
            alpha=alpha,
            stratify=scheme,
            random_state=seed,
        )
        # Only tested comparisons count: an untested row carries a raw p-value
        # but was never a question the data could answer, so including it would
        # measure the wrong denominator in both directions.
        pairwise = result.pairwise
        tested = pairwise[pairwise["tested"]]
        unadjusted += int((tested["p_value"] < alpha).sum())
        adjusted += int(tested["significant"].sum())
        comparisons += len(tested)
        untested += int((~pairwise["tested"]).sum())
        runs_with_a_finding += int(pairwise["significant"].any())

        gaps = result.table[result.table["quantity"].str.contains("difference")]
        gap_excluding_zero += int((~gaps["includes_zero"]).sum())
        gap_total += len(gaps)

        # Same counters, split by metric. The denominators differ per metric
        # because not every metric yields a testable comparison on every draw.
        for metric, block in tested.groupby("metric", sort=False):
            counts = per_metric[str(metric)]
            counts["unadjusted"] += int((block["p_value"] < alpha).sum())
            counts["adjusted"] += int(block["significant"].sum())
            counts["comparisons"] += len(block)
        for metric, block in pairwise[~pairwise["tested"]].groupby("metric", sort=False):
            per_metric[str(metric)]["untested"] += len(block)
        for metric, block in gaps.groupby("metric", sort=False):
            counts = per_metric[str(metric)]
            counts["gap_excluding_zero"] += int((~block["includes_zero"]).sum())
            counts["gap_total"] += len(block)

    summary = {
        "stratify": scheme,
        "seeds": seeds,
        "n_rows": n_rows,
        "n_boot": n_boot,
        "alpha": alpha,
        "comparisons": comparisons,
        "untested": untested,
        "false_rate_unadjusted": unadjusted / comparisons if comparisons else np.nan,
        "false_rate_adjusted": adjusted / comparisons if comparisons else np.nan,
        "runs_with_a_false_finding": runs_with_a_finding,
        "max_gap_ci_excludes_zero": gap_excluding_zero / gap_total if gap_total else np.nan,
    }

    metric_rows = []
    for metric in sorted(per_metric):
        counts = per_metric[metric]
        n_comparisons = counts["comparisons"]
        n_gaps = counts["gap_total"]
        metric_rows.append(
            {
                "stratify": scheme,
                "metric": metric,
                "seeds": seeds,
                "n_rows": n_rows,
                "n_boot": n_boot,
                "alpha": alpha,
                "comparisons": n_comparisons,
                "untested": counts["untested"],
                "false_rate_unadjusted": (
                    counts["unadjusted"] / n_comparisons if n_comparisons else np.nan
                ),
                "false_rate_adjusted": (
                    counts["adjusted"] / n_comparisons if n_comparisons else np.nan
                ),
                "max_gap_ci_excludes_zero": (
                    counts["gap_excluding_zero"] / n_gaps if n_gaps else np.nan
                ),
            }
        )
    return summary, metric_rows


def exchangeable_arms(n: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two arms that predict the same rows equally well, and differ only by noise.

    The paired test asks whether a technique changed anything. Here nothing was
    changed: both arms score the same cohort with independent draws from one
    process, so every paired difference has population value zero. This is the
    scenario the max-gap interval cannot survive — a gap is biased upward under
    resampling — and the one the paired difference should, because both arms
    carry that bias and it cancels.
    """
    baseline = null_cohort(n, seed)
    arm = baseline.copy()
    rng = np.random.default_rng(seed + 10_000)
    arm["y_proba"] = rng.random(n)
    arm["y_pred"] = (arm["y_proba"] > 0.5).astype(int)
    return baseline, arm


def evaluate_paired(seeds: int, n_rows: int, n_boot: int, alpha: float) -> dict:
    """Count false findings from the paired test on exchangeable arms."""
    unadjusted = adjusted = quantities = 0
    runs_with_a_finding = 0
    gap_excluding_zero = gap_total = 0

    for seed in range(seeds):
        baseline, arm = exchangeable_arms(n_rows, seed)
        table = paired_arm_differences(
            baseline,
            arm,
            SENSITIVE,
            n_boot=n_boot,
            alpha=alpha,
            random_state=seed,
        ).table
        tested = table[table["p_value"].notna() & ~table["degenerate"]]
        unadjusted += int((tested["p_value"] < alpha).sum())
        adjusted += int(tested["significant"].sum())
        quantities += len(tested)
        runs_with_a_finding += int(table["significant"].any())

        gaps = tested[tested["quantity"].str.contains("difference")]
        gap_excluding_zero += int(gaps["excludes_zero"].sum())
        gap_total += len(gaps)

    return {
        "scenario": "paired_exchangeable_arms",
        "seeds": seeds,
        "n_rows": n_rows,
        "n_boot": n_boot,
        "alpha": alpha,
        "quantities": quantities,
        "false_rate_unadjusted": unadjusted / quantities if quantities else np.nan,
        "false_rate_adjusted": adjusted / quantities if quantities else np.nan,
        "runs_with_a_false_finding": runs_with_a_finding,
        "gap_ci_excludes_zero": gap_excluding_zero / gap_total if gap_total else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Null-calibration study for the fairness bootstrap"
    )
    parser.add_argument("--seeds", type=int, default=20, help="Independent null cohorts to draw")
    parser.add_argument("--n-rows", type=int, default=400, help="Rows per cohort")
    parser.add_argument(
        "--n-boot",
        type=int,
        nargs="+",
        default=[4000],
        help=(
            "Bootstrap replicates per cohort. Accepts several values to sweep the "
            "replicate count; the default matches the count the cardiac results run at."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided level")
    parser.add_argument(
        "--skip-paired",
        action="store_true",
        help="Skip the paired scenario (exchangeable arms) and run the single-arm study only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path for the summary table; the per-metric table lands beside it",
    )
    args = parser.parse_args()

    # The module warns per run about small strata and p-value resolution; on a
    # sweep those are expected and would drown the results.
    logging.disable(logging.WARNING)

    summary_rows = []
    metric_rows = []
    for n_boot in args.n_boot:
        for scheme in SCHEMES:
            summary, per_metric = evaluate_scheme(
                scheme, args.seeds, args.n_rows, n_boot, args.alpha
            )
            summary_rows.append(summary)
            metric_rows.extend(per_metric)

    table = pd.DataFrame(summary_rows)
    by_metric = pd.DataFrame(metric_rows)

    rate_formatters = {
        "false_rate_unadjusted": "{:.2%}".format,
        "false_rate_adjusted": "{:.2%}".format,
        "max_gap_ci_excludes_zero": "{:.1%}".format,
    }
    print(table.to_string(index=False, formatters=rate_formatters))
    print(
        f"\nNominal per-comparison rate is {args.alpha:.0%}. A max-gap interval near 100% is the "
        "expected failure this study documents: that statistic cannot be read as a test."
    )

    print("\nPer-metric detail:")
    print(by_metric.to_string(index=False, formatters=rate_formatters))

    paired = pd.DataFrame()
    if not args.skip_paired:
        paired = pd.DataFrame(
            [evaluate_paired(args.seeds, args.n_rows, n_boot, args.alpha) for n_boot in args.n_boot]
        )
        print("\nPaired test on exchangeable arms:")
        print(
            paired.to_string(
                index=False,
                formatters={
                    "false_rate_unadjusted": "{:.2%}".format,
                    "false_rate_adjusted": "{:.2%}".format,
                    "gap_ci_excludes_zero": "{:.1%}".format,
                },
            )
        )
        print(
            "\nThe gap column is the point of this scenario: the same max-gap statistic whose "
            "single-arm interval never contains zero is testable once it is differenced against "
            "a paired baseline."
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        metric_out = out.with_name(f"{out.stem}_by_metric{out.suffix}")
        by_metric.to_csv(metric_out, index=False)
        saved = [out, metric_out]
        if not paired.empty:
            paired_out = out.with_name(f"{out.stem}_paired{out.suffix}")
            paired.to_csv(paired_out, index=False)
            saved.append(paired_out)
        print("\nSaved to " + ", ".join(str(path) for path in saved))


if __name__ == "__main__":
    main()
