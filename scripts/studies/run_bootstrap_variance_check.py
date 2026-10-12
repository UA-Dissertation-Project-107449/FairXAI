#!/usr/bin/env python3
"""Does each resampling scheme get the spread of a pairwise difference right?

Follow-up to run_bootstrap_calibration.py. On null cohorts, predictive parity
reports too many false findings before adjustment, but only under the
group x outcome scheme. The suspected cause: that scheme fixes each group's
outcome prevalence in every replicate, predictive parity depends on
prevalence, so the bootstrap leaves out part of the metric's sampling variance.

The check compares two spreads of every pairwise difference. The true one is
the standard deviation of the point difference across independent null
cohorts. The bootstrap one is the replicate standard error each scheme reports
on a single cohort. A ratio below one means the scheme's intervals are too
narrow. The cause holds if the ratio falls below one for predictive parity
under group x outcome and nowhere else.

Two tables come out: one row per cohort, scheme and comparison, and a summary
per scheme, metric and quantity with the ratio, the raw false-finding rate and
the rate a normal test on the reported standard error would give.

Usage:
    python3 scripts/studies/run_bootstrap_variance_check.py
    python3 scripts/studies/run_bootstrap_variance_check.py --seeds 300 --n-boot 400
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_bootstrap_calibration import SCHEMES, SENSITIVE, null_cohort  # noqa: E402

from fairxai.fairness.uncertainty import bootstrap_fairness_metrics  # noqa: E402

KEY = ["attribute", "metric", "quantity", "group_a", "group_b"]


def collect(seeds: int, n_rows: int, n_boot: int, alpha: float) -> pd.DataFrame:
    """Pairwise rows of every scheme on every null cohort, tested rows only."""
    frames = []
    for seed in range(seeds):
        cohort = null_cohort(n_rows, seed)
        for scheme in SCHEMES:
            result = bootstrap_fairness_metrics(
                cohort,
                SENSITIVE,
                n_boot=n_boot,
                alpha=alpha,
                stratify=scheme,
                random_state=seed,
            )
            rows = result.pairwise[result.pairwise["tested"]].copy()
            rows["seed"] = seed
            rows["scheme"] = scheme
            frames.append(rows[["seed", "scheme", *KEY, "difference", "se", "p_value"]])
        if (seed + 1) % 10 == 0:
            logging.info("%d/%d cohorts", seed + 1, seeds)
    return pd.concat(frames, ignore_index=True)


def summarise(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Bootstrap spread against true spread, per scheme, metric and quantity.

    The true spread of a comparison is taken across cohorts from one scheme's
    rows; the point difference does not depend on the scheme, only which
    comparisons were tested does.
    """
    z = float(np.abs(np.quantile(np.random.default_rng(0).standard_normal(200_000), alpha / 2)))
    per_key = (
        rows.groupby(["scheme", *KEY])
        .agg(
            true_sd=("difference", "std"),
            boot_se=("se", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            cohorts=("seed", "nunique"),
        )
        .reset_index()
    )
    per_key["ratio"] = per_key["boot_se"] / per_key["true_sd"]
    rows = rows.assign(
        raw_finding=rows["p_value"] < alpha,
        normal_finding=(rows["difference"].abs() / rows["se"]) > z,
    )
    rates = (
        rows.groupby(["scheme", "metric", "quantity"])
        .agg(
            comparisons=("p_value", "size"),
            raw_rate=("raw_finding", "mean"),
            normal_rate=("normal_finding", "mean"),
        )
        .reset_index()
    )
    ratios = (
        per_key.groupby(["scheme", "metric", "quantity"])
        .agg(
            ratio_mean=("ratio", "mean"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
            keys=("ratio", "size"),
        )
        .reset_index()
    )
    return ratios.merge(rates, on=["scheme", "metric", "quantity"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=150, help="Independent null cohorts")
    parser.add_argument("--n-rows", type=int, default=400, help="Rows per cohort")
    parser.add_argument("--n-boot", type=int, default=200, help="Replicates per cohort")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided level")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/cardiac/studies/null_calibration/variance_check"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The p-value resolution warning repeats on every call and is about the
    # adjusted column, which this check does not read.
    logging.getLogger().addFilter(lambda r: "under-resolved" not in r.getMessage())

    rows = collect(args.seeds, args.n_rows, args.n_boot, args.alpha)
    summary = summarise(rows, args.alpha)
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "variance_check_rows.csv", index=False)
    summary.to_csv(args.out / "variance_check_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"wrote {args.out} (seeds={args.seeds}, B={args.n_boot}, n={args.n_rows})")


if __name__ == "__main__":
    main()
