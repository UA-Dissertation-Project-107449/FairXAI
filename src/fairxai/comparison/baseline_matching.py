"""Baseline matching primitives shared by comparison tables and plots."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def normalize_sensitive_attr(attr: object) -> str:
    """Normalize attribute names across baseline-assess and experiment JSONs."""
    attr_str = str(attr)
    if attr_str.endswith("_cat"):
        return attr_str[: -len("_cat")]
    return attr_str


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def baseline_key_from_row(row, include_variant: bool = True) -> tuple:
    key = (
        row["dataset"],
        row.get("model_type", "logistic_regression"),
        row["binning_strategy"],
        row["training_method"],
    )
    if include_variant:
        return key[:2] + (row.get("model_variant", "default"),) + key[2:]
    return key


def build_baseline_lookups(df: pd.DataFrame) -> tuple[dict[tuple, pd.Series], dict[tuple, pd.Series]]:
    """Build exact and no-variant baseline lookup maps."""
    exact: dict[tuple, pd.Series] = {}
    no_variant: dict[tuple, pd.Series] = {}
    if df is None or df.empty or "mitigation_technique" not in df.columns:
        return exact, no_variant

    baseline_rows = df[df["mitigation_technique"].astype(str) == "baseline"]
    for _, row in baseline_rows.iterrows():
        exact[baseline_key_from_row(row, include_variant=True)] = row
        no_variant.setdefault(baseline_key_from_row(row, include_variant=False), row)
    return exact, no_variant


def find_matching_baseline(
    row,
    exact_lookup: dict[tuple, pd.Series],
    no_variant_lookup: dict[tuple, pd.Series],
) -> tuple[pd.Series | None, str | None]:
    """Find baseline for an experiment row; exact variant match wins."""
    exact_key = baseline_key_from_row(row, include_variant=True)
    if exact_key in exact_lookup:
        return exact_lookup[exact_key], "combinatorial_exact"

    fallback_key = baseline_key_from_row(row, include_variant=False)
    if fallback_key in no_variant_lookup:
        return no_variant_lookup[fallback_key], "combinatorial_no_variant"

    return None, None

