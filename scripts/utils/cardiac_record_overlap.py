#!/usr/bin/env python3
"""Measure fingerprint overlap between two standardized cardiac datasets.

Chapter-3 P0 (lines 285, 450): Cleveland is one of the five UCI sources merged
into the Kaggle Heart compilation, so the standalone Cleveland file and the
Kaggle file cannot be assumed statistically independent. This quantifies a
six-field fingerprint intersection so comparisons and aggregate claims can be
caveated.

Matching is done on continuous/stable clinical columns that survive schema
harmonisation unchanged (age, sex, resting BP, cholesterol, max heart rate, ST
depression). Categorical re-encodings (cp, slope) and imputed zeros are excluded
from the key because they differ across source curations.

Run (from Code/FairXAI/, venv active):

    python3 scripts/utils/cardiac_record_overlap.py
    python3 scripts/utils/cardiac_record_overlap.py --a <fileA.csv> --b <fileB.csv>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_A = Path("data/raw/cardiac/cleveland_standardized.csv")
DEFAULT_B = Path("data/raw/cardiac/kaggle_heart_standardized.csv")

# Standardized column names shared across the harmonised cardiac files.
KEY_COLS = ["age_raw", "sex_bin", "trestbps", "chol", "thalach", "oldpeak"]


def _key_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    k = df[cols].copy()
    k["oldpeak"] = pd.to_numeric(k["oldpeak"], errors="coerce").round(1)
    for c in cols:
        if c == "oldpeak":
            continue
        k[c] = pd.to_numeric(k[c], errors="coerce").round().astype("Int64")
    return k


def _keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return _key_frame(df, cols).apply(
        lambda row: tuple(None if pd.isna(value) else value for value in row),
        axis=1,
    )


def overlap(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> dict:
    count_a = Counter(_keys(a, cols))
    count_b = Counter(_keys(b, cols))
    shared = count_a & count_b
    matched_pairs = sum(shared.values())

    return {
        "fingerprint_columns": cols,
        "matching_method": "one-to-one multiset fingerprint matching",
        "a_rows": int(len(a)),
        "b_rows": int(len(b)),
        "a_unique_fingerprints": len(count_a),
        "b_unique_fingerprints": len(count_b),
        "intersection_unique_fingerprints": len(shared),
        "matched_row_pairs": matched_pairs,
        "a_pct_matched": round(100 * matched_pairs / len(a), 1) if len(a) else 0.0,
        "b_pct_matched": round(100 * matched_pairs / len(b), 1) if len(b) else 0.0,
    }


def print_report(a_path: Path, b_path: Path, r: dict) -> None:
    print("=== Cardiac record overlap ===\n")
    print(f"A: {a_path}  ({r['a_rows']} rows)")
    print(f"B: {b_path}  ({r['b_rows']} rows)")
    print(f"Fingerprint columns: {', '.join(r['fingerprint_columns'])}\n")
    print(
        f"Unique fingerprints A / B   : {r['a_unique_fingerprints']} / {r['b_unique_fingerprints']}"
    )
    print(f"Shared unique fingerprints  : {r['intersection_unique_fingerprints']}")
    print(f"One-to-one matched row pairs : {r['matched_row_pairs']}")
    print(f"Matched share of A / B       : {r['a_pct_matched']}% / {r['b_pct_matched']}%")
    print(
        "\nInterpretation: A's matched share estimates reuse in the compilation. "
        "The six fields are a fingerprint, not proof of exact record or patient "
        "identity; duplicates are paired at most once."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, default=DEFAULT_A)
    p.add_argument("--b", type=Path, default=DEFAULT_B)
    p.add_argument("--keys", nargs="+", default=KEY_COLS, help="Override key columns.")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    for label, path in (("A", args.a), ("B", args.b)):
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 2

    a = pd.read_csv(args.a)
    b = pd.read_csv(args.b)
    missing = [c for c in args.keys if c not in a.columns or c not in b.columns]
    if missing:
        print(
            f"ERROR: key columns absent from one file: {missing}\n"
            f"A columns: {list(a.columns)}\nB columns: {list(b.columns)}",
            file=sys.stderr,
        )
        return 2

    r = overlap(a, b, args.keys)
    print_report(args.a, args.b, r)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(r, indent=2))
        print(f"\nWrote JSON summary -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
