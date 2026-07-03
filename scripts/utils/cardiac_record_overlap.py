#!/usr/bin/env python3
"""Measure record-level overlap between two standardized cardiac datasets.

Chapter-3 P0 (lines 285, 450): Cleveland is one of the five UCI sources merged
into the Kaggle Heart compilation, so the standalone Cleveland file and the
Kaggle file cannot be assumed statistically independent. This quantifies the
exact intersection so comparisons and aggregate claims can be caveated.

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
    return _key_frame(df, cols).apply(lambda r: tuple(None if pd.isna(v) else v for v in r), axis=1)


def overlap(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> dict:
    ka = _keys(a, cols)
    kb = _keys(b, cols)
    set_a, set_b = set(ka), set(kb)
    inter = set_a & set_b

    # Count how many rows in each file fall in the intersection (duplicates
    # within a file mean row-count and unique-key-count can differ).
    a_rows_in = int(ka.isin(inter).sum())
    b_rows_in = int(kb.isin(inter).sum())

    return {
        "key_columns": cols,
        "a_rows": int(len(a)),
        "b_rows": int(len(b)),
        "a_unique_keys": len(set_a),
        "b_unique_keys": len(set_b),
        "intersection_unique_keys": len(inter),
        "a_rows_in_intersection": a_rows_in,
        "b_rows_in_intersection": b_rows_in,
        "a_pct_in_intersection": round(100 * a_rows_in / len(a), 1),
        "b_pct_in_intersection": round(100 * b_rows_in / len(b), 1),
    }


def print_report(a_path: Path, b_path: Path, r: dict) -> None:
    print("=== Cardiac record overlap ===\n")
    print(f"A: {a_path}  ({r['a_rows']} rows)")
    print(f"B: {b_path}  ({r['b_rows']} rows)")
    print(f"Key columns: {', '.join(r['key_columns'])}\n")
    print(f"Unique keys A / B          : {r['a_unique_keys']} / {r['b_unique_keys']}")
    print(f"Intersection (unique keys) : {r['intersection_unique_keys']}")
    print(
        f"A rows inside intersection : {r['a_rows_in_intersection']} "
        f"({r['a_pct_in_intersection']}% of A)"
    )
    print(
        f"B rows inside intersection : {r['b_rows_in_intersection']} "
        f"({r['b_pct_in_intersection']}% of B)"
    )
    print(
        "\nInterpretation: A's share inside the intersection estimates how much of "
        "the standalone file is re-used in the compilation; treat the two as "
        "non-independent to that degree."
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
