#!/usr/bin/env python3
"""Compare the UCI Cleveland source against the working standardized file.

Answers the chapter-3 provenance question (Decision 1): is the working
``cleveland_standardized.csv`` a faithful complete-case derivative of the raw
UCI ``processed.cleveland.data``, and exactly which records / encodings differ?

The working file is known to originate from a curated Kaggle mirror
(``heart_cleveland_upload.csv``) that dropped rows with missing ``ca``/``thal``
and re-encoded some categoricals. This script quantifies that transformation so
the dissertation can cite raw UCI provenance with evidence, and decide whether
to switch the pipeline onto the raw UCI file.

Run (from Code/FairXAI/, venv active):

    python3 scripts/utils/cleveland_provenance.py

Outputs a human-readable report to stdout and, with --json, a machine-readable
summary for archival alongside the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Raw UCI processed.cleveland.data: 14 used attributes, no header, '?' = missing.
UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Continuous/stable columns unaffected by categorical re-encoding; used as the
# join key to match records across the two files.
MATCH_KEYS = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]
COMPARABLE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target_bin",
]
INTEGER_COLUMNS = [column for column in COMPARABLE_COLUMNS if column != "oldpeak"]

DEFAULT_UCI = Path("data/external/cardiac/heart_disease_uci/processed.cleveland.data")
DEFAULT_WORKING = Path("data/raw/cardiac/cleveland_standardized.csv")


def load_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values="?")
    df["_missing_ca_thal"] = df[["ca", "thal"]].isna().any(axis=1)
    # Binary target: UCI num 0 -> no disease, 1-4 -> disease.
    df["target_bin"] = (df["num"] > 0).astype(int)
    return df


def load_working(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Working file stores age as age_raw and a pre-binarised heart_disease target.
    df = df.rename(columns={"age_raw": "age", "heart_disease": "target_bin"})
    return df


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    k = df[MATCH_KEYS].copy()
    k["oldpeak"] = k["oldpeak"].round(1)
    for c in ("age", "sex", "trestbps", "chol", "thalach"):
        k[c] = pd.to_numeric(k[c], errors="coerce").round().astype("Int64")
    return k


def _key_tuples(df: pd.DataFrame) -> list[tuple]:
    return [
        tuple(None if pd.isna(value) else value for value in row)
        for row in _key_frame(df).itertuples(index=False, name=None)
    ]


def _normalise_comparable(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COMPARABLE_COLUMNS].copy()
    out["oldpeak"] = pd.to_numeric(out["oldpeak"], errors="coerce").round(1)
    for column in INTEGER_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").round().astype("Int64")
    return out


def _expected_complete_cases(uci: pd.DataFrame) -> pd.DataFrame:
    complete = uci.loc[~uci["_missing_ca_thal"]].copy()
    complete["cp"] = pd.to_numeric(complete["cp"], errors="coerce") - 1
    complete["slope"] = pd.to_numeric(complete["slope"], errors="coerce") - 1
    complete["thal"] = pd.to_numeric(complete["thal"], errors="coerce").map(
        {3.0: 0, 6.0: 1, 7.0: 2}
    )
    return _normalise_comparable(complete)


def _working_comparable(working: pd.DataFrame) -> pd.DataFrame:
    return _normalise_comparable(working)


def _row_counter(df: pd.DataFrame) -> Counter:
    rows = (
        tuple(None if pd.isna(value) else value for value in row)
        for row in df.itertuples(index=False, name=None)
    )
    return Counter(rows)


def _unmatched_indices(reference: pd.DataFrame, candidate: pd.DataFrame) -> list[int]:
    """Return reference indices without a one-to-one stable-key match."""
    remaining = Counter(_key_tuples(candidate))
    unmatched = []
    for index, key in zip(reference.index, _key_tuples(reference), strict=True):
        if remaining[key]:
            remaining[key] -= 1
        else:
            unmatched.append(int(index))
    return unmatched


def _add_occurrence_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_key_occurrence"] = out.groupby(MATCH_KEYS, dropna=False).cumcount()
    return out


def compare(uci: pd.DataFrame, working: pd.DataFrame) -> dict:
    uci = uci.copy()
    working = working.copy()
    expected = _expected_complete_cases(uci)
    actual = _working_comparable(working)

    expected_rows = _row_counter(expected)
    actual_rows = _row_counter(actual)
    missing_expected_rows = sum((expected_rows - actual_rows).values())
    unexpected_working_rows = sum((actual_rows - expected_rows).values())

    dropped_indices = _unmatched_indices(uci, working)
    working_only_indices = _unmatched_indices(working, uci)
    dropped = uci.loc[dropped_indices]
    missing_source = uci.loc[uci["_missing_ca_thal"]]
    dropped_key_counts = Counter(_key_tuples(dropped))
    missing_key_counts = Counter(_key_tuples(missing_source))

    expected_keyed = _add_occurrence_index(expected)
    actual_keyed = _add_occurrence_index(actual)
    matched = expected_keyed.merge(
        actual_keyed,
        on=[*MATCH_KEYS, "_key_occurrence"],
        how="outer",
        suffixes=("_expected", "_working"),
        indicator=True,
        validate="one_to_one",
    )
    shared = matched.loc[matched["_merge"] == "both"]

    mapping_rules = {
        "cp_minus_one": ("cp_expected", "cp_working", "working cp == UCI cp - 1"),
        "slope_minus_one": (
            "slope_expected",
            "slope_working",
            "working slope == UCI slope - 1",
        ),
        "target_binarisation": (
            "target_bin_expected",
            "target_bin_working",
            "working target == (UCI num > 0)",
        ),
        "thal_reencoding": (
            "thal_expected",
            "thal_working",
            "UCI thal {3, 6, 7} == working {0, 1, 2}",
        ),
    }
    mapping_checks = {}
    for label, (expected_col, working_col, rule) in mapping_rules.items():
        values_equal = shared[expected_col].eq(shared[working_col]) | (
            shared[expected_col].isna() & shared[working_col].isna()
        )
        mismatches = int((~values_equal).sum())
        mapping_checks[label] = {
            "rule": rule,
            "shared_rows_checked": int(len(shared)),
            "mismatches": mismatches,
            "verified": mismatches == 0,
        }

    # Encoding checks on the shared records.
    def rng(df, col):
        s = pd.to_numeric(df[col], errors="coerce")
        return (int(s.min()), int(s.max()))

    report = {
        "uci_rows": int(len(uci)),
        "uci_complete_case_rows": int(len(expected)),
        "working_rows": int(len(working)),
        "uci_missing_ca_thal": int(uci["_missing_ca_thal"].sum()),
        "uci_rows_absent_from_working": len(dropped_indices),
        "working_rows_absent_from_uci": len(working_only_indices),
        "uci_complete_case_rows_missing_from_working": missing_expected_rows,
        "working_rows_not_in_uci_complete_cases": unexpected_working_rows,
        "complete_case_stable_keys_equal_both_directions": bool(
            not _unmatched_indices(expected, actual) and not _unmatched_indices(actual, expected)
        ),
        "complete_case_exact_row_multisets_equal": expected_rows == actual_rows,
        "dropped_all_missing_ca_thal": bool(dropped["_missing_ca_thal"].all() and len(dropped) > 0),
        "dropped_rows_exactly_missing_ca_thal": bool(
            dropped_key_counts == missing_key_counts and not working_only_indices
        ),
        "dropped_row_indices_uci": dropped_indices,
        "mapping_checks": mapping_checks,
        "encoding": {
            "uci_cp_range": rng(uci, "cp"),
            "working_cp_range": rng(working, "cp"),
            "uci_slope_range": rng(uci, "slope"),
            "working_slope_range": rng(working, "slope"),
            "uci_num_range": rng(uci, "num"),
            "working_target_range": rng(working, "target_bin"),
        },
        "target_balance": {
            "uci_after_dropping_missing": _balance(uci.loc[~uci["_missing_ca_thal"], "target_bin"]),
            "working": _balance(working["target_bin"]),
        },
    }
    return report


def _balance(s: pd.Series) -> dict:
    vc = s.value_counts().to_dict()
    n = int(s.shape[0])
    return {
        "n": n,
        "negatives": int(vc.get(0, 0)),
        "positives": int(vc.get(1, 0)),
    }


def print_report(r: dict) -> None:
    print("=== Cleveland provenance: UCI raw vs working standardized ===\n")
    print(f"UCI processed.cleveland.data rows : {r['uci_rows']}")
    print(f"Working standardized rows         : {r['working_rows']}")
    print(f"UCI rows with missing ca/thal     : {r['uci_missing_ca_thal']}")
    print(f"UCI rows absent from working      : {r['uci_rows_absent_from_working']}")
    verdict = (
        "YES — exact complete-case rows after verified re-encodings"
        if r["dropped_rows_exactly_missing_ca_thal"]
        and r["complete_case_exact_row_multisets_equal"]
        else "NO — complete-case rows or encodings differ (investigate)"
    )
    print(f"Working == UCI complete-case?     : {verdict}\n")

    e = r["encoding"]
    print("Encoding deltas (raw UCI -> working):")
    print(f"  cp    : UCI {e['uci_cp_range']} -> working {e['working_cp_range']}")
    print(f"  slope : UCI {e['uci_slope_range']} -> working {e['working_slope_range']}")
    print(f"  target: UCI num {e['uci_num_range']} -> working binary {e['working_target_range']}")
    print("\nRow-wise mapping checks:")
    for label, check in r["mapping_checks"].items():
        print(
            f"  {label:<21}: {check['mismatches']} mismatch(es) / "
            f"{check['shared_rows_checked']} matched rows"
        )
    tb = r["target_balance"]
    print(
        f"\nTarget balance (neg/pos): "
        f"UCI-complete-case {tb['uci_after_dropping_missing']['negatives']}/"
        f"{tb['uci_after_dropping_missing']['positives']}  |  "
        f"working {tb['working']['negatives']}/{tb['working']['positives']}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uci", type=Path, default=DEFAULT_UCI)
    p.add_argument("--working", type=Path, default=DEFAULT_WORKING)
    p.add_argument("--json", type=Path, default=None, help="Write summary JSON here.")
    args = p.parse_args(argv)

    for label, path in (("UCI", args.uci), ("working", args.working)):
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 2

    uci = load_uci(args.uci)
    working = load_working(args.working)
    report = compare(uci, working)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON summary -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
