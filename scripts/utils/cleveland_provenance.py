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

DEFAULT_UCI = Path(
    "data/external/cardiac/temp_cleveland_uci/heart+disease/processed.cleveland.data"
)
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
    return [tuple(r) for r in _key_frame(df).itertuples(index=False, name=None)]


def compare(uci: pd.DataFrame, working: pd.DataFrame) -> dict:
    uci_keys = _key_tuples(uci)
    work_keys = set(_key_tuples(working))

    uci["_in_working"] = [k in work_keys for k in uci_keys]
    dropped = uci[~uci["_in_working"]]

    # Encoding checks on the shared records.
    def rng(df, col):
        s = pd.to_numeric(df[col], errors="coerce")
        return (int(s.min()), int(s.max()))

    report = {
        "uci_rows": int(len(uci)),
        "working_rows": int(len(working)),
        "uci_missing_ca_thal": int(uci["_missing_ca_thal"].sum()),
        "uci_rows_absent_from_working": int((~uci["_in_working"]).sum()),
        "dropped_all_missing_ca_thal": bool(dropped["_missing_ca_thal"].all() and len(dropped) > 0),
        "dropped_row_indices_uci": dropped.index.tolist(),
        "encoding": {
            "uci_cp_range": rng(uci, "cp"),
            "working_cp_range": rng(working, "cp"),
            "uci_slope_range": rng(uci, "slope"),
            "working_slope_range": rng(working, "slope"),
            "uci_num_range": rng(uci, "num"),
            "working_target_range": rng(working, "target_bin"),
        },
        "target_balance": {
            "uci_after_dropping_missing": _balance(uci.loc[uci["_in_working"], "target_bin"]),
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
        "YES — the dropped rows are exactly the missing-ca/thal records"
        if r["dropped_all_missing_ca_thal"]
        else "NO — dropped rows do NOT all coincide with missing ca/thal (investigate)"
    )
    print(f"Working == UCI complete-case?     : {verdict}\n")

    e = r["encoding"]
    print("Encoding deltas (raw UCI -> working):")
    print(f"  cp    : UCI {e['uci_cp_range']} -> working {e['working_cp_range']}")
    print(f"  slope : UCI {e['uci_slope_range']} -> working {e['working_slope_range']}")
    print(
        f"  target: UCI num {e['uci_num_range']} " f"-> working binary {e['working_target_range']}"
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
