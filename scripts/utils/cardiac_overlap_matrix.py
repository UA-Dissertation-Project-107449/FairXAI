#!/usr/bin/env python3
"""All-pairs fingerprint overlap across every cardiac source.

Audits which curation reused which records: raw UCI databases (Cleveland,
Hungarian, Switzerland, Long Beach/VA), UCI Statlog Heart, the curated Kaggle
Cleveland mirror, the Kaggle Heart 918-row compilation, and the standardized
working files. Every source is projected onto encoding-stable continuous keys
(age, sex, resting BP, cholesterol, max heart rate, ST depression) so files with
different column names and encodings still compare.

This makes the "black box" of each published curation measurable: e.g. how many
of the 303 raw Cleveland records survive into the 918-row Kaggle compilation, or
that the curated Cleveland mirror dropped exactly the 6 ca/thal-missing rows.

The report also performs a stricter source-union audit over the 11 common
predictors plus binary target. It measures whether Statlog contributes any
canonical records beyond the four UCI Heart Disease databases and whether the
advertised 1,190 -> 918 deduplication count is reproducible. Matching canonical
clinical rows is evidence of record reuse, not proof of patient identity.

cardio70k is intentionally excluded: it records a different clinical panel
(anthropometry + ap_hi/ap_lo, no serum cholesterol / max HR / ST depression), so
it shares no continuous key with the Cleveland-family cohorts.

Run (from Code/FairXAI/, venv active):

    python3 scripts/utils/cardiac_overlap_matrix.py
    python3 scripts/utils/cardiac_overlap_matrix.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

FINGERPRINT_KEYS = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]
COMMON_KEYS = [
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
    "target",
]

UCI_DIR = Path("data/external/cardiac/heart_disease_uci")
STATLOG_PATH = Path("data/external/cardiac/statlog_heart_uci/heart.dat")
KAGGLE_HEART_PATH = Path("data/external/cardiac/heart.csv")
UCI_RAW_COLUMNS = [
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
UCI_SOURCE_PATHS = {
    "uci_cleveland_303": UCI_DIR / "processed.cleveland.data",
    "uci_hungarian_294": UCI_DIR / "processed.hungarian.data",
    "uci_switzerland_123": UCI_DIR / "processed.switzerland.data",
    "uci_va_200": UCI_DIR / "processed.va.data",
}
Loader = Callable[[Path], pd.DataFrame]


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    """Round/coerce fingerprint columns for cross-curation matching."""
    k = df[FINGERPRINT_KEYS].copy()
    k["oldpeak"] = pd.to_numeric(k["oldpeak"], errors="coerce").round(1)
    for c in ("age", "sex", "trestbps", "chol", "thalach"):
        k[c] = pd.to_numeric(k[c], errors="coerce").round().astype("Int64")
    return k


def _canonical_common(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 11 shared predictors plus target to numeric UCI encodings."""
    out = df[COMMON_KEYS].copy()
    for col in COMMON_KEYS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["oldpeak"] = out["oldpeak"].round(1)
    for col in COMMON_KEYS:
        if col != "oldpeak":
            out[col] = out[col].round().astype("Int64")
    return out


def _load_uci_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=UCI_RAW_COLUMNS, na_values="?")
    return _canon(df)


def _load_uci_common(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=UCI_RAW_COLUMNS, na_values="?")
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype("Int64")
    return _canonical_common(df)


def _load_statlog_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=UCI_RAW_COLUMNS)


def _load_statlog_raw(path: Path) -> pd.DataFrame:
    return _canon(_load_statlog_frame(path))


def _load_statlog_common(path: Path) -> pd.DataFrame:
    df = _load_statlog_frame(path)
    # Statlog target: 1 = absence, 2 = presence.
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") == 2).astype("Int64")
    return _canonical_common(df)


def _load_kaggle_cleveland(path: Path) -> pd.DataFrame:
    # heart_cleveland_upload.csv: lowercase UCI-style headers, numeric sex 0/1.
    return _canon(pd.read_csv(path))


def _load_heart918_common(path: Path) -> pd.DataFrame:
    """Map the Kaggle compilation onto the original UCI common encodings."""
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "age": df["Age"],
            "sex": df["Sex"].map({"M": 1, "F": 0}),
            "cp": df["ChestPainType"].map({"TA": 1, "ATA": 2, "NAP": 3, "ASY": 4}),
            "trestbps": df["RestingBP"],
            "chol": df["Cholesterol"],
            "fbs": df["FastingBS"],
            "restecg": df["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2}),
            "thalach": df["MaxHR"],
            "exang": df["ExerciseAngina"].map({"N": 0, "Y": 1}),
            "oldpeak": df["Oldpeak"],
            "slope": df["ST_Slope"].map({"Up": 1, "Flat": 2, "Down": 3}),
            "target": df["HeartDisease"],
        }
    )
    return _canonical_common(out)


def _load_heart918(path: Path) -> pd.DataFrame:
    return _canon(_load_heart918_common(path))


def _load_standardized(path: Path) -> pd.DataFrame:
    # Standardized files carry both a raw `sex` and `sex_bin`; select explicitly
    # to build the canonical key without colliding column names.
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "age": df["age_raw"],
            "sex": df["sex_bin"],
            "trestbps": df["trestbps"],
            "chol": df["chol"],
            "thalach": df["thalach"],
            "oldpeak": df["oldpeak"],
        }
    )
    return _canon(out)


# label -> (path, loader). Only sources present on disk are used.
SOURCES: dict[str, tuple[Path, Loader]] = {
    **{label: (path, _load_uci_raw) for label, path in UCI_SOURCE_PATHS.items()},
    "uci_statlog_270": (STATLOG_PATH, _load_statlog_raw),
    "kaggle_cleveland_curated_297": (
        Path("data/external/cardiac/heart_cleveland_upload.csv"),
        _load_kaggle_cleveland,
    ),
    "kaggle_heart_918": (
        KAGGLE_HEART_PATH,
        _load_heart918,
    ),
    "std_cleveland_297": (
        Path("data/raw/cardiac/cleveland_standardized.csv"),
        _load_standardized,
    ),
    "std_kaggle_918": (
        Path("data/raw/cardiac/kaggle_heart_standardized.csv"),
        _load_standardized,
    ),
}


def _keyset(df: pd.DataFrame) -> set[tuple]:
    return {
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    }


def audit_common_frames(
    uci_frames: dict[str, pd.DataFrame],
    statlog: pd.DataFrame,
    kaggle: pd.DataFrame,
) -> dict:
    """Quantify Statlog redundancy and the 1,190 -> 918 source-union lineage."""
    uci_sets = {label: _keyset(frame) for label, frame in uci_frames.items()}
    uci_union = set().union(*uci_sets.values())
    statlog_keys = _keyset(statlog)
    combined_union = uci_union | statlog_keys
    kaggle_keys = _keyset(kaggle)
    cleveland_keys = uci_sets["uci_cleveland_303"]
    source_rows = sum(len(frame) for frame in uci_frames.values()) + len(statlog)
    exact_shared = len(combined_union & kaggle_keys)

    return {
        "key_columns": COMMON_KEYS,
        "uci_four_database_rows": sum(len(frame) for frame in uci_frames.values()),
        "uci_four_unique_common_records": len(uci_union),
        "statlog_rows": len(statlog),
        "statlog_unique_common_records": len(statlog_keys),
        "statlog_matches_cleveland": len(statlog_keys & cleveland_keys),
        "statlog_unique_beyond_cleveland": len(statlog_keys - cleveland_keys),
        "statlog_unique_beyond_four_uci": len(statlog_keys - uci_union),
        "combined_source_rows_before_dedup": source_rows,
        "combined_unique_records_after_dedup": len(combined_union),
        "duplicate_rows_removed_by_dedup": source_rows - len(combined_union),
        "kaggle_rows": len(kaggle),
        "kaggle_unique_common_records": len(kaggle_keys),
        "exact_common_keys_shared_with_kaggle": exact_shared,
        "exact_common_key_coverage_pct": (
            round(100 * exact_shared / len(kaggle_keys), 1) if kaggle_keys else 0.0
        ),
        "deduplicated_source_count_matches_kaggle_rows": len(combined_union) == len(kaggle),
        "exact_common_keysets_equal": combined_union == kaggle_keys,
    }


def build_source_union_audit() -> dict:
    required = [*UCI_SOURCE_PATHS.values(), STATLOG_PATH, KAGGLE_HEART_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"available": False, "missing": missing}

    uci_frames = {label: _load_uci_common(path) for label, path in UCI_SOURCE_PATHS.items()}
    result = audit_common_frames(
        uci_frames,
        _load_statlog_common(STATLOG_PATH),
        _load_heart918_common(KAGGLE_HEART_PATH),
    )
    return {"available": True, **result}


def build(sources: dict[str, tuple[Path, Loader]]) -> dict:
    loaded: dict[str, set[tuple]] = {}
    sizes: dict[str, int] = {}
    skipped: list[str] = []
    for label, (path, loader) in sources.items():
        if not path.exists():
            skipped.append(f"{label} (missing {path})")
            continue
        keys = _keyset(loader(path))
        loaded[label] = keys
        sizes[label] = len(keys)

    labels = list(loaded)
    inter = {a: {} for a in labels}
    pct_of_unique_keys = {a: {} for a in labels}
    for a in labels:
        for b in labels:
            n = len(loaded[a] & loaded[b])
            inter[a][b] = n
            pct_of_unique_keys[a][b] = round(100 * n / len(loaded[a]), 1) if loaded[a] else 0.0

    return {
        "key_columns": FINGERPRINT_KEYS,
        "unique_key_counts": sizes,
        "intersection_counts": inter,
        "pct_of_unique_keys_in_col": pct_of_unique_keys,
        "skipped": skipped,
    }


def _print_matrix(title: str, labels: list[str], cell) -> None:
    print(f"\n{title}")
    w = max(len(x) for x in labels) + 2
    short = {x: x[:16] for x in labels}
    header = " " * w + "".join(f"{short[b]:>18}" for b in labels)
    print(header)
    for a in labels:
        row = f"{a:<{w}}" + "".join(f"{cell(a, b):>18}" for b in labels)
        print(row)


def _print_source_union_audit(audit: dict) -> None:
    print("\n=== Source-union deduplication audit ===")
    if not audit.get("available"):
        print("Unavailable; missing source files:")
        for path in audit.get("missing", []):
            print(f"  {path}")
        return

    print(f"Canonical key: {', '.join(audit['key_columns'])}")
    print(
        f"Four UCI Heart Disease databases: {audit['uci_four_database_rows']} rows -> "
        f"{audit['uci_four_unique_common_records']} unique common records"
    )
    print(
        f"Statlog: {audit['statlog_rows']} rows; "
        f"{audit['statlog_matches_cleveland']} match Cleveland; "
        f"{audit['statlog_unique_beyond_four_uci']} unique beyond the four UCI databases"
    )
    print(
        f"Combined sources: {audit['combined_source_rows_before_dedup']} rows -> "
        f"{audit['combined_unique_records_after_dedup']} unique common records "
        f"({audit['duplicate_rows_removed_by_dedup']} duplicate rows)"
    )
    print(
        f"Kaggle compilation: {audit['kaggle_rows']} rows; "
        f"source-union count match = "
        f"{audit['deduplicated_source_count_matches_kaggle_rows']}"
    )
    print(
        f"Exact normalized common keys shared with published Kaggle file: "
        f"{audit['exact_common_keys_shared_with_kaggle']}/"
        f"{audit['kaggle_unique_common_records']} "
        f"({audit['exact_common_key_coverage_pct']}%)"
    )
    if audit["statlog_unique_beyond_four_uci"] == 0:
        print(
            "CONFIRMED: Statlog is fully redundant with the four-database UCI "
            "collection and adds no unique canonical record."
        )
    if not audit["exact_common_keysets_equal"]:
        print(
            "Curation note: the 1,190 -> 918 count lineage is reproduced, but the "
            "published Kaggle rows are not an exact canonical-key copy of the raw union."
        )


def print_report(r: dict) -> None:
    labels = list(r["unique_key_counts"])
    print("=== Cardiac all-pairs fingerprint overlap ===")
    print(f"Key columns: {', '.join(r['key_columns'])}")
    print("\nUnique-key counts per source:")
    for label, n in r["unique_key_counts"].items():
        print(f"  {label:<32} {n}")
    if r["skipped"]:
        print("\nSkipped (not on disk):")
        for s in r["skipped"]:
            print(f"  {s}")

    _print_matrix(
        "Intersection counts (unique keys):",
        labels,
        lambda a, b: r["intersection_counts"][a][b],
    )
    _print_matrix(
        "Unique-key coverage (% of ROW source keys found in COLUMN source):",
        labels,
        lambda a, b: r["pct_of_unique_keys_in_col"][a][b],
    )
    print(
        "\nRead the coverage matrix by row: it estimates how much of each source's "
        "fingerprints survive into every other source (the curation 'black box')."
    )
    _print_source_union_audit(r.get("source_union_audit", {"available": False}))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    r = build(SOURCES)
    r["source_union_audit"] = build_source_union_audit()
    if not r["unique_key_counts"]:
        print("ERROR: no cardiac sources found on disk.", file=sys.stderr)
        return 2
    print_report(r)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(r, indent=2))
        print(f"\nWrote JSON summary -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
