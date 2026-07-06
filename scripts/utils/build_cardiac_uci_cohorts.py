#!/usr/bin/env python3
"""Build the analytical cardiac cohorts directly from raw UCI Heart Disease files.

Replaces the black-box curated cohorts (Kaggle-curated Cleveland, fedesoriano 918)
with two cohorts derived from the UCI ``processed.*.data`` distributions, so the
dissertation owns every dedup / encoding / missing-value decision:

- ``cleveland_uci``  : raw ``processed.cleveland.data`` (303 rows), full panel.
- ``four_site_uci``  : Cleveland + Hungarian + Switzerland + VA, deduplicated on the
  exact 12-field canonical key to 918 rows, common panel.

Dedup uses the same canonical key as ``cardiac_overlap_matrix.py`` so the 918 count
is guaranteed identical to the provenance audit. Statlog is intentionally excluded
(the audit proved it is fully contained in Cleveland, 0 unique records).

Run (from Code/FairXAI/, venv active):

    python3 scripts/utils/build_cardiac_uci_cohorts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse the audit's raw columns + canonical-key builder so dedup matches exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardiac_overlap_matrix import UCI_DIR, UCI_RAW_COLUMNS, canonical_common  # noqa: E402

# Concat order matters: Cleveland first so it is fully retained under keep="first".
SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
DEFAULT_OUT_DIR = Path("data/external/cardiac")

# Expected raw invariants; the build refuses to write if the sources drift.
EXPECTED_SITE_ROWS = {"cleveland": 303, "hungarian": 294, "switzerland": 123, "va": 200}
EXPECTED_FOUR_SITE_ROWS = 918

# Columns whose UCI 1-based codes are shifted to the 0-based scheme used by the
# historical curated files (keeps encodings comparable across cohorts).
_SHIFT_TO_ZERO_BASED = ["cp", "slope"]
# thal ships as 3=normal / 6=fixed defect / 7=reversible defect; remap to a
# canonical contiguous 0/1/2 categorical scheme (redesign §9).
_THAL_REMAP = {3: 0, 6: 1, 7: 2}


def _load_site(site: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=UCI_RAW_COLUMNS, na_values="?")
    df.insert(0, "source_site", site)
    return df


def _dedup_four_site(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact-canonical-key duplicates (keep first), reproducing the 918 count."""
    keyed = df.copy()
    keyed["target"] = (pd.to_numeric(keyed["num"], errors="coerce") > 0).astype("Int64")
    key = canonical_common(keyed)  # 12-field canonical representation
    dup_mask = key.duplicated(keep="first")
    return df[~dup_mask].reset_index(drop=True), int(dup_mask.sum())


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared encoding: 0-based cp/slope, thal 0/1/2, chol 0 -> missing.

    Runs AFTER dedup so the dedup key stays in the raw representation the audit used.
    """
    out = df.copy()
    for col in _SHIFT_TO_ZERO_BASED:
        out[col] = pd.to_numeric(out[col], errors="coerce") - 1
    out["thal"] = pd.to_numeric(out["thal"], errors="coerce").map(_THAL_REMAP)
    # A recorded 0 for serum cholesterol or resting blood pressure is a
    # physiological impossibility used as a missing-value sentinel (Switzerland/VA),
    # not a real measurement. Fold it into missing so it is imputed, not dropped.
    for col in ("chol", "trestbps"):
        out[col] = pd.to_numeric(out[col], errors="coerce").replace(0, pd.NA)
    # Keep target/sex as clean integers so schema target_mapping ("0".."4") matches.
    out["num"] = pd.to_numeric(out["num"], errors="coerce").round().astype("Int64")
    out["sex"] = pd.to_numeric(out["sex"], errors="coerce").round().astype("Int64")
    return out


def _missing_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    n = len(df)
    report = {}
    for col in UCI_RAW_COLUMNS:
        if col not in df.columns:
            continue
        miss = int(df[col].isna().sum())
        report[col] = {"missing": miss, "rate_pct": round(100 * miss / n, 1) if n else 0.0}
    return report


def _validate(site_rows: dict[str, int], four_site_rows: int) -> None:
    """Refuse to write cohorts if the raw sources drift from known invariants."""
    if site_rows != EXPECTED_SITE_ROWS:
        raise ValueError(
            f"Unexpected raw site row counts {site_rows}; expected {EXPECTED_SITE_ROWS}. "
            "Refusing to write cohorts — verify the UCI source files."
        )
    if four_site_rows != EXPECTED_FOUR_SITE_ROWS:
        raise ValueError(
            f"Four-site dedup produced {four_site_rows} rows; expected "
            f"{EXPECTED_FOUR_SITE_ROWS}. Refusing to write cohorts."
        )


def build(uci_dir: Path = UCI_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    sites = {site: _load_site(site, uci_dir / fname) for site, fname in SITE_FILES.items()}
    site_rows = {site: len(df) for site, df in sites.items()}

    combined = pd.concat(sites.values(), ignore_index=True)
    four_site_raw, dup_removed = _dedup_four_site(combined)

    # Validate raw + dedup invariants BEFORE writing any output.
    _validate(site_rows, len(four_site_raw))

    cleveland = _finalize(sites["cleveland"])
    four_site = _finalize(four_site_raw)

    cleveland_out = out_dir / "cleveland_uci.csv"
    four_site_out = out_dir / "four_site_uci.csv"
    manifest_out = out_dir / "cardiac_uci_manifest.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    cleveland.to_csv(cleveland_out, index=False)
    four_site.to_csv(four_site_out, index=False)

    manifest = {
        "source_dir": str(uci_dir),
        "site_source_rows": site_rows,
        "four_site_source_rows": sum(site_rows.values()),
        "four_site_dedup_removed": dup_removed,
        "four_site_final_rows": len(four_site),
        "four_site_rows_per_site": four_site["source_site"].value_counts().to_dict(),
        "cleveland_rows": len(cleveland),
        "encoding": (
            "cp,slope -> 0-based; thal 3/6/7 -> 0/1/2; "
            "chol==0 -> missing; trestbps==0 -> missing; num kept 0-4"
        ),
        "cleveland_missing": _missing_report(cleveland),
        "four_site_missing": _missing_report(four_site),
        "outputs": {"cleveland_uci": str(cleveland_out), "four_site_uci": str(four_site_out)},
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uci-dir", type=Path, default=UCI_DIR)
    args = p.parse_args(argv)

    if not args.uci_dir.exists():
        print(f"ERROR: UCI dir not found: {args.uci_dir}", file=sys.stderr)
        return 2

    m = build(args.uci_dir)
    print("=== Cardiac UCI cohort build ===")
    print(f"Cleveland: {m['cleveland_rows']} rows -> {m['outputs']['cleveland_uci']}")
    print(
        f"Four-site: {m['four_site_source_rows']} rows -> {m['four_site_final_rows']} "
        f"({m['four_site_dedup_removed']} dup removed) -> {m['outputs']['four_site_uci']}"
    )
    print(f"Per-site after dedup: {m['four_site_rows_per_site']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
