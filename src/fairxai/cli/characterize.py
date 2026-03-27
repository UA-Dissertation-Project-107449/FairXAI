"""CLI entrypoint for WebApp-compatible dataset characterization."""

from __future__ import annotations

import argparse
import json
import sys

from fairxai.profiling.domain_characterization import characterize_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Characterize dataset and compute EBM difficulty")
    parser.add_argument("--filename", required=True, help="Dataset filename or full CSV path")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the output JSON (<jobId>.json) will be written",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Optional base directory for resolving relative --filename values",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional target column override (defaults to heart_disease or last column)",
    )
    parser.add_argument(
        "--index-column",
        default=None,
        help="Optional index/identifier column to exclude from metric feature computation",
    )
    parser.add_argument(
        "--ebm-model-path",
        default=None,
        help="Optional path to the EBM model (.joblib)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print resulting JSON to stdout",
    )
    parser.add_argument(
        "--include-triage",
        action="store_true",
        help="Optionally run the recommendation engine and append triage_report",
    )
    parser.add_argument(
        "--sensitive-columns",
        nargs="*",
        default=None,
        help="Optional sensitive columns passed to triage generation",
    )
    parser.add_argument(
        "--triage-project-root",
        default=None,
        help="Optional project root for recommendation config resolution",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = characterize_dataset(
            filename=args.filename,
            output_dir=args.output_dir,
            datasets_dir=args.datasets_dir,
            target_column=args.target_column,
            index_column=args.index_column,
            ebm_model_path=args.ebm_model_path,
            include_triage=args.include_triage,
            sensitive_columns=args.sensitive_columns,
            triage_project_root=args.triage_project_root,
        )
    except Exception as exc:
        print(f"[ERROR] Characterization failed: {exc}", file=sys.stderr)
        return 1

    if args.print_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
