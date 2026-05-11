"""Unified CLI entry point: ``fairxai <subcommand> [args]``."""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fairxai",
        description="FairXAI WebApp integration CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- characterize -------------------------------------------------------
    char = sub.add_parser("characterize", help="Compute complexity metrics + EBM difficulty")
    char.add_argument("--filename", required=True)
    char.add_argument("--output-dir", required=True)
    char.add_argument("--datasets-dir", default=None)
    char.add_argument("--target-column", default=None)
    char.add_argument("--index-column", default=None)
    char.add_argument("--ebm-model-path", default=None)
    char.add_argument("--print-json", action="store_true")
    char.add_argument("--include-triage", action="store_true")
    char.add_argument("--sensitive-columns", nargs="*", default=None)
    char.add_argument("--triage-project-root", default=None)

    # --- binning ------------------------------------------------------------
    binn = sub.add_parser("binning", help="Attribute binning subgroup analysis")
    binn.add_argument("--filename", required=True)
    binn.add_argument("--target-column", required=True)
    binn.add_argument("--attribute", required=True, help="Numerical column to bin")
    binn.add_argument("--strategy", required=True, help="e.g. quantile_5, equal_width_3")
    binn.add_argument("--datasets-dir", default=None)
    binn.add_argument("--min-group-size", type=int, default=10)

    # --- clustering ---------------------------------------------------------
    clust = sub.add_parser("clustering", help="Cluster-based subgroup discovery")
    clust.add_argument("--filename", required=True)
    clust.add_argument("--target-column", required=True)
    clust.add_argument("--datasets-dir", default=None)
    clust.add_argument(
        "--pca2d-json",
        default=None,
        help="JSON string of existing [[x,y,label],...] PCA coords to reuse",
    )
    clust.add_argument(
        "--pca2d-file",
        default=None,
        help="Path to JSON file with existing [[x,y,label],...] PCA coords to reuse",
    )

    return parser


def _resolve_csv(filename: str, datasets_dir: str | None) -> str:
    from pathlib import Path

    p = Path(filename)
    if p.is_absolute() or p.exists():
        return str(p)
    if datasets_dir:
        candidate = Path(datasets_dir) / filename
        if candidate.exists():
            return str(candidate)
    return filename


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "characterize":
            from fairxai.integration.characterize import characterize_dataset

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
            if args.print_json:
                print(json.dumps(result, indent=2))

        elif args.command == "binning":
            from fairxai.integration.binning import run_binning

            csv_path = _resolve_csv(args.filename, args.datasets_dir)
            result = run_binning(
                csv_path=csv_path,
                target_column=args.target_column,
                attribute=args.attribute,
                strategy=args.strategy,
                min_group_size=args.min_group_size,
            )
            print(json.dumps(result))

        elif args.command == "clustering":
            from fairxai.integration.clustering import run_clustering

            csv_path = _resolve_csv(args.filename, args.datasets_dir)
            if args.pca2d_file:
                with open(args.pca2d_file) as f:
                    pca2d = json.load(f)
            elif args.pca2d_json:
                pca2d = json.loads(args.pca2d_json)
            else:
                pca2d = None
            result = run_clustering(
                csv_path=csv_path,
                target_column=args.target_column,
                pca2d=pca2d,
            )
            print(json.dumps(result))

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
