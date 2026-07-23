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

    # --- profile -----------------------------------------------------------
    profile = sub.add_parser("profile", help="Read lightweight upload metadata")
    profile.add_argument("--filename", required=True)
    profile.add_argument("--datasets-dir", default=None)

    # --- characterize -------------------------------------------------------
    char = sub.add_parser("characterize", help="Compute complexity metrics + EBM difficulty")
    char.add_argument("--filename", required=True)
    char.add_argument("--output-dir", required=True)
    char.add_argument("--datasets-dir", default=None)
    char.add_argument("--target-column", default=None)
    char.add_argument("--index-column", default=None)
    char.add_argument("--ebm-model-path", default=None)
    char.add_argument("--print-json", action="store_true")

    # --- triage --------------------------------------------------------------
    triage = sub.add_parser(
        "triage",
        help="Generate a fairness triage report",
        description=(
            "Emit the fairness triage report as JSON on stdout. Nothing is written "
            "to disk. Text columns are rejected as target or sensitive columns; "
            "string identifiers remain valid index columns."
        ),
    )
    triage.add_argument("--filename", required=True)
    triage.add_argument("--target-column", required=True)
    triage.add_argument("--datasets-dir", default=None)
    triage.add_argument("--index-column", default=None)
    triage.add_argument("--sensitive-columns", nargs="*", default=None)
    triage.add_argument("--triage-project-root", default=None)

    # --- binning ------------------------------------------------------------
    binn = sub.add_parser("binning", help="Attribute binning subgroup analysis")
    binn.add_argument("--filename", required=True)
    binn.add_argument("--target-column", required=True)
    binn.add_argument("--attribute", required=True, help="Numerical column to bin")
    binn.add_argument("--strategy", required=True, help="e.g. quantile_5, equal_width_10")
    binn.add_argument("--datasets-dir", default=None)
    binn.add_argument("--min-group-size", type=int, default=10)

    # --- clustering ---------------------------------------------------------
    clust = sub.add_parser("clustering", help="Cluster-based subgroup discovery")
    clust.add_argument("--filename", required=True)
    clust.add_argument("--target-column", required=True)
    clust.add_argument("--datasets-dir", default=None)
    clust.add_argument(
        "--method",
        default="auto",
        choices=["auto", "kmeans", "hierarchical", "dbscan", "gaussian_mixture"],
        help="Clustering method to run. auto tries all supported methods and selects best.",
    )
    clust.add_argument("--index-column", default=None)
    clust.add_argument(
        "--sensitive-columns",
        nargs="*",
        default=None,
        help="Columns to keep out of the feature set (cluster-vs-attribute stays a finding)",
    )
    clust.add_argument(
        "--pca2d-json",
        default=None,
        help=(
            "JSON string of a stored projection: either bare [[x,y,label],...] "
            'coords, or {"points": [...], "feature_columns": [...]}'
        ),
    )
    clust.add_argument(
        "--pca2d-file",
        default=None,
        help="Path to a JSON file in either --pca2d-json form",
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


def _unpack_projection(stored: object) -> tuple[list | None, list[str] | None]:
    """Split a stored projection into coords and the columns it was built from.

    Accepts the bare ``[[x, y, label], ...]`` list that older callers send; that
    form carries no column list, and the clustering adapter treats the resulting
    ``None`` as "unknown" and recomputes rather than reusing coords it cannot
    vouch for.
    """
    if stored is None:
        return None, None
    if isinstance(stored, dict):
        points = stored.get("points")
        columns = stored.get("feature_columns")
        return (
            list(points) if points is not None else None,
            [str(col) for col in columns] if columns is not None else None,
        )
    return list(stored), None  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "profile":
            from fairxai.integration.profile import profile_dataset

            result = profile_dataset(
                filename=args.filename,
                datasets_dir=args.datasets_dir,
            )
            print(json.dumps(result))

        elif args.command == "characterize":
            from fairxai.integration.characterize import characterize_dataset

            result = characterize_dataset(
                filename=args.filename,
                output_dir=args.output_dir,
                datasets_dir=args.datasets_dir,
                target_column=args.target_column,
                index_column=args.index_column,
                ebm_model_path=args.ebm_model_path,
            )
            if args.print_json:
                print(json.dumps(result, indent=2))

        elif args.command == "triage":
            from fairxai.integration.triage import triage_dataset

            result = triage_dataset(
                filename=args.filename,
                target_column=args.target_column,
                datasets_dir=args.datasets_dir,
                index_column=args.index_column,
                sensitive_columns=args.sensitive_columns,
                project_root=args.triage_project_root,
            )
            # The report is this subcommand's only output, so it is always
            # printed — matching characterize's --print-json formatting.
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
                    stored_projection = json.load(f)
            elif args.pca2d_json:
                stored_projection = json.loads(args.pca2d_json)
            else:
                stored_projection = None
            pca2d, pca2d_feature_columns = _unpack_projection(stored_projection)
            result = run_clustering(
                csv_path=csv_path,
                target_column=args.target_column,
                pca2d=pca2d,
                method=args.method,
                index_column=args.index_column,
                sensitive_columns=args.sensitive_columns,
                pca2d_feature_columns=pca2d_feature_columns,
            )
            print(json.dumps(result))

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
