"""DEPRECATED entry point for ``fairxai-characterize``.

Superseded by ``fairxai characterize`` and ``fairxai triage``.  This shim exists
only so FairXAI, the WebApp, and the HPC scripts can migrate independently
instead of in one coordinated maintenance window.  It reproduces the old
combined behaviour — characterization plus optional inline triage merged into
the same output JSON — on top of the new split API.

Remove this module, its console-script entry in ``pyproject.toml``, and its test
once every caller invokes the ``fairxai`` subcommands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fairxai.integration.triage import triage_dataset
from fairxai.profiling.domain_characterization import characterize_dataset

_DEPRECATION_NOTICE = (
    "DEPRECATION: 'fairxai-characterize' is deprecated and will be removed. "
    "Use 'fairxai characterize' for metrics and 'fairxai triage' for the "
    "fairness triage report."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Characterize dataset and compute EBM difficulty",
    )
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
        help="[DEPRECATED] Run triage inline; prefer a separate 'fairxai triage' call",
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


def _attach_triage(result: dict[str, Any], args: argparse.Namespace) -> None:
    """Merge a triage report into *result*, mirroring the pre-split contract.

    Triage failure is non-fatal: the characterization result is preserved and
    the error is reported through ``triage_status``/``triage_error``.
    """
    result["triage_status"] = "not_requested"
    if not args.include_triage:
        return

    try:
        result["triage_report"] = triage_dataset(
            filename=args.filename,
            # The old contract triaged whatever column characterization settled
            # on, so an omitted (or unresolvable) --target-column still worked.
            # ``fairxai triage`` is deliberately stricter, so resolve here.
            target_column=result["target_column"],
            datasets_dir=args.datasets_dir,
            index_column=args.index_column,
            sensitive_columns=args.sensitive_columns,
            project_root=args.triage_project_root,
        )
        result["triage_status"] = "success"
    except Exception as exc:
        result["triage_error"] = str(exc)
        result["triage_status"] = "failed"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(_DEPRECATION_NOTICE, file=sys.stderr)

    try:
        result = characterize_dataset(
            filename=args.filename,
            output_dir=args.output_dir,
            datasets_dir=args.datasets_dir,
            target_column=args.target_column,
            index_column=args.index_column,
            ebm_model_path=args.ebm_model_path,
        )
    except Exception as exc:
        print(f"Characterization failed: {exc}", file=sys.stderr)
        return 1

    _attach_triage(result, args)

    # characterize_dataset already wrote the file; rewrite it so the triage keys
    # land in the JSON the WebApp reads back from disk.
    output_path = Path(args.output_dir) / f"{result['jobId']}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=4)

    if args.print_json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
