"""WebApp adapter for standalone fairness triage.

Triage is a separate operation from characterization: it produces only the
recommendation report and writes nothing to disk.  Callers that need both run
``fairxai characterize`` and ``fairxai triage`` in sequence.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from fairxai.profiling.domain_characterization import (
    build_triage_report,
    is_analysis_role_eligible,
    resolve_input_csv,
)


def _reject_text_roles(df: pd.DataFrame, target_column: str, sensitive: list[str]) -> None:
    """Raise ``ValueError`` if any role column is free text.

    Index columns are intentionally exempt — all-unique string identifiers are
    legitimate index choices.
    """
    if not is_analysis_role_eligible(target_column, df[target_column]):
        raise ValueError(
            f"Target column '{target_column}' is free text and cannot be used as a target."
        )

    text_sensitive = [c for c in sensitive if not is_analysis_role_eligible(c, df[c])]
    if text_sensitive:
        raise ValueError(
            "Free-text columns cannot be used as sensitive attributes: " + ", ".join(text_sensitive)
        )


def triage_dataset(
    filename: str,
    target_column: str,
    datasets_dir: str | Path | None = None,
    index_column: str | None = None,
    sensitive_columns: list[str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run fairness triage on one CSV and return the report as a plain dict."""
    csv_path = resolve_input_csv(filename=filename, datasets_dir=datasets_dir)
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path.name}")

    sensitive = [c for c in (sensitive_columns or []) if c]
    missing = [c for c in sensitive if c not in df.columns]
    if missing:
        raise ValueError(f"Sensitive columns not found in {csv_path.name}: {', '.join(missing)}")

    _reject_text_roles(df, target_column, sensitive)

    report, _feature_summary = build_triage_report(
        csv_path=csv_path,
        dataset_name=csv_path.stem,
        target_column=target_column,
        index_column=index_column,
        sensitive_columns=sensitive,
        project_root=project_root,
    )
    return report
