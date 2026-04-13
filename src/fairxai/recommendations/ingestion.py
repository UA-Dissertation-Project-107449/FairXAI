"""Dataset ingestion: auto-detection of column roles/types + user overrides.

Provides heuristic detection of column types (numerical, categorical, binary,
identifier, datetime) and semantic roles (label, sensitive, feature, identifier),
plus a schema-based fast-path for already-registered datasets.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .models import ColumnMeta, ColumnRole, ColumnType, DatasetIngestion

logger = logging.getLogger(__name__)

# Heuristic constants
_IDENTIFIER_UNIQUENESS_RATIO = 0.99  # ≥ 99 % unique → likely identifier
_CATEGORICAL_MAX_CARDINALITY = 20  # numeric col with ≤ 20 unique → categorical
_BINARY_VALUES = 2  # exactly 2 unique non-null values
_LABEL_NAME_HINTS = {
    "target",
    "label",
    "class",
    "outcome",
    "y",
    "diagnosis",
    "disease",
    "heart_disease",
    "condition",
    "result",
}
_SENSITIVE_NAME_HINTS = {
    "sex",
    "gender",
    "age",
    "age_group",
    "race",
    "ethnicity",
    "religion",
    "nationality",
    "disability",
    "marital_status",
}


# ---------------------------------------------------------------------------
# CSV inspection helpers
# ---------------------------------------------------------------------------


def _detect_separator(sample: str) -> str:
    """Guess CSV separator from a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _detect_header(sample: str, separator: str) -> bool:
    """Heuristic: if the first row looks non-numeric while later rows are numeric,
    it is probably a header line."""
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        # Fallback: check if first row values parse as float
        first_line = sample.split("\n")[0]
        tokens = first_line.split(separator)
        numeric_count = sum(1 for t in tokens if _is_float(t.strip()))
        return numeric_count < len(tokens) / 2


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------


def _detect_column_type(series: pd.Series, n_rows: int) -> ColumnType:
    """Infer the type of a single column from its values."""
    non_null = series.dropna()
    if non_null.empty:
        return ColumnType.UNKNOWN

    n_unique = non_null.nunique()

    # Datetime check
    if non_null.dtype == "object":
        sample = non_null.head(20)
        try:
            pd.to_datetime(sample, format="mixed")
            return ColumnType.DATETIME
        except (ValueError, TypeError):
            pass

    # Identifier: (almost) all unique values
    if n_unique >= n_rows * _IDENTIFIER_UNIQUENESS_RATIO and n_rows > 10:
        return ColumnType.IDENTIFIER

    # Binary
    if n_unique == _BINARY_VALUES:
        return ColumnType.BINARY

    # Numeric
    if pd.api.types.is_numeric_dtype(non_null):
        if n_unique <= _CATEGORICAL_MAX_CARDINALITY:
            return ColumnType.CATEGORICAL
        return ColumnType.NUMERICAL

    # String / object with low cardinality → categorical
    if n_unique <= _CATEGORICAL_MAX_CARDINALITY:
        return ColumnType.CATEGORICAL

    # High cardinality string → text
    return ColumnType.TEXT


def _detect_column_role(
    name: str,
    col_type: ColumnType,
    is_last: bool,
) -> ColumnRole:
    """Guess a column's semantic role from its name and detected type."""
    lower = name.lower().strip()

    if col_type == ColumnType.IDENTIFIER:
        return ColumnRole.IDENTIFIER

    if lower in _LABEL_NAME_HINTS:
        return ColumnRole.LABEL

    if lower in _SENSITIVE_NAME_HINTS:
        return ColumnRole.SENSITIVE

    # Fallback: last binary column is often the label
    if is_last and col_type == ColumnType.BINARY:
        return ColumnRole.LABEL

    return ColumnRole.FEATURE


# ---------------------------------------------------------------------------
# Public API: auto-detection
# ---------------------------------------------------------------------------


class DatasetIngestor:
    """Detect column metadata from a raw CSV file."""

    def __init__(self, sniff_bytes: int = 8192):
        self.sniff_bytes = sniff_bytes

    def ingest(
        self,
        filepath: str,
        *,
        label_column: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        identifier_columns: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ) -> DatasetIngestion:
        """Auto-detect column types/roles and build a ``DatasetIngestion``.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        label_column : str, optional
            Override: force this column as the label.
        sensitive_columns : list[str], optional
            Override: force these columns as sensitive.
        identifier_columns : list[str], optional
            Override: force these columns as identifiers.
        dataset_name : str, optional
            Human-readable name (defaults to filename stem).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # ---- sniff separator & header ----
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            sample = fh.read(self.sniff_bytes)

        separator = _detect_separator(sample)
        has_header = _detect_header(sample, separator)

        # ---- load DataFrame ----
        df = pd.read_csv(
            path,
            sep=separator,
            header=0 if has_header else None,
            low_memory=False,
        )
        n_rows, n_cols = df.shape

        # ---- detect each column ----
        columns: List[ColumnMeta] = []
        for idx, col_name in enumerate(df.columns):
            series = df[col_name]
            is_last = idx == n_cols - 1
            col_type = _detect_column_type(series, n_rows)
            col_role = _detect_column_role(str(col_name), col_type, is_last)

            columns.append(
                ColumnMeta(
                    name=str(col_name),
                    detected_type=col_type,
                    role=col_role,
                    n_unique=int(series.nunique()),
                    n_missing=int(series.isna().sum()),
                    sample_values=series.dropna().head(5).tolist(),
                )
            )

        # ---- apply explicit overrides ----
        if label_column:
            for c in columns:
                if c.name == label_column:
                    c.role = ColumnRole.LABEL
                    c.user_confirmed = True
                elif c.role == ColumnRole.LABEL:
                    # Un-set any auto-detected label that clashes
                    c.role = ColumnRole.FEATURE

        if sensitive_columns:
            for c in columns:
                if c.name in sensitive_columns:
                    c.role = ColumnRole.SENSITIVE
                    c.user_confirmed = True

        if identifier_columns:
            for c in columns:
                if c.name in identifier_columns:
                    c.role = ColumnRole.IDENTIFIER
                    c.user_confirmed = True

        # ---- derive convenience lists ----
        resolved_label = next((c.name for c in columns if c.role == ColumnRole.LABEL), None)
        resolved_sensitive = [c.name for c in columns if c.role == ColumnRole.SENSITIVE]
        resolved_identifiers = [c.name for c in columns if c.role == ColumnRole.IDENTIFIER]

        return DatasetIngestion(
            filepath=str(path),
            columns=columns,
            label_column=resolved_label,
            sensitive_columns=resolved_sensitive,
            identifier_columns=resolved_identifiers,
            has_header=has_header,
            separator=separator,
            n_rows=n_rows,
            n_cols=n_cols,
            dataset_name=dataset_name or path.stem,
        )


# ---------------------------------------------------------------------------
# Schema-based fast path (for already-registered datasets)
# ---------------------------------------------------------------------------


def ingestion_from_schema(
    schema_path: str,
    dataset_key: str,
    data_dir: Optional[str] = None,
) -> DatasetIngestion:
    """Build a ``DatasetIngestion`` from an existing JSON schema file.

    This mirrors the structure of ``configs/schema/cardiac.json``.
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    if "datasets" not in schema or dataset_key not in schema["datasets"]:
        raise KeyError(f"Dataset '{dataset_key}' not found in schema {path}")

    ds = schema["datasets"][dataset_key]

    # Resolve file path
    filename = ds.get("filename", "")
    if data_dir:
        filepath = str(Path(data_dir) / filename)
    else:
        filepath = filename

    # Build column metadata from schema declarations
    sensitive_attrs = ds.get("sensitive_attributes", {})
    exclude_cols = set(ds.get("exclude_features", []))
    label_col = ds.get("label") or ds.get("target")
    clinical_features = ds.get("clinical_features", [])

    columns: List[ColumnMeta] = []

    # Sensitive columns first
    for attr_name, attr_info in sensitive_attrs.items():
        attr_type_str = attr_info.get("type", "unknown")
        if attr_type_str == "continuous":
            col_type = ColumnType.NUMERICAL
        elif attr_type_str == "categorical":
            mapping = attr_info.get("mapping", {})
            col_type = ColumnType.BINARY if len(mapping) == 2 else ColumnType.CATEGORICAL
        else:
            col_type = ColumnType.UNKNOWN

        columns.append(
            ColumnMeta(
                name=attr_name,
                detected_type=col_type,
                role=ColumnRole.SENSITIVE,
                user_confirmed=True,
            )
        )

    # Label column
    if label_col:
        columns.append(
            ColumnMeta(
                name=label_col,
                detected_type=ColumnType.BINARY,
                role=ColumnRole.LABEL,
                user_confirmed=True,
            )
        )

    # Clinical / remaining features
    for feat in clinical_features:
        if feat in exclude_cols or feat == label_col:
            continue
        # Already listed as sensitive?
        if any(c.name == feat for c in columns):
            continue
        columns.append(
            ColumnMeta(
                name=feat,
                detected_type=ColumnType.UNKNOWN,  # known at load time
                role=ColumnRole.FEATURE,
                user_confirmed=True,
            )
        )

    sensitive_names = list(sensitive_attrs.keys())

    return DatasetIngestion(
        filepath=filepath,
        columns=columns,
        label_column=label_col,
        sensitive_columns=sensitive_names,
        identifier_columns=[],
        has_header=True,
        separator=",",
        dataset_name=dataset_key,
    )


# ---------------------------------------------------------------------------
# User override / confirmation helper
# ---------------------------------------------------------------------------


def confirm_ingestion(
    ingestion: DatasetIngestion,
    overrides: Dict[str, Dict[str, str]],
) -> DatasetIngestion:
    """Apply user corrections to auto-detected ingestion.

    Parameters
    ----------
    ingestion : DatasetIngestion
        The auto-detected ingestion object.
    overrides : dict
        Mapping of ``column_name`` → ``{"role": "...", "type": "..."}``.
        Values use the string forms of ``ColumnRole`` / ``ColumnType``.

    Returns
    -------
    DatasetIngestion
        A new ingestion with overrides applied and ``user_confirmed`` set.
    """
    for col in ingestion.columns:
        if col.name in overrides:
            patch = overrides[col.name]
            if "role" in patch:
                col.role = ColumnRole(patch["role"])
            if "type" in patch:
                col.detected_type = ColumnType(patch["type"])
            col.user_confirmed = True

    # Re-derive convenience lists after overrides
    ingestion.label_column = next(
        (c.name for c in ingestion.columns if c.role == ColumnRole.LABEL), None
    )
    ingestion.sensitive_columns = [
        c.name for c in ingestion.columns if c.role == ColumnRole.SENSITIVE
    ]
    ingestion.identifier_columns = [
        c.name for c in ingestion.columns if c.role == ColumnRole.IDENTIFIER
    ]

    return ingestion
