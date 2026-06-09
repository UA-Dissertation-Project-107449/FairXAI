"""Data loading utilities for pipeline datasets."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .schemas import get_sex_mapping, harmonize_cardiac_schema


class CardiacDataLoader:
    """Loader for cardiac disease datasets with schema mapping."""

    def __init__(self, config_path: str, feature_map_path: str | None = None):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.datasets = self.config.get("datasets", {})
        self.cardiac_datasets = self.config.get(
            "cardiac_relevant_datasets", ["cleveland", "kaggle_heart"]
        )
        self.feature_map = None
        if feature_map_path:
            try:
                with open(feature_map_path, "r") as f:
                    self.feature_map = yaml.safe_load(f) or {}
            except FileNotFoundError:
                logging.warning(f"Feature map not found: {feature_map_path}")

    def load_dataset(self, dataset_name: str, data_dir: str) -> pd.DataFrame:
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_config = self.datasets[dataset_name]
        filename = dataset_config.get("filename")

        # Prefer data/external/cardiac/{filename}, then data/external/{filename}
        p1 = Path(data_dir) / "cardiac" / filename
        p2 = Path(data_dir) / filename
        filepath = p1 if p1.exists() else p2
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        logging.info(f"Loading {dataset_name} from {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline()
        sep = ";" if header_line.count(";") > header_line.count(",") else ","
        df = pd.read_csv(filepath, sep=sep)
        df["_dataset_source"] = dataset_name
        df["_dataset_file"] = filename

        # Harmonize base schema and apply sensitive/target standardization
        df = harmonize_cardiac_schema(df, dataset_name)
        df = self._apply_sensitive_standardization(df, dataset_name)
        df = self._apply_target_standardization(df, dataset_name)
        df = self._apply_feature_mapping(df, dataset_name)
        df = self._apply_clinical_value_maps(df, dataset_name)
        self._log_unmapped_columns(df, dataset_name)
        self._log_missing_core_columns(df, dataset_name)
        df = self._apply_feature_rules(df, dataset_name)
        return df

    def load_all_cardiac_datasets(self, data_dir: str) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        for name in self.cardiac_datasets:
            try:
                datasets[name] = self.load_dataset(name, data_dir)
                logging.info(f"[SUCCESS] Loaded {name}: {len(datasets[name])} rows")
            except Exception as e:
                logging.error(f"Failed to load {name}: {e}")
        return datasets

    def _apply_sensitive_standardization(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        sens = self.datasets.get(dataset_name, {}).get("sensitive_attributes", {})

        # Age binning to age_group if age_raw exists
        age_key = "age" if "age" in sens else "Age" if "Age" in sens else None
        if age_key and "age_raw" in df.columns:
            bins = sens[age_key].get("bins", [0, 40, 50, 60, 70, 120])
            labels = sens[age_key].get("labels", ["<40", "40-49", "50-59", "60-69", "70+"])
            df["age_group"] = pd.cut(df["age_raw"], bins=bins, labels=labels, include_lowest=True)

        # Sex mapping (normalize to 0/1 in `sex`, keep labels in `sex_extended`)
        # Canonical fallback mappings come from configs/domain/cardiac.yaml via
        # schemas.get_sex_mapping(); per-dataset overrides in the schema JSON
        # take precedence when present.
        sex_key = (
            "sex"
            if "sex" in sens
            else (
                "Sex"
                if "Sex" in sens
                else "Gender" if "Gender" in sens else "gender" if "gender" in sens else None
            )
        )
        if sex_key and sex_key in df.columns:
            raw = df[sex_key]
            mapping_cfg = sens.get(sex_key, {}).get("mapping", {})
            if pd.api.types.is_numeric_dtype(raw):
                raw_num = pd.to_numeric(raw, errors="coerce")
                if mapping_cfg:
                    mapping_num = {int(k): v for k, v in mapping_cfg.items()}
                    sex_label = raw_num.map(mapping_num)
                else:
                    sex_label = raw_num.map(get_sex_mapping("numeric"))
            else:
                raw_str = raw.astype(str).str.strip()
                if mapping_cfg:
                    mapping_str = {str(k): v for k, v in mapping_cfg.items()}
                    sex_label = raw_str.map(mapping_str)
                else:
                    sex_label = raw_str.map(get_sex_mapping("string"))

            if sex_label.isna().all():
                raw_str = raw.astype(str).str.strip()
                sex_label = raw_str.map(get_sex_mapping("string"))

            df["sex_extended"] = sex_label
            df["sex"] = sex_label.map({"Female": 0, "Male": 1})
            df["sex_bin"] = df["sex"]

        return df

    def _apply_target_standardization(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        cfg = self.datasets.get(dataset_name, {})
        tgt_col = cfg.get("label") or cfg.get("target")
        mapping = cfg.get("target_mapping")
        if tgt_col and tgt_col in df.columns:
            if mapping:
                mapped = df[tgt_col].astype(str).map(mapping)
                df["heart_disease"] = mapped.map({"no_disease": 0, "disease": 1})
            else:
                df["heart_disease"] = pd.to_numeric(df[tgt_col], errors="coerce")
        return df

    def _apply_feature_rules(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        cfg = self.datasets.get(dataset_name, {})
        unified = self.config.get("unified_schema", {})

        include = cfg.get("include_features") or []
        exclude = cfg.get("exclude_features") or []
        unified_exclude = unified.get("exclude_features") or []
        label_col = cfg.get("label") or cfg.get("target")

        required_cols = set(
            [
                "heart_disease",
                "age_raw",
                "age_group",
                "sex",
                "sex_extended",
                "sex_bin",
                "ethnicity",
                "group_cluster",
                "_dataset_source",
                "_dataset_file",
            ]
        )

        if include:
            keep_cols = set(include) | required_cols
            keep_cols = [col for col in df.columns if col in keep_cols]
            df = df[keep_cols].copy()

        drop_cols = set(exclude) | set(unified_exclude)
        if label_col and label_col != "heart_disease":
            drop_cols.add(label_col)

        drop_cols = [col for col in drop_cols if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def _apply_feature_mapping(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        if not self.feature_map:
            return df

        def _extract_mappings(section: dict) -> dict[str, list]:
            mappings: dict[str, list] = {}
            for entry in section.values():
                canonical = entry.get("canonical")
                aliases = entry.get("aliases") or []
                if canonical:
                    mappings[canonical] = list(dict.fromkeys([canonical] + aliases))
            return mappings

        mappings = {}
        mappings.update(_extract_mappings(self.feature_map.get("sensitive", {})))
        mappings.update(_extract_mappings(self.feature_map.get("target", {})))
        mappings.update(_extract_mappings(self.feature_map.get("common", {})))

        dataset_specific = self.feature_map.get("dataset_specific", {})
        for entry in dataset_specific.get(dataset_name, {}).values():
            canonical = entry.get("canonical")
            aliases = entry.get("aliases") or []
            if canonical:
                mappings[canonical] = list(dict.fromkeys([canonical] + aliases))

        for canonical, aliases in mappings.items():
            present = [col for col in aliases if col in df.columns]
            if not present:
                continue
            if canonical in df.columns:
                present = [canonical] + [col for col in present if col != canonical]
            combined = df[present[0]].copy()
            for col in present[1:]:
                if df[col].isna().all():
                    continue
                combined = combined.fillna(df[col])
            df[canonical] = combined
            drop_cols = [col for col in present if col != canonical]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        return df

    def _apply_clinical_value_maps(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Encode string-coded clinical categoricals to the canonical numeric scheme.

        Datasets such as Kaggle Heart store cp/restecg/exang/slope as strings
        ('ATA', 'Normal', ...); Cleveland already uses numeric codes.  Only
        object-dtype columns are mapped, so numeric source columns pass through
        untouched.  Mappings come from ``value_map`` entries in the feature map.
        """
        if not self.feature_map:
            return df

        for entry in self.feature_map.get("common", {}).values():
            canonical = entry.get("canonical")
            value_map = entry.get("value_map")
            if not canonical or not value_map or canonical not in df.columns:
                continue
            col = df[canonical]
            if pd.api.types.is_numeric_dtype(col):
                continue
            normalized = col.astype(str).str.strip()
            mapped = normalized.map({str(k): v for k, v in value_map.items()})
            unmapped = sorted(normalized[mapped.isna() & col.notna()].unique())
            if unmapped:
                logging.warning(
                    "[%s] %s: unmapped categorical values %s (set to NaN)",
                    dataset_name,
                    canonical,
                    unmapped,
                )
            df[canonical] = mapped
        return df

    def _log_unmapped_columns(self, df: pd.DataFrame, dataset_name: str) -> None:
        if not self.feature_map:
            return

        def _collect_aliases(section: dict) -> set:
            aliases = set()
            for entry in section.values():
                canonical = entry.get("canonical")
                if canonical:
                    aliases.add(canonical)
                for alias in entry.get("aliases") or []:
                    aliases.add(alias)
            return aliases

        known = set()
        known |= _collect_aliases(self.feature_map.get("sensitive", {}))
        known |= _collect_aliases(self.feature_map.get("target", {}))
        known |= _collect_aliases(self.feature_map.get("common", {}))

        dataset_specific = self.feature_map.get("dataset_specific", {})
        if dataset_name in dataset_specific:
            known |= _collect_aliases(dataset_specific.get(dataset_name, {}))

        cfg = self.datasets.get(dataset_name, {})
        label_col = cfg.get("label") or cfg.get("target")
        if label_col:
            known.add(label_col)

        known |= {
            "heart_disease",
            "age_raw",
            "age_group",
            "sex",
            "sex_extended",
            "sex_bin",
            "_dataset_source",
            "_dataset_file",
        }

        unmapped = [col for col in df.columns if col not in known and not col.startswith("_")]
        if unmapped:
            logging.info(f"Unmapped columns for {dataset_name}: {sorted(unmapped)}")

    def _log_missing_core_columns(self, df: pd.DataFrame, dataset_name: str) -> None:
        core_cols = ["heart_disease", "age_raw", "age_group", "sex"]
        missing = [col for col in core_cols if col not in df.columns]
        if missing:
            logging.warning(f"{dataset_name}: missing core columns after mapping: {missing}")


class DermatologyDataLoader:
    """Loader for PAD-UFES dermatology image metadata."""

    def __init__(self, config_path: str, feature_map_path: str | None = None):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.datasets = self.config.get("datasets", {})
        self.dermatology_datasets = self.config.get(
            "dermatology_relevant_datasets", ["pad_ufes_20"]
        )
        self.last_image_reports: dict[str, dict[str, Any]] = {}

    def load_dataset(self, dataset_name: str, data_dir: str) -> pd.DataFrame:
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        cfg = self.datasets[dataset_name]
        dataset_dir = self._resolve_dataset_dir(Path(data_dir), cfg)
        metadata_path = dataset_dir / cfg.get("metadata_filename", "metadata.csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        logging.info("Loading %s metadata from %s", dataset_name, metadata_path)
        df = pd.read_csv(metadata_path)
        df["_dataset_source"] = dataset_name
        df["_dataset_file"] = str(metadata_path.relative_to(dataset_dir))

        df = self._standardize_pad(df, dataset_name, dataset_dir, cfg)
        return df

    def load_all_dermatology_datasets(self, data_dir: str) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        for name in self.dermatology_datasets:
            try:
                datasets[name] = self.load_dataset(name, data_dir)
                logging.info("[SUCCESS] Loaded %s: %d rows", name, len(datasets[name]))
            except Exception as e:
                logging.error("Failed to load %s: %s", name, e)
        return datasets

    def _resolve_dataset_dir(self, data_dir: Path, cfg: dict[str, Any]) -> Path:
        relative_dir = cfg.get("relative_dir")
        candidates = []
        if relative_dir:
            candidates.append(data_dir / relative_dir)
        candidates.append(data_dir)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _standardize_pad(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        dataset_dir: Path,
        cfg: dict[str, Any],
    ) -> pd.DataFrame:
        target_col = cfg.get("target", "diagnostic")
        positive_labels = {
            str(v).upper() for v in cfg.get("positive_labels", ["BCC", "SCC", "MEL"])
        }
        negative_labels = {
            str(v).upper() for v in cfg.get("negative_labels", ["ACK", "NEV", "SEK"])
        }
        image_id_col = cfg.get("image_id_column", "img_id")

        required = [target_col, image_id_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{dataset_name}: missing required metadata columns: {missing}")

        out = df.copy()
        out["diagnostic_label"] = out[target_col].astype(str).str.strip().str.upper()
        mapping = {label: 1 for label in positive_labels}
        mapping.update({label: 0 for label in negative_labels})
        out["skin_cancer"] = out["diagnostic_label"].map(mapping)
        unmapped = sorted(set(out.loc[out["skin_cancer"].isna(), "diagnostic_label"]))
        if unmapped:
            logging.warning(
                "%s: dropping %d rows with unmapped diagnostics: %s",
                dataset_name,
                out["skin_cancer"].isna().sum(),
                unmapped,
            )
            out = out[out["skin_cancer"].notna()].copy()
        out["skin_cancer"] = out["skin_cancer"].astype(int)

        if "age" in out.columns:
            out["age_raw"] = pd.to_numeric(out["age"], errors="coerce")
            out["age_group"] = pd.cut(
                out["age_raw"],
                bins=cfg.get("age_bins", [0, 20, 40, 60, 80, 120]),
                labels=cfg.get("age_labels", ["<20", "20-39", "40-59", "60-79", "80+"]),
                include_lowest=True,
            ).astype("object")
            out["age_group"] = out["age_group"].fillna("unknown")
        else:
            out["age_raw"] = pd.NA
            out["age_group"] = "unknown"

        if "gender" in out.columns:
            sex_label = out["gender"].astype("string").str.strip().str.upper()
            sex_label = sex_label.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
            out["sex_extended"] = sex_label.map({"FEMALE": "Female", "MALE": "Male"}).fillna(
                "unknown"
            )
        else:
            out["sex_extended"] = "unknown"
        out["sex"] = out["sex_extended"].map({"Female": 0, "Male": 1}).fillna(-1).astype(int)
        out["sex_bin"] = out["sex"]

        fst_col = (
            "fitspatrick"
            if "fitspatrick" in out.columns
            else "fitzpatrick" if "fitzpatrick" in out.columns else None
        )
        if fst_col:
            fst = pd.to_numeric(out[fst_col], errors="coerce")
            out["fitzpatrick"] = fst.astype("Int64").astype("string").replace("<NA>", "unknown")
            out["fitzpatrick_group"] = fst.map(self._fitzpatrick_group).fillna("unknown")
        else:
            out["fitzpatrick"] = "unknown"
            out["fitzpatrick_group"] = "unknown"

        image_index = self._build_image_index(dataset_dir, cfg)
        image_paths = out[image_id_col].astype(str).map(image_index)
        out["image_path"] = image_paths
        missing_mask = out["image_path"].isna()
        report = {
            "dataset": dataset_name,
            "metadata_rows": int(len(out)),
            "image_files_indexed": int(len(image_index)),
            "found_images": int((~missing_mask).sum()),
            "missing_images": int(missing_mask.sum()),
            "missing_image_ids": out.loc[missing_mask, image_id_col].astype(str).head(50).tolist(),
            "layout_patterns": cfg.get("image_globs", ["images/*.png", "images/imgs_part_*/*.png"]),
        }
        self.last_image_reports[dataset_name] = report
        if missing_mask.any():
            logging.warning(
                "%s: filtering %d rows with missing images", dataset_name, int(missing_mask.sum())
            )
            out = out[~missing_mask].copy()

        out["patient_id"] = out.get(
            "patient_id", pd.Series(range(len(out)), index=out.index)
        ).astype(str)
        out["lesion_id"] = out.get("lesion_id", out["patient_id"]).astype(str)

        keep_cols = cfg.get("include_features") or []
        if keep_cols:
            required_cols = {
                "patient_id",
                "lesion_id",
                "image_path",
                "diagnostic_label",
                "skin_cancer",
                "age_raw",
                "age_group",
                "sex",
                "sex_extended",
                "sex_bin",
                "fitzpatrick",
                "fitzpatrick_group",
                "_dataset_source",
                "_dataset_file",
            }
            keep = [col for col in out.columns if col in set(keep_cols) | required_cols]
            out = out[keep].copy()

        return out.reset_index(drop=True)

    def _build_image_index(self, dataset_dir: Path, cfg: dict[str, Any]) -> dict[str, str]:
        globs = cfg.get("image_globs") or ["images/*.png", "images/imgs_part_*/*.png"]
        index: dict[str, str] = {}
        duplicates: set[str] = set()
        for pattern in globs:
            for path in dataset_dir.glob(pattern):
                if not path.is_file():
                    continue
                key = path.name
                if key in index and index[key] != str(path):
                    duplicates.add(key)
                index[key] = str(path)
        if duplicates:
            logging.warning("Duplicate image basenames found: %s", sorted(duplicates)[:20])
        return index

    @staticmethod
    def _fitzpatrick_group(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= 2:
            return "I-II"
        if value <= 4:
            return "III-IV"
        return "V-VI"


def get_dataset_summary(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    """Basic summary used by loading script."""
    return {
        "dataset_name": dataset_name,
        "n_samples": int(len(df)),
        "n_features": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
    }


def load_standardized_raw(dataset: str, root: str) -> pd.DataFrame:
    """
    Load standardized raw cardiac dataset and harmonize schema.
    Expected location: {root}/data/raw/cardiac/{dataset}_standardized.csv
    """
    path = os.path.join(root, "data", "raw", "cardiac", f"{dataset}_standardized.csv")
    df = pd.read_csv(path)
    return harmonize_cardiac_schema(df, dataset)


def load_processed_splits(
    dataset: str, root: str, scaled: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train/test splits for a cardiac dataset.
    scaled=True loads *_train_scaled.csv and *_test_scaled.csv; otherwise loads *_train.csv and *_test.csv
    Paths (legacy): {root}/data/processed/cardiac/{dataset}_train[_scaled].csv
           {root}/data/processed/cardiac/{dataset}_test[_scaled].csv
    Use load_processed_dataset for binning-aware paths.
    """
    suffix = "_scaled" if scaled else ""
    droot = os.path.join(root, "data", "processed", "cardiac")
    train_path = os.path.join(droot, f"{dataset}_train{suffix}.csv")
    test_path = os.path.join(droot, f"{dataset}_test{suffix}.csv")
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    return X_train, X_test


def load_processed_dataset(
    dataset: str,
    root: str,
    area: str = "cardiac",
    binning: str | None = None,
    scaled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed splits, supporting per-binning subdirectories.

    Paths: {root}/data/processed/{area}/{dataset}_{binning}/{dataset}_train[_scaled].csv
           (if binning is None, uses {dataset}/…)
    """
    suffix = "_scaled" if scaled else ""
    base = Path(root) / "data" / "processed" / area
    subdir = f"{dataset}_{binning}" if binning else dataset
    data_dir = base / subdir
    train_path = data_dir / f"{dataset}_train{suffix}.csv"
    test_path = data_dir / f"{dataset}_test{suffix}.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed split not found under {data_dir} (looked for {train_path.name} & {test_path.name})"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)
