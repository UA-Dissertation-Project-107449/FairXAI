"""Data loading utilities for cardiac datasets."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

from .schemas import harmonize_cardiac_schema


class CardiacDataLoader:
    """Loader for cardiac disease datasets with schema mapping."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.datasets = self.config.get('datasets', {})
        self.cardiac_datasets = self.config.get('cardiac_relevant_datasets', ["cleveland", "kaggle_heart"])

    def load_dataset(self, dataset_name: str, data_dir: str) -> pd.DataFrame:
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_config = self.datasets[dataset_name]
        filename = dataset_config.get('filename')

        # Prefer data/external/cardiac/{filename}, then data/external/{filename}
        p1 = Path(data_dir) / 'cardiac' / filename
        p2 = Path(data_dir) / filename
        filepath = p1 if p1.exists() else p2
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        logging.info(f"Loading {dataset_name} from {filepath}")
        df = pd.read_csv(filepath)
        df['_dataset_source'] = dataset_name
        df['_dataset_file'] = filename

        # Harmonize base schema and apply sensitive/target standardization
        df = harmonize_cardiac_schema(df, dataset_name)
        df = self._apply_sensitive_standardization(df, dataset_name)
        df = self._apply_target_standardization(df, dataset_name)
        return df

    def load_all_cardiac_datasets(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        datasets: Dict[str, pd.DataFrame] = {}
        for name in self.cardiac_datasets:
            try:
                datasets[name] = self.load_dataset(name, data_dir)
                logging.info(f"✓ Loaded {name}: {len(datasets[name])} rows")
            except Exception as e:
                logging.error(f"✗ Failed to load {name}: {e}")
        return datasets

    def _apply_sensitive_standardization(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        sens = self.datasets.get(dataset_name, {}).get('sensitive_attributes', {})

        # Age binning to age_group if age_raw exists
        age_key = 'age' if 'age' in sens else 'Age' if 'Age' in sens else None
        if age_key and 'age_raw' in df.columns:
            bins = sens[age_key].get('bins', [0, 40, 50, 60, 70, 120])
            labels = sens[age_key].get('labels', ['<40', '40-49', '50-59', '60-69', '70+'])
            df['age_group'] = pd.cut(df['age_raw'], bins=bins, labels=labels, include_lowest=True)

        # Sex mapping to "sex"
        sex_key = 'sex' if 'sex' in sens else 'Sex' if 'Sex' in sens else 'Gender' if 'Gender' in sens else None
        if sex_key:
            mapping = sens[sex_key].get('mapping', {})
            if sex_key in df.columns and 'sex' not in df.columns:
                if pd.api.types.is_numeric_dtype(df[sex_key]):
                    # Convert mapping keys to ints when source is numeric
                    mapping = {int(k): v for k, v in mapping.items()}
                df['sex'] = df[sex_key].map(mapping).fillna(df[sex_key])

        # Extended and binary encodings
        if 'sex' in df.columns:
            df['sex_extended'] = df['sex'].astype('object')
            df['sex_bin'] = df['sex'].map({'Female': 0, 'Male': 1})

        return df

    def _apply_target_standardization(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        cfg = self.datasets.get(dataset_name, {})
        tgt_col = cfg.get('target')
        mapping = cfg.get('target_mapping')
        if tgt_col and tgt_col in df.columns:
            if mapping:
                mapped = df[tgt_col].astype(str).map(mapping)
                df['heart_disease'] = mapped.map({'no_disease': 0, 'disease': 1})
            else:
                df['heart_disease'] = pd.to_numeric(df[tgt_col], errors='coerce')
        return df


def get_dataset_summary(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Basic summary used by loading script."""
    return {
        'dataset_name': dataset_name,
        'n_samples': int(len(df)),
        'n_features': int(len(df.columns)),
        'columns': list(df.columns),
        'missing_total': int(df.isnull().sum().sum())
    }


def load_standardized_raw(dataset: str, root: str) -> pd.DataFrame:
    """
    Load standardized raw cardiac dataset and harmonize schema.
    Expected location: {root}/data/raw/cardiac/{dataset}_standardized.csv
    """
    path = os.path.join(root, "data", "raw", "cardiac", f"{dataset}_standardized.csv")
    df = pd.read_csv(path)
    return harmonize_cardiac_schema(df, dataset)


def load_processed_splits(dataset: str, root: str, scaled: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    binning: Optional[str] = None,
    scaled: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        raise FileNotFoundError(f"Processed split not found under {data_dir} (looked for {train_path.name} & {test_path.name})")
    return pd.read_csv(train_path), pd.read_csv(test_path)
