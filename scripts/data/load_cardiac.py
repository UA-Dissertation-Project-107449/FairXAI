#!/usr/bin/env python3
"""
Load cardiac datasets and perform initial data profiling.

This script:
1. Loads all cardiac-relevant datasets (Cleveland, Kaggle Heart)
2. Standardizes sensitive attributes and target variables
3. Generates dataset summaries
4. Saves unified datasets to data/raw/cardiac/
5. Creates initial data profiling report

Usage:
    python scripts/data/load_cardiac.py
"""

import sys
import logging
from pathlib import Path
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.data.loaders import CardiacDataLoader, get_dataset_summary
from fairxai.utils.config import load_yaml_config


def setup_logging(log_dir: Path):
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'data_loading.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("Cardiac data loading")
    logging.info("="*60)


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    pipeline_cfg = load_yaml_config(str(project_root / 'configs/pipelines/cardiac.yaml'))
    config_path = project_root / pipeline_cfg['runtime']['schema_mapping_json']
    data_external = project_root / pipeline_cfg['paths']['external_dir']
    data_raw_cardiac = project_root / pipeline_cfg['paths']['raw_dir']
    log_dir = project_root / 'logs/cardiac'
    results_profiling = project_root / 'results/cardiac/profiling'
    
    # Setup
    setup_logging(log_dir)
    data_raw_cardiac.mkdir(parents=True, exist_ok=True)
    results_profiling.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logging.info("Initializing cardiac loader...")
    loader = CardiacDataLoader(str(config_path))
    
    logging.info(f"Loading cardiac datasets from: {data_external}")
    datasets = loader.load_all_cardiac_datasets(str(data_external))
    
    if not datasets:
        logging.error("No datasets loaded. Exiting.")
        return
    
    # Process each dataset
    summaries = []
    standardized_datasets = {}
    
    for dataset_name, df_raw in datasets.items():
        logging.info(f"\n--- Processing: {dataset_name} ---")
        
        # Generate raw summary
        summary = get_dataset_summary(df_raw, dataset_name)
        summaries.append(summary)
        
        logging.info(f"Shape: {df_raw.shape}")
        logging.info(f"Missing values: {df_raw.isnull().sum().sum()} total")
        
        # Datasets returned by loader are already harmonized and standardized
        try:
            # Verify expected columns
            assert 'age_group' in df_raw.columns, "Missing age_group"
            assert 'sex' in df_raw.columns, "Missing sex"
            assert 'heart_disease' in df_raw.columns, "Missing heart_disease"

            standardized_datasets[dataset_name] = df_raw

            # Save to raw/cardiac
            output_file = data_raw_cardiac / f"{dataset_name}_standardized.csv"
            df_raw.to_csv(output_file, index=False)
            logging.info(f"✓ Saved standardized dataset to: {output_file}")

            # Quick stats
            logging.info(f"  Age groups: {df_raw['age_group'].value_counts().to_dict()}")
            logging.info(f"  Sex: {df_raw['sex'].value_counts().to_dict()}")
            logging.info(f"  Heart disease: {df_raw['heart_disease'].value_counts().to_dict()}")

        except Exception as e:
            logging.error(f"✗ Failed to verify/save {dataset_name}: {e}")
            continue
    
    # Save summaries
    summary_file = results_profiling / 'dataset_summaries.json'
    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    logging.info(f"\n✓ Dataset summaries saved to: {summary_file}")
    
    # Generate combined report
    report = {
        'total_datasets': len(datasets),
        'standardized_datasets': len(standardized_datasets),
        'datasets': {}
    }
    
    for name, df in standardized_datasets.items():
        report['datasets'][name] = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'age_groups': df['age_group'].value_counts().to_dict(),
            'sex_distribution': df['sex'].value_counts().to_dict(),
            'disease_prevalence': df['heart_disease'].value_counts().to_dict(),
            'missing_values_total': int(df.isnull().sum().sum())
        }
    
    report_file = results_profiling / 'loading_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info(f"✓ Loading report saved to: {report_file}")
    
    logging.info("\n" + "="*60)
    logging.info("DATA LOADING COMPLETE")
    logging.info("="*60)
    logging.info(f"Standardized datasets saved to: {data_raw_cardiac}")
    logging.info(f"Profiling results saved to: {results_profiling}")
    

if __name__ == "__main__":
    main()
