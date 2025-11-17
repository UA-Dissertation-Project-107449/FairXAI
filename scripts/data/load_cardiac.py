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
    logging.info("CARDIAC DATA LOADING PIPELINE")
    logging.info("="*60)


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / 'experiments/cardiac/configs/schema_mapping.json'
    data_external = project_root / 'data/external'
    data_raw_cardiac = project_root / 'data/raw/cardiac'
    log_dir = project_root / 'logs/cardiac'
    results_profiling = project_root / 'results/cardiac/data_profiling'
    
    # Setup
    setup_logging(log_dir)
    data_raw_cardiac.mkdir(parents=True, exist_ok=True)
    results_profiling.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logging.info("Initializing CardiacDataLoader...")
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
        
        # Standardize
        try:
            df_std = loader.standardize_sensitive_attributes(df_raw, dataset_name)
            df_std = loader.standardize_target(df_std, dataset_name)
            
            # Verify standardization
            assert 'age_group' in df_std.columns, "Missing age_group"
            assert 'sex' in df_std.columns, "Missing sex"
            assert 'heart_disease' in df_std.columns, "Missing heart_disease"
            
            standardized_datasets[dataset_name] = df_std
            
            # Save to raw/cardiac
            output_file = data_raw_cardiac / f"{dataset_name}_standardized.csv"
            df_std.to_csv(output_file, index=False)
            logging.info(f"✓ Saved standardized dataset to: {output_file}")
            
            # Quick stats
            logging.info(f"  Age groups: {df_std['age_group'].value_counts().to_dict()}")
            logging.info(f"  Sex: {df_std['sex'].value_counts().to_dict()}")
            logging.info(f"  Heart disease: {df_std['heart_disease'].value_counts().to_dict()}")
            
        except Exception as e:
            logging.error(f"✗ Failed to standardize {dataset_name}: {e}")
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
