# Data Directory

This directory contains datasets used by FairXAI pipelines.

## Structure

- `external/` - Original source datasets
- `raw/` - Standardized datasets with unified schema (generated)
- `processed/` - Train/test splits and scaled data (generated)

## Datasets

Cardiac pipeline (current):
- Cleveland
- Kaggle Heart
- Cardio70k

Dermatology pipeline: TODO

## Regenerating data

Run the pipeline scripts (see scripts/README.md) to rebuild raw and processed data from `external/` sources.
