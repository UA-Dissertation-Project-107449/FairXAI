# Data Directory

This directory contains cardiac disease datasets used in the FairXAI pipeline.

## Structure

- `external/` - Original datasets
- `raw/` - Standardized datasets with unified schema (gitignored)
- `processed/` - Train/test splits ready for modeling (gitignored)

## Datasets

To reproduce the pipeline, run:
```bash
python scripts/data/load_cardiac.py
python scripts/data/preprocess_cardiac.py
```

This will regenerate all processed data from the original sources.
