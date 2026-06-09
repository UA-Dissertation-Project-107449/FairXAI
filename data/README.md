# Data Directory

Data staging area for FairXAI pipelines. Files here are inputs or generated
artifacts, not reusable package code.

See [../docs/README.md](../docs/README.md) for the full docs index.

## Layout

| Path | Purpose |
|------|---------|
| `external/` | Original source datasets or externally downloaded files |
| `raw/` | Standardized raw datasets, e.g. `data/raw/cardiac/*_standardized.csv` |
| `processed/` | Train/test/scaled splits, usually `data/processed/cardiac/<dataset>_<binning>/` |

## Current Datasets

Active cardiac pipeline:

- Cleveland
- Kaggle Heart
- Cardio70k

Dermatology data acquisition is scaffolded, but cardiac is the only active end-to-end pipeline.

## Regenerate

```bash
# Raw + profiling + preprocessing through stage 4
bash scripts/cardiac/cardiac_pipeline.sh --go-until preprocess

# Cleveland-only preprocessing path
python3 flows/cardiac_pipeline.py --datasets cleveland --go-until preprocess
```

Pipeline run artifacts live under `output/cardiac/runs/<run_id>/`; reusable processed splits live under `data/processed/cardiac/`.
