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

Cardiac (`configs/pipelines/cardiac.yaml`):

- `cleveland_uci` and `four_site_uci`: the default cohorts, built from the raw
  UCI files by `scripts/utils/build_cardiac_uci_cohorts.py`.
- `cardio70k`: opt-in, pass `--datasets cardio70k`.
- `cleveland` and `kaggle_heart`: the older curated files, kept only for the
  provenance tools in `scripts/utils/`.

Dermatology (`configs/pipelines/dermatology.yaml`):

- `pad_ufes_20`: the default image dataset.
- `scin`: declared in `configs/schema/dermatology.json` with its own
  standardizer, but not in the pipeline's default dataset list.

## Regenerate

```bash
# Raw + profiling + preprocessing through stage 4
bash scripts/cardiac/cardiac_pipeline.sh --go-until preprocess

# Cleveland-only preprocessing path
python3 flows/cardiac_pipeline.py --datasets cleveland_uci --go-until preprocess
```

Pipeline run artifacts live under `output/cardiac/runs/<run_id>/`; reusable processed splits live under `data/processed/cardiac/`.
