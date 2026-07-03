# scripts/utils

Standalone provenance, overlap, and archival utilities for the cardiac datasets.
All are keepable/re-runnable; run from the repo root (`Code/FairXAI/`) with the
venv active. The three analysis utilities accept `--json` to persist a
machine-readable summary; `archive_run.py` records directly in
`archive_manifest.json`.

## Scripts

| Script | Purpose |
|---|---|
| `cleveland_provenance.py` | Diff raw UCI `processed.cleveland.data` (303) against the working `cleveland_standardized.csv` (297): confirms complete-case derivation, the 6 dropped ca/thal-missing rows, and cp/slope/target encoding deltas. |
| `cardiac_record_overlap.py` | One-to-one multiset overlap between two standardized cardiac files using a six-field clinical fingerprint. This measures likely reuse, not exact record or patient identity. |
| `cardiac_overlap_matrix.py` | All-pairs fingerprint overlap plus a strict 11-predictor + target source-union audit across the four UCI Heart Disease databases, UCI Statlog Heart, curated Kaggle files, and standardized working files. |
| `archive_run.py` | Copy (or move) a completed run from `output/<domain>/runs/` into `output/<domain>/archived_runs/`, optionally renamed, recording provenance in `archive_manifest.json`. |

## Tests and local data

Synthetic unit tests exercise the matching and provenance logic in CI. The
end-to-end smoke test against the gitignored cardiac files is marked
`local_data`, excluded by both CI workflows, and can be run locally with:

```bash
pytest tests/unit/test_utils_archive_overlap.py -m local_data
```

## UCI Statlog Heart source

The overlap matrix expects the official 270-row Statlog file at:

`data/external/cardiac/statlog_heart_uci/heart.dat`

Source: UCI Machine Learning Repository, Statlog (Heart), DOI
[`10.24432/C57303`](https://doi.org/10.24432/C57303).

Direct file:
`https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat`

Verified SHA-256:
`f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9`

`data/external/` is intentionally gitignored, so this source remains local like
the other raw cardiac distributions. The matrix reports the missing path instead
of silently treating Statlog as an unmatched remainder.

## cardio70k subsample cap (MAX_SAMPLES) — read before full-data runs

`scripts/common/preprocess_data.py` applies a **stratified subsample cap of
10,000 rows by default** (`--max-samples`), *before* clinical-constraint
filtering. This exists because full cardio70k (70,000 rows) OOMs SVM-RBF
training on consumer hardware (the kernel matrix alone reserves ~18 GB against
16 GB RAM).

Consequences and correct usage:

- **A default cardiac run silently profiles only a 10k subsample of cardio70k.**
  The previously reported "9,822 rows after filtering" was this 10k subsample
  minus ~1.8% invalid rows — **not** the full-data cohort.
- **Full-data profiling** (no training, no OOM): set `MAX_SAMPLES` high and stop
  before training, e.g.
  ```bash
  MAX_SAMPLES=1000000 GO_UNTIL=preprocess bash scripts/cardiac/cardiac_pipeline.sh --datasets cardio70k
  ```
  The Prefect flow has the same override through `--max-samples 1000000` (or
  the `MAX_SAMPLES` environment variable).
  On the full 70,000 rows, clinical constraints drop **1,251** records
  (ap_lo 1,006 + ap_hi 228 + height 29 + weight 7, with overlaps) →
  **final analytical n = 68,749**.
- **Do not add `cardio70k` to `cardiac_relevant_datasets`** in
  `configs/schema/cardiac.json`. That list drives the default bulk load/train
  loop; adding cardio70k would pull all 70k rows into SVM-RBF training and OOM
  the machine. cardio70k stays opt-in via `--datasets cardio70k`.
