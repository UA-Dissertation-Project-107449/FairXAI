# Scripts

Entry points for running FairXAI pipelines and experiments.

## Structure

```
scripts/
├── common/                 # Pipeline-agnostic runners
├── cardiac/                # Cardiac wrappers (add --pipeline cardiac)
├── dermatology/            # Dermatology wrappers (TODO)
└── experiments/            # Experiment runners (age binning, mitigation, combinatorial, comparison)
```

## Run order (cardiac)

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `load_data.py` | Download / locate raw CSVs |
| 2 | `profile_data.py` | Generate profiling JSONs (complexity, imbalance, …) |
| 3 | `generate_recommendations.py` | Pre-model fairness triage (see `src/fairxai/recommendations/README.md`) |
| 4 | `preprocess_data.py` | Clean, encode, bin — all binning variants for combinatorial |
| 5 | `train_baseline.py` | Train baseline models |
| 6 | `assess_baseline_fairness.py` | Compute fairness metrics on baselines |
| 7 | `age_binning_analysis.py` | Age-binning sensitivity analysis |
| 8 | `mitigation_comparison.py` | Pre-/in-/post-processing mitigation comparison |
| 9 | `run_combinatorial.py` | Combinatorial experiment matrix |
| 10 | `compare_experiments.py` | Cross-experiment comparison |

Phase 3 is controlled by the `RUN_RECOMMENDATIONS` env var (default `true`).

## Verbosity

All scripts accept a `-v` / `-vv` flag (stacks with `action='count'`):

| Level | Flag | Console output |
|-------|------|----------------|
| 0 (default) | — | `[PHASE]`/`[SUCCESS]`/`[ERROR]` tags + WARNING and above |
| 1 | `-v` | All INFO+ messages |
| 2 | `-vv` | All DEBUG+ messages |

File logs always capture **DEBUG+** regardless of verbosity.  
Dedicated `*_warnings.log` and `*_errors.log` files are always written alongside the main log.

**Bash pipeline** — set `VERBOSE=0`, `1`, or `2` (legacy `true`/`false` still accepted):

```bash
VERBOSE=2 bash scripts/cardiac/cardiac_pipeline.sh   # debug output
```

**Prefect flow:**

```bash
python3 flows/cardiac_pipeline.py -vv   # debug
python3 flows/cardiac_pipeline.py -v    # info
python3 flows/cardiac_pipeline.py       # quiet (default)
```

## Outputs

All outputs are written under `results/<pipeline>/runs/<run_id>/` when `RUN_ID` is set. If not set, outputs go to the default pipeline folders under `results/<pipeline>/`.

## Utility scripts

Two helper scripts live at the **project root** (not inside `scripts/`):

- **`setup.sh`** — bootstraps the virtual environment, checks Python ≥ 3.10, installs `requirements.txt`.
- **`cleanup.sh`** — removes generated outputs (`results/`, `data/processed/`, `data/raw/`, `logs/`). Flags: `--results-only`, `--keep-latest`, `--nuke-env`, `--dry-run`, `-y`.

## Notes

- `RUN_ID` should be a single value for the whole run to keep outputs grouped.
- Dermatology pipeline runners are TODO.
