# Architecture & Design Decisions

## Project Structure

### Directory Organization

- **`src/fairxai/`**: Core Python package
  - `data/`: Data loading, preprocessing, validation
  - `models/`: Model architectures and base classes
  - `fairness/`: Fairness metrics, analysis, and mitigation algorithms
  - `explainability/`: Explainability and interpretability methods
  - `utils/`: Helper functions and utilities

- **`scripts/`**: Standalone Python scripts for running experiments
  - Run main code, can be called from bash or notebooks
  - Better for production-ready code and batch processing

- **`notebooks/`**: Jupyter notebooks for exploration and visualization
  - `runs/`: Output notebooks from script runs (for documentation)
  - Parallel to scripts, not replacing them

- **`experiments/`**: Experiment tracking and configuration
  - `configs/`: YAML/JSON configs for reproducibility
  - `baseline/`, `mitigation/`, `explainability/`: Experiment-specific outputs

- **`data/`**: Dataset management (gitignored for large files)
  - `raw/`: Original datasets
  - `processed/`: Cleaned/preprocessed data
  - `external/`: External reference data

- **`models/`**: Trained model checkpoints (gitignored)

- **`results/`**: All outputs and analysis results
  - `fairness/`, `explainability/`: Metric results by domain
  - `plots/`: Generated visualizations
  - `reports/`: Summary reports

- **`logs/`**: Application logs from script execution

## Workflow

1. **Scripts** (primary): Python scripts in `scripts/` handle core logic
2. **Bash**: Orchestrate scripts, run pipelines
3. **Notebooks** (secondary): Show results, explore interactively, document findings

## Key Principles

- Keep `src/` clean and modular for reusability
- Use `experiments/` for tracking different approaches
- All outputs (results, models, logs) are explicit and organized
- Large files are gitignored to keep repo lightweight
