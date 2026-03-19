# FairXAI: Fair and Explainable AI for Healthcare Decision Support

## Overview

This dissertation project develops and evaluates technical solutions to make AI-based healthcare systems fair in their predictions and transparent in their decision-making.

### Key Objectives

- **Fairness**: Ensure equitable performance across diverse patient groups (demographic, individual, counterfactual)
- **Explainability**: Make model decisions understandable to healthcare professionals
- **Bias Detection & Mitigation**: Identify and reduce algorithmic bias in medical AI

### Research Scope

- State-of-the-art review on bias and fairness
- Implementation of bias mitigation techniques (pre-, in-, post-processing)
- Integration of explainability methods (model-agnostic and inherently interpretable designs)
- Evaluation of fairness gains vs. performance trade-offs

## Project Structure

See `docs/DECISIONS.md` for detailed architecture decisions.

```
FairXAI/
├── src/fairxai/          # Core package
├── scripts/              # Entry points and pipelines (see scripts/README.md)
├── configs/              # Pipeline/dataset/experiment configs
├── notebooks/            # Jupyter notebooks & runs
├── data/                 # Datasets (external/raw/processed)
├── output/               # Outputs (metrics, plots, reports)
├── models/               # Trained checkpoints
├── logs/                 # Execution logs
└── docs/                 # Documentation
```

## Getting Started

FairXAI uses `pyproject.toml` as the package configuration source.

- `setup.py` is not required for standard development/deployment workflows.
- Editable installs (`pip install -e ...`) expose local source changes immediately.

### Installation Profiles

From the repository root:

```bash
# Core package only
pip install -e .

# Characterization/runtime extras (recommended for profiling + WebApp integration)
pip install -e ".[experiment]"

# Local research/dev tooling (plots, notebooks, explainability extras)
pip install -e ".[dev]"

# Prefect orchestration extras (optional)
pip install -e ".[orchestration]"

# HPC profile (includes experiment extras)
pip install -e ".[hpc]"
```

GPU acceleration is environment-specific and should be installed manually only on compatible hosts.

### Running the Existing Cardiac Pipeline

For full end-to-end runs:

```bash
bash scripts/cardiac/cardiac_pipeline.sh
```

Resume or stop at a specific stage:

```bash
RESUME_FROM=profile GO_UNTIL=recommend bash scripts/cardiac/cardiac_pipeline.sh
```

For script-by-script details, see:
- scripts/README.md (script behavior and run_id usage)
- data/README.md (dataset layout and regeneration)
- configs/README.md (config files and how they are used)

## WebApp Characterization Strategy

WebApp request-time characterization is exposed through a lightweight CLI entrypoint that calls FairXAI profiling internals directly.

Design properties:
- Pipeline logic remains unchanged for research and batch experimentation.
- The WebApp entrypoint is a thin adapter for job execution, not a second profiling implementation.
- Both paths use the same profiling code.

Recommended split:
- Pipeline (`scripts/cardiac/cardiac_pipeline.sh`): full multi-stage research workflow, checkpoints, resume/go-until.
- WebApp characterize CLI: focused single-job path (CSV in, JSON out) for low latency and simpler operations.

CLI usage:

```bash
python3 -m fairxai.cli.characterize \
	--filename cleveland_standardized.csv \
	--datasets-dir data/raw/cardiac \
	--output-dir /tmp/fairxai_characterize
```

Console script (after install):

```bash
fairxai-characterize \
	--filename cleveland_standardized.csv \
	--datasets-dir data/raw/cardiac \
	--output-dir /tmp/fairxai_characterize
```

When to use stage controls instead:
- Use stage controls (`RESUME_FROM`, `GO_UNTIL`) when you need the run-scoped artifacts and checkpoint lifecycle.
- Avoid using full pipeline orchestration for request-time WebApp jobs unless those artifacts are explicitly required.

Contract note for migration from Domain_characterization:
- Keep required metrics available to WebApp.
- Allow additive fields in output JSON.
- Treat `ebmDifficulty` as optional/deferred during initial cutover if needed.

## Datasets

Cardiac pipeline datasets:
- Cleveland
- Kaggle Heart
- Cardio70k

Dermatology pipeline: TODO

## Related Work

- Transfer Learning & Fine-tuning strategies
- Group, individual, and counterfactual fairness definitions
- Explainability vs. performance trade-offs

---

*Dissertation by Miguel | IEETA R&D, University of Aveiro*
