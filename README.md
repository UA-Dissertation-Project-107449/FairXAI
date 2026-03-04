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

For how to run the pipeline, see:
- scripts/README.md (script-by-script and run_id usage)
- data/README.md (dataset layout and regeneration)
- configs/README.md (config files and how they are used)

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
