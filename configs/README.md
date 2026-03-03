# Configs

Configuration files for pipelines, datasets, and experiments.

## Structure

```
configs/
├── datasets/               # Dataset registry and metadata
├── domain/                 # Domain-specific mappings (feature/target)
├── experiments/            # Experiment configs (age binning, mitigation, combinatorial)
├── models/                 # Model defaults and hyperparameters
├── pipelines/              # Pipeline runtime settings
├── profiling/              # Profiling tunables (complexity metrics)
└── schema/                 # Unified schema definitions
```

## Usage

- Pipeline runners load `pipelines/<name>.yaml` for paths and runtime settings.
- Dataset registry lives in `datasets/registry.yaml`.
- Feature/target mappings are under `domain/`.
- Experiments consume files in `experiments/`.
- Profiling tunables (max samples, random seed, solver) live in `profiling/complexity.yaml`.

## Notes

- Dermatology configs are TBD.
