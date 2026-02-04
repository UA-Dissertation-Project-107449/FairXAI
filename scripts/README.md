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

1. Load data
2. Preprocess data (use all binnings for combinatorial)
3. Profile data
4. Train baseline
5. Assess baseline fairness
6. Age binning analysis
7. Mitigation comparison
8. Combinatorial experiments
9. Compare experiments

## Outputs

All outputs are written under `results/<pipeline>/runs/<run_id>/` when `RUN_ID` is set. If not set, outputs go to the default pipeline folders under `results/<pipeline>/`.

## Notes

- `RUN_ID` should be a single value for the whole run to keep outputs grouped.
- Dermatology pipeline runners are TODO.
