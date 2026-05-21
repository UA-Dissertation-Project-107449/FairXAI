# Comparison Module

Helpers for canonical experiment evidence tables, baseline matching, metric
deltas, and figure naming.

## Files

| File | Purpose |
|------|---------|
| `baseline_matching.py` | Baseline lookup keys and safe numeric coercion |
| `metric_tables.py` | Canonical metric/evidence table builders |
| `plot_frames.py` | Plot-ready metric frames |
| `config.py` | Comparison config loader and deep merge |
| `naming.py` | Slugs and configured figure filenames |
| `__init__.py` | Public exports |

## Public API

- `baseline_key_from_row`
- `build_metric_plot_frame`
- `build_baseline_lookups`
- `figure_filename`
- `find_matching_baseline`
- `load_comparison_config`
- `normalize_sensitive_attr`
- `safe_float`
- `safe_int`
- `slugify_token`
- `write_canonical_comparison_outputs`

## Config And Artifacts

- Config: `configs/experiments/comparison.yaml`
- Main script: `scripts/cardiac/compare.py`
- Canonical outputs: `output/cardiac/runs/<run_id>/experiments/comparisons/data/`
- Figure outputs: `output/cardiac/runs/<run_id>/experiments/comparisons/plots/` and dissertation figure roots.

## Usage

```python
from pathlib import Path

from fairxai.comparison import write_canonical_comparison_outputs

write_canonical_comparison_outputs(
    full_df=full_df,
    per_group_df=per_group_df,
    output_dir=Path("output/cardiac/runs/run_id/experiments/comparisons/data"),
    config=comparison_config,
)
```

## Related

- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
- Plot guide: [../../../docs/reference/plots.md](../../../docs/reference/plots.md)
