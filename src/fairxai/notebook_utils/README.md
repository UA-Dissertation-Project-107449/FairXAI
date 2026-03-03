# Notebook Utilities Module

Shared helpers for FairXAI notebooks, focused on repeatable loading,
context resolution, profiling access, and plotting convenience utilities.

This package centralizes notebook-side helpers so exploratory notebooks can
reuse the same conventions as scripts.

## Files

| File | Purpose |
|------|---------|
| `context.py` | Project-root resolution and domain/pipeline context helpers |
| `data.py` | Notebook-friendly dataset loading and stage summaries |
| `profiling.py` | Profiling-output loading and convenience accessors |
| `__init__.py` | Public convenience exports, plotting label helpers, schema-aware utilities |

## Key Utility Areas

- **Context resolution**
  - Resolve project root and dataset context from notebook location.

- **Data access**
  - Load external/raw/processed datasets in canonical structures.

- **Schema-aware helpers**
  - Resolve age units and age-group order from schema mappings.

- **Visualization helpers**
  - Add bar/point labels consistently across notebook charts.
  - Reuse shared palette and units from `fairxai.viz.style`.

## Typical Usage

```python
from fairxai.notebook_utils import (
    resolve_project_root,
    load_processed_scaled_datasets,
    age_group_order,
    add_bar_labels,
)

root = resolve_project_root()
datasets = load_processed_scaled_datasets(root)
```

## Notes

- This package is notebook-facing convenience code; production pipeline logic remains in script and module packages (`data`, `pipeline`, `experiments`).
- Helpers are intentionally pragmatic and may include light formatting helpers not used in batch scripts.
