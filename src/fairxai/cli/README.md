# CLI Module

Shared runner helpers used by FairXAI scripts.

This module provides common utilities for project-root resolution, pipeline
config loading, logging setup, and run tracking (`latest_run`, run IDs, run
history).

## Files

| File | Purpose |
|------|---------|
| `runner_base.py` | Core runner helpers (`get_project_root`, `load_pipeline_config`, `setup_phase_logging`) |
| `runner_utils.py` | Run ID/pointer/history/archive helpers for reproducible pipeline runs |
| `__init__.py` | Public API re-exports |

## Public API

- `get_project_root(current_file)`
- `load_pipeline_config(root, pipeline='cardiac')`
- `setup_phase_logging(root, log_name, verbose=0, log_subdir='cardiac')`
- `resolve_run_id(explicit=None)`
- `get_run_root(base_results, run_id)`
- `resolve_latest_run_dir(base_results)`
- `update_latest_pointer(base_results, run_dir, logger)`
- `append_run_history(base_results, record)`
- `archive_latest_run(base_dir, enabled, logger)`

## Logging Integration

`setup_phase_logging` delegates to `fairxai.utils.logging_utils.setup_logging`
and writes phase logs under:

```text
logs/{log_subdir}/{log_name}
```

## Run Pointer Contract

Run utilities support both mechanisms:

- symlink pointer: `latest_run -> runs/{run_id}`
- fallback pointer file: `latest_run.txt`

History records are appended to:

- `run_history.jsonl`

## Usage Example

```python
from pathlib import Path
from fairxai.cli import get_project_root, setup_phase_logging, resolve_run_id

root = get_project_root(Path(__file__))
run_id = resolve_run_id()
setup_phase_logging(root, "baseline.log", verbose=1, log_subdir=f"baseline/{run_id}")
```
