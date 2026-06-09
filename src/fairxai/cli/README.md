# CLI Module

Shared helpers for script entry points: project-root resolution, pipeline config
loading, run ID handling, latest-run pointers, log setup, and run history.

## Files

| File | Purpose |
|------|---------|
| `runner_base.py` | Project root, pipeline config, phase/study logging setup |
| `runner_utils.py` | Run ID, run roots, latest pointers, run history, archive helpers |
| `memory_utils.py` | Memory-aware worker/job helpers for larger studies |
| `characterize.py` | WebApp-compatible characterization CLI |
| `main.py` | Unified console-script entry point |
| `__init__.py` | Public exports |

## Public API

Exported by `fairxai.cli`:

- `get_project_root`
- `load_pipeline_config`
- `setup_phase_logging`
- `setup_study_logging`
- `append_run_history`
- `archive_latest_run`
- `get_run_root`
- `resolve_latest_run_dir`
- `resolve_run_id`
- `update_latest_pointer`

## Runtime Contracts

- Run roots are `output/<pipeline>/runs/<run_id>/`.
- Latest-run pointers are `output/<pipeline>/latest_run` and `output/<pipeline>/latest_run.txt`.
- Log pointers mirror the same pattern under `logs/<pipeline>/`.
- Scripts should use these helpers instead of hardcoding run paths.

## Usage

```python
from pathlib import Path

from fairxai.cli import get_run_root, resolve_run_id

run_id = resolve_run_id()
run_root = get_run_root(Path("output/cardiac"), run_id)
```

```bash
fairxai-characterize \
  --filename cleveland_standardized.csv \
  --datasets-dir data/raw/cardiac \
  --output-dir /tmp/fairxai_characterize
```

## Related

- Pipeline controls: [../../../docs/architecture/pipeline-flow-control.md](../../../docs/architecture/pipeline-flow-control.md)
- Testing guide: [../../../docs/guides/testing.md](../../../docs/guides/testing.md)
