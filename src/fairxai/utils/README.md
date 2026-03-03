# Utils Module

Shared utilities used across FairXAI scripts and packages.

This module contains core cross-cutting helpers for logging setup and YAML
configuration loading.

## Files

| File | Purpose |
|------|---------|
| `config.py` | YAML config loading helpers |
| `logging_utils.py` | Centralized logging setup (console + file handlers, verbosity levels) |
| `__init__.py` | Public re-exports for utility APIs |

## Public API

- `setup_logging(log_file, verbose=0)`
  - Configures project logging handlers and verbosity behavior.

Additional utilities are imported directly from submodules as needed:
- `load_yaml_config` from `config.py`

## Logging Notes

`logging_utils` is the common logging entrypoint used by script runners and
pipeline phases; this keeps handler behavior and output formatting consistent
across modules.

## Usage Example

```python
from pathlib import Path
from fairxai.utils import setup_logging
from fairxai.utils.config import load_yaml_config

setup_logging(Path("logs/run.log"), verbose=1)
cfg = load_yaml_config("configs/pipelines/cardiac.yaml")
```
