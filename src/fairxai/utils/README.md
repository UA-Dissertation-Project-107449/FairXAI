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

## Warning & Error Capture

`setup_logging` installs structured formatters for the dedicated
`*_warnings.log` and `*_errors.log` sidecar files.

### Category prefixes
Every captured Python warning is prefixed with its category class:
```
2026-03-04 ... - WARNING - [DeprecationWarning] some/file.py:42: ...
```
Every logged exception is prefixed with its exception class:
```
2026-03-04 ... - ERROR - [KeyError] failed to load config key 'bins'
```
This is achieved via a custom `warnings.showwarning` hook (installed after
`logging.captureWarnings(True)`) that attaches `warning_category` as a
structured `extra` field, and `_WarningFormatter` / `_ErrorFormatter`
subclasses that read it at format time.

### Suppressing noisy third-party warnings
Some explainer backends (e.g. SHAP `PermutationExplainer`) spawn internal
threads via joblib. `warnings.catch_warnings` is **not** effective there
because its `__exit__` restores the global filter list while worker threads
are still running.

The correct pattern is a permanent filter registered **before** the
explainer is constructed:
```python
warnings.filterwarnings("ignore", message=".*least populated class.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_splits.*", category=UserWarning)
explainer = shap.Explainer(model, df)
shap_values = explainer(df)
```
This writes directly to `warnings.filters` and persists for the process
lifetime and all its threads.

## Usage Example

```python
from pathlib import Path
from fairxai.utils import setup_logging
from fairxai.utils.config import load_yaml_config

setup_logging(Path("logs/run.log"), verbose=1)
cfg = load_yaml_config("configs/pipelines/cardiac.yaml")
```
