# Utils Module

Shared utilities used across FairXAI scripts and packages.

This module contains core cross-cutting helpers for logging setup and YAML
configuration loading.

## Files

| File | Purpose |
|------|---------|
| `config.py` | YAML config loading helpers |
| `logging_utils.py` | Centralized logging setup (console + file handlers, verbosity levels) |
| `gpu.py` | GPU/accelerator detection utility |
| `__init__.py` | Public re-exports for utility APIs |

## Public API

- `setup_logging(log_file, verbose=0)`
  - Configures project logging handlers and verbosity behavior.
- `detect_accelerator(requested='auto') -> str`
  - Returns `'cuda'` or `'cpu'`. `'auto'` probes `nvidia-smi`; falls back to `'cpu'` on any error.
  - Pass `'cuda'` or `'cpu'` explicitly to skip detection.

Additional utilities are imported directly from submodules as needed:
- `load_yaml_config` from `config.py`

## GPU Detection

`gpu.py` is used by the combinatorial experiment runner and model wrappers (XGBoost, cuML RF)
to determine which compute backend to use at runtime:

```python
from fairxai.utils.gpu import detect_accelerator

device = detect_accelerator("auto")   # 'cuda' if nvidia-smi succeeds, else 'cpu'
device = detect_accelerator("cuda")   # explicit override — no probe
device = detect_accelerator("cpu")    # force CPU path
```

The function is side-effect free and safe to call multiple times. It does not import any GPU
libraries — it only probes `nvidia-smi` to avoid import errors on CPU-only machines.

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
