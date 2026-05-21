# Utils Module

Shared configuration, logging, warning/error capture, and accelerator detection
helpers.

## Files

| File | Purpose |
|------|---------|
| `config.py` | YAML config loader |
| `logging_utils.py` | Structured phase/study logging, warning/error files, summaries |
| `gpu.py` | Accelerator detection and CUDA/cpu fallback helpers |
| `__init__.py` | Public exports |

## Public API

- `detect_accelerator`
- `setup_logging`

`load_yaml_config` is available from `fairxai.utils.config`.

## Runtime Contracts

- Package modules should use `logging.getLogger(__name__)`.
- Scripts should create phase logs through CLI/logging helpers.
- Warning and error logs are generated alongside phase logs.
- Accelerator detection should degrade to CPU instead of failing local runs.

## Usage

```python
from fairxai.utils import detect_accelerator
from fairxai.utils.config import load_yaml_config

device = detect_accelerator()
cfg = load_yaml_config("configs/pipelines/cardiac.yaml")
```

## Related

- Style guide: [../../../docs/guides/style-guide.md](../../../docs/guides/style-guide.md)
- Testing: [../../../docs/guides/testing.md](../../../docs/guides/testing.md)
