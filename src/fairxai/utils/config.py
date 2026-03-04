from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(path: str) -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {p}")
    with open(p, 'r') as f:
        return yaml.safe_load(f)
