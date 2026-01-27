import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {p}")
    with open(p, 'r') as f:
        return yaml.safe_load(f)
