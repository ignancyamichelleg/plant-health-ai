"""Load YAML configs: an experiment file is merged on top of configs/base.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Return base updated with override; nested dicts are merged, not replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(experiment_path: str | None = None) -> dict:
    with open(BASE_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if experiment_path:
        with open(experiment_path, encoding="utf-8") as f:
            config = _deep_merge(config, yaml.safe_load(f) or {})
    return config