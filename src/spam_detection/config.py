"""Project paths and configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml

# src/spam_detection/config.py -> project root (requires an editable install: pip install -e .)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML configuration file."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from the config relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path