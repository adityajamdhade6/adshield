"""Load YAML configuration from configs/ for model parameters, rules and thresholds."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR_ENV = "ADSHIELD_CONFIG_DIR"


def default_config_dir() -> Path:
    """``$ADSHIELD_CONFIG_DIR`` if set, else ``configs/`` at the repository root.

    The repository default only resolves for an editable (``pip install -e .``)
    install or a source checkout; a deployed wheel must set the environment variable.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override) if override else Path(__file__).resolve().parents[2] / "configs"


def load_yaml(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Read ``<config_dir>/<name>`` and return its top-level mapping.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if
    it is empty or its top level is not a mapping.
    """
    path = (config_dir or default_config_dir()) / name
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data
