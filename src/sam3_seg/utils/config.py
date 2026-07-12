"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root, regardless of where code is executed from.

    This file lives at:  <root>/src/sam3_video_segmentation/utils/config.py
    so the root is four parents up.
    """
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config and resolve all relative paths against the project root."""
    root = project_root()
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Turn every entry under `paths:` into an absolute Path object
    cfg["paths"] = {k: root / v for k, v in cfg.get("paths", {}).items()}
    cfg["_root"] = root
    return cfg