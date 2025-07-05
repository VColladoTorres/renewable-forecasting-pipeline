# src/utils/yaml_cfg.py
from __future__ import annotations
from pathlib import Path
import yaml

def load(cfg_path: str | Path) -> dict:
    """Carga YAML usando UTF-8 (windows-safe)."""
    cfg_path = Path(cfg_path)
    text = cfg_path.read_text(encoding="utf-8")
    return yaml.safe_load(text)
