"""Utilities: seed fixing, YAML config loading, experiment artifact logging."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    """Pin deterministic seeds across Python, NumPy, and scikit-learn consumers."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load experiment YAML config."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "experiment_config.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Create results/data directories and return resolved Path map."""
    if config is None:
        config = load_config()
    paths = {}
    for key, rel in config["paths"].items():
        p = PROJECT_ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def log_artifact(name: str, payload: dict[str, Any], metrics_dir: Path) -> Path:
    """Append-friendly JSON artifact logger."""
    out = metrics_dir / f"{name}.json"
    save_json(payload, out)
    return out
