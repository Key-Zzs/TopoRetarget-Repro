"""Small report writers shared by Stage 6 commands."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def write_flat_csv(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("field", "value"))
        for key, value in payload.items():
            writer.writerow((key, json.dumps(json_ready(value), sort_keys=True, default=str)))
    return destination


__all__ = ["json_ready", "write_flat_csv", "write_json"]
