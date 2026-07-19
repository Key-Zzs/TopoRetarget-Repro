"""Small report writers used by Stage 5 CLI and local acceptance scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_grab_report(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return destination


__all__ = ["write_grab_report"]
