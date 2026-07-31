"""Dependency-light numerical dashboard fallback for headless Stage-16 runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_dashboard(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = html.escape(json.dumps(payload, indent=2, sort_keys=True, default=str))
    destination.write_text(
        "<!doctype html><meta charset='utf-8'><title>Stage 16 numerical dashboard</title>"
        "<style>body{font-family:system-ui;margin:2rem}"
        "pre{padding:1rem;background:#111;color:#eee;overflow:auto}</style>"
        "<h1>Stage 16 numerical dashboard</h1>"
        "<p>Fallback when renderer evidence is unavailable.</p>"
        f"<pre>{rendered}</pre>",
        encoding="utf-8",
    )
    return destination


__all__ = ["write_dashboard"]
