#!/usr/bin/env python3
"""Run the Stage 16-D materializer from the Isaac Lab script directory."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "materialize_stage16d_trajectory.py"),
        run_name="__main__",
    )
