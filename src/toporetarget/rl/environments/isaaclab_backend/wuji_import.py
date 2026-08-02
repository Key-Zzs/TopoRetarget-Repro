"""Wuji import recipe builders without an eager Isaac dependency."""

from __future__ import annotations

import sys
from pathlib import Path


def build_wuji_import_command(
    repo_root: Path,
    *,
    upstream_root: Path,
    python: str | None = None,
    accept_eula: bool = False,
) -> list[str]:
    command = [
        python or sys.executable,
        str(repo_root / "scripts/rl/isaaclab/import_wuji_hand2.py"),
        "--config",
        str(repo_root / "configs/rl/stage16/isaaclab_asset_validation.yaml"),
        "--upstream-root",
        str(upstream_root),
    ]
    if accept_eula:
        command.append("--accept-eula")
    return command


def require_isaaclab() -> object:
    """Import Isaac Lab only when runtime code explicitly asks for it."""

    try:
        import isaaclab
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Lab is optional; run inside the frozen toporetarget-isaaclab environment"
        ) from exc
    return isaaclab
