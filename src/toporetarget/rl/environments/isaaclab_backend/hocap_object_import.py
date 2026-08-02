"""HO-Cap object import recipe builders without an eager Isaac dependency."""

from __future__ import annotations

import sys
from pathlib import Path


def build_hocap_import_command(
    repo_root: Path, *, python: str | None = None, accept_eula: bool = False
) -> list[str]:
    command = [
        python or sys.executable,
        str(repo_root / "scripts/rl/isaaclab/import_hocap_objects.py"),
        "--config",
        str(repo_root / "configs/rl/stage16/isaaclab_asset_validation.yaml"),
    ]
    if accept_eula:
        command.append("--accept-eula")
    return command
