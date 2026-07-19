from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Resolved external paths; construction does not touch the filesystem."""

    repo_root: Path
    storage_root: Path
    maniptrans_root: Path
    artimano_asset_root: Path
    paper_path: Path
