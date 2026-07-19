"""Arti-MANO adapters; asset acquisition remains owned by Stage 0."""

from __future__ import annotations

from pathlib import Path

from .base import RobotHandModel
from .registry import get_robot_registry


def load_artimano_model(
    side: str,
    *,
    asset_root: str | Path | None = None,
    config_root: str | Path | None = None,
) -> RobotHandModel:
    normalized = side.lower()
    name = {
        "right": "artimano_rh",
        "rh": "artimano_rh",
        "left": "artimano_lh",
        "lh": "artimano_lh",
    }.get(normalized)
    if name is None:
        raise ValueError(f"unsupported Arti-MANO side: {side!r}")
    return get_robot_registry(config_root=config_root).load(name, asset_root=asset_root)


__all__ = ["load_artimano_model"]
