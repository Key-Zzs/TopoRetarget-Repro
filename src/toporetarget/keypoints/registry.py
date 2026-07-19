"""Explicit YAML-backed layout and mapping-profile registry."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from toporetarget.keypoints.layouts import KeypointLayoutDefinition
from toporetarget.keypoints.profiles import MappingProfile


def _default_config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "keypoints"


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"keypoint config must be a mapping: {path}")
    return loaded


@lru_cache(maxsize=8)
def _load_layouts_cached(root_string: str) -> dict[str, KeypointLayoutDefinition]:
    root = Path(root_string)
    layouts: dict[str, KeypointLayoutDefinition] = {}
    for path in sorted((root / "layouts").glob("*.yaml")):
        layout = KeypointLayoutDefinition.from_mapping(_read_yaml(path))
        if layout.name in layouts:
            raise ValueError(f"duplicate keypoint layout: {layout.name}")
        layouts[layout.name] = layout
        for alias in layout.aliases:
            if alias in layouts:
                raise ValueError(f"duplicate keypoint layout alias: {alias}")
            layouts[alias] = layout
    if "mediapipe21" not in layouts:
        raise ValueError("mediapipe21 layout is not registered")
    return layouts


def load_layouts(config_root: str | Path | None = None) -> dict[str, KeypointLayoutDefinition]:
    """Load and validate all tracked layout definitions."""

    root = Path(config_root).expanduser() if config_root is not None else _default_config_root()
    return dict(_load_layouts_cached(str(root.resolve())))


def get_layout(name: str, config_root: str | Path | None = None) -> KeypointLayoutDefinition:
    try:
        return load_layouts(config_root)[name]
    except KeyError as exc:
        raise KeyError(f"unknown keypoint layout: {name}") from exc


@lru_cache(maxsize=8)
def _load_profiles_cached(root_string: str) -> dict[str, MappingProfile]:
    root = Path(root_string)
    profiles: dict[str, MappingProfile] = {}
    layouts = load_layouts(root)
    for path in sorted((root / "mappings").glob("*.yaml")):
        profile = MappingProfile.from_mapping(_read_yaml(path), path=path)
        profile.validate(layouts)
        if profile.profile_id in profiles:
            raise ValueError(f"duplicate mapping profile: {profile.profile_id}")
        profiles[profile.profile_id] = profile
    return profiles


def load_profiles(config_root: str | Path | None = None) -> dict[str, MappingProfile]:
    """Load and validate all tracked MANO mapping profiles."""

    root = Path(config_root).expanduser() if config_root is not None else _default_config_root()
    return dict(_load_profiles_cached(str(root.resolve())))


__all__ = ["get_layout", "load_layouts", "load_profiles"]
