from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path

import yaml

from toporetarget.config.models import PathConfig

_ENV_KEYS = {
    "storage_root": "REF2DEX_STORAGE_ROOT",
    "maniptrans_root": "MANIPTRANS_ROOT",
    "artimano_asset_root": "TOPORETARGET_ARTIMANO_ASSET_ROOT",
    "paper_path": "TOPORETARGET_PAPER_PATH",
}


def _repo_root(repo_root: Path | None) -> Path:
    return (repo_root or Path.cwd()).resolve()


def load_path_config(
    repo_root: Path | None = None,
    *,
    overrides: Mapping[str, str | Path | None] | None = None,
    environ: Mapping[str, str] | None = None,
    local_path: Path | None = None,
) -> PathConfig:
    """Resolve paths without scanning or validating them.

    The explicit precedence is CLI overrides, environment, local YAML, then safe defaults.
    ``overrides`` is intentionally a mapping so the CLI can pass only options supplied by a user.
    """

    root = _repo_root(repo_root)
    env = os.environ if environ is None else environ
    local_file = local_path or root / ".local" / "config.yaml"
    local_values: dict[str, object] = {}
    if local_file.is_file():
        with local_file.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Path config must be a mapping: {local_file}")
        local_values = loaded

    defaults: dict[str, Path] = {
        "storage_root": root / ".local" / "external" / "Ref2Dex_storage",
        "maniptrans_root": root / ".local" / "external" / "ManipTrans",
        "artimano_asset_root": root / "third_party" / "robot_hands" / "artimano",
        "paper_path": root / "docs" / "TopoRetarget.pdf",
    }
    resolved: dict[str, Path] = {}
    supplied = overrides or {}
    for key, env_key in _ENV_KEYS.items():
        value: object = supplied.get(key)
        if value is None:
            value = env.get(env_key)
        if value is None and key == "artimano_asset_root":
            value = env.get("ARTIMANO_ASSET_ROOT")
            if value is not None:
                warnings.warn(
                    "ARTIMANO_ASSET_ROOT is deprecated; use TOPORETARGET_ARTIMANO_ASSET_ROOT",
                    DeprecationWarning,
                    stacklevel=2,
                )
        if value is None:
            value = local_values.get(key)
        if value is not None and not isinstance(value, (str, Path)):
            raise ValueError(f"Path value for {key} must be a string: {value!r}")
        resolved[key] = Path(value).expanduser() if value is not None else defaults[key]
    return PathConfig(repo_root=root, **resolved)
