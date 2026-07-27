"""YAML-driven robot-hand registry with lazy URDF/asset loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from toporetarget.paths.assets import (
    AssetResolution,
    check_artimano_assets,
    resolve_artimano_asset,
)

from .base import RobotHandModel
from .spec import RobotHandSpec
from .urdf.parser import parse_urdf


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "robots"


class RobotHandRegistry:
    """Discover robot YAML files without parsing URDFs during listing/import."""

    def __init__(
        self, config_root: str | Path | None = None, *, repo_root: str | Path | None = None
    ) -> None:
        self.config_root = (
            Path(config_root).expanduser().resolve() if config_root is not None else _default_root()
        )
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else self.config_root.parents[1]
        )

    def _paths(self) -> list[Path]:
        return sorted(path for path in self.config_root.glob("*.yaml") if path.is_file())

    def names(self) -> tuple[str, ...]:
        return tuple(self._load_spec(path).name for path in self._paths())

    def _load_spec(self, path: Path) -> RobotHandSpec:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"robot config must be a mapping: {path}")
        return RobotHandSpec.from_mapping(loaded)

    def specs(self) -> list[RobotHandSpec]:
        return [self._load_spec(path) for path in self._paths()]

    def get_spec(self, name: str) -> RobotHandSpec:
        for path in self._paths():
            spec = self._load_spec(path)
            if spec.name == name:
                return spec
        raise KeyError(f"unknown robot {name!r}; choose from {', '.join(self.names())}")

    def _asset_resolution(
        self, spec: RobotHandSpec, override: str | Path | None
    ) -> AssetResolution:
        if spec.asset_id == "artimano":
            return resolve_artimano_asset(self.repo_root, asset_root=override)
        root = (
            Path(override).expanduser().resolve()
            if override is not None
            else self.repo_root / "third_party" / "robot_hands" / spec.asset_id
        )
        return AssetResolution(
            spec.asset_id,
            "override" if override is not None else "tracked",
            root,
            override is not None,
            False,
        )

    def availability(
        self, spec: RobotHandSpec, *, asset_root: str | Path | None = None
    ) -> dict[str, Any]:
        resolution = self._asset_resolution(spec, asset_root)
        root = resolution.root
        urdf = self._find_urdf(root, spec)
        manifest = root / "asset_manifest.json"
        return {
            "asset_root": str(root),
            "resolved_asset_source": resolution.source,
            "resolved_asset_root": str(root),
            "legacy_fallback_used": resolution.legacy_fallback_used,
            "warnings": list(resolution.warnings),
            "source_commit": resolution.as_dict().get("source_commit"),
            "license": resolution.as_dict().get("license"),
            "asset_manifest_hash": resolution.asset_manifest_hash,
            "urdf": spec.urdf_relative_path,
            "resolved_urdf": None if urdf is None else str(urdf),
            "urdf_exists": urdf is not None and urdf.is_file(),
            "manifest_exists": manifest.is_file(),
            "available": urdf is not None and urdf.is_file() and manifest.is_file(),
        }

    @staticmethod
    def _find_urdf(root: Path, spec: RobotHandSpec) -> Path | None:
        candidates = [root / spec.urdf_relative_path]
        # Legacy .local imports kept the source-root flat layout.
        candidates.append(root / Path(spec.urdf_relative_path).name)
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def list(self, *, asset_root: str | Path | None = None) -> list[dict[str, Any]]:
        result = []
        for spec in self.specs():
            result.append(
                {
                    "name": spec.name,
                    "side": spec.side,
                    "urdf": spec.urdf_relative_path,
                    "expected_dofs": len(spec.dof_order),
                    "semantic_layout": spec.semantic_keypoint_layout,
                    "config_status": "ok",
                    "asset": self.availability(spec, asset_root=asset_root),
                }
            )
        return result

    def load(self, name: str, *, asset_root: str | Path | None = None) -> RobotHandModel:
        spec = self.get_spec(name)
        resolution = self._asset_resolution(spec, asset_root)
        root = resolution.root
        if spec.asset_id == "artimano":
            asset_check = check_artimano_assets(root)
            if asset_check.status != "ok":
                raise RuntimeError(
                    f"{spec.name}: asset manifest check failed: {asset_check.message}; "
                    f"missing={asset_check.missing_files}, changed={asset_check.changed_files}"
                )
        urdf_path = self._find_urdf(root, spec)
        if urdf_path is None:
            raise FileNotFoundError(f"{spec.name}: URDF not found below {root}")
        urdf = parse_urdf(urdf_path, asset_root=root)
        manifest_path = root / "asset_manifest.json"
        manifest = None
        if manifest_path.is_file():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"asset manifest must be a mapping: {manifest_path}")
            manifest = loaded
        return RobotHandModel(
            spec,
            urdf,
            asset_root=root,
            asset_manifest=manifest,
            config_root=self.config_root,
            asset_resolution=resolution,
        )


def get_robot_registry(
    *, config_root: str | Path | None = None, repo_root: str | Path | None = None
) -> RobotHandRegistry:
    return RobotHandRegistry(config_root=config_root, repo_root=repo_root)


__all__ = ["RobotHandRegistry", "get_robot_registry"]
