"""Source-backed support contracts; generic planes and fixtures are forbidden."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import yaml


class SupportClassification(str, Enum):
    SUPPORT_EXPLICIT_VALIDATED = "SUPPORT_EXPLICIT_VALIDATED"
    SUPPORT_RECOVERED_VALIDATED = "SUPPORT_RECOVERED_VALIDATED"
    HAND_SUPPORTED_VALIDATED = "HAND_SUPPORTED_VALIDATED"
    UNSUPPORTED_REFERENCE = "UNSUPPORTED_REFERENCE"
    SUPPORT_UNKNOWN = "SUPPORT_UNKNOWN"


class SupportMode(str, Enum):
    SOURCE_SUPPORT = "SOURCE_SUPPORT"
    HAND_SUPPORTED = "HAND_SUPPORTED"
    CONTACT_READY_ONLY_VALIDATED = "CONTACT_READY_ONLY_VALIDATED"
    BLOCKED = "BLOCKED"


_SUPPORT_NAME_TOKENS = ("support", "table", "furniture", "environment", "scene")
_GEOMETRY_SUFFIXES = {".obj", ".ply", ".stl", ".usd", ".usda", ".usdc"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceSupportAssetV1:
    """A recoverable source asset only; this is not a runtime convenience plane."""

    source_path: str
    mesh_sha256: str
    pose_world: tuple[float, ...]
    scale: tuple[float, float, float]
    collision_approximation: str
    friction_source: str
    static: bool = True

    def __post_init__(self) -> None:
        path = Path(self.source_path)
        name = path.name.lower()
        if not path.is_absolute() or path.suffix.lower() not in _GEOMETRY_SUFFIXES:
            raise ValueError("SOURCE_SUPPORT_ASSET_PATH_INVALID")
        if not any(token in name for token in _SUPPORT_NAME_TOKENS):
            raise ValueError("SOURCE_SUPPORT_ASSET_PROVENANCE_NOT_EXPLICIT")
        if len(self.mesh_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.mesh_sha256
        ):
            raise ValueError("SOURCE_SUPPORT_ASSET_HASH_INVALID")
        if len(self.pose_world) not in {7, 16} or not all(
            isinstance(value, float) for value in self.pose_world
        ):
            raise ValueError("SOURCE_SUPPORT_ASSET_POSE_INVALID")
        if len(self.scale) != 3 or any(value <= 0.0 for value in self.scale):
            raise ValueError("SOURCE_SUPPORT_ASSET_SCALE_INVALID")
        if self.collision_approximation not in {
            "triangle_mesh",
            "convex_decomposition",
            "simple_proxy",
        }:
            raise ValueError("SOURCE_SUPPORT_COLLISION_APPROXIMATION_INVALID")
        if not self.static:
            raise ValueError("SOURCE_SUPPORT_ASSET_MUST_BE_STATIC")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSupportContractV1:
    """Versioned evidence behind source support reconstruction or its absence."""

    identifier: str = "Stage16SourceSupportContractV1"
    evidence_hierarchy: tuple[str, ...] = (
        "source_explicit_metadata",
        "source_scene_geometry",
        "local_rgbd_or_scene_reconstruction",
        "dynamic_or_geometric_inference",
    )
    assets: tuple[SourceSupportAssetV1, ...] = ()
    hidden_support: bool = False

    def __post_init__(self) -> None:
        if self.identifier != "Stage16SourceSupportContractV1":
            raise ValueError("SOURCE_SUPPORT_CONTRACT_IDENTIFIER_INVALID")
        if self.evidence_hierarchy != (
            "source_explicit_metadata",
            "source_scene_geometry",
            "local_rgbd_or_scene_reconstruction",
            "dynamic_or_geometric_inference",
        ):
            raise ValueError("SOURCE_SUPPORT_EVIDENCE_HIERARCHY_DRIFT")
        if self.hidden_support:
            raise ValueError("SOURCE_SUPPORT_HIDDEN_SUPPORT_FORBIDDEN")
        paths = [asset.source_path for asset in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("SOURCE_SUPPORT_ASSET_DUPLICATE")

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["assets"] = [asset.as_dict() for asset in self.assets]
        return value


def _walk_mapping(value: object, *, prefix: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            location = f"{prefix}.{key_text}" if prefix else key_text
            yield location, child
            yield from _walk_mapping(child, prefix=location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping(child, prefix=f"{prefix}[{index}]")


def discover_source_support_evidence(sequence_dir: Path) -> dict[str, object]:
    """Inspect already-local source metadata/assets without manufacturing support."""

    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"SOURCE_SUPPORT_SEQUENCE_MISSING:{sequence_dir}")
    metadata_files = sorted(sequence_dir.glob("*.yaml")) + sorted(sequence_dir.glob("*.yml"))
    metadata_hits: list[dict[str, object]] = []
    for path in metadata_files:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        for location, value in _walk_mapping(loaded):
            text = f"{location}:{value}".lower()
            if any(token in text for token in _SUPPORT_NAME_TOKENS):
                metadata_hits.append({"path": str(path), "field": location, "value": value})
    geometry_files = sorted(
        path
        for path in sequence_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _GEOMETRY_SUFFIXES
        and any(token in path.name.lower() for token in _SUPPORT_NAME_TOKENS)
    )
    return {
        "schema_version": "Stage16SourceSupportEvidenceDiscoveryV1",
        "sequence_dir": str(sequence_dir.resolve()),
        "metadata_files_checked": [str(path.resolve()) for path in metadata_files],
        "metadata_support_hits": metadata_hits,
        "source_scene_geometry_candidates": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in geometry_files
        ],
        "local_rgbd_or_scene_reconstruction": "NOT_PRESENT_IN_SEQUENCE_DIRECTORY",
        "network_download_performed": False,
    }


__all__ = [
    "SourceSupportAssetV1",
    "SourceSupportContractV1",
    "SupportClassification",
    "SupportMode",
    "discover_source_support_evidence",
    "sha256_file",
]
