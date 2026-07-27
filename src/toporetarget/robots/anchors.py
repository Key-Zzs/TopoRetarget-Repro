"""Robot target anchors backed by the canonical Stage 3 MediaPipe-21 layout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from toporetarget.keypoints.registry import get_layout


@dataclass(frozen=True)
class AnchorDefinition:
    semantic_name: str
    anchor_type: str
    link_name: str | None = None
    joint_name: str | None = None
    local_xyz: tuple[float, float, float] | None = None
    source: str = ""
    parent: str | None = None
    finger: str | None = None
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> AnchorDefinition:
        local = values.get("local_xyz")
        local_xyz = None
        if local is not None:
            if len(local) != 3:
                raise ValueError("local_xyz must contain exactly three values")
            local_xyz = (float(local[0]), float(local[1]), float(local[2]))
        return cls(
            semantic_name=str(values["semantic_name"]),
            anchor_type=str(values["anchor_type"]),
            link_name=None if values.get("link_name") is None else str(values["link_name"]),
            joint_name=None if values.get("joint_name") is None else str(values["joint_name"]),
            local_xyz=local_xyz,
            source=str(values.get("source", "")),
            parent=None if values.get("parent") is None else str(values["parent"]),
            finger=None if values.get("finger") is None else str(values["finger"]),
            confidence=(None if values.get("confidence") is None else float(values["confidence"])),
            provenance=dict(values.get("provenance", {})),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            notes=str(values.get("notes", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_name": self.semantic_name,
            "anchor_type": self.anchor_type,
            "link_name": self.link_name,
            "joint_name": self.joint_name,
            "local_xyz": None if self.local_xyz is None else list(self.local_xyz),
            "source": self.source,
            "parent": self.parent,
            "finger": self.finger,
            "confidence": self.confidence,
            "provenance": dict(self.provenance or {}),
            "assumptions": list(self.assumptions),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AnchorProfile:
    profile_id: str
    version: str
    layout_name: str
    anchors: tuple[AnchorDefinition, ...]
    notes: str = ""

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> AnchorProfile:
        profile = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            layout_name=str(values["layout_name"]),
            anchors=tuple(AnchorDefinition.from_mapping(item) for item in values["anchors"]),
            notes=str(values.get("notes", "")),
        )
        profile.validate()
        return profile

    def validate(self) -> AnchorProfile:
        layout = get_layout(self.layout_name)
        names = tuple(item.semantic_name for item in self.anchors)
        if names != layout.semantic_names:
            raise ValueError(f"{self.profile_id}: anchors must exactly follow {layout.name} order")
        valid_types = {"link_origin", "joint_origin", "link_local_point"}
        for anchor in self.anchors:
            if anchor.anchor_type not in valid_types:
                raise ValueError(f"{self.profile_id}: unsupported anchor type {anchor.anchor_type}")
            if anchor.anchor_type == "link_origin" and not anchor.link_name:
                raise ValueError(f"{anchor.semantic_name}: link_origin requires link_name")
            if anchor.anchor_type == "joint_origin" and not anchor.joint_name:
                raise ValueError(f"{anchor.semantic_name}: joint_origin requires joint_name")
            if anchor.anchor_type == "link_local_point" and (
                not anchor.link_name or anchor.local_xyz is None
            ):
                raise ValueError(
                    f"{anchor.semantic_name}: link_local_point requires link and local_xyz"
                )
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "layout_name": self.layout_name,
            "anchors": [item.as_dict() for item in self.anchors],
            "notes": self.notes,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RobotKeypointSet:
    """Keypoint tensor plus provenance metadata for report/storage boundaries."""

    positions: Any
    metadata: dict[str, Any]

    @property
    def shape(self) -> Any:
        return self.positions.shape

    def as_dict(self) -> dict[str, Any]:
        values = self.positions
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()
        elif hasattr(values, "tolist"):
            values = values.tolist()
        return {"positions": values, "metadata": self.metadata}


def load_anchor_profile(profile_id: str, *, config_root: str | Path | None = None) -> AnchorProfile:
    if config_root is None:
        root = Path(__file__).resolve().parents[3] / "configs" / "robots"
    else:
        root = Path(config_root).expanduser()
    candidates = [
        root / "anchors" / f"{profile_id}.yaml",
        root / "keypoints" / f"{profile_id}.yaml",
        root / f"{profile_id}.yaml",
    ]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"anchor profile not found: {profile_id} below {root}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"anchor profile must be a mapping: {path}")
    return AnchorProfile.from_mapping(loaded)


__all__ = ["AnchorDefinition", "AnchorProfile", "RobotKeypointSet", "load_anchor_profile"]
