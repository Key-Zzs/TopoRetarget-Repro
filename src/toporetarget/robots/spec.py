"""Serializable, path-independent robot-hand specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True)
class RobotHandSpec:
    """Tracked robot configuration; local asset roots are resolved separately."""

    name: str
    version: str
    side: str
    asset_id: str
    urdf_relative_path: str
    base_link: str
    semantic_keypoint_layout: str
    keypoint_anchor_profile: str
    dof_order: tuple[str, ...]
    neutral_q: tuple[float, ...]
    expected_link_count: int
    expected_total_joint_count: int
    expected_actuated_joint_count: int
    expected_fixed_joint_count: int
    expected_tip_links: tuple[str, ...]
    visual_geometry_policy: dict[str, Any] = field(default_factory=dict)
    collision_geometry_policy: dict[str, Any] = field(default_factory=dict)
    self_collision: dict[str, Any] = field(default_factory=dict)
    upstream_provenance: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.urdf_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("urdf_relative_path must remain a repository-relative asset path")
        if len(self.neutral_q) != len(self.dof_order):
            raise ValueError("neutral_q must contain exactly one value per configured DoF")
        if len(set(self.dof_order)) != len(self.dof_order):
            raise ValueError("dof_order entries must be unique")
        counts = (
            self.expected_link_count,
            self.expected_total_joint_count,
            self.expected_actuated_joint_count,
            self.expected_fixed_joint_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("expected topology counts must be non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> RobotHandSpec:
        return cls(
            name=str(values["name"]),
            version=str(values.get("version", "1.0.0")),
            side=str(values["side"]),
            asset_id=str(values["asset_id"]),
            urdf_relative_path=str(values["urdf_relative_path"]),
            base_link=str(values["base_link"]),
            semantic_keypoint_layout=str(values["semantic_keypoint_layout"]),
            keypoint_anchor_profile=str(values["keypoint_anchor_profile"]),
            dof_order=tuple(str(item) for item in values["dof_order"]),
            neutral_q=tuple(float(item) for item in values["neutral_q"]),
            expected_link_count=int(values["expected_link_count"]),
            expected_total_joint_count=int(values["expected_total_joint_count"]),
            expected_actuated_joint_count=int(values["expected_actuated_joint_count"]),
            expected_fixed_joint_count=int(values["expected_fixed_joint_count"]),
            expected_tip_links=tuple(str(item) for item in values.get("expected_tip_links", [])),
            visual_geometry_policy=dict(values.get("visual_geometry_policy", {})),
            collision_geometry_policy=dict(values.get("collision_geometry_policy", {})),
            self_collision=dict(values.get("self_collision", {})),
            upstream_provenance=dict(values.get("upstream_provenance", {})),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            notes=str(values.get("notes", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "side": self.side,
            "asset_id": self.asset_id,
            "urdf_relative_path": self.urdf_relative_path,
            "base_link": self.base_link,
            "semantic_keypoint_layout": self.semantic_keypoint_layout,
            "keypoint_anchor_profile": self.keypoint_anchor_profile,
            "dof_order": list(self.dof_order),
            "neutral_q": list(self.neutral_q),
            "expected_link_count": self.expected_link_count,
            "expected_total_joint_count": self.expected_total_joint_count,
            "expected_actuated_joint_count": self.expected_actuated_joint_count,
            "expected_fixed_joint_count": self.expected_fixed_joint_count,
            "expected_tip_links": list(self.expected_tip_links),
            "visual_geometry_policy": _canonical(self.visual_geometry_policy),
            "collision_geometry_policy": _canonical(self.collision_geometry_policy),
            "self_collision": _canonical(self.self_collision),
            "upstream_provenance": _canonical(self.upstream_provenance),
            "assumptions": list(self.assumptions),
            "notes": self.notes,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            _canonical(self.as_dict()), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def config_hash(self) -> str:
        return self.sha256


__all__ = ["RobotHandSpec"]
