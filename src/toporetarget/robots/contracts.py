"""Data-driven contracts shared by all target-hand implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


def _tuple_strings(values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or ()))


@dataclass(frozen=True)
class RobotKinematicSpec:
    """Kinematic conventions that are independent of a particular URDF parser."""

    root_link: str
    actuated_joint_order: tuple[str, ...]
    neutral_q: tuple[float, ...]
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.actuated_joint_order) != len(self.neutral_q):
            raise ValueError("kinematic joint order and neutral_q must have equal length")
        if len(set(self.actuated_joint_order)) != len(self.actuated_joint_order):
            raise ValueError("kinematic joint order must contain unique names")

    @classmethod
    def from_mapping(cls, values: dict[str, Any], *, root_link: str) -> RobotKinematicSpec:
        limits: dict[str, tuple[float, float]] = {}
        for name, values_for_joint in dict(values.get("joint_limits", {})).items():
            if len(values_for_joint) != 2:
                raise ValueError(f"joint limit for {name!r} must contain lower and upper")
            limits[str(name)] = (float(values_for_joint[0]), float(values_for_joint[1]))
        return cls(
            root_link=root_link,
            actuated_joint_order=_tuple_strings(values.get("dof_order")),
            neutral_q=tuple(float(value) for value in values.get("neutral_q", ())),
            joint_limits=limits,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_link": self.root_link,
            "actuated_joint_order": list(self.actuated_joint_order),
            "neutral_q": list(self.neutral_q),
            "joint_limits": {
                name: [bounds[0], bounds[1]] for name, bounds in sorted(self.joint_limits.items())
            },
        }


@dataclass(frozen=True)
class RobotSemanticAnchorProfile:
    """Semantic keypoint profile declared by a target hand."""

    profile_id: str
    layout_name: str
    version: str = "1.0.0"
    source: str = ""
    assumptions: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "layout_name": self.layout_name,
            "version": self.version,
            "source": self.source,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class RobotSurfaceProfile:
    """Visual and surface-contact declarations; no contact query is implied."""

    visual_geometry: dict[str, Any] = field(default_factory=dict)
    surface_contact: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "visual_geometry": dict(self.visual_geometry),
            "surface_contact": dict(self.surface_contact),
        }


@dataclass(frozen=True)
class RobotCollisionProfile:
    """Collision geometry and self-collision declarations."""

    geometry: dict[str, Any] = field(default_factory=dict)
    self_collision: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometry": dict(self.geometry),
            "self_collision": dict(self.self_collision),
        }


@dataclass(frozen=True)
class RobotSimulationSpec:
    """Optional simulator assets and explicit joint mapping metadata."""

    mjcf_relative_path: str | None = None
    simulator_joint_mapping: dict[str, str] = field(default_factory=dict)
    root_link: str | None = None
    qpos_order: tuple[str, ...] = ()
    actuator_order: tuple[str, ...] = ()
    tip_sites: tuple[str, ...] = ()
    collision_source: str | None = None
    excluded_collision_pairs: tuple[tuple[str, str], ...] = ()
    timestep_hints: dict[str, Any] = field(default_factory=dict)
    known_limitations: tuple[str, ...] = ()
    source_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mjcf_relative_path is not None:
            relative = PurePosixPath(self.mjcf_relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("mjcf_relative_path must be repository-relative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mjcf_relative_path": self.mjcf_relative_path,
            "simulator_joint_mapping": dict(sorted(self.simulator_joint_mapping.items())),
            "root_link": self.root_link,
            "qpos_order": list(self.qpos_order),
            "actuator_order": list(self.actuator_order),
            "tip_sites": list(self.tip_sites),
            "collision_source": self.collision_source,
            "excluded_collision_pairs": [list(pair) for pair in self.excluded_collision_pairs],
            "timestep_hints": dict(self.timestep_hints),
            "known_limitations": list(self.known_limitations),
            "source_hash": self.source_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RobotHandAssetBundle:
    """Path-independent description of a hand's asset bundle."""

    asset_id: str
    root_relative_path: str
    urdf_relative_path: str
    optional_mjcf_relative_path: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("root_relative_path", self.root_relative_path),
            ("urdf_relative_path", self.urdf_relative_path),
        ):
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{name} must be a repository-relative path")

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "root_relative_path": self.root_relative_path,
            "urdf_relative_path": self.urdf_relative_path,
            "optional_mjcf_relative_path": self.optional_mjcf_relative_path,
            "provenance": dict(self.provenance),
        }


__all__ = [
    "RobotCollisionProfile",
    "RobotHandAssetBundle",
    "RobotKinematicSpec",
    "RobotSemanticAnchorProfile",
    "RobotSimulationSpec",
    "RobotSurfaceProfile",
]
