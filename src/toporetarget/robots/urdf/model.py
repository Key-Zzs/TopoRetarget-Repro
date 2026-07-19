"""In-memory URDF model objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointLimit:
    lower: float = -float("inf")
    upper: float = float("inf")
    effort: float | None = None
    velocity: float | None = None


@dataclass(frozen=True)
class GeometrySpec:
    geometry_type: str
    origin: np.ndarray
    parameters: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None
    resolved_path: Path | None = None
    source_hash: str | None = None


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visuals: tuple[GeometrySpec, ...] = ()
    collisions: tuple[GeometrySpec, ...] = ()


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limit: JointLimit
    dof_index: int | None = None

    @property
    def actuated(self) -> bool:
        return self.joint_type in {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class UrdfModel:
    name: str
    links: dict[str, LinkSpec]
    joints: tuple[JointSpec, ...]
    root_link: str
    link_order: tuple[str, ...]
    urdf_path: Path

    @property
    def link_names(self) -> tuple[str, ...]:
        return self.link_order

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    @property
    def joint_by_name(self) -> dict[str, JointSpec]:
        return {joint.name: joint for joint in self.joints}

    @property
    def parent_joint_by_child(self) -> dict[str, JointSpec]:
        return {joint.child: joint for joint in self.joints}

    @property
    def actuated_joints(self) -> tuple[JointSpec, ...]:
        return tuple(joint for joint in self.joints if joint.actuated)

    @property
    def fixed_joints(self) -> tuple[JointSpec, ...]:
        return tuple(joint for joint in self.joints if joint.joint_type == "fixed")


__all__ = ["GeometrySpec", "JointLimit", "JointSpec", "LinkSpec", "UrdfModel"]
