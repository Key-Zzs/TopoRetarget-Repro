"""Serializable, path-independent robot-hand specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .contracts import (
    RobotCollisionProfile,
    RobotHandAssetBundle,
    RobotKinematicSpec,
    RobotSemanticAnchorProfile,
    RobotSimulationSpec,
    RobotSurfaceProfile,
)


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
    asset_root_relative_path: str = ""
    optional_mjcf_relative_path: str | None = None
    qpos_order_profile: str | None = None
    surface_profile_path: str | None = None
    urdf_collision_profile: str | None = None
    mjcf_collision_profile: str | None = None
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    surface_contact_profile: dict[str, Any] = field(default_factory=dict)
    simulator_joint_mapping: dict[str, str] = field(default_factory=dict)
    simulation_metadata: dict[str, Any] = field(default_factory=dict)

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
            asset_root_relative_path=str(
                values.get(
                    "asset_root_relative_path", f"third_party/robot_hands/{values['asset_id']}"
                )
            ),
            optional_mjcf_relative_path=(
                None
                if values.get("optional_mjcf_relative_path") is None
                else str(values["optional_mjcf_relative_path"])
            ),
            qpos_order_profile=(
                None
                if values.get("qpos_order_profile") is None
                else str(values["qpos_order_profile"])
            ),
            surface_profile_path=(
                None if values.get("surface_profile") is None else str(values["surface_profile"])
            ),
            urdf_collision_profile=(
                None
                if values.get("urdf_collision_profile") is None
                else str(values["urdf_collision_profile"])
            ),
            mjcf_collision_profile=(
                None
                if values.get("mjcf_collision_profile") is None
                else str(values["mjcf_collision_profile"])
            ),
            joint_limits={
                str(name): (float(bounds[0]), float(bounds[1]))
                for name, bounds in dict(values.get("joint_limits", {})).items()
            },
            surface_contact_profile=dict(values.get("surface_contact_profile", {})),
            simulator_joint_mapping={
                str(name): str(mapped)
                for name, mapped in dict(values.get("simulator_joint_mapping", {})).items()
            },
            simulation_metadata=dict(values.get("simulation", {})),
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
            "asset_root_relative_path": self.asset_root_relative_path,
            "optional_mjcf_relative_path": self.optional_mjcf_relative_path,
            "qpos_order_profile": self.qpos_order_profile,
            "surface_profile": self.surface_profile_path,
            "urdf_collision_profile": self.urdf_collision_profile,
            "mjcf_collision_profile": self.mjcf_collision_profile,
            "joint_limits": {
                name: [bounds[0], bounds[1]] for name, bounds in sorted(self.joint_limits.items())
            },
            "surface_contact_profile": _canonical(self.surface_contact_profile),
            "simulator_joint_mapping": dict(sorted(self.simulator_joint_mapping.items())),
            "simulation": _canonical(self.simulation_metadata),
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

    @property
    def asset_bundle(self) -> RobotHandAssetBundle:
        return RobotHandAssetBundle(
            asset_id=self.asset_id,
            root_relative_path=self.asset_root_relative_path,
            urdf_relative_path=self.urdf_relative_path,
            optional_mjcf_relative_path=self.optional_mjcf_relative_path,
            provenance=self.upstream_provenance,
        )

    @property
    def kinematics(self) -> RobotKinematicSpec:
        return RobotKinematicSpec(
            root_link=self.base_link,
            actuated_joint_order=self.dof_order,
            neutral_q=self.neutral_q,
            joint_limits=self.joint_limits,
        )

    @property
    def semantic_anchors(self) -> RobotSemanticAnchorProfile:
        return RobotSemanticAnchorProfile(
            profile_id=self.keypoint_anchor_profile,
            layout_name=self.semantic_keypoint_layout,
            source=str(self.upstream_provenance.get("anchor_source", "")),
            assumptions=self.assumptions,
        )

    @property
    def surface_profile(self) -> RobotSurfaceProfile:
        return RobotSurfaceProfile(
            visual_geometry=self.visual_geometry_policy,
            surface_contact=self.surface_contact_profile,
        )

    @property
    def collision_profile(self) -> RobotCollisionProfile:
        return RobotCollisionProfile(
            geometry=self.collision_geometry_policy,
            self_collision=self.self_collision,
        )

    @property
    def simulation(self) -> RobotSimulationSpec:
        metadata = dict(self.simulation_metadata)
        pair_values: list[tuple[str, str]] = []
        for pair in metadata.get("excluded_collision_pairs", ()):
            if len(pair) != 2:
                raise ValueError("excluded_collision_pairs entries must contain two link names")
            pair_values.append((str(pair[0]), str(pair[1])))
        pairs = tuple(pair_values)
        qpos_order = tuple(str(item) for item in metadata.get("qpos_order", self.dof_order))
        actuator_order = tuple(str(item) for item in metadata.get("actuator_order", ()))
        joint_mapping = dict(self.simulator_joint_mapping)
        if not joint_mapping and len(qpos_order) == len(actuator_order):
            joint_mapping = dict(zip(qpos_order, actuator_order, strict=True))
        return RobotSimulationSpec(
            mjcf_relative_path=self.optional_mjcf_relative_path,
            simulator_joint_mapping=joint_mapping,
            root_link=metadata.get("root_link", self.base_link),
            qpos_order=qpos_order,
            actuator_order=actuator_order,
            tip_sites=tuple(
                str(item) for item in metadata.get("tip_sites", self.expected_tip_links)
            ),
            collision_source=metadata.get("collision_source"),
            excluded_collision_pairs=pairs,
            timestep_hints=dict(metadata.get("timestep_hints", {})),
            known_limitations=tuple(str(item) for item in metadata.get("known_limitations", ())),
            source_hash=metadata.get("source_hash"),
            metadata=metadata,
        )


__all__ = ["RobotHandSpec"]
