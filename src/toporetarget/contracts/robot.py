"""RobotHandPlugin v1, built from the existing generic robot-hand spec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from toporetarget.robots.contracts import (
    RobotCollisionProfile,
    RobotHandAssetBundle,
    RobotKinematicSpec,
    RobotSemanticAnchorProfile,
    RobotSimulationSpec,
    RobotSurfaceProfile,
)
from toporetarget.robots.spec import RobotHandSpec

from .version import ROBOT_HAND_PLUGIN_V1, ROBOT_REFERENCE_V2


@dataclass(frozen=True)
class RobotCapabilities:
    kinematics_ready: bool = False
    retarget_ready: bool = False
    collision_ready: bool = False
    simulation_ready: bool = False
    rl_ready: bool = False

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class RobotReferenceExportProfile:
    schema_version: str = ROBOT_REFERENCE_V2
    formats: tuple[str, ...] = ("npz", "zarr")
    coordinate_frame: str = "robot_base"
    required_arrays: tuple[str, ...] = (
        "qpos_reference",
        "base_pose",
        "object_pose_base",
        "tracked_link_positions",
        "timestamps",
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formats": list(self.formats),
            "coordinate_frame": self.coordinate_frame,
            "required_arrays": list(self.required_arrays),
        }


@dataclass(frozen=True)
class RobotHandPlugin:
    """Complete target-hand declaration consumed by future adapters/workflows."""

    spec: RobotHandSpec
    asset: RobotHandAssetBundle
    kinematics: RobotKinematicSpec
    semantic_anchors: RobotSemanticAnchorProfile
    surface_profile: RobotSurfaceProfile
    collision_profile: RobotCollisionProfile
    simulation_profile: RobotSimulationSpec
    reference_export_profile: RobotReferenceExportProfile = field(
        default_factory=RobotReferenceExportProfile
    )
    capabilities: RobotCapabilities = field(default_factory=RobotCapabilities)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def plugin_version(self) -> str:
        return ROBOT_HAND_PLUGIN_V1

    @classmethod
    def from_spec(cls, spec: RobotHandSpec) -> RobotHandPlugin:
        simulation = spec.simulation
        capabilities = RobotCapabilities(
            kinematics_ready=True,
            retarget_ready=True,
            collision_ready=bool(spec.collision_geometry_policy),
            simulation_ready=bool(spec.optional_mjcf_relative_path),
            # The current repository has no PPO training contract for either
            # target hand.  Keep this false until Stage 16 supplies evidence.
            rl_ready=False,
        )
        return cls(
            spec=spec,
            asset=spec.asset_bundle,
            kinematics=spec.kinematics,
            semantic_anchors=spec.semantic_anchors,
            surface_profile=spec.surface_profile,
            collision_profile=spec.collision_profile,
            simulation_profile=simulation,
            capabilities=capabilities,
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.spec.asset_id != self.asset.asset_id:
            errors.append("asset asset_id does not match RobotHandSpec")
        if len(self.kinematics.actuated_joint_order) != len(self.spec.dof_order):
            errors.append("kinematics joint order does not match RobotHandSpec")
        if not self.reference_export_profile.formats:
            errors.append("reference export profile must declare at least one format")
        if errors:
            raise ValueError("; ".join(errors))
        return errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": ROBOT_HAND_PLUGIN_V1,
            "name": self.name,
            "spec": self.spec.as_dict(),
            "asset": self.asset.as_dict(),
            "kinematics": self.kinematics.as_dict(),
            "semantic_anchors": self.semantic_anchors.as_dict(),
            "surface_profile": self.surface_profile.as_dict(),
            "collision_profile": self.collision_profile.as_dict(),
            "simulation_profile": self.simulation_profile.as_dict(),
            "reference_export_profile": self.reference_export_profile.as_dict(),
            "capabilities": self.capabilities.as_dict(),
        }


class RobotHandPluginRegistry:
    """Registry facade over the existing YAML robot registry."""

    def __init__(self, registry: Any | None = None) -> None:
        if registry is None:
            from toporetarget.robots.registry import RobotHandRegistry

            registry = RobotHandRegistry()
        self.registry = registry

    def names(self) -> tuple[str, ...]:
        return self.registry.names()

    def get(self, name: str) -> RobotHandPlugin:
        return RobotHandPlugin.from_spec(self.registry.get_spec(name))

    def list(self, *, asset_root: str | None = None) -> list[dict[str, Any]]:
        result = []
        for name in self.names():
            plugin = self.get(name)
            row = plugin.as_dict()
            row["asset_availability"] = self.registry.availability(
                plugin.spec, asset_root=asset_root
            )
            result.append(row)
        return result

    def load(self, name: str, **kwargs: Any) -> Any:
        """Load the established FK model while exposing the frozen plugin spec."""

        return self.registry.load(name, **kwargs)


def get_robot_plugin_registry(**kwargs: Any) -> RobotHandPluginRegistry:
    return RobotHandPluginRegistry(**kwargs)


__all__ = [
    "ROBOT_HAND_PLUGIN_V1",
    "ROBOT_REFERENCE_V2",
    "RobotCapabilities",
    "RobotHandPlugin",
    "RobotHandPluginRegistry",
    "RobotReferenceExportProfile",
    "get_robot_plugin_registry",
]
