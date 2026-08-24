"""Versioned, dataset-independent support-resolution data types.

The support resolver is deliberately separate from the Stage 16 task and reward
contracts.  A caller can therefore audit support provenance and geometry before
constructing a runtime scene, without changing control or object state
semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SupportType(str, Enum):
    SOURCE_EXPLICIT_SUPPORT = "SOURCE_EXPLICIT_SUPPORT"
    SOURCE_RECONSTRUCTED_SUPPORT = "SOURCE_RECONSTRUCTED_SUPPORT"
    # Historical source name retained as a library alias, never emitted by the
    # current production resolver.
    SOURCE_RECOVERED_SUPPORT = "SOURCE_RECONSTRUCTED_SUPPORT"
    INFERRED_PLANAR_SUPPORT = "INFERRED_PLANAR_SUPPORT"
    HAND_SUPPORTED_ONLY = "HAND_SUPPORTED_ONLY"
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN = "UNRESOLVED"


class SupportResolutionStatus(str, Enum):
    SOURCE_SUPPORT_VALIDATED = "SOURCE_SUPPORT_VALIDATED"
    INFERRED_SUPPORT_VALIDATED = "INFERRED_SUPPORT_VALIDATED"
    INFERRED_SUPPORT_VALIDATED_TRANSFER_DEFERRED = "INFERRED_SUPPORT_VALIDATED_TRANSFER_DEFERRED"
    SUPPORT_RECONSTRUCTION_BLOCKED = "SUPPORT_RECONSTRUCTION_BLOCKED"
    SUPPORT_UNRESOLVED = "SUPPORT_UNRESOLVED"
    SUPPORT_UNKNOWN = "SUPPORT_UNRESOLVED"


class SupportResolutionMode(str, Enum):
    AUTO = "auto"
    SOURCE_ONLY = "source_only"
    INFERRED_PLANAR = "inferred_planar"


class SupportPatchType(str, Enum):
    POINT_SUPPORT = "POINT_SUPPORT"
    EDGE_SUPPORT = "EDGE_SUPPORT"
    AREA_SUPPORT = "AREA_SUPPORT"
    UNSTABLE_SUPPORT_PATCH = "UNSTABLE_SUPPORT_PATCH"


@dataclass(frozen=True)
class SupportCollisionPolicyV1:
    """Pairwise collision behavior for one resolved support authority."""

    schema_version: str
    support_type: SupportType
    object_support_collision: bool
    hand_support_collision: bool
    hand_support_geometry_diagnostics: str = "DIAGNOSTIC_ONLY"
    global_support_collision_disabled: bool = False

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["support_type"] = self.support_type.value
        return value


@dataclass(frozen=True)
class SupportCollisionContractV1:
    schema_version: str = "SupportCollisionContractV1"
    implementation: str = "pairwise_collision_filtering"

    def policy(self, support_type: SupportType | str) -> SupportCollisionPolicyV1:
        selected = SupportType(support_type)
        if selected in {
            SupportType.SOURCE_EXPLICIT_SUPPORT,
            SupportType.SOURCE_RECONSTRUCTED_SUPPORT,
        }:
            return SupportCollisionPolicyV1(
                schema_version=self.schema_version,
                support_type=selected,
                object_support_collision=True,
                hand_support_collision=True,
            )
        if selected is SupportType.INFERRED_PLANAR_SUPPORT:
            return SupportCollisionPolicyV1(
                schema_version=self.schema_version,
                support_type=selected,
                object_support_collision=True,
                hand_support_collision=False,
            )
        return SupportCollisionPolicyV1(
            schema_version=self.schema_version,
            support_type=selected,
            object_support_collision=False,
            hand_support_collision=False,
        )


@dataclass(frozen=True)
class StablePreContactDetectionContractV1:
    """Shared engineering thresholds for stable pre-contact detection.

    These values are frozen in the tracked config before clip-level results are
    generated.  They are intentionally conservative and dataset-agnostic.
    """

    schema_version: str = "StablePreContactDetectionContractV1"
    min_consecutive_frames: int = 8
    max_linear_speed_mps: float = 0.03
    max_angular_speed_radps: float = 0.03
    max_translation_step_m: float = 0.003
    max_rotation_step_rad: float = 0.03
    max_height_mad_m: float = 0.0025
    max_height_range_m: float = 0.01
    support_patch_tolerance_m: float = 0.0015

    def __post_init__(self) -> None:
        if self.min_consecutive_frames < 2:
            raise ValueError("STABLE_INTERVAL_MIN_LENGTH_INVALID")
        for name in (
            "max_linear_speed_mps",
            "max_angular_speed_radps",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "max_height_mad_m",
            "max_height_range_m",
            "support_patch_tolerance_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"STABLE_INTERVAL_THRESHOLD_INVALID:{name}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportExtentContractV1:
    schema_version: str = "SupportExtentContractV1"
    support_extent_margin_m: float = 0.02
    table_thickness_m: float = 0.02

    def __post_init__(self) -> None:
        if self.support_extent_margin_m <= 0.0 or self.table_thickness_m <= 0.0:
            raise ValueError("SUPPORT_EXTENT_PARAMETER_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NominalSupportMaterialV1:
    schema_version: str = "NOMINAL_UNCALIBRATED_SUPPORT_MATERIAL_V1"
    static_friction: float = 0.8
    dynamic_friction: float = 0.6
    restitution: float = 0.0
    provenance: str = "global engineering nominal; not friction optimized"

    def __post_init__(self) -> None:
        if not 0.0 <= self.dynamic_friction <= self.static_friction:
            raise ValueError("SUPPORT_MATERIAL_FRICTION_INVALID")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("SUPPORT_MATERIAL_RESTITUTION_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportPlaneConsistencyGateV1:
    schema_version: str = "SupportPlaneConsistencyGateV1"
    max_mad_m: float = 0.0025
    max_range_m: float = 0.01
    max_object_table_penetration_m: float = 0.002
    max_object_table_gap_m: float = 0.005
    max_hand_table_penetration_m: float = 0.002

    def __post_init__(self) -> None:
        for name in (
            "max_mad_m",
            "max_range_m",
            "max_object_table_penetration_m",
            "max_object_table_gap_m",
            "max_hand_table_penetration_m",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"SUPPORT_GEOMETRY_GATE_INVALID:{name}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportInterval:
    start_frame: int
    end_frame_exclusive: int
    reason: str
    threshold_kind: str = "ENGINEERING_THRESHOLD"

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"frame_count": self.frame_count}


@dataclass(frozen=True)
class StableIntervalResult:
    status: str
    interval: SupportInterval | None
    candidate_intervals: tuple[SupportInterval, ...] = ()
    linear_speed_mps: tuple[float, ...] = ()
    angular_speed_radps: tuple[float, ...] = ()
    translation_step_m: tuple[float, ...] = ()
    rotation_step_rad: tuple[float, ...] = ()
    contact_mask_used: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "interval": self.interval.as_dict() if self.interval else None,
            "candidate_intervals": [item.as_dict() for item in self.candidate_intervals],
            "linear_speed_mps": list(self.linear_speed_mps),
            "angular_speed_radps": list(self.angular_speed_radps),
            "translation_step_m": list(self.translation_step_m),
            "rotation_step_rad": list(self.rotation_step_rad),
            "contact_mask_used": self.contact_mask_used,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SupportPlaneFit:
    plane_normal: tuple[float, float, float]
    plane_offset: float
    h_visual: tuple[float, ...]
    h_collision: tuple[float, ...]
    h_visual_stats: dict[str, float]
    h_collision_stats: dict[str, float]
    delta_support_geometry: float
    support_patch_type: SupportPatchType
    support_patch_vertex_count: int
    support_patch_projected_area_m2: float
    patch_connected_components: int | None
    stable_interval: SupportInterval
    applicability: str = "PLANAR_SUPPORT_APPLICABLE"

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["support_patch_type"] = self.support_patch_type.value
        return value


@dataclass(frozen=True)
class FinitePlanarSupportProxy:
    table_pose: tuple[float, ...]
    table_extent: tuple[float, float]
    table_thickness: float
    plane_normal: tuple[float, float, float]
    plane_offset: float
    material: NominalSupportMaterialV1 = field(default_factory=NominalSupportMaterialV1)
    representation: str = "static_kinematic_rigid_box"

    def __post_init__(self) -> None:
        if len(self.table_pose) != 7 or len(self.table_extent) != 2:
            raise ValueError("SUPPORT_PROXY_POSE_OR_EXTENT_INVALID")
        if self.table_extent[0] <= 0.0 or self.table_extent[1] <= 0.0:
            raise ValueError("SUPPORT_PROXY_EXTENT_INVALID")
        if self.table_thickness <= 0.0:
            raise ValueError("SUPPORT_PROXY_THICKNESS_INVALID")

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["material"] = self.material.as_dict()
        return value


@dataclass(frozen=True)
class GeometryValidation:
    object_table: dict[str, object]
    hand_table: dict[str, object]
    visual_collision_consistent: bool
    status: str
    hand_table_diagnostic_only: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicsValidation:
    object_only_with_support: dict[str, object]
    object_only_without_support: dict[str, object]
    causal_comparison: dict[str, object]
    status: str
    external_guidance: bool = False
    object_rollout_state_writes: int = 0
    hidden_attachment: bool = False
    kinematic_support_force: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportResolutionResult:
    status: str
    support_type: SupportType
    support_source: str
    source_explicit: bool
    source_recovered: bool
    support_inferred: bool
    confidence: float
    plane_normal: tuple[float, float, float] | None
    plane_offset: float | None
    table_pose: tuple[float, ...] | None
    table_extent: tuple[float, float] | None
    table_thickness: float | None
    support_interval: SupportInterval | None
    visual_mesh_evidence: dict[str, object]
    collision_mesh_evidence: dict[str, object]
    geometry_validation: dict[str, object]
    physics_validation: dict[str, object]
    provenance: dict[str, object]
    hashes: dict[str, str]
    stable_interval: StableIntervalResult | None = None
    plane_fit: SupportPlaneFit | None = None
    table_proxy: FinitePlanarSupportProxy | None = None
    transfer_status: str = "NOT_RUN"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        value: dict[str, Any] = asdict(self)
        value["support_type"] = self.support_type.value
        value["support_interval"] = (
            self.support_interval.as_dict() if self.support_interval is not None else None
        )
        value["stable_interval"] = (
            self.stable_interval.as_dict() if self.stable_interval is not None else None
        )
        value["plane_fit"] = self.plane_fit.as_dict() if self.plane_fit is not None else None
        value["table_proxy"] = self.table_proxy.as_dict() if self.table_proxy is not None else None
        return value


def jsonable(value: object) -> object:
    """Convert numpy scalars/arrays and enums for report serialization."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    return value


__all__ = [
    "FinitePlanarSupportProxy",
    "GeometryValidation",
    "NominalSupportMaterialV1",
    "PhysicsValidation",
    "StableIntervalResult",
    "StablePreContactDetectionContractV1",
    "SupportExtentContractV1",
    "SupportCollisionContractV1",
    "SupportCollisionPolicyV1",
    "SupportInterval",
    "SupportPatchType",
    "SupportPlaneConsistencyGateV1",
    "SupportPlaneFit",
    "SupportResolutionMode",
    "SupportResolutionResult",
    "SupportResolutionStatus",
    "SupportType",
    "jsonable",
]
