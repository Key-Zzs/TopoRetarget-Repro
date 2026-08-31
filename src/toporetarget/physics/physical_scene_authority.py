"""Dataset-independent physical-scene authority contracts.

The semantic retarget result and the physical-scene admission decision are
separate authorities.  This module contains the small, deterministic pieces
used by the certification script so that a visualization or a sparse tracked
point set cannot accidentally become collision authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np


class PhysicalSceneStatus(str, Enum):
    OBJECT_COLLISION_AUTHORITY_FAIL = "OBJECT_COLLISION_AUTHORITY_FAIL"
    ROBOT_COLLISION_AUTHORITY_FAIL = "ROBOT_COLLISION_AUTHORITY_FAIL"
    SUPPORT_EXPECTATION_UNRESOLVED = "SUPPORT_EXPECTATION_UNRESOLVED"
    SUPPORT_AUTHORITY_UNRESOLVED = "SUPPORT_AUTHORITY_UNRESOLVED"
    SUPPORT_DYNAMICS_UNSTABLE = "SUPPORT_DYNAMICS_UNSTABLE"
    RESET_SEVERE_PENETRATION = "RESET_SEVERE_PENETRATION"
    COLLISION_FILTER_INVALID = "COLLISION_FILTER_INVALID"
    RUNTIME_BINDING_INVALID = "RUNTIME_BINDING_INVALID"
    UNSUPPORTED_SUPPORT_TYPE = "UNSUPPORTED_SUPPORT_TYPE"
    PHYSICAL_SCENE_READY = "PHYSICAL_SCENE_READY"
    INCONCLUSIVE = "INCONCLUSIVE"


class ContactState(str, Enum):
    NO_CONTACT = "NO_CONTACT"
    INTENDED_CONTACT = "INTENDED_CONTACT"
    SHALLOW_OVERLAP_ACCEPTABLE = "SHALLOW_OVERLAP_ACCEPTABLE"
    SEVERE_PENETRATION = "SEVERE_PENETRATION"
    INCONCLUSIVE = "INCONCLUSIVE"


class SupportExpectation(str, Enum):
    STATIC_ENVIRONMENT_SUPPORT = "STATIC_ENVIRONMENT_SUPPORT"
    HAND_SUPPORTED = "HAND_SUPPORTED"
    OTHER_OBJECT_SUPPORTED = "OTHER_OBJECT_SUPPORTED"
    MIXED_SUPPORT = "MIXED_SUPPORT"
    UNSUPPORTED_DYNAMIC = "UNSUPPORTED_DYNAMIC"
    UNRESOLVED = "UNRESOLVED"


class SupportAuthority(str, Enum):
    SOURCE_EXPLICIT_SUPPORT = "SOURCE_EXPLICIT_SUPPORT"
    SOURCE_RECONSTRUCTED_SUPPORT = "SOURCE_RECONSTRUCTED_SUPPORT"
    INFERRED_ENVIRONMENT_SUPPORT = "INFERRED_ENVIRONMENT_SUPPORT"
    HAND_SUPPORTED = "HAND_SUPPORTED"
    OTHER_OBJECT_SUPPORTED = "OTHER_OBJECT_SUPPORTED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class PhysicalSceneAuthorityContractV1:
    """Frozen engineering thresholds; none are fitted from a canary outcome."""

    schema_version: str = "PhysicalSceneAuthorityV1"
    units: str = "metre, seconds, radians, kilograms, newtons"
    minimum_support_extent_m: float = 0.01
    support_normal_cosine_min: float = 0.995
    com_margin_tolerance_m: float = 0.002
    object_table_gap_max_m: float = 0.005
    object_table_penetration_max_m: float = 0.002
    shallow_overlap_max_m: float = 0.002
    severe_penetration_max_m: float = 0.01
    reset_velocity_max_mps: float = 0.25
    reset_angular_velocity_max_radps: float = 1.0
    support_duration_s: float = 3.0
    support_rotation_drift_max_rad: float = 0.05
    support_linear_speed_max_mps: float = 0.05
    support_angular_speed_max_radps: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_support_extent_m",
            "support_normal_cosine_min",
            "com_margin_tolerance_m",
            "object_table_gap_max_m",
            "object_table_penetration_max_m",
            "shallow_overlap_max_m",
            "severe_penetration_max_m",
            "reset_velocity_max_mps",
            "reset_angular_velocity_max_radps",
            "support_duration_s",
            "support_rotation_drift_max_rad",
            "support_linear_speed_max_mps",
            "support_angular_speed_max_radps",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"PHYSICAL_SCENE_CONTRACT_INVALID:{name}")
        if not 0.0 < self.support_normal_cosine_min <= 1.0:
            raise ValueError("PHYSICAL_SCENE_CONTRACT_INVALID:support_normal_cosine_min")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeCollisionShapeV1:
    """One collision shape enumerated from the composed runtime USD stage."""

    role: str
    articulation_path: str
    link_name: str
    collision_prim: str
    shape_type: str
    source_asset: str
    local_transform: tuple[tuple[float, ...], ...]
    world_transform: tuple[tuple[float, ...], ...]
    collision_enabled: bool
    rigid_body: bool
    contact_report_enabled: bool | None
    collision_group: int | str | None
    collision_mask: int | str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _array(value: Sequence[float] | np.ndarray, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"PHYSICAL_SCENE_ARRAY_INVALID:{name}")
    return array


def classify_contact_state(
    *,
    max_penetration_m: float | None,
    intended_contact: bool,
    reset_exploded: bool = False,
    contract: PhysicalSceneAuthorityContractV1 | None = None,
) -> ContactState:
    """Classify reset contact without conflating intended touch and penetration."""

    active = contract or PhysicalSceneAuthorityContractV1()
    if max_penetration_m is None or not np.isfinite(max_penetration_m):
        return ContactState.INCONCLUSIVE
    penetration = float(max_penetration_m)
    if penetration < 0.0:
        raise ValueError("PHYSICAL_SCENE_PENETRATION_NEGATIVE")
    if reset_exploded or penetration > active.severe_penetration_max_m:
        return ContactState.SEVERE_PENETRATION
    if penetration > active.shallow_overlap_max_m:
        return ContactState.INCONCLUSIVE
    if penetration > 0.0:
        return ContactState.SHALLOW_OVERLAP_ACCEPTABLE
    return ContactState.INTENDED_CONTACT if intended_contact else ContactState.NO_CONTACT


def validate_runtime_collision_shapes(
    shapes: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, object]:
    """Validate the fields required for collision authority, not visual display."""

    errors: list[str] = []
    for index, shape in enumerate(shapes):
        prefix = f"shape[{index}]"
        for key in ("articulation_path", "collision_prim", "shape_type", "source_asset"):
            if not str(shape.get(key, "")):
                errors.append(f"{prefix}:{key}_missing")
        for key in ("local_transform", "world_transform"):
            try:
                value = shape.get(key)
                _array(
                    value if value is not None else (),
                    shape=(4, 4),
                    name=f"{prefix}:{key}",
                )
            except (TypeError, ValueError):
                errors.append(f"{prefix}:{key}_invalid")
        if not bool(shape.get("collision_enabled", False)):
            errors.append(f"{prefix}:collision_disabled")
        if not bool(shape.get("rigid_body", False)):
            errors.append(f"{prefix}:rigid_body_missing")
        if role == "robot" and not str(shape.get("link_name", "")):
            errors.append(f"{prefix}:link_name_missing")
        shape_type = str(shape.get("shape_type", ""))
        if shape_type not in {"convex_hull", "convexMesh", "box", "sphere", "capsule"}:
            errors.append(f"{prefix}:shape_type_unsupported:{shape_type}")
    return {
        "status": "PASS" if shapes and not errors else "FAIL",
        "role": role,
        "shape_count": len(shapes),
        "errors": errors,
        "authority": "runtime_composed_usd_collision_shapes",
    }


def resolve_support_expectation(evidence: Mapping[str, Any]) -> dict[str, object]:
    """Resolve source semantics using evidence, retaining ambiguity explicitly."""

    explicit = bool(evidence.get("source_explicit_support", False))
    hand = bool(evidence.get("hand_supported", False))
    other = bool(evidence.get("other_object_supported", False))
    environment = bool(evidence.get("static_environment_support", False))
    dynamic = bool(evidence.get("unsupported_dynamic", False))
    candidates = [
        name
        for name, enabled in (
            (SupportExpectation.STATIC_ENVIRONMENT_SUPPORT.value, environment),
            (SupportExpectation.HAND_SUPPORTED.value, hand),
            (SupportExpectation.OTHER_OBJECT_SUPPORTED.value, other),
        )
        if enabled
    ]
    if dynamic and not candidates:
        selected = SupportExpectation.UNSUPPORTED_DYNAMIC
    elif len(candidates) > 1:
        selected = SupportExpectation.MIXED_SUPPORT
    elif candidates:
        selected = SupportExpectation(candidates[0])
    else:
        selected = SupportExpectation.UNRESOLVED
    return {
        "expectation": selected.value,
        "source_explicit_support": explicit,
        "evidence": dict(evidence),
        "candidate_expectations": candidates,
        "status": "PASS" if selected is not SupportExpectation.UNRESOLVED else "FAIL",
    }


def support_collision_policy(support_type: SupportAuthority | str) -> dict[str, object]:
    selected = SupportAuthority(support_type)
    if selected in {
        SupportAuthority.SOURCE_EXPLICIT_SUPPORT,
        SupportAuthority.SOURCE_RECONSTRUCTED_SUPPORT,
    }:
        return {
            "support_type": selected.value,
            "object_support_collision": True,
            "hand_support_collision": True,
            "global_support_collision_disabled": False,
            "pairwise_filter_authority": "explicit_source_support_pairs",
        }
    if selected is SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT:
        return {
            "support_type": selected.value,
            "object_support_collision": True,
            "hand_support_collision": False,
            "global_support_collision_disabled": False,
            "pairwise_filter_authority": "object_support_on_hand_support_off",
        }
    if selected in {
        SupportAuthority.HAND_SUPPORTED,
        SupportAuthority.OTHER_OBJECT_SUPPORTED,
    }:
        return {
            "support_type": selected.value,
            "object_support_collision": False,
            "hand_support_collision": False,
            "global_support_collision_disabled": False,
            "pairwise_filter_authority": "non_environment_support",
        }
    return {
        "support_type": selected.value,
        "object_support_collision": False,
        "hand_support_collision": False,
        "global_support_collision_disabled": False,
        "pairwise_filter_authority": "fail_closed_unresolved",
    }


def validate_support_geometry(
    *,
    plane_normal_world: Sequence[float],
    gravity_world_mps2: Sequence[float],
    support_center_world: Sequence[float],
    support_extent_m: Sequence[float],
    object_footprint_world: Sequence[Sequence[float]],
    center_of_mass_world: Sequence[float],
    object_min_signed_distance_m: float,
    object_max_signed_distance_m: float,
    contract: PhysicalSceneAuthorityContractV1 | None = None,
) -> dict[str, object]:
    """Validate finite planar support and COM margin in the support frame."""

    active = contract or PhysicalSceneAuthorityContractV1()
    normal = _array(plane_normal_world, shape=(3,), name="plane_normal_world")
    gravity = _array(gravity_world_mps2, shape=(3,), name="gravity_world_mps2")
    center = _array(support_center_world, shape=(3,), name="support_center_world")
    extent = _array(support_extent_m, shape=(2,), name="support_extent_m")
    footprint = np.asarray(object_footprint_world, dtype=np.float64)
    com = _array(center_of_mass_world, shape=(3,), name="center_of_mass_world")
    if footprint.ndim != 2 or footprint.shape[1] != 3 or len(footprint) < 3:
        raise ValueError("PHYSICAL_SCENE_SUPPORT_FOOTPRINT_INVALID")
    if not np.isfinite(footprint).all() or np.any(extent <= 0.0):
        raise ValueError("PHYSICAL_SCENE_SUPPORT_GEOMETRY_NONFINITE")
    normal /= np.linalg.norm(normal)
    gravity /= np.linalg.norm(gravity)
    alignment = float(np.dot(normal, -gravity))
    if abs(normal[2]) < 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    else:
        reference = np.array([1.0, 0.0, 0.0])
    tangent_u = np.cross(normal, reference)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    relative = footprint - center
    footprint_uv = np.column_stack((relative @ tangent_u, relative @ tangent_v))
    com_relative = com - center
    com_uv = np.asarray([com_relative @ tangent_u, com_relative @ tangent_v])
    half_extent = extent / 2.0
    com_margin = np.min(half_extent - np.abs(com_uv))
    footprint_margin = np.min(half_extent[None, :] - np.abs(footprint_uv), axis=0)
    extent_ok = bool(np.all(extent >= active.minimum_support_extent_m))
    normal_ok = bool(alignment >= active.support_normal_cosine_min)
    com_inside = bool(np.all(com_margin >= -active.com_margin_tolerance_m))
    object_clear = bool(object_min_signed_distance_m >= -active.object_table_penetration_max_m)
    object_near = bool(object_min_signed_distance_m <= active.object_table_gap_max_m)
    status = (
        "PASS"
        if extent_ok and normal_ok and com_inside and object_clear and object_near
        else "FAIL"
    )
    return {
        "status": status,
        "plane_normal_world": normal.tolist(),
        "gravity_world_mps2": gravity.tolist(),
        "normal_alignment_cosine": alignment,
        "support_center_world": center.tolist(),
        "support_extent_m": extent.tolist(),
        "object_footprint_vertex_count": len(footprint),
        "object_footprint_margin_uv_m": footprint_margin.tolist(),
        "center_of_mass_world": com.tolist(),
        "center_of_mass_uv_m": com_uv.tolist(),
        "center_of_mass_support_margin_m": float(np.min(com_margin)),
        "center_of_mass_inside_support": com_inside,
        "object_min_signed_distance_m": float(object_min_signed_distance_m),
        "object_max_signed_distance_m": float(object_max_signed_distance_m),
        "extent_ok": extent_ok,
        "normal_ok": normal_ok,
        "object_clear_of_severe_penetration": object_clear,
        "object_within_support_gap": object_near,
        "thresholds": active.as_dict(),
    }


def admit_physical_scene(
    *,
    runtime_binding_status: str,
    robot_collision_status: str,
    object_collision_status: str,
    collision_filter_status: str,
    reset_contact_state: ContactState | str,
    support_expectation: SupportExpectation | str,
    support_authority: SupportAuthority | str,
    support_dynamics_status: str,
) -> dict[str, object]:
    """Apply the fail-closed physical-scene admission order."""

    try:
        selected_support_authority = SupportAuthority(support_authority)
    except ValueError:
        selected_support_authority = None
    checks = {
        "runtime_binding": runtime_binding_status == "PASS",
        "robot_collision": robot_collision_status == "PASS",
        "object_collision": object_collision_status == "PASS",
        "collision_filter": collision_filter_status == "PASS",
        "reset_contact": ContactState(reset_contact_state)
        not in {ContactState.SEVERE_PENETRATION, ContactState.INCONCLUSIVE},
        "support_expectation": SupportExpectation(support_expectation)
        is not SupportExpectation.UNRESOLVED,
        "support_authority": selected_support_authority is not None
        and selected_support_authority is not SupportAuthority.UNRESOLVED,
        "support_dynamics": support_dynamics_status == "PASS",
    }
    if not checks["runtime_binding"]:
        status = PhysicalSceneStatus.RUNTIME_BINDING_INVALID
    elif not checks["object_collision"]:
        status = PhysicalSceneStatus.OBJECT_COLLISION_AUTHORITY_FAIL
    elif not checks["robot_collision"]:
        status = PhysicalSceneStatus.ROBOT_COLLISION_AUTHORITY_FAIL
    elif ContactState(reset_contact_state) is ContactState.SEVERE_PENETRATION:
        status = PhysicalSceneStatus.RESET_SEVERE_PENETRATION
    elif ContactState(reset_contact_state) is ContactState.INCONCLUSIVE:
        status = PhysicalSceneStatus.INCONCLUSIVE
    elif not checks["collision_filter"]:
        status = PhysicalSceneStatus.COLLISION_FILTER_INVALID
    elif SupportExpectation(support_expectation) is SupportExpectation.UNRESOLVED:
        status = PhysicalSceneStatus.SUPPORT_EXPECTATION_UNRESOLVED
    elif selected_support_authority is None:
        status = PhysicalSceneStatus.UNSUPPORTED_SUPPORT_TYPE
    elif selected_support_authority is SupportAuthority.UNRESOLVED:
        status = PhysicalSceneStatus.SUPPORT_AUTHORITY_UNRESOLVED
    elif not checks["support_dynamics"]:
        status = PhysicalSceneStatus.SUPPORT_DYNAMICS_UNSTABLE
    else:
        status = PhysicalSceneStatus.PHYSICAL_SCENE_READY
    return {
        "status": status.value,
        "checks": checks,
        "fail_closed": status is not PhysicalSceneStatus.PHYSICAL_SCENE_READY,
    }


__all__ = [
    "ContactState",
    "PhysicalSceneAuthorityContractV1",
    "PhysicalSceneStatus",
    "RuntimeCollisionShapeV1",
    "SupportAuthority",
    "SupportExpectation",
    "admit_physical_scene",
    "canonical_json",
    "classify_contact_state",
    "resolve_support_expectation",
    "sha256_json",
    "support_collision_policy",
    "validate_runtime_collision_shapes",
    "validate_support_geometry",
]
