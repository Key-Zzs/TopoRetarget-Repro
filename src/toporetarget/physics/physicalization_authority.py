"""Frozen physicalization and settled-dynamics authorities.

The semantic retarget authority, the object-dynamics provenance authority, and
the support-dynamics authority are intentionally separate.  This module keeps
the deterministic, backend-neutral decisions in one place so that the Isaac
runner can only supply telemetry and cannot silently tune a canary into a
pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, cast

import numpy as np


class DynamicsAuthorityStatus(str, Enum):
    PASS = "PASS"
    INCOMPLETE = "INCOMPLETE"
    RUNTIME_DEFAULT_MISMATCH = "RUNTIME_DEFAULT_MISMATCH"


class SupportExistenceStatus(str, Enum):
    SOURCE_EXPLICIT_SUPPORT = "SOURCE_EXPLICIT_SUPPORT"
    HAND_SUPPORTED = "HAND_SUPPORTED"
    OTHER_OBJECT_SUPPORTED = "OTHER_OBJECT_SUPPORTED"
    ENVIRONMENT_SUPPORT_REQUIRED = "ENVIRONMENT_SUPPORT_REQUIRED"
    MIXED_SUPPORT = "MIXED_SUPPORT"
    UNSUPPORTED_DYNAMIC = "UNSUPPORTED_DYNAMIC"
    UNRESOLVED = "UNRESOLVED"


class PhysicalizationMode(str, Enum):
    SUPPORT_ONLY = "SUPPORT_ONLY"
    COMMON_SCENE_SE3 = "COMMON_SCENE_SE3"
    RELATIVE_OBJECT_PROJECTION = "RELATIVE_OBJECT_PROJECTION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DeviationStatus(str, Enum):
    PASS = "PASS"
    REJECTED = "REJECTED"
    INVALID = "INVALID"


class RetargetReuseStatus(str, Enum):
    REUSE_GEOMETRIC_RETARGET = "REUSE_GEOMETRIC_RETARGET"
    REQUIRES_EXACT_RETARGET = "REQUIRES_EXACT_RETARGET"
    HELD_OUT_REQUIRES_RELATIVE_RETARGET_CHANGE = (
        "HELDOUT_PHYSICALIZATION_REQUIRES_RELATIVE_RETARGET_CHANGE"
    )
    INCONCLUSIVE = "INCONCLUSIVE"


class SettledDynamicsStatus(str, Enum):
    STABLE_STATIC = "STABLE_STATIC"
    SETTLED_AFTER_TRANSIENT = "SETTLED_AFTER_TRANSIENT"
    PERSISTENT_ROLLING_SLIDING = "PERSISTENT_ROLLING_SLIDING"
    TIPPED = "TIPPED"
    FELL_OFF = "FELL_OFF"
    NO_CONTACT = "NO_CONTACT"
    FELL_THROUGH = "FELL_THROUGH"
    EXPLOSIVE = "EXPLOSIVE"
    NOT_SETTLED = "NOT_SETTLED"
    INCONCLUSIVE = "INCONCLUSIVE"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"PHYSICALIZATION_ARRAY_INVALID:{name}")
    return array


def _finite_scalar(value: object, name: str) -> float:
    result = float(cast(Any, value))
    if not math.isfinite(result):
        raise ValueError(f"PHYSICALIZATION_SCALAR_INVALID:{name}")
    return result


@dataclass(frozen=True)
class ObjectDynamicsAuthorityContractV1:
    schema_version: str = "ObjectDynamicsAuthorityV1"
    required_runtime_fields: tuple[str, ...] = (
        "mass_kg",
        "center_of_mass_m",
        "diagonal_inertia_kgm2",
        "collision_method",
        "generated_usd",
        "generated_sha256",
    )
    # The canonical object USD is intentionally gravity-neutral; the runtime
    # scene must enable gravity in its spawn override.  Runtime parity is
    # checked separately by ``audit_runtime_default_provenance``.
    require_gravity_enabled_in_scene: bool = False
    require_free_rigid_body: bool = True
    require_collision_shape: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicsProvenanceV1:
    object_id: str
    source_mesh_sha256: str
    generated_usd_sha256: str
    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    diagonal_inertia_kgm2: tuple[float, float, float]
    collision_method: str
    static_friction: float | None
    dynamic_friction: float | None
    restitution: float | None
    gravity_enabled: bool
    free_rigid_body: bool
    source_of_truth: str
    physical_classification: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.object_id or len(self.source_mesh_sha256) != 64:
            raise ValueError("OBJECT_DYNAMICS_PROVENANCE_ID_OR_HASH_INVALID")
        if len(self.generated_usd_sha256) != 64:
            raise ValueError("OBJECT_DYNAMICS_PROVENANCE_USD_HASH_INVALID")
        if self.mass_kg <= 0.0 or not math.isfinite(self.mass_kg):
            raise ValueError("OBJECT_DYNAMICS_MASS_INVALID")
        _finite_array(self.center_of_mass_m, (3,), "center_of_mass_m")
        inertia = _finite_array(self.diagonal_inertia_kgm2, (3,), "diagonal_inertia_kgm2")
        if np.any(inertia <= 0.0):
            raise ValueError("OBJECT_DYNAMICS_INERTIA_INVALID")
        if not self.collision_method:
            raise ValueError("OBJECT_DYNAMICS_COLLISION_METHOD_MISSING")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportExistenceContractV1:
    schema_version: str = "SupportExistenceAuthorityV1"
    minimum_stationary_initial_frames: int = 8
    max_initial_linear_speed_mps: float = 0.03
    max_initial_angular_speed_radps: float = 0.15
    max_initial_rotation_span_rad: float = 0.02
    environment_geometry_required_for_inferred_support: bool = True

    def __post_init__(self) -> None:
        if self.minimum_stationary_initial_frames < 2:
            raise ValueError("SUPPORT_EXISTENCE_FRAME_COUNT_INVALID")
        if self.max_initial_linear_speed_mps <= 0.0:
            raise ValueError("SUPPORT_EXISTENCE_LINEAR_THRESHOLD_INVALID")
        if self.max_initial_angular_speed_radps <= 0.0:
            raise ValueError("SUPPORT_EXISTENCE_ANGULAR_THRESHOLD_INVALID")
        if self.max_initial_rotation_span_rad <= 0.0:
            raise ValueError("SUPPORT_EXISTENCE_ROTATION_SPAN_THRESHOLD_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalizationDeviationBudgetV1:
    schema_version: str = "PhysicalizationDeviationBudgetV1"
    max_support_translation_m: float = 0.050
    max_support_normal_change_rad: float = 0.08726646259971647
    max_common_scene_translation_m: float = 0.050
    max_common_scene_rotation_rad: float = 0.08726646259971647
    max_relative_object_translation_m: float = 0.0
    max_relative_object_rotation_rad: float = 0.0
    freeze_before_canary_outcomes: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_support_translation_m",
            "max_support_normal_change_rad",
            "max_common_scene_translation_m",
            "max_common_scene_rotation_rad",
            "max_relative_object_translation_m",
            "max_relative_object_rotation_rad",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"PHYSICALIZATION_BUDGET_INVALID:{name}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalizationCandidateV1:
    candidate_id: str
    mode: PhysicalizationMode
    support_translation_m: float
    support_normal_change_rad: float
    common_scene_translation_m: float = 0.0
    common_scene_rotation_rad: float = 0.0
    relative_object_translation_m: float = 0.0
    relative_object_rotation_rad: float = 0.0
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value


def audit_object_dynamics_provenance(
    asset: Mapping[str, Any],
    *,
    contract: ObjectDynamicsAuthorityContractV1 | None = None,
) -> dict[str, object]:
    """Validate the object asset manifest without claiming physical ground truth."""

    active = contract or ObjectDynamicsAuthorityContractV1()
    aliases = {
        "diagonal_inertia_kgm2": asset.get("principal_inertia_kgm2"),
        "generated_sha256": asset.get("generated_usd_sha256"),
    }
    missing = [
        key for key in active.required_runtime_fields if not asset.get(key) and not aliases.get(key)
    ]
    errors: list[str] = []
    if missing:
        errors.extend(f"missing:{key}" for key in missing)
    try:
        mass = _finite_scalar(asset.get("mass_kg"), "mass_kg")
        com = _finite_array(asset.get("center_of_mass_m"), (3,), "center_of_mass_m")
        inertia = _finite_array(
            asset.get("diagonal_inertia_kgm2", asset.get("principal_inertia_kgm2")),
            (3,),
            "diagonal_inertia_kgm2",
        )
        if mass <= 0.0 or np.any(inertia <= 0.0):
            errors.append("mass_or_inertia_nonpositive")
    except (TypeError, ValueError):
        mass = float("nan")
        com = np.full(3, np.nan)
        inertia = np.full(3, np.nan)
        errors.append("numeric_field_invalid")
    rigid = asset.get("rigid_body", {})
    if not isinstance(rigid, Mapping):
        rigid = {}
    gravity_enabled = bool(rigid.get("gravity_enabled", asset.get("gravity_enabled", True)))
    free_rigid_body = bool(rigid.get("free", asset.get("free_rigid_body", False)))
    collision_count = int(asset.get("collision_prim_count", 0) or 0)
    if active.require_gravity_enabled_in_scene and not gravity_enabled:
        errors.append("gravity_disabled_in_runtime_asset")
    if active.require_free_rigid_body and not free_rigid_body:
        errors.append("free_rigid_body_missing")
    if active.require_collision_shape and collision_count < 1:
        errors.append("collision_shape_missing")
    return {
        "schema_version": active.schema_version,
        "status": DynamicsAuthorityStatus.PASS.value
        if not errors
        else DynamicsAuthorityStatus.INCOMPLETE.value,
        "errors": errors,
        "object_id": asset.get("object_id"),
        "source_mesh_sha256": asset.get("visual_mesh_sha256", asset.get("source_mesh_sha256")),
        "generated_usd_sha256": asset.get("generated_sha256", asset.get("generated_usd_sha256")),
        "mass_kg": mass,
        "center_of_mass_m": com.tolist(),
        "diagonal_inertia_kgm2": inertia.tolist(),
        "collision_method": asset.get("collision_method"),
        "collision_prim_count": collision_count,
        "gravity_enabled": gravity_enabled,
        "free_rigid_body": free_rigid_body,
        "physical_classification": asset.get(
            "physical_classification", "UNSPECIFIED_PHYSICAL_PROVENANCE"
        ),
        "ground_truth_physics_available": not bool(
            "ENGINEERING_NOMINAL" in str(asset.get("physical_classification", ""))
        ),
        "warnings": list(asset.get("warnings", [])),
        "contract": active.as_dict(),
    }


def audit_runtime_default_provenance(
    declared: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    atol: float = 1.0e-6,
) -> dict[str, object]:
    """Compare runtime Isaac defaults with the frozen asset declaration."""

    errors: list[str] = []
    compared: dict[str, object] = {}
    for key in ("mass_kg", "center_of_mass_m", "diagonal_inertia_kgm2"):
        if key not in declared or key not in runtime:
            errors.append(f"missing:{key}")
            continue
        lhs = np.asarray(declared[key], dtype=np.float64)
        rhs = np.asarray(runtime[key], dtype=np.float64)
        if lhs.shape != rhs.shape or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
            errors.append(f"invalid:{key}")
            continue
        delta = float(np.max(np.abs(lhs - rhs)))
        compared[key] = {"declared": lhs.tolist(), "runtime": rhs.tolist(), "max_abs_delta": delta}
        if delta > atol:
            errors.append(f"mismatch:{key}")
    for key in ("gravity_enabled", "collision_enabled", "rigid_body"):
        if key in declared or key in runtime:
            if declared.get(key) != runtime.get(key):
                errors.append(f"mismatch:{key}")
            compared[key] = {"declared": declared.get(key), "runtime": runtime.get(key)}
    return {
        "schema_version": "ObjectDynamicsRuntimeDefaultAuditV1",
        "status": (
            DynamicsAuthorityStatus.PASS.value
            if not errors
            else DynamicsAuthorityStatus.RUNTIME_DEFAULT_MISMATCH.value
        ),
        "errors": errors,
        "compared": compared,
        "atol": atol,
    }


def resolve_support_existence(
    evidence: Mapping[str, Any],
    *,
    contract: SupportExistenceContractV1 | None = None,
) -> dict[str, object]:
    """Resolve support existence, allowing stationary environment support.

    An environment-support decision requires both a stationary initial interval
    and an independently supplied finite environment geometry.  This avoids
    converting a missing annotation into an invented table while allowing the
    valid ``G05_1``-style stationary scene to proceed.
    """

    active = contract or SupportExistenceContractV1()
    explicit = bool(evidence.get("source_explicit_support", False))
    hand = bool(evidence.get("hand_supported", False))
    other = bool(evidence.get("other_object_supported", False))
    stationary_frames = int(evidence.get("stationary_initial_frames", 0) or 0)
    linear = float(evidence.get("initial_linear_speed_max_mps", float("inf")))
    angular = float(evidence.get("initial_angular_speed_max_radps", float("inf")))
    rotation_span = float(evidence.get("initial_rotation_span_rad", 0.0))
    geometry = bool(
        evidence.get("finite_environment_geometry_available", False)
        or evidence.get("support_proxy_available", False)
    )
    candidates: list[str] = []
    if explicit:
        candidates.append(SupportExistenceStatus.SOURCE_EXPLICIT_SUPPORT.value)
    if hand:
        candidates.append(SupportExistenceStatus.HAND_SUPPORTED.value)
    if other:
        candidates.append(SupportExistenceStatus.OTHER_OBJECT_SUPPORTED.value)
    stationary = bool(
        stationary_frames >= active.minimum_stationary_initial_frames
        and math.isfinite(linear)
        and math.isfinite(angular)
        and math.isfinite(rotation_span)
        and linear <= active.max_initial_linear_speed_mps
        and angular <= active.max_initial_angular_speed_radps
        and rotation_span <= active.max_initial_rotation_span_rad
    )
    if len(candidates) > 1:
        selected = SupportExistenceStatus.MIXED_SUPPORT
    elif candidates:
        selected = SupportExistenceStatus(candidates[0])
    elif bool(evidence.get("unsupported_dynamic", False)):
        selected = SupportExistenceStatus.UNSUPPORTED_DYNAMIC
    elif stationary and geometry and active.environment_geometry_required_for_inferred_support:
        selected = SupportExistenceStatus.ENVIRONMENT_SUPPORT_REQUIRED
    else:
        selected = SupportExistenceStatus.UNRESOLVED
    return {
        "schema_version": active.schema_version,
        "status": selected.value,
        "pass": selected is not SupportExistenceStatus.UNRESOLVED,
        "candidate_sources": candidates,
        "stationary_initial_interval_valid": stationary,
        "initial_rotation_span_rad": rotation_span,
        "finite_environment_geometry_available": geometry,
        "evidence": dict(evidence),
        "contract": active.as_dict(),
    }


def evaluate_physicalization_candidate(
    candidate: PhysicalizationCandidateV1,
    budget: PhysicalizationDeviationBudgetV1,
) -> dict[str, object]:
    reasons = list(candidate.rejection_reasons)
    if candidate.support_translation_m > budget.max_support_translation_m:
        reasons.append("support_translation_budget_exceeded")
    if candidate.support_normal_change_rad > budget.max_support_normal_change_rad:
        reasons.append("support_normal_budget_exceeded")
    if candidate.common_scene_translation_m > budget.max_common_scene_translation_m:
        reasons.append("common_scene_translation_budget_exceeded")
    if candidate.common_scene_rotation_rad > budget.max_common_scene_rotation_rad:
        reasons.append("common_scene_rotation_budget_exceeded")
    if candidate.relative_object_translation_m > budget.max_relative_object_translation_m:
        reasons.append("relative_object_translation_requires_retarget")
    if candidate.relative_object_rotation_rad > budget.max_relative_object_rotation_rad:
        reasons.append("relative_object_rotation_requires_retarget")
    if candidate.mode is PhysicalizationMode.RELATIVE_OBJECT_PROJECTION:
        reasons.append("relative_object_projection_requires_exact_retarget")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return {
        "candidate_id": candidate.candidate_id,
        "mode": candidate.mode.value,
        "status": DeviationStatus.PASS.value
        if not unique_reasons
        else DeviationStatus.REJECTED.value,
        "accepted": not unique_reasons,
        "rejection_reasons": list(unique_reasons),
        "candidate": candidate.as_dict(),
        "budget": budget.as_dict(),
    }


def select_physicalization_candidate(
    candidates: Sequence[PhysicalizationCandidateV1],
    budget: PhysicalizationDeviationBudgetV1,
) -> dict[str, object]:
    """Select the first deterministic acceptable candidate; never optimize per canary."""

    evaluations = [
        evaluate_physicalization_candidate(candidate, budget) for candidate in candidates
    ]
    selected = next((item for item in evaluations if bool(item["accepted"])), None)
    rejection_counts: dict[str, int] = {}
    for item in evaluations:
        for reason in cast(list[str], item["rejection_reasons"]):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return {
        "schema_version": "SupportPhysicalizationV1",
        "status": "PASS" if selected is not None else "FAIL",
        "candidate_count": len(evaluations),
        "evaluations": evaluations,
        "rejection_counts": rejection_counts,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_mode": selected["mode"] if selected else None,
        "selection_rule": "first_accepted_in_frozen_order",
        "budget": budget.as_dict(),
    }


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("RETARGET_QUATERNION_INVALID")
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("RETARGET_QUATERNION_ZERO")
    w, x, y, z = (q / norm[:, None]).T
    result = np.empty((len(q), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - z * w)
    result[:, 0, 2] = 2 * (x * z + y * w)
    result[:, 1, 0] = 2 * (x * y + z * w)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - x * w)
    result[:, 2, 0] = 2 * (x * z - y * w)
    result[:, 2, 1] = 2 * (y * z + x * w)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def compare_retarget_reuse(
    *,
    hand_translation_before_m: Sequence[Sequence[float]],
    hand_quaternion_before_wxyz: Sequence[Sequence[float]],
    object_translation_before_m: Sequence[Sequence[float]],
    object_quaternion_before_wxyz: Sequence[Sequence[float]],
    hand_translation_after_m: Sequence[Sequence[float]],
    hand_quaternion_after_wxyz: Sequence[Sequence[float]],
    object_translation_after_m: Sequence[Sequence[float]],
    object_quaternion_after_wxyz: Sequence[Sequence[float]],
    budget: PhysicalizationDeviationBudgetV1,
    mode: PhysicalizationMode | str,
) -> dict[str, object]:
    """Compare the full hand-object relative trajectory before and after support-only changes."""

    mode_value = PhysicalizationMode(mode)
    arrays = [
        _finite_array(value, (len(hand_translation_before_m), 3), name)
        for value, name in (
            (hand_translation_before_m, "hand_translation_before"),
            (object_translation_before_m, "object_translation_before"),
            (hand_translation_after_m, "hand_translation_after"),
            (object_translation_after_m, "object_translation_after"),
        )
    ]
    if not all(len(value) == len(arrays[0]) for value in arrays):
        raise ValueError("RETARGET_TRAJECTORY_LENGTH_MISMATCH")
    hb_t, ob_t, ha_t, oa_t = arrays
    hb_r, ob_r, ha_r, oa_r = (
        _quaternion_matrix(np.asarray(value, dtype=np.float64))
        for value in (
            hand_quaternion_before_wxyz,
            object_quaternion_before_wxyz,
            hand_quaternion_after_wxyz,
            object_quaternion_after_wxyz,
        )
    )
    before_t = np.einsum("tji,tj->ti", hb_r, ob_t - hb_t)
    after_t = np.einsum("tji,tj->ti", ha_r, oa_t - ha_t)
    before_r = np.einsum("tij,tjk->tik", np.transpose(hb_r, (0, 2, 1)), ob_r)
    after_r = np.einsum("tij,tjk->tik", np.transpose(ha_r, (0, 2, 1)), oa_r)
    relative_translation_error = np.linalg.norm(after_t - before_t, axis=1)
    relative_rotation_trace = np.einsum("tij,tij->t", after_r, before_r)
    relative_rotation_error = np.arccos(np.clip((relative_rotation_trace - 1.0) / 2.0, -1.0, 1.0))
    max_translation = float(np.max(relative_translation_error))
    max_rotation = float(np.max(relative_rotation_error))
    exact_required = mode_value is PhysicalizationMode.RELATIVE_OBJECT_PROJECTION
    within_budget = bool(
        max_translation <= budget.max_relative_object_translation_m
        and max_rotation <= budget.max_relative_object_rotation_rad
    )
    if exact_required:
        status = RetargetReuseStatus.HELD_OUT_REQUIRES_RELATIVE_RETARGET_CHANGE
    elif within_budget:
        status = RetargetReuseStatus.REUSE_GEOMETRIC_RETARGET
    else:
        status = RetargetReuseStatus.REQUIRES_EXACT_RETARGET
    return {
        "schema_version": "RetargetReuseDecisionV1",
        "status": status.value,
        "mode": mode_value.value,
        "frame_count": len(before_t),
        "max_relative_translation_error_m": max_translation,
        "p95_relative_translation_error_m": float(np.percentile(relative_translation_error, 95.0)),
        "max_relative_rotation_error_rad": max_rotation,
        "p95_relative_rotation_error_rad": float(np.percentile(relative_rotation_error, 95.0)),
        "within_frozen_budget": within_budget,
        "full_trajectory_compared": True,
        "budget": budget.as_dict(),
    }


@dataclass(frozen=True)
class SettledSupportDynamicsQualificationV2:
    schema_version: str = "SettledSupportDynamicsQualificationV2"
    dt_s: float = 1.0 / 120.0
    initial_window_s: tuple[float, float] = (0.0, 0.5)
    settling_window_s: tuple[float, float] = (0.5, 2.0)
    terminal_window_s: tuple[float, float] = (2.0, 3.0)
    terminal_linear_speed_p95_max_mps: float = 0.02
    terminal_angular_speed_p95_max_radps: float = 0.35
    terminal_translation_span_max_m: float = 0.005
    terminal_rotation_span_max_rad: float = 0.03
    terminal_contact_fraction_min: float = 0.95
    # The 1.10 allowance is frozen from the positive-control envelope: it
    # tolerates a small terminal numerical tail while still rejecting a
    # genuinely growing trajectory.
    terminal_not_worse_than_settling_ratio: float = 1.10
    settling_trend_required: bool = True

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("SETTLED_DYNAMICS_DT_INVALID")
        windows = (self.initial_window_s, self.settling_window_s, self.terminal_window_s)
        if any(end <= start for start, end in windows):
            raise ValueError("SETTLED_DYNAMICS_WINDOW_INVALID")
        if not self.initial_window_s[1] <= self.settling_window_s[0]:
            raise ValueError("SETTLED_DYNAMICS_WINDOW_ORDER_INVALID")
        if not self.settling_window_s[1] <= self.terminal_window_s[0]:
            raise ValueError("SETTLED_DYNAMICS_WINDOW_ORDER_INVALID")
        if self.terminal_contact_fraction_min <= 0.0 or self.terminal_contact_fraction_min > 1.0:
            raise ValueError("SETTLED_DYNAMICS_CONTACT_THRESHOLD_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _window_mask(times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    start, end = window
    return (times > start) & (times <= end)


def _quat_span(quaternions: np.ndarray) -> float:
    matrices = _quaternion_matrix(quaternions)
    relative = np.einsum("ij,tjk->tik", matrices[0].T, matrices)
    trace = np.trace(relative, axis1=1, axis2=2)
    return float(np.max(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def qualify_settled_support_dynamics_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    mass_kg: float,
    contract: SettledSupportDynamicsQualificationV2 | None = None,
) -> dict[str, object]:
    """Reduce full telemetry using seconds-domain terminal windows.

    Impact peaks are retained as diagnostics.  They are not primary failure
    gates; terminal speed, terminal pose span, contact persistence, trend, and
    explicit fall/tip-over flags are the primary decision.
    """

    active = contract or SettledSupportDynamicsQualificationV2()
    if not records:
        return {
            "schema_version": active.schema_version,
            "status": SettledDynamicsStatus.INCONCLUSIVE.value,
            "pass": False,
            "reason": "empty_telemetry",
        }
    required = {
        "time_s",
        "position_world_m",
        "orientation_world_wxyz",
        "linear_velocity_world_mps",
        "angular_velocity_world_radps",
        "support_contact",
    }
    missing = sorted(key for key in required if any(key not in row for row in records))
    if missing:
        return {
            "schema_version": active.schema_version,
            "status": SettledDynamicsStatus.INCONCLUSIVE.value,
            "pass": False,
            "reason": "missing_telemetry_fields",
            "missing_fields": missing,
        }
    try:
        times = np.asarray([row["time_s"] for row in records], dtype=np.float64)
        positions = np.asarray([row["position_world_m"] for row in records], dtype=np.float64)
        orientations = np.asarray(
            [row["orientation_world_wxyz"] for row in records], dtype=np.float64
        )
        linear = np.asarray([row["linear_velocity_world_mps"] for row in records], dtype=np.float64)
        angular = np.asarray(
            [row["angular_velocity_world_radps"] for row in records], dtype=np.float64
        )
        contact = np.asarray([bool(row["support_contact"]) for row in records], dtype=bool)
    except (TypeError, ValueError):
        return {
            "schema_version": active.schema_version,
            "status": SettledDynamicsStatus.INCONCLUSIVE.value,
            "pass": False,
            "reason": "non_numeric_telemetry",
        }
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
        or positions.shape != (len(records), 3)
        or orientations.shape != (len(records), 4)
        or linear.shape != (len(records), 3)
        or angular.shape != (len(records), 3)
        or not all(np.isfinite(value).all() for value in (positions, orientations, linear, angular))
    ):
        return {
            "schema_version": active.schema_version,
            "status": SettledDynamicsStatus.INCONCLUSIVE.value,
            "pass": False,
            "reason": "invalid_telemetry_shape_or_finiteness",
        }
    observed_dt = float(np.median(np.diff(times)))
    dt_consistent = bool(
        np.max(np.abs(np.diff(times) - observed_dt)) <= max(1.0e-9, active.dt_s * 1.0e-3)
    )
    if not dt_consistent:
        return {
            "schema_version": active.schema_version,
            "status": SettledDynamicsStatus.INCONCLUSIVE.value,
            "pass": False,
            "reason": "inconsistent_physics_dt",
            "observed_dt_s": observed_dt,
            "contract_dt_s": active.dt_s,
        }
    initial_mask = _window_mask(times, active.initial_window_s)
    settling_mask = _window_mask(times, active.settling_window_s)
    terminal_mask = _window_mask(times, active.terminal_window_s)
    if not initial_mask.any() or not settling_mask.any() or not terminal_mask.any():
        return {
            "schema_version": active.schema_version,
            "status": SettledDynamicsStatus.INCONCLUSIVE.value,
            "pass": False,
            "reason": "seconds_window_not_covered",
            "window_counts": {
                "initial": int(initial_mask.sum()),
                "settling": int(settling_mask.sum()),
                "terminal": int(terminal_mask.sum()),
            },
        }
    linear_speed = np.linalg.norm(linear, axis=1)
    angular_speed = np.linalg.norm(angular, axis=1)
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    rotation_from_initial = 2.0 * np.arccos(
        np.clip(np.abs(orientations @ orientations[0]), -1.0, 1.0)
    )
    terminal_positions = positions[terminal_mask]
    terminal_orientations = orientations[terminal_mask]
    position_span = float(
        np.max(np.linalg.norm(terminal_positions - terminal_positions[0], axis=1))
    )
    rotation_span = _quat_span(terminal_orientations)
    initial_p95_v = float(np.percentile(linear_speed[initial_mask], 95.0))
    settling_p95_v = float(np.percentile(linear_speed[settling_mask], 95.0))
    terminal_p95_v = float(np.percentile(linear_speed[terminal_mask], 95.0))
    initial_p95_w = float(np.percentile(angular_speed[initial_mask], 95.0))
    settling_p95_w = float(np.percentile(angular_speed[settling_mask], 95.0))
    terminal_p95_w = float(np.percentile(angular_speed[terminal_mask], 95.0))
    settling_ratio_v = terminal_p95_v / max(settling_p95_v, 1.0e-12)
    settling_ratio_w = terminal_p95_w / max(settling_p95_w, 1.0e-12)
    terminal_contact_fraction = float(np.mean(contact[terminal_mask]))
    flags = {
        "tip_over": bool(any(row.get("tip_over", False) for row in records)),
        "fell_off": bool(any(row.get("fell_off", False) for row in records)),
        "fell_through": bool(any(row.get("fell_through", False) for row in records)),
        "explosive": bool(any(row.get("explosive", False) for row in records)),
        "persistent_rolling": bool(any(row.get("persistent_rolling", False) for row in records)),
        "persistent_sliding": bool(any(row.get("persistent_sliding", False) for row in records)),
    }
    trend = bool(
        terminal_p95_v <= settling_p95_v * active.terminal_not_worse_than_settling_ratio
        and terminal_p95_w <= settling_p95_w * active.terminal_not_worse_than_settling_ratio
        and terminal_p95_v <= initial_p95_v * active.terminal_not_worse_than_settling_ratio
        and terminal_p95_w <= initial_p95_w * active.terminal_not_worse_than_settling_ratio
    )
    primary_pass = bool(
        terminal_p95_v <= active.terminal_linear_speed_p95_max_mps
        and terminal_p95_w <= active.terminal_angular_speed_p95_max_radps
        and position_span <= active.terminal_translation_span_max_m
        and rotation_span <= active.terminal_rotation_span_max_rad
        and terminal_contact_fraction >= active.terminal_contact_fraction_min
        and (trend or not active.settling_trend_required)
        and not any(flags.values())
    )
    if flags["explosive"]:
        status = SettledDynamicsStatus.EXPLOSIVE
    elif flags["fell_through"]:
        status = SettledDynamicsStatus.FELL_THROUGH
    elif flags["fell_off"]:
        status = SettledDynamicsStatus.FELL_OFF
    elif flags["tip_over"]:
        status = SettledDynamicsStatus.TIPPED
    elif terminal_contact_fraction < active.terminal_contact_fraction_min:
        status = SettledDynamicsStatus.NO_CONTACT
    elif flags["persistent_rolling"] or flags["persistent_sliding"]:
        status = SettledDynamicsStatus.PERSISTENT_ROLLING_SLIDING
    elif primary_pass:
        status = (
            SettledDynamicsStatus.STABLE_STATIC
            if initial_p95_v <= active.terminal_linear_speed_p95_max_mps
            and initial_p95_w <= active.terminal_angular_speed_p95_max_radps
            else SettledDynamicsStatus.SETTLED_AFTER_TRANSIENT
        )
    else:
        status = SettledDynamicsStatus.NOT_SETTLED
    return {
        "schema_version": active.schema_version,
        "status": status.value,
        "pass": primary_pass,
        "dt_s_observed": observed_dt,
        "dt_s_contract": active.dt_s,
        "window_counts": {
            "initial": int(initial_mask.sum()),
            "settling": int(settling_mask.sum()),
            "terminal": int(terminal_mask.sum()),
        },
        "terminal_contact_fraction": terminal_contact_fraction,
        "terminal_linear_speed_p95_mps": terminal_p95_v,
        "terminal_linear_speed_max_mps": float(np.max(linear_speed[terminal_mask])),
        "terminal_angular_speed_p95_radps": terminal_p95_w,
        "terminal_angular_speed_max_radps": float(np.max(angular_speed[terminal_mask])),
        "terminal_translation_span_m": position_span,
        "terminal_rotation_span_rad": rotation_span,
        "global_rotation_from_initial_max_rad": float(np.max(rotation_from_initial)),
        "initial_linear_speed_p95_mps": initial_p95_v,
        "settling_linear_speed_p95_mps": settling_p95_v,
        "initial_angular_speed_p95_radps": initial_p95_w,
        "settling_angular_speed_p95_radps": settling_p95_w,
        "terminal_to_settling_linear_p95_ratio": settling_ratio_v,
        "terminal_to_settling_angular_p95_ratio": settling_ratio_w,
        "settling_trend": trend,
        "impact_peaks_diagnostic_only": True,
        "flags": flags,
        "mass_kg": float(mass_kg),
        "contract": active.as_dict(),
    }


def build_physical_scene_protocol_v2(
    *,
    dynamics_contract: ObjectDynamicsAuthorityContractV1,
    support_contract: SupportExistenceContractV1,
    deviation_budget: PhysicalizationDeviationBudgetV1,
    settled_contract: SettledSupportDynamicsQualificationV2,
) -> tuple[dict[str, object], str]:
    protocol = {
        "schema_version": "PhysicalSceneProtocolV2",
        "authority_order": [
            "ObjectDynamicsAuthorityV1",
            "SupportExistenceAuthorityV1",
            "SupportPhysicalizationV1",
            "PhysicalizationDeviationBudgetV1",
            "SettledSupportDynamicsQualificationV2",
            "RetargetReuseDecisionV1",
        ],
        "dynamics_contract": dynamics_contract.as_dict(),
        "support_existence_contract": support_contract.as_dict(),
        "deviation_budget": deviation_budget.as_dict(),
        "settled_dynamics_contract": settled_contract.as_dict(),
        "frozen_before_canary_outcomes": True,
        "relative_object_change_requires_exact_retarget": True,
        "peak_metrics_primary_gate": False,
    }
    return protocol, sha256_json(protocol)


__all__ = [
    "DeviationStatus",
    "DynamicsAuthorityStatus",
    "DynamicsProvenanceV1",
    "ObjectDynamicsAuthorityContractV1",
    "PhysicalizationCandidateV1",
    "PhysicalizationDeviationBudgetV1",
    "PhysicalizationMode",
    "RetargetReuseStatus",
    "SettledDynamicsStatus",
    "SettledSupportDynamicsQualificationV2",
    "SupportExistenceContractV1",
    "SupportExistenceStatus",
    "audit_object_dynamics_provenance",
    "audit_runtime_default_provenance",
    "build_physical_scene_protocol_v2",
    "canonical_json",
    "compare_retarget_reuse",
    "evaluate_physicalization_candidate",
    "qualify_settled_support_dynamics_v2",
    "resolve_support_existence",
    "select_physicalization_candidate",
    "sha256_json",
]
