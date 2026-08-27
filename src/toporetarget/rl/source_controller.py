"""Production source-controller admission, fidelity, and AUTO routing contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class SourceControllerMode(StrEnum):
    """Supported source-controller routing modes."""

    AUTO = "AUTO"
    ZERO_RESIDUAL = "ZERO_RESIDUAL"
    CORRECTED_L0 = "CORRECTED_L0"


class SourceControllerDecision(StrEnum):
    """Historical V1/P2 terminal decisions, retained for report replay."""

    ZERO_RESIDUAL_SOURCE_CONTROLLER_SUFFICIENT = "ZERO_RESIDUAL_SOURCE_CONTROLLER_SUFFICIENT"
    AUTO_ZERO_RESIDUAL_THEN_L0_FALLBACK = "AUTO_ZERO_RESIDUAL_THEN_L0_FALLBACK"
    CORRECTED_L0_REQUIRED = "CORRECTED_L0_REQUIRED"
    L0_AUTHORITY_INCONCLUSIVE = "L0_AUTHORITY_INCONCLUSIVE"


class SourceControllerExecutability(StrEnum):
    """Hard source-controller admission result."""

    PASS = "PASS"
    FAIL = "FAIL"


class SourceControllerFidelity(StrEnum):
    """Non-gating source-controller quality result."""

    PASS = "PASS"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


class SourceControllerRouteV2(StrEnum):
    """Terminal AUTO V2 route selected without downstream outcomes."""

    ZERO_RESIDUAL = "ZERO_RESIDUAL"
    CORRECTED_L0 = "CORRECTED_L0"
    SOURCE_CONTROLLER_HARD_FAILURE = "SOURCE_CONTROLLER_HARD_FAILURE"


@dataclass(frozen=True)
class SourceControllerSafetyContractV1:
    """Historical V1 safety description, retained for receipt compatibility."""

    real_finger_joint_limits: bool = True
    actuator_effort_limits: bool = True
    actuator_velocity_limits: bool = True
    action_bounds: bool = True
    controller_stability: bool = True
    singularity_detection: bool = True
    collision_safety: bool = True
    finite_state_checks: bool = True
    virtual_wrist_angle_authority: str = "continuous_equivalent_branch_v1"

    def as_dict(self) -> dict[str, bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class SourceControllerExecutableContractV2:
    """Hard admission dimensions; task imitation is intentionally absent."""

    state_finite: bool = True
    target_finite: bool = True
    command_finite: bool = True
    action_finite: bool = True
    reference_index_advances: bool = True
    trajectory_rows_readable: bool = True
    controller_state_fresh: bool = True
    real_finger_joint_limits_safe: bool = True
    virtual_wrist_translation_limits_safe: bool = True
    actuator_effort_limits_safe: bool = True
    actuator_velocity_limits_safe: bool = True
    action_bounds_safe: bool = True
    singularity_safety_pass: bool = True
    catastrophic_collision_safe: bool = True
    nonfinite_dynamics_absent: bool = True
    controller_divergence_absent: bool = True
    virtual_wrist_angle_authority: str = "continuous_equivalent_branch_v1"

    def as_dict(self) -> dict[str, bool | str]:
        return asdict(self)


EXECUTABILITY_V2_REQUIRED_TRUE = tuple(
    name
    for name, value in SourceControllerExecutableContractV2().as_dict().items()
    if isinstance(value, bool) and value
)

FIDELITY_V2_CHECKS = (
    "wrist_position_tracking_pass",
    "wrist_rotation_tracking_pass",
    "finger_tracking_pass",
    "link_tracking_pass",
    "source_contact_recall_pass",
    "object_tracking_pass",
    "interaction_progression_pass",
    "command_clamp_pass",
    "actuator_saturation_pass",
    "reference_completion_pass",
)


def real_finger_joint_limit_safety_v2(
    joint_position: np.ndarray,
    joint_target: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    *,
    solver_tolerance_rad: float,
) -> dict[str, object]:
    """Audit strict targets and bounded PhysX joint-constraint penetration."""

    position = np.asarray(joint_position, dtype=np.float64)
    target = np.asarray(joint_target, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if (
        position.ndim != 2
        or target.shape != position.shape
        or lower.shape != (position.shape[1],)
        or upper.shape != lower.shape
        or not 0.0 <= solver_tolerance_rad <= 0.005
    ):
        raise ValueError("SOURCE_CONTROLLER_FINGER_JOINT_LIMIT_AUDIT_INVALID")
    finite = bool(
        np.isfinite(position).all()
        and np.isfinite(target).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    )
    target_safe = bool(finite and np.all(target >= lower[None]) and np.all(target <= upper[None]))
    excursion = np.maximum(
        np.maximum(lower[None] - position, position - upper[None]),
        0.0,
    )
    maximum = float(np.max(excursion, initial=0.0)) if finite else float("inf")
    failures = np.argwhere(excursion > solver_tolerance_rad) if finite else np.empty((0, 2))
    first = None if len(failures) == 0 else failures[0]
    return {
        "real_finger_joint_targets_within_authored_limits": target_safe,
        "maximum_finger_joint_limit_excursion_rad": maximum,
        "real_finger_joint_limit_solver_tolerance_rad": solver_tolerance_rad,
        "first_true_joint_limit_failure_frame": (None if first is None else int(first[0])),
        "first_true_joint_limit_failure_joint_index": (None if first is None else int(first[1])),
        "real_finger_joint_limits_safe": bool(target_safe and maximum <= solver_tolerance_rad),
    }


def source_controller_executability_v2(
    receipt: Mapping[str, object],
) -> SourceControllerExecutability:
    """Evaluate only finite execution and true physical safety constraints."""

    passed = all(receipt.get(name) is True for name in EXECUTABILITY_V2_REQUIRED_TRUE)
    return SourceControllerExecutability.PASS if passed else SourceControllerExecutability.FAIL


def source_controller_fidelity_v2(receipt: Mapping[str, object]) -> SourceControllerFidelity:
    """Classify source quality without changing downstream admission.

    Full fidelity requires every declared diagnostic. An executable controller
    with partial evidence is degraded; a non-executable controller is failed.
    """

    if source_controller_executability_v2(receipt) is SourceControllerExecutability.FAIL:
        return SourceControllerFidelity.FAIL
    if all(receipt.get(name) is True for name in FIDELITY_V2_CHECKS):
        return SourceControllerFidelity.PASS
    return SourceControllerFidelity.DEGRADED


def _finite_metric(receipt: Mapping[str, object], name: str, default: float) -> float:
    value = receipt.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if result == result and abs(result) != float("inf") else default


def source_side_rank_v2(receipt: Mapping[str, object]) -> tuple[Any, ...]:
    """Return a downstream-outcome-free rank; larger tuples are preferred."""

    wrist = _finite_metric(receipt, "normalized_wrist_tracking_error", float("inf"))
    finger = _finite_metric(receipt, "normalized_finger_tracking_error", float("inf"))
    clamp = _finite_metric(receipt, "command_clamp_fraction", 1.0)
    saturation = _finite_metric(receipt, "actuator_saturation_fraction", 1.0)
    contact = _finite_metric(receipt, "source_contact_recall", 0.0)
    object_fidelity = _finite_metric(receipt, "object_tracking_score", 0.0)
    return (
        source_controller_executability_v2(receipt) is SourceControllerExecutability.PASS,
        receipt.get("reference_completion_pass") is True,
        -(wrist + finger),
        -(clamp + saturation),
        contact,
        object_fidelity,
    )


def select_source_controller_route_v2(
    zero_residual: Mapping[str, object],
    corrected_l0: Mapping[str, object] | None = None,
) -> SourceControllerRouteV2:
    """Select AUTO V2 using source-side evidence only."""

    zero_ok = (
        source_controller_executability_v2(zero_residual) is SourceControllerExecutability.PASS
    )
    l0_ok = corrected_l0 is not None and (
        source_controller_executability_v2(corrected_l0) is SourceControllerExecutability.PASS
    )
    if zero_ok and l0_ok:
        assert corrected_l0 is not None
        return (
            SourceControllerRouteV2.CORRECTED_L0
            if source_side_rank_v2(corrected_l0) > source_side_rank_v2(zero_residual)
            else SourceControllerRouteV2.ZERO_RESIDUAL
        )
    if zero_ok:
        return SourceControllerRouteV2.ZERO_RESIDUAL
    if l0_ok:
        return SourceControllerRouteV2.CORRECTED_L0
    return SourceControllerRouteV2.SOURCE_CONTROLLER_HARD_FAILURE


def make_zero_output_residual_actor_(actor_critic: Any) -> None:
    """Make the current residual-policy actor identically zero in place."""

    actor = getattr(actor_critic, "actor", None)
    parameters = getattr(actor, "parameters", None)
    if actor is None or not callable(parameters):
        raise TypeError("ZERO_RESIDUAL_ACTOR_REQUIRES_ACTOR_MODULE")
    for parameter in parameters():
        parameter.detach().zero_()


def qualification_pass(receipt: Mapping[str, object]) -> bool:
    """Replay the historical V1 source-controller task-fidelity gate."""

    required_true = (
        "reference_tracking_pass",
        "contact_execution_pass",
        "reference_progression_pass",
        "finite_safe",
        "controller_authority_pass",
        "joint_limits_safe",
        "actuator_limits_safe",
        "action_bounds_safe",
        "collision_safety_pass",
    )
    return all(receipt.get(name) is True for name in required_true)


def select_source_controller_route(
    zero_residual: Sequence[Mapping[str, object]],
    corrected_l0: Sequence[Mapping[str, object]],
) -> SourceControllerDecision:
    """Replay the historical V1/P2 global route selection."""

    if not zero_residual:
        return SourceControllerDecision.L0_AUTHORITY_INCONCLUSIVE
    zero_by_clip = {str(row.get("clip_id")): qualification_pass(row) for row in zero_residual}
    if all(zero_by_clip.values()):
        return SourceControllerDecision.ZERO_RESIDUAL_SOURCE_CONTROLLER_SUFFICIENT
    l0_by_clip = {str(row.get("clip_id")): qualification_pass(row) for row in corrected_l0}
    failed = [clip for clip, passed in zero_by_clip.items() if not passed]
    if any(clip not in l0_by_clip for clip in failed):
        return SourceControllerDecision.L0_AUTHORITY_INCONCLUSIVE
    if all(l0_by_clip[clip] for clip in failed):
        if any(zero_by_clip.values()):
            return SourceControllerDecision.AUTO_ZERO_RESIDUAL_THEN_L0_FALLBACK
        return SourceControllerDecision.CORRECTED_L0_REQUIRED
    return SourceControllerDecision.L0_AUTHORITY_INCONCLUSIVE


def selected_mode_for_clip(
    zero_residual_receipt: Mapping[str, object],
) -> SourceControllerMode:
    """Replay the historical V1 per-clip route."""

    return (
        SourceControllerMode.ZERO_RESIDUAL
        if qualification_pass(zero_residual_receipt)
        else SourceControllerMode.CORRECTED_L0
    )


__all__ = [
    "EXECUTABILITY_V2_REQUIRED_TRUE",
    "FIDELITY_V2_CHECKS",
    "SourceControllerDecision",
    "SourceControllerExecutableContractV2",
    "SourceControllerExecutability",
    "SourceControllerFidelity",
    "SourceControllerMode",
    "SourceControllerRouteV2",
    "SourceControllerSafetyContractV1",
    "qualification_pass",
    "make_zero_output_residual_actor_",
    "select_source_controller_route",
    "select_source_controller_route_v2",
    "selected_mode_for_clip",
    "source_controller_executability_v2",
    "source_controller_fidelity_v2",
    "source_side_rank_v2",
]
