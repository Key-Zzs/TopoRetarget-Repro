"""Production source-controller AUTO routing and safety invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class SourceControllerMode(StrEnum):
    """Supported source-controller routing modes."""

    AUTO = "AUTO"
    ZERO_RESIDUAL = "ZERO_RESIDUAL"
    CORRECTED_L0 = "CORRECTED_L0"


class SourceControllerDecision(StrEnum):
    """P2 terminal decisions allowed by Hardening Protocol V2."""

    ZERO_RESIDUAL_SOURCE_CONTROLLER_SUFFICIENT = "ZERO_RESIDUAL_SOURCE_CONTROLLER_SUFFICIENT"
    AUTO_ZERO_RESIDUAL_THEN_L0_FALLBACK = "AUTO_ZERO_RESIDUAL_THEN_L0_FALLBACK"
    CORRECTED_L0_REQUIRED = "CORRECTED_L0_REQUIRED"
    L0_AUTHORITY_INCONCLUSIVE = "L0_AUTHORITY_INCONCLUSIVE"


@dataclass(frozen=True)
class SourceControllerSafetyContractV1:
    """Safety constraints that AUTO routing is never allowed to relax."""

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
        return {
            "real_finger_joint_limits": self.real_finger_joint_limits,
            "actuator_effort_limits": self.actuator_effort_limits,
            "actuator_velocity_limits": self.actuator_velocity_limits,
            "action_bounds": self.action_bounds,
            "controller_stability": self.controller_stability,
            "singularity_detection": self.singularity_detection,
            "collision_safety": self.collision_safety,
            "finite_state_checks": self.finite_state_checks,
            "virtual_wrist_angle_authority": self.virtual_wrist_angle_authority,
        }


def qualification_pass(receipt: Mapping[str, object]) -> bool:
    """Return whether one source-controller receipt passes execution gates."""

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
    """Select one global production route from matched per-clip receipts.

    Missing corrected-L0 evidence after any zero-residual failure is
    inconclusive and conservatively routes production through corrected L0.
    """

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
    """Apply AUTO without consulting any downstream PPO/PF outcome."""

    return (
        SourceControllerMode.ZERO_RESIDUAL
        if qualification_pass(zero_residual_receipt)
        else SourceControllerMode.CORRECTED_L0
    )


__all__ = [
    "SourceControllerDecision",
    "SourceControllerMode",
    "SourceControllerSafetyContractV1",
    "qualification_pass",
    "select_source_controller_route",
    "selected_mode_for_clip",
]
