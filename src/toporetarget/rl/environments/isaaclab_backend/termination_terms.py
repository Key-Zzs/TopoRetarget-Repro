"""Frozen Stage 16-C failure-first termination logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .tensor_math import quaternion_geodesic


@dataclass(frozen=True)
class Stage16TerminationProfileV1:
    object_position_error_max_m: float = 0.05
    object_axis_point_error_max_m: float = 0.05
    object_orientation_error_max_rad: float = 0.7853981633974483
    wrist_position_error_max_m: float = 0.20
    wrist_orientation_error_max_rad: float = 1.5707963267948966

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


TERMINATION_REASONS = (
    "NONE",
    "FAILURE_NUMERICAL",
    "FAILURE_OBJECT_POSITION",
    "FAILURE_OBJECT_AXIS_POINT",
    "FAILURE_OBJECT_ORIENTATION",
    "FAILURE_WRIST_POSITION_SAFETY",
    "FAILURE_WRIST_ORIENTATION_SAFETY",
    "SUCCESS_REFERENCE_COMPLETE",
)


def stage16_termination(
    *,
    object_position: torch.Tensor,
    object_quaternion_wxyz: torch.Tensor,
    object_axis_points: torch.Tensor,
    object_position_ref: torch.Tensor,
    object_quaternion_ref_wxyz: torch.Tensor,
    object_axis_points_ref: torch.Tensor,
    wrist_position: torch.Tensor,
    wrist_quaternion_wxyz: torch.Tensor,
    wrist_position_ref: torch.Tensor,
    wrist_quaternion_ref_wxyz: torch.Tensor,
    reference_index: torch.Tensor,
    final_reference_index: int,
    profile: Stage16TerminationProfileV1 = Stage16TerminationProfileV1(),
) -> dict[str, torch.Tensor]:
    """Evaluate every failure before success and retain one primary reason."""

    object_position_error = torch.linalg.vector_norm(object_position - object_position_ref, dim=-1)
    axis_error = torch.linalg.vector_norm(object_axis_points - object_axis_points_ref, dim=-1).amax(
        dim=-1
    )
    object_orientation_error = quaternion_geodesic(
        object_quaternion_wxyz, object_quaternion_ref_wxyz
    )
    wrist_position_error = torch.linalg.vector_norm(wrist_position - wrist_position_ref, dim=-1)
    wrist_orientation_error = quaternion_geodesic(wrist_quaternion_wxyz, wrist_quaternion_ref_wxyz)
    finite = torch.isfinite(
        torch.cat(
            (
                object_position,
                object_quaternion_wxyz,
                object_axis_points.flatten(1),
                wrist_position,
                wrist_quaternion_wxyz,
            ),
            dim=-1,
        )
    ).all(dim=-1)
    failures = torch.stack(
        (
            ~finite,
            object_position_error > profile.object_position_error_max_m,
            axis_error > profile.object_axis_point_error_max_m,
            object_orientation_error > profile.object_orientation_error_max_rad,
            wrist_position_error > profile.wrist_position_error_max_m,
            wrist_orientation_error > profile.wrist_orientation_error_max_rad,
        ),
        dim=-1,
    )
    failed = failures.any(dim=-1)
    priority = torch.arange(1, failures.shape[-1] + 1, device=failures.device)
    codes = torch.where(failures, priority, torch.full_like(priority, 99)).amin(dim=-1)
    success = (~failed) & (reference_index >= final_reference_index)
    primary = torch.where(
        success, torch.full_like(codes, 7), torch.where(failed, codes, torch.zeros_like(codes))
    )
    return {
        "terminated": failed,
        "success": success,
        "primary_reason_code": primary,
        "all_failure_conditions": failures,
        "object_position_error_m": object_position_error,
        "object_axis_error_m": axis_error,
        "object_orientation_error_rad": object_orientation_error,
        "wrist_position_error_m": wrist_position_error,
        "wrist_orientation_error_rad": wrist_orientation_error,
    }


def source_controller_admission_dones_v2(
    termination: dict[str, torch.Tensor],
    *,
    reference_index: torch.Tensor,
    final_reference_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate hard execution stops from task-fidelity termination reasons.

    Codes 2--4 are object-tracking diagnostics. Codes 1, 5, and 6 retain
    hard authority for numerical and wrist-safety failures.
    """

    reason = termination["primary_reason_code"]
    hard_terminated = (reason == 1) | (reason == 5) | (reason == 6)
    reference_complete = reference_index >= final_reference_index
    return hard_terminated, reference_complete & ~hard_terminated


__all__ = [
    "Stage16TerminationProfileV1",
    "TERMINATION_REASONS",
    "source_controller_admission_dones_v2",
    "stage16_termination",
]
