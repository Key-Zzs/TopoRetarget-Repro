"""Numerical C.5A replication metrics without simulator dependencies."""

from __future__ import annotations

import torch


def quaternion_geodesic_rad(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("quaternion comparison requires matching [..., 4] tensors")
    exact_same_rotation = torch.logical_or(
        torch.all(first == second, dim=-1),
        torch.all(first == -second, dim=-1),
    )
    first_n = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(1.0e-12)
    second_n = second / torch.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1.0e-12)
    dot = (first_n * second_n).sum(dim=-1).abs().clamp(max=1.0)
    # A byte-identical unit quaternion can produce dot=0.99999994 after
    # float32 normalization, which would invent a 6.9e-4 rad error.  Exact
    # equality (including q/-q) is an exact SO(3) equality, not a tolerance.
    angle = 2.0 * torch.arccos(dot)
    return torch.where(exact_same_rotation, torch.zeros_like(angle), angle)


def max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError("difference requires matching shapes")
    return float((first - second).abs().amax().detach().cpu())


def state_differences(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Compare standard root/articulation fields used by C.5A qualifications."""

    result: dict[str, float] = {
        "wrist_position_m": max_abs_difference(
            first["robot_root_state"][..., :3], second["robot_root_state"][..., :3]
        ),
        "object_position_m": max_abs_difference(
            first["active_object_root_state"][..., :3], second["active_object_root_state"][..., :3]
        ),
        "quaternion_geodesic_rad": float(
            torch.maximum(
                quaternion_geodesic_rad(
                    first["robot_root_state"][..., 3:7], second["robot_root_state"][..., 3:7]
                ).amax(),
                quaternion_geodesic_rad(
                    first["active_object_root_state"][..., 3:7],
                    second["active_object_root_state"][..., 3:7],
                ).amax(),
            )
            .detach()
            .cpu()
        ),
        "joint_position_rad": max_abs_difference(
            first["robot_joint_pos"], second["robot_joint_pos"]
        ),
        "linear_velocity_si": max_abs_difference(
            torch.cat(
                (
                    first["robot_root_state"][..., 7:10],
                    first["active_object_root_state"][..., 7:10],
                ),
                dim=-1,
            ),
            torch.cat(
                (
                    second["robot_root_state"][..., 7:10],
                    second["active_object_root_state"][..., 7:10],
                ),
                dim=-1,
            ),
        ),
        "angular_velocity_si": max_abs_difference(
            torch.cat(
                (
                    first["robot_root_state"][..., 10:13],
                    first["active_object_root_state"][..., 10:13],
                ),
                dim=-1,
            ),
            torch.cat(
                (
                    second["robot_root_state"][..., 10:13],
                    second["active_object_root_state"][..., 10:13],
                ),
                dim=-1,
            ),
        ),
    }
    return result


__all__ = ["max_abs_difference", "quaternion_geodesic_rad", "state_differences"]
