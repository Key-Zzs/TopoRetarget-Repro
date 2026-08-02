"""Vectorized frozen Stage 16 world-wrist reward terms."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .tensor_math import quaternion_geodesic


@dataclass(frozen=True)
class Stage16WorldWristRewardProfileV1:
    object_weight: float = 8.0
    object_sigma_m: float = 0.04
    link_weight: float = 1.0
    link_sigma_m: float = 0.025
    finger_weight: float = 1.0
    finger_sigma_normalized: float = 0.10
    wrist_position_weight: float = 2.0
    wrist_position_sigma_m: float = 0.02
    wrist_rotation_weight: float = 1.0
    wrist_rotation_sigma_rad: float = 0.17453292519943295
    smoothness_weight: float = -0.01

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def world_wrist_reward_terms(
    *,
    object_axis_points: torch.Tensor,
    object_axis_points_ref: torch.Tensor,
    tracked_links: torch.Tensor,
    tracked_links_ref: torch.Tensor,
    finger_q: torch.Tensor,
    finger_q_ref: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    wrist_position: torch.Tensor,
    wrist_quaternion_wxyz: torch.Tensor,
    wrist_position_ref: torch.Tensor,
    wrist_quaternion_ref_wxyz: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    second_previous_action: torch.Tensor,
    profile: Stage16WorldWristRewardProfileV1 = Stage16WorldWristRewardProfileV1(),
) -> dict[str, torch.Tensor]:
    """Return per-environment rewards without simulator-only hidden inputs."""

    axis_error = torch.linalg.vector_norm(object_axis_points - object_axis_points_ref, dim=-1).mean(
        dim=-1
    )
    link_error = torch.linalg.vector_norm(tracked_links - tracked_links_ref, dim=-1)
    normalized_finger_error = (finger_q - finger_q_ref) / (joint_upper - joint_lower)
    wrist_position_error = torch.linalg.vector_norm(wrist_position - wrist_position_ref, dim=-1)
    wrist_rotation_error = quaternion_geodesic(wrist_quaternion_wxyz, wrist_quaternion_ref_wxyz)
    object_term = torch.exp(-(axis_error / profile.object_sigma_m).square())
    link_term = torch.exp(-(link_error / profile.link_sigma_m).square()).mean(dim=-1)
    finger_term = torch.exp(
        -(normalized_finger_error / profile.finger_sigma_normalized).square()
    ).mean(dim=-1)
    wrist_position_term = torch.exp(
        -(wrist_position_error / profile.wrist_position_sigma_m).square()
    )
    wrist_rotation_term = torch.exp(
        -(wrist_rotation_error / profile.wrist_rotation_sigma_rad).square()
    )
    smoothness = (action - previous_action).square().sum(dim=-1) + (
        action - 2.0 * previous_action + second_previous_action
    ).square().sum(dim=-1)
    total = (
        profile.object_weight * object_term
        + profile.link_weight * link_term
        + profile.finger_weight * finger_term
        + profile.wrist_position_weight * wrist_position_term
        + profile.wrist_rotation_weight * wrist_rotation_term
        + profile.smoothness_weight * smoothness
    )
    terms = {
        "object": object_term,
        "tracked_links": link_term,
        "finger_joints": finger_term,
        "wrist_position": wrist_position_term,
        "wrist_rotation": wrist_rotation_term,
        "smoothness": smoothness,
        "total": total,
    }
    if not all(bool(torch.isfinite(value).all()) for value in terms.values()):
        raise FloatingPointError("Stage 16-C reward became non-finite")
    return terms


__all__ = ["Stage16WorldWristRewardProfileV1", "world_wrist_reward_terms"]
