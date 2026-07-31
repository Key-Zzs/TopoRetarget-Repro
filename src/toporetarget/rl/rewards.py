"""Literal Table-4 reward profile and independent diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAPER_LITERAL_REWARD_ID = "paper_literal_reward_v1"


@dataclass(frozen=True)
class RewardProfile:
    object_weight: float = 8.0
    object_sigma_m: float = 0.04
    link_weight: float = 1.0
    link_sigma_m: float = 0.025
    joint_weight: float = 1.0
    joint_sigma_normalized: float = 0.1
    smoothness_weight: float = -0.01


PAPER_LITERAL_REWARD = RewardProfile()


def gaussian_tracking_kernel(error: np.ndarray | float, sigma: float) -> np.ndarray:
    """Paper's psi(e; sigma) = exp(-||e/sigma||^2)."""

    value = np.asarray(error, dtype=np.float64)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if value.ndim == 0:
        return np.exp(-np.square(value / sigma))
    return np.exp(-np.sum(np.square(value / sigma), axis=-1))


def paper_literal_reward(
    *,
    object_axis_points: np.ndarray,
    object_axis_points_ref: np.ndarray,
    link_positions: np.ndarray,
    link_positions_ref: np.ndarray,
    q: np.ndarray,
    q_ref: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    second_previous_action: np.ndarray,
    profile: RewardProfile = PAPER_LITERAL_REWARD,
) -> dict[str, float]:
    """Evaluate the PDF's Table-4 terms without reward-shaped substitutions."""

    axes = np.asarray(object_axis_points, dtype=np.float64)
    axes_ref = np.asarray(object_axis_points_ref, dtype=np.float64)
    links = np.asarray(link_positions, dtype=np.float64)
    links_ref = np.asarray(link_positions_ref, dtype=np.float64)
    current_q = np.asarray(q, dtype=np.float64)
    target_q = np.asarray(q_ref, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if axes.shape != (6, 3) or axes_ref.shape != (6, 3):
        raise ValueError("object axis points must have shape [6,3]")
    if links.shape != links_ref.shape or links.ndim != 2 or links.shape[1] != 3:
        raise ValueError("link positions must have matching shape [L,3]")
    if (
        current_q.shape != target_q.shape
        or current_q.shape != lower.shape
        or lower.shape != upper.shape
    ):
        raise ValueError("joint arrays must have identical shape")
    if np.any(upper <= lower):
        raise ValueError("joint upper limits must exceed lower limits")
    axis_distances = np.linalg.norm(axes - axes_ref, axis=-1)
    # The PDF places the 1/6 aggregate inside psi, not a squared mean error.
    object_term = float(
        gaussian_tracking_kernel(float(np.mean(axis_distances)), profile.object_sigma_m)
    )
    link_distances = np.linalg.norm(links - links_ref, axis=-1)
    link_term = float(np.mean(gaussian_tracking_kernel(link_distances, profile.link_sigma_m)))
    normalized_joint_error = (current_q - target_q) / (upper - lower)
    joint_term = float(
        np.mean(gaussian_tracking_kernel(normalized_joint_error, profile.joint_sigma_normalized))
    )
    first_difference = np.asarray(action) - np.asarray(previous_action)
    second_difference = (
        np.asarray(action) - 2.0 * np.asarray(previous_action) + np.asarray(second_previous_action)
    )
    smoothness = float(np.sum(first_difference**2) + np.sum(second_difference**2))
    total = (
        profile.object_weight * object_term
        + profile.link_weight * link_term
        + profile.joint_weight * joint_term
        + profile.smoothness_weight * smoothness
    )
    values = {
        "object": object_term,
        "link": link_term,
        "joint": joint_term,
        "smoothness": smoothness,
        "weighted_object": profile.object_weight * object_term,
        "weighted_link": profile.link_weight * link_term,
        "weighted_joint": profile.joint_weight * joint_term,
        "weighted_smoothness": profile.smoothness_weight * smoothness,
        "total": float(total),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError("paper literal reward is non-finite")
    return values


__all__ = [
    "PAPER_LITERAL_REWARD",
    "PAPER_LITERAL_REWARD_ID",
    "RewardProfile",
    "gaussian_tracking_kernel",
    "paper_literal_reward",
]
