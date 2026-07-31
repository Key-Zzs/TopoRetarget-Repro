"""Residual-action and global PD qualification helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTION_SCALE_CANDIDATES = (0.05, 0.10, 0.20)


def residual_target(
    q_ref: np.ndarray,
    action: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    *,
    action_scale_fraction: float,
) -> np.ndarray:
    """Apply one globally selected residual scale in joint-range units."""

    if action_scale_fraction not in ACTION_SCALE_CANDIDATES:
        raise ValueError("action scale must be a predefined global candidate")
    low = np.asarray(joint_lower, dtype=np.float64)
    high = np.asarray(joint_upper, dtype=np.float64)
    target = (
        np.asarray(q_ref, dtype=np.float64)
        + np.asarray(action, dtype=np.float64) * (high - low) * action_scale_fraction
    )
    return np.clip(target, low, high)


@dataclass(frozen=True)
class PDQualificationResult:
    action_scale_fraction: float
    settling_time_s: float
    overshoot_fraction: float
    saturated_fraction: float
    stable: bool


def choose_global_action_scale(results: list[PDQualificationResult]) -> PDQualificationResult:
    """Select pre-training using non-learning stability metrics, never clip results."""

    valid = [item for item in results if item.stable]
    if not valid:
        raise ValueError("no globally stable action-scale candidate")
    return min(
        valid,
        key=lambda item: (item.saturated_fraction, item.overshoot_fraction, item.settling_time_s),
    )


__all__ = [
    "ACTION_SCALE_CANDIDATES",
    "PDQualificationResult",
    "choose_global_action_scale",
    "residual_target",
]
