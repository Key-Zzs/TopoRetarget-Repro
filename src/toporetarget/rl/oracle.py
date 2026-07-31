"""Non-learning residual controllers used only for controllability diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OracleResidualController:
    """A shared, object-blind joint residual diagnostic.

    The controller uses the same normalized residual action as PPO.  It may use
    the current measured joint state and the allowed reference lookahead, but it
    never writes object state, qpos, or an object action.  This is an environment
    controllability probe, never an RL result.
    """

    joint_gain: float = 0.50
    feedforward_gain: float = 1.0
    action_scale_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.joint_gain < 0.0 or self.feedforward_gain < 0.0:
            raise ValueError("oracle gains must be non-negative")
        if self.action_scale_fraction not in (0.05, 0.10, 0.20):
            raise ValueError("oracle action scale must be a frozen global candidate")

    def action(
        self,
        *,
        q: np.ndarray,
        q_ref: np.ndarray,
        q_ref_next: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ) -> np.ndarray:
        """Return a clipped residual action in PPO's normalized action space."""

        low = np.asarray(joint_lower, dtype=np.float64)
        high = np.asarray(joint_upper, dtype=np.float64)
        span = high - low
        if np.any(span <= 0.0):
            raise ValueError("joint ranges must be positive")
        current = np.asarray(q, dtype=np.float64)
        reference = np.asarray(q_ref, dtype=np.float64)
        next_reference = np.asarray(q_ref_next, dtype=np.float64)
        if current.shape != reference.shape or reference.shape != next_reference.shape:
            raise ValueError("oracle joint arrays must have equal shape")
        target_delta = self.feedforward_gain * (next_reference - reference)
        feedback_delta = self.joint_gain * (next_reference - current)
        action = (target_delta + feedback_delta) / (span * self.action_scale_fraction)
        return np.clip(action, -1.0, 1.0)


def oracle_action(
    controller: OracleResidualController,
    *,
    state: dict[str, np.ndarray],
    reference_q: np.ndarray,
    next_reference_q: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> np.ndarray:
    """Small adapter that makes the object-blind access path explicit."""

    return controller.action(
        q=state["q"],
        q_ref=reference_q,
        q_ref_next=next_reference_q,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
    )


__all__ = ["OracleResidualController", "oracle_action"]
