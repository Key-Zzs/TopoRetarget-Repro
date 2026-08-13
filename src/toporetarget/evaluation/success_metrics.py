"""Trajectory success logic for Evaluation Suite V2."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .contracts import EvaluationSuiteV2, PhysicsEpisodeEvidence


def _trajectory_mean(series: np.ndarray, *, name: str) -> float:
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional series")
    return float(values.mean())


def trajectory_success(
    metrics: Mapping[str, np.ndarray],
    *,
    complete: bool,
    physics: PhysicsEpisodeEvidence,
    suite: EvaluationSuiteV2 | None = None,
    bimanual_right_metrics: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Evaluate strict V2 success; an incomplete trajectory always fails kinematics."""

    contract = suite or EvaluationSuiteV2()
    means = {
        "E_r_mean_deg": _trajectory_mean(metrics["e_r_deg"], name="e_r_deg"),
        "E_t_mean_cm": _trajectory_mean(metrics["e_t_cm"], name="e_t_cm"),
        "E_j_mean_cm": _trajectory_mean(metrics["e_j_cm"], name="e_j_cm"),
        "E_ft_mean_cm": _trajectory_mean(metrics["e_ft_cm"], name="e_ft_cm"),
    }
    object_pass = (
        means["E_r_mean_deg"] < contract.object_rotation_threshold_deg
        and means["E_t_mean_cm"] < contract.object_translation_threshold_cm
    )
    left_hand_pass = (
        means["E_j_mean_cm"] < contract.hand_joint_threshold_cm
        and means["E_ft_mean_cm"] < contract.fingertip_threshold_cm
    )
    right_hand_pass = True
    if bimanual_right_metrics is not None:
        right_hand_pass = (
            _trajectory_mean(bimanual_right_metrics["e_j_cm"], name="right_e_j_cm")
            < contract.hand_joint_threshold_cm
            and _trajectory_mean(bimanual_right_metrics["e_ft_cm"], name="right_e_ft_cm")
            < contract.fingertip_threshold_cm
        )
    kinematic_success = bool(complete and object_pass and left_hand_pass and right_hand_pass)
    physics_success = physics.success
    return {
        **means,
        "complete": bool(complete),
        "object_pass": object_pass,
        "left_hand_pass": left_hand_pass,
        "right_hand_pass": right_hand_pass,
        "kinematic_success": kinematic_success,
        "physics_success": physics_success,
        "qualified_success": bool(kinematic_success and physics_success),
    }
