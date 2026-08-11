from __future__ import annotations

import numpy as np

from toporetarget.evaluation import (
    EvaluationSuiteV2,
    PhysicsEpisodeEvidence,
    quaternion_geodesic_deg,
    trajectory_success,
)


def _physics(**overrides: bool) -> PhysicsEpisodeEvidence:
    values = {
        "terminal_contact_pass": True,
        "terminal_stability_pass": True,
        "contact_causality_pass": True,
        "inter_finger_penetration_pass": True,
        "absolute_hand_object_penetration_pass": True,
        "action_bounds_pass": True,
        "no_hidden_force": True,
        "no_object_rollout_state_write": True,
        "no_wrist_root_teleport": True,
    }
    values.update(overrides)
    return PhysicsEpisodeEvidence(**values)


def _metrics(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "e_r_deg": np.full(4, value),
        "e_t_cm": np.full(4, value),
        "e_j_cm": np.full(4, value),
        "e_ft_cm": np.full(4, value),
    }


def test_so3_geodesic_and_strict_kinematic_threshold() -> None:
    actual = np.array([[1.0, 0.0, 0.0, 0.0]])
    reference = np.array([[0.0, 1.0, 0.0, 0.0]])
    assert np.allclose(quaternion_geodesic_deg(actual, reference), [180.0])
    suite = EvaluationSuiteV2()
    metrics = _metrics()
    metrics["e_r_deg"][:] = suite.object_rotation_threshold_deg
    result = trajectory_success(metrics, complete=True, physics=_physics(), suite=suite)
    assert result["kinematic_success"] is False


def test_incomplete_and_physics_and_bimanual_are_fail_closed() -> None:
    metrics = _metrics()
    assert (
        trajectory_success(metrics, complete=False, physics=_physics())["kinematic_success"]
        is False
    )
    assert (
        trajectory_success(metrics, complete=True, physics=_physics(action_bounds_pass=False))[
            "physics_success"
        ]
        is False
    )
    right = {"e_j_cm": np.full(2, 9.0), "e_ft_cm": np.full(2, 1.0)}
    assert (
        trajectory_success(
            metrics, complete=True, physics=_physics(), bimanual_right_metrics=right
        )["kinematic_success"]
        is False
    )
