import numpy as np

from toporetarget.evaluation.policy_failure_localization import (
    action_saturation,
    force_feasibility,
    forgetting,
    reward_product_error,
    tracking_errors,
    viability_probability,
)


def test_action_boundaries_and_tracking():
    result = action_saturation(np.vstack([np.zeros(26), np.ones(26)]))
    assert result["exact_all"] == 0.5
    errors = tracking_errors(
        np.zeros((2, 7)), np.zeros((2, 7)), np.zeros((2, 20)), np.ones((2, 20))
    )
    assert np.all(errors["wrist_translation_m"] == 0) and np.all(
        errors["finger_joint_abs_rad"] == 1
    )


def test_force_proxy_and_reward_product():
    status, count, rank, _ = force_feasibility(
        np.array([[0, 0, 0], [1, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0]]),
        np.zeros(3),
        np.array([0, 0, -1]),
    )
    assert count == 2 and rank == 2 and status in {"GRASP_FEASIBLE", "GRASP_MARGINAL"}
    assert reward_product_error(np.array([[0.5, 0.5, 0.5, 0.5]]), np.array([0.0625])) == 0


def test_viability_and_forgetting():
    mass = viability_probability(
        ["REFERENCE_STATE_VIABLE", "REFERENCE_STATE_NONVIABLE"], [0.25, 0.75]
    )
    assert mass["viable"] == 0.25 and forgetting(0.5, 0.1)
