from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.objective_v2 import (
    ObjectiveV2Candidate,
    ObjectiveV2Measurements,
    ProductionObjectiveContextBindingV2,
    RetargetNonRegressionBudgetAuthorityV1,
    compare_candidate_states,
    evaluate_candidate,
    interaction_hinge,
    interaction_retention_limit,
)


def authority() -> RetargetNonRegressionBudgetAuthorityV1:
    return RetargetNonRegressionBudgetAuthorityV1.from_frozen_v1(
        collision_hard_bound_m=0.03,
        collision_soft_tolerance_m=0.001,
    )


def measurement(**updates: float) -> ObjectiveV2Measurements:
    values: dict[str, object] = {
        "interaction_e_im": 1.0e-4,
        "secondary_objective": 1.0,
        "bone_direction_p95_rad": 0.1,
        "wrist_position_m": 0.001,
        "wrist_rotation_rad": 0.01,
        "temporal_base_translation_m": 0.001,
        "temporal_base_rotation_rad": 0.01,
        "temporal_q_inf_rad": 0.01,
        "temporal_excess_keypoint_m": 0.001,
        "collision_min_signed_distance_m": -0.001,
        "joint_limit_min_margin_rad": 0.01,
        "rotation_determinant": 1.0,
        "unit_scale_ratio": 1.0,
        "per_components": {"bone": 0.2, "temporal": 0.8},
    }
    values.update(updates)
    return ObjectiveV2Measurements(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.0e-4, 1.0), (1.0e-4, 0.0), (0.5e-4, 0.0)],
)
def test_interaction_hinge_uses_frozen_target(value: float, expected: float) -> None:
    assert interaction_hinge(value, 1.0e-4) == pytest.approx(expected)


def test_thresholded_candidate_has_no_low_control_pressure() -> None:
    candidate = ObjectiveV2Candidate.candidate_b()
    frozen = authority()
    assert candidate.primary_value(1.0e-4, frozen.interaction_target) == 0.0
    assert candidate.primary_value(0.5e-4, frozen.interaction_target) == 0.0
    assert candidate.primary_value(2.0e-4, frozen.interaction_target) > 0.0


def test_candidate_a_remains_interaction_minimizing_below_target() -> None:
    candidate = ObjectiveV2Candidate.candidate_a()
    frozen = authority()
    assert candidate.primary_value(0.5e-4, frozen.interaction_target) == pytest.approx(0.5)


def test_evaluator_reports_decomposition_and_constraint_margins() -> None:
    receipt = evaluate_candidate(ObjectiveV2Candidate.candidate_b(), measurement(), authority())
    assert receipt["feasible"] is True
    assert receipt["primary_objective"] == 0.0
    assert receipt["secondary_objective"] == 1.0
    assert receipt["per_components"] == {"bone": 0.2, "temporal": 0.8}
    assert set(receipt["constraint_margins"]) >= {
        "collision_hard_m",
        "joint_limit_rad",
        "bone_direction_rad",
        "temporal_q_inf_rad",
    }


@pytest.mark.parametrize(
    ("field", "value", "violation"),
    [
        ("collision_min_signed_distance_m", -0.031, "collision_hard_m"),
        ("joint_limit_min_margin_rad", -0.001, "joint_limit_rad"),
        ("bone_direction_p95_rad", 0.9, "bone_direction_rad"),
        ("temporal_q_inf_rad", 0.051, "temporal_q_inf_rad"),
    ],
)
def test_each_hard_budget_violation_is_detected(field: str, value: float, violation: str) -> None:
    receipt = evaluate_candidate(
        ObjectiveV2Candidate.candidate_b(), measurement(**{field: value}), authority()
    )
    assert receipt["feasible"] is False
    assert violation in receipt["violated_constraints"]


def test_lexicographic_retention_uses_target_or_primary_optimum() -> None:
    assert interaction_retention_limit(0.8e-4, 1.0e-4) == 1.0e-4
    assert interaction_retention_limit(1.2e-4, 1.0e-4) == pytest.approx(1.200001e-4)


def test_thresholded_low_states_are_ranked_by_secondary() -> None:
    comparison = compare_candidate_states(
        ObjectiveV2Candidate.candidate_b(),
        measurement(interaction_e_im=0.8e-4, secondary_objective=2.0),
        measurement(interaction_e_im=0.2e-4, secondary_objective=3.0),
        authority(),
    )
    assert comparison["preferred"] == "LEFT"


def test_context_binding_requires_all_continuous_inputs() -> None:
    with pytest.raises(ValueError, match="omitted"):
        ProductionObjectiveContextBindingV2(
            active_frame_id=11,
            local_ordinal=1,
            current_source_frame_id=11,
            previous_source_frame_id=10,
            previous_runtime_base_scene=np.eye(4),
            previous_robot_qpos=np.zeros(2),
            continuous_predicted_translation_scene=np.zeros(3),
            continuous_predicted_rotation_scene=np.eye(3),
            continuous_predicted_qpos=None,
            base_correction_reference=np.zeros(6),
            object_pose_scene=np.eye(4),
            object_id="object",
            robot_name="robot",
            robot_side="right",
            robot_dof_names=("q0", "q1"),
            robot_mapping_authority="asset",
            source_hand_id="hand",
        ).validate()


def test_context_binding_preserves_frame_zero_no_temporal_branch() -> None:
    binding = ProductionObjectiveContextBindingV2(
        active_frame_id=10,
        local_ordinal=0,
        current_source_frame_id=10,
        previous_source_frame_id=None,
        previous_runtime_base_scene=None,
        previous_robot_qpos=None,
        continuous_predicted_translation_scene=None,
        continuous_predicted_rotation_scene=None,
        continuous_predicted_qpos=None,
        base_correction_reference=None,
        object_pose_scene=np.eye(4),
        object_id="object",
        robot_name="robot",
        robot_side="right",
        robot_dof_names=("q0", "q1"),
        robot_mapping_authority="asset",
        source_hand_id="hand",
    ).validate()
    assert binding.continuous_prediction_base is None
