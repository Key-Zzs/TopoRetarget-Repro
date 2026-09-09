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
    evaluate_candidate_b2,
    interaction_hinge,
    interaction_retention_limit,
    objective_v2_certification_state,
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


def test_candidate_b2_keeps_candidate_b_interaction_hinge() -> None:
    candidate = ObjectiveV2Candidate.candidate_b2()
    frozen = authority()
    assert candidate.primary_value(0.5e-4, frozen.interaction_target) == 0.0
    assert candidate.primary_value(1.0e-4, frozen.interaction_target) == 0.0
    assert candidate.primary_value(2.0e-4, frozen.interaction_target) == pytest.approx(1.0)


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


def test_b2_prediction_correction_can_exceed_profile_while_actual_continuity_passes() -> None:
    receipt = evaluate_candidate_b2(
        ObjectiveV2Candidate.candidate_b2(),
        measurement(
            temporal_base_translation_m=0.02,
            temporal_base_rotation_rad=0.2,
            temporal_q_inf_rad=0.3,
            temporal_excess_keypoint_m=0.04,
        ),
        authority(),
        actual_translation_step_m=0.01,
        actual_rotation_step_rad=0.1,
        actual_q_step_inf_rad=0.3,
        semantic_translation_step_limit_m=0.05,
        semantic_rotation_step_limit_rad=float(np.deg2rad(90.0)),
    )
    assert receipt["feasible"] is True
    assert receipt["prediction_correction_profile"]["hard_gate"] is False
    assert "temporal_q_inf_rad" not in receipt["constraint_margins"]
    assert receipt["actual_trajectory_continuity"]["q_step_hard_limit_rad"] is None


@pytest.mark.parametrize(
    ("translation", "rotation", "violation"),
    [
        (0.051, 0.1, "actual_temporal_translation_m"),
        (0.01, float(np.deg2rad(91.0)), "actual_temporal_rotation_rad"),
    ],
)
def test_b2_actual_semantic_continuity_is_hard(
    translation: float, rotation: float, violation: str
) -> None:
    receipt = evaluate_candidate_b2(
        ObjectiveV2Candidate.candidate_b2(),
        measurement(),
        authority(),
        actual_translation_step_m=translation,
        actual_rotation_step_rad=rotation,
        actual_q_step_inf_rad=0.01,
        semantic_translation_step_limit_m=0.05,
        semantic_rotation_step_limit_rad=float(np.deg2rad(90.0)),
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


def test_objective_v2_certification_state_fails_closed() -> None:
    assert objective_v2_certification_state(
        objective_frozen=False,
        sparse_status="NOT_RUN",
        window_status="NOT_RUN",
        frame0_status="NOT_RUN",
    ) == {
        "sparse_allowed": False,
        "window_allowed": False,
        "frame0_allowed": False,
        "full_dev2_authorized": False,
    }
    sparse_fail = objective_v2_certification_state(
        objective_frozen=True,
        sparse_status="FAIL",
        window_status="NOT_RUN",
        frame0_status="NOT_RUN",
    )
    assert sparse_fail["window_allowed"] is False
    window_fail = objective_v2_certification_state(
        objective_frozen=True,
        sparse_status="PASS",
        window_status="FAIL",
        frame0_status="NOT_RUN",
    )
    assert window_fail["frame0_allowed"] is False
    frame0_fail = objective_v2_certification_state(
        objective_frozen=True,
        sparse_status="PASS",
        window_status="PASS",
        frame0_status="FAIL",
    )
    assert frame0_fail["full_dev2_authorized"] is False
    all_pass = objective_v2_certification_state(
        objective_frozen=True,
        sparse_status="PASS",
        window_status="PASS",
        frame0_status="PASS",
    )
    assert all_pass["full_dev2_authorized"] is True
