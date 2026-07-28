from __future__ import annotations

import numpy as np

from toporetarget.retarget.wuji_closeout import (
    ablation_conclusion,
    build_w2_attribution,
    decompose_q_steps,
    detect_qstep_transitions,
    jump_and_return_map,
    recommendation_gates,
    synthetic_window_fixture,
)


def _attribution_fixture(final_steps: list[float], correction_steps: list[float]) -> dict:
    q = np.zeros((3, 20), dtype=np.float64)
    warm = np.zeros_like(q)
    final_accum = 0.0
    correction_accum = 0.0
    for index, (final_step, correction_step) in enumerate(
        zip(final_steps, correction_steps, strict=True)
    ):
        final_accum += final_step
        correction_accum += correction_step
        q[index + 1, 0] = final_accum
        warm[index + 1, 0] = final_accum - correction_accum
    zeros = np.zeros((3, 20, 3), dtype=np.float64)
    zeros[..., 0] = 1.0
    keypoints = np.zeros((3, 21, 3), dtype=np.float64)
    return build_w2_attribution(
        warm_arrays={"qpos": warm},
        final_arrays={
            "qpos": q,
            "frame_indices": np.arange(3),
            "retry_profile": np.array(["none"] * 3),
            "retry_attempt": np.zeros(3),
            "active_set_rounds": np.zeros(3),
            "e_bone": np.zeros(3),
            "e_im": np.zeros(3),
        },
        joint_names=tuple(f"r_index_finger_joint_{i}" for i in range(20)),
        joint_lower=np.full(20, -1.0),
        joint_upper=np.full(20, 1.0),
        source_bone_directions=zeros,
        warm_bone_directions=zeros,
        final_bone_directions=zeros,
        warm_keypoints_scene=keypoints,
        final_keypoints_scene=keypoints,
        global_frame_offset=212,
        timestamps=np.arange(3, dtype=np.float64) / 120.0,
    )


def test_qstep_decomposition_detection_and_jump_return() -> None:
    warm = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    final = np.array([[0.0, 0.0], [0.2, 0.0], [0.1, 0.0]])
    result = decompose_q_steps(warm, final)
    assert result["decomposition_pass"]
    assert detect_qstep_transitions(final, threshold=0.05) == [1, 2]
    assert jump_and_return_map(result["delta_q_final"], threshold=0.05) == {1: [0]}


def test_attribution_source_and_correction_driven_categories() -> None:
    warm_driven = _attribution_fixture([0.08, 0.0], [0.005, 0.0])
    assert warm_driven["aggregate"]["warm_driven_count"] == 1
    assert warm_driven["aggregate"]["correction_driven_count"] == 0
    correction_driven = _attribution_fixture([0.08, 0.0], [0.08, 0.0])
    assert correction_driven["aggregate"]["correction_driven_count"] == 1
    assert correction_driven["aggregate"]["blocks_recommendation"]


def test_attribution_categories_include_limit_and_mixed() -> None:
    from toporetarget.retarget.wuji_closeout import _classify

    limit = _classify(
        warm=np.array([0.02]),
        correction=np.array([0.02]),
        final=np.array([0.04]),
        margins=np.array([0.01]),
        warm_keypoint_step=0.01,
        final_keypoint_step=0.03,
        jump_and_return=False,
    )
    mixed = _classify(
        warm=np.array([0.04]),
        correction=np.array([0.03]),
        final=np.array([0.07]),
        margins=np.array([0.5]),
        warm_keypoint_step=0.01,
        final_keypoint_step=0.01,
        jump_and_return=False,
    )
    assert limit[0] == "REACHABILITY_OR_LIMIT_DRIVEN"
    assert mixed[0] == "MIXED_WARM_AND_CORRECTION"


def test_ablation_labels_and_synthetic_window_are_deterministic() -> None:
    rows = []
    for profile, jumps, correction in (
        ("B0", (0.02, 0.02), 0.04),
        ("B1", (0.011, 0.0), 0.02),
        ("B2", (0.0, 0.0), 0.01),
    ):
        rows.extend(
            {
                "profile": profile,
                "solve": True,
                "base_jump_m": jump,
                "rotation_jump_rad": 0.0,
                "excess_keypoint_step_m": 0.0,
                "q_correction_linf_rad": correction,
            }
            for jump in jumps
        )
    assert ablation_conclusion(rows)["label"] == "TRANSPORT_AND_TEMPORAL_BOTH_REQUIRED"
    first = synthetic_window_fixture()
    second = synthetic_window_fixture()
    assert first == second
    assert first["routing_to_window"]
    assert first["center_continuity_pass"]
    assert first["checkpoint_resume_pass"]


def test_recommendation_gate_allows_warm_absolute_q_step_and_rejects_quality() -> None:
    formal = {
        "frame_count": 60,
        "all_optimizer_converged": True,
        "all_single_frame_feasible": True,
        "all_trajectory_continuous": True,
        "all_accepted": True,
        "q_bounds_pass": True,
        "slack_bounds_pass": True,
        "full_collision_pass": True,
        "unqueried_violation_count": 0,
        "all_finite": True,
        "max_base_translation_correction_m": 0.0,
        "max_base_rotation_correction_rad": 0.0,
        "max_correction_q_linf_rad": 0.04,
        "max_excess_keypoint_m": 0.0,
        "jump_and_return_count": 0,
        "baseline_mean_eim": 1.0,
        "continuous_mean_eim": 1.0,
        "baseline_mean_ebone": 1.0,
        "continuous_mean_ebone": 1.0,
        "baseline_max_penetration_m": 0.01,
        "continuous_max_penetration_m": 0.01,
        "baseline_penetration_rate": 0.1,
        "continuous_penetration_rate": 0.1,
        "baseline_joint_limit_saturation": 0.1,
        "continuous_joint_limit_saturation": 0.1,
        "max_base_jump_reduction": 0.8,
        "max_rotation_jump_reduction": 0.8,
        "max_keypoint_jump_reduction": 0.8,
        "q_jerk_reduction": 0.5,
        "base_jerk_reduction": 0.5,
    }
    attribution = {
        "aggregate": {
            "correction_driven_count": 0,
            "decomposition_max_error_rad": 0.0,
            "correction_continuity_gate_pass": True,
        }
    }
    passed = recommendation_gates(
        formal_rows=[formal],
        attribution=attribution,
        ablation_complete=True,
        synthetic_pass=True,
        real_window_pass=True,
        determinism_pass=True,
    )
    assert passed["passed"]
    formal["continuous_mean_eim"] = 1.06
    failed = recommendation_gates(
        formal_rows=[formal],
        attribution=attribution,
        ablation_complete=True,
        synthetic_pass=True,
        real_window_pass=True,
        determinism_pass=True,
    )
    assert not failed["quality_gate"]
    assert not failed["passed"]
