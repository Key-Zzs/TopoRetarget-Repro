from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.reference_tracking.reference_kinematics import (
    derive_angular_velocity_world_wxyz,
)
from toporetarget.rl.stage16_pf_df import (
    FINGER_ORDER,
    angular_episode_audit,
    contact_timing_metrics,
    evaluate_demonstration_fidelity,
    evaluate_physical_functionality,
    terminal_threshold_pass,
    true_runs,
)


def _contact_mask(start: int, *, frames: int = 12, fingers: tuple[int, ...] = (0, 1)) -> np.ndarray:
    mask = np.zeros((frames, 5), dtype=bool)
    mask[start:, fingers] = True
    return mask


def test_contact_timing_preserves_identity_time_authority_and_delay_sign() -> None:
    frames = 12
    metrics = contact_timing_metrics(
        raw_contact=_contact_mask(1),
        retarget_contact=_contact_mask(3),
        actual_contact=_contact_mask(7),
        actual_valid=np.ones(frames, dtype=bool),
        lift_onset=6,
        timestamps_s=np.arange(frames, dtype=np.float64) * 0.05,
        raw_timestamps_s=np.arange(frames, dtype=np.float64) * 0.00625,
        raw_frame_float=np.arange(frames, dtype=np.float64) / 8.0 + 10.0,
    )
    assert metrics["raw_ready"] == 1
    assert metrics["retarget_ready"] == 3
    assert metrics["actual_ready"] == 7
    assert metrics["raw_margin_frames"] == 5
    assert metrics["retarget_margin_frames"] == 3
    assert metrics["actual_margin_frames"] == -1
    assert metrics["raw_to_retarget_delay_frames"] == 2
    assert metrics["retarget_to_actual_delay_frames"] == 4
    assert metrics["lift_runtime_time_s"] == pytest.approx(0.3)
    assert metrics["lift_raw_time_s"] == pytest.approx(0.0375)
    assert metrics["lift_raw_frame_float"] == pytest.approx(10.75)
    assert metrics["source_required_fingers_at_lift"] == list(FINGER_ORDER[:2])
    assert metrics["actual_persistent_fingers_at_lift"] == []


def test_contact_timing_reports_each_finger_and_never_invents_missing_onsets() -> None:
    raw = _contact_mask(2, fingers=(1, 2))
    retarget = _contact_mask(2, fingers=(1, 2))
    actual = _contact_mask(2, fingers=(1, 2))
    metrics = contact_timing_metrics(
        raw_contact=raw,
        retarget_contact=retarget,
        actual_contact=actual,
        actual_valid=np.ones(12, dtype=bool),
        lift_onset=5,
        timestamps_s=np.arange(12, dtype=np.float64) * 0.05,
        raw_timestamps_s=np.arange(12, dtype=np.float64) * 0.00625,
        raw_frame_float=np.arange(12, dtype=np.float64),
    )
    rows = {row["finger"]: row for row in metrics["per_finger"]}
    assert tuple(rows) == FINGER_ORDER
    assert rows["thumb"]["raw_onset"] is None
    assert rows["index"]["raw_onset"] == 2
    assert rows["middle"]["actual_persistent"] == 2
    assert metrics["named_source_contact_match_at_lift"] is True


def _z_rotation_quaternion(angle: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.cos(angle / 2.0),
            np.zeros_like(angle),
            np.zeros_like(angle),
            np.sin(angle / 2.0),
        ),
        axis=-1,
    )


def _angular_audit(*, timestamps: np.ndarray, rate: float = 0.4) -> dict[str, object]:
    quaternion = _z_rotation_quaternion(rate * timestamps)
    omega = derive_angular_velocity_world_wxyz(quaternion, timestamps)
    pose = np.concatenate((np.zeros((len(timestamps), 3)), quaternion), axis=-1)
    twist = np.concatenate((np.zeros((len(timestamps), 3)), omega), axis=-1)
    phase = np.asarray(
        [
            "APPROACH",
            "APPROACH",
            "CONTACT",
            "CONTACT",
            "GRASP",
            "GRASP",
            "LIFT",
            "LIFT",
            "MANIPULATION",
        ]
    )[: len(timestamps)]
    return angular_episode_audit(
        actual_object_pose_wxyz=pose,
        actual_object_twist_world=twist,
        reference_object_pose_wxyz=pose,
        reference_object_twist_world=twist,
        wrist_pose_wxyz=pose,
        wrist_twist_world=twist,
        timestamps_s=timestamps,
        phase=phase,
        hand_object_contact=np.ones(len(timestamps), dtype=bool),
        valid=np.ones(len(timestamps), dtype=bool),
        contact_angular_limit_radps=0.5,
        free_angular_limit_radps=0.25,
    )


def test_angular_audit_constant_known_world_rotation_is_consistent() -> None:
    result = _angular_audit(timestamps=np.arange(9, dtype=np.float64) * 0.05)
    assert result["measurement_consistency"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["Delta_omega_trace"]["max"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["terminal"]["trace_pass_under_v1"] is True


def test_angular_audit_handles_so3_wraparound_and_nonuniform_timestamps() -> None:
    timestamps = np.asarray((0.0, 0.04, 0.11, 0.16, 0.23, 0.31, 0.36, 0.44, 0.5))
    angle = np.deg2rad(np.asarray((170, 175, 179, 183, 188, 194, 199, 204, 210)))
    quaternion = _z_rotation_quaternion(angle)
    omega = derive_angular_velocity_world_wxyz(quaternion, timestamps)
    assert np.isfinite(omega).all()
    assert np.max(np.linalg.norm(omega, axis=-1)) < 3.0
    pose = np.concatenate((np.zeros((len(timestamps), 3)), quaternion), axis=-1)
    twist = np.concatenate((np.zeros((len(timestamps), 3)), omega), axis=-1)
    result = angular_episode_audit(
        actual_object_pose_wxyz=pose,
        actual_object_twist_world=twist,
        reference_object_pose_wxyz=pose,
        reference_object_twist_world=twist,
        wrist_pose_wxyz=pose,
        wrist_twist_world=twist,
        timestamps_s=timestamps,
        phase=np.asarray(("APPROACH",) * 9),
        hand_object_contact=np.ones(9, dtype=bool),
        valid=np.ones(9, dtype=bool),
        contact_angular_limit_radps=0.5,
        free_angular_limit_radps=0.25,
    )
    assert result["measurement_consistency"]["max"] == pytest.approx(0.0, abs=1.0e-12)


def test_exceedance_runs_and_terminal_boundary_handling() -> None:
    assert true_runs(np.asarray((False, True, True, False, True))) == [(1, 3), (4, 5)]
    assert terminal_threshold_pass(
        np.asarray((1.0, 1.0, 0.1, 0.1)),
        contact=np.ones(4, dtype=bool),
        valid=np.ones(4, dtype=bool),
        contact_limit=0.5,
        free_limit=0.25,
        terminal_steps=2,
    )


@pytest.mark.parametrize(
    ("prelift", "lift", "causal", "hidden", "expected_reason"),
    (
        (True, 0.06, True, True, None),
        (False, 0.06, True, True, "prelift_multifinger_grasp_ready"),
        (True, 0.0, True, True, "lift_success"),
        (True, 0.06, False, True, "causal_execution"),
        (True, 0.06, True, False, "no_hidden_control"),
    ),
)
def test_physical_functionality_gates_late_grazing_lift_and_hidden_control(
    prelift: bool, lift: float, causal: bool, hidden: bool, expected_reason: str | None
) -> None:
    result = evaluate_physical_functionality(
        causal_execution=causal,
        geometry_safe=True,
        action_bounds_safe=True,
        prelift_multifinger_grasp_ready=prelift,
        lift_dz_m=lift,
        no_hidden_control=hidden,
    )
    assert result["pf"] is (expected_reason is None)
    if expected_reason is not None:
        assert expected_reason in result["pf_failure_reasons"]


def test_pf_and_df_are_orthogonal() -> None:
    pf_failure = evaluate_physical_functionality(
        causal_execution=True,
        geometry_safe=True,
        action_bounds_safe=True,
        prelift_multifinger_grasp_ready=False,
        lift_dz_m=0.0,
        no_hidden_control=True,
    )
    df_pass = evaluate_demonstration_fidelity(
        e_r_mean_deg=0.0,
        e_t_mean_cm=0.0,
        e_j_mean_cm=0.0,
        e_ft_mean_cm=0.0,
        linear_pass_under_v1=True,
        angular_trace_pass_under_v1=True,
        angular_pose_pass_under_v1=True,
    )
    assert pf_failure["pf"] is False
    assert df_pass["df_pose"] is True
    pf_pass = evaluate_physical_functionality(
        causal_execution=True,
        geometry_safe=True,
        action_bounds_safe=True,
        prelift_multifinger_grasp_ready=True,
        lift_dz_m=0.06,
        no_hidden_control=True,
    )
    df_failure = evaluate_demonstration_fidelity(
        e_r_mean_deg=31.0,
        e_t_mean_cm=0.0,
        e_j_mean_cm=0.0,
        e_ft_mean_cm=0.0,
        linear_pass_under_v1=False,
        angular_trace_pass_under_v1=False,
        angular_pose_pass_under_v1=False,
    )
    assert pf_pass["pf"] is True
    assert df_failure["df_pose"] is False
    assert df_failure["DF_OVERALL_BOOL"] == "NOT_DEFINED"
