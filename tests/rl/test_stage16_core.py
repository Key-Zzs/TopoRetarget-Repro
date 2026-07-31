from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.axis_points import object_axis_points_from_poses
from toporetarget.rl.contracts import Stage16ReferenceClip, Stage16ReferenceValidationError
from toporetarget.rl.observations import (
    ObservationContract,
    ObservationDelayBuffer,
    build_observation,
)
from toporetarget.rl.randomization import DomainRandomizationConfig, sample_randomization
from toporetarget.rl.resampling import resample_reference_20hz
from toporetarget.rl.rewards import paper_literal_reward
from toporetarget.rl.state_machine import RecoveryBudget, Stage16RecoveryStateMachine
from toporetarget.rl.termination import TerminationInput, TerminationType, classify_termination


def reference() -> Stage16ReferenceClip:
    timestamps = np.asarray([0.0, 0.05, 0.10])
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, 0, 3] = [0.0, 0.01, 0.02]
    return Stage16ReferenceClip(
        timestamps=timestamps,
        q_finger_ref=np.zeros((3, 2)),
        object_pose_base_ref=poses,
        object_axis_points_base_ref=object_axis_points_from_poses(poses),
        tracked_link_positions_base_ref=np.zeros((3, 1, 3)),
        joint_order=("j0", "j1"),
        tracked_link_names=("palm",),
        provenance={"dataset": "synthetic_test"},
    )


def test_reference_resampling_and_static_gate(tmp_path) -> None:
    clip = reference()
    result = resample_reference_20hz(clip)
    assert result.validate(expected_hz=20.0)["valid"]
    path = result.to_npz(tmp_path / "reference.npz")
    assert Stage16ReferenceClip.from_npz(path).content_hash() == result.content_hash()
    with pytest.raises(Stage16ReferenceValidationError, match="STATIC_REFERENCE_NOT_RL_ELIGIBLE"):
        Stage16ReferenceClip(
            timestamps=np.asarray([0.0]),
            q_finger_ref=np.zeros((1, 2)),
            object_pose_base_ref=np.eye(4)[None],
            object_axis_points_base_ref=np.zeros((1, 6, 3)),
            tracked_link_positions_base_ref=np.zeros((1, 1, 3)),
            joint_order=("j0", "j1"),
            tracked_link_names=("palm",),
            provenance={"dataset": "static"},
        ).validate()


def test_observation_order_dimension_and_delay() -> None:
    clip = reference()
    contract = ObservationContract(2, 1)
    observed = build_observation(
        q=np.asarray([1.0, 2.0]),
        qdot=np.asarray([3.0, 4.0]),
        previous_action=np.asarray([5.0, 6.0]),
        current_object_axis_points=clip.object_axis_points_base_ref[0],
        reference=clip,
        reference_index=2,
    )
    assert observed.shape == (contract.dimension,)
    assert observed[:6].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    delay = ObservationDelayBuffer(2)
    first = delay.push(np.array([1]), np.array([2]), np.array([[3]]))
    delay.push(np.array([4]), np.array([5]), np.array([[6]]))
    third = delay.push(np.array([7]), np.array([8]), np.array([[9]]))
    assert first[0].item() == 1 and third[0].item() == 1


def test_literal_reward_monotonic_and_finite() -> None:
    zeros_axes = np.zeros((6, 3))
    zeros_links = np.zeros((2, 3))
    kwargs = dict(
        object_axis_points=zeros_axes,
        object_axis_points_ref=zeros_axes,
        link_positions=zeros_links,
        link_positions_ref=zeros_links,
        q=np.zeros(2),
        q_ref=np.zeros(2),
        joint_lower=-np.ones(2),
        joint_upper=np.ones(2),
        action=np.zeros(2),
        previous_action=np.zeros(2),
        second_previous_action=np.zeros(2),
    )
    perfect = paper_literal_reward(**kwargs)
    displaced = paper_literal_reward(**(kwargs | {"object_axis_points": np.full((6, 3), 0.04)}))
    assert perfect["object"] == pytest.approx(1.0)
    assert displaced["object"] < perfect["object"]
    assert np.isfinite(perfect["total"])


def test_termination_boundaries_are_exclusive() -> None:
    common = dict(
        step=1,
        reference_index=0,
        reference_frame_count=3,
        object_height_m=0.06,
        object_linear_velocity_mps=10.0,
        object_angular_velocity_radps=500.0,
        object_position_error_m=0.05,
        object_orientation_error_rad=np.deg2rad(45.0),
        max_axis_point_error_m=0.05,
    )
    assert classify_termination(TerminationInput(**common)) is None
    assert (
        classify_termination(TerminationInput(**(common | {"object_height_m": 0.06 - 1e-8})))
        == TerminationType.FAILURE_OBJECT_HEIGHT
    )
    assert (
        classify_termination(TerminationInput(**(common | {"reference_index": 2})))
        == TerminationType.SUCCESS_REFERENCE_COMPLETE
    )


def test_randomization_ranges_and_recovery_bound() -> None:
    sample = sample_randomization(np.random.default_rng(4), DomainRandomizationConfig())
    assert 0 <= sample["observation_delay_steps"] <= 2
    assert 0.75 <= sample["pd_stiffness_scale"] <= 1.5
    machine = Stage16RecoveryStateMachine(
        RecoveryBudget(repairs_per_class=1, reruns_per_phase=1, backend_switches=1, major_repairs=2)
    )
    from toporetarget.rl.failure_classifier import FailureClass

    machine.record(
        phase="test",
        failure_class=FailureClass.DATA_UNAVAILABLE,
        evidence={},
        repair="record",
        rerun_scope="none",
    )
    second = machine.record(
        phase="test",
        failure_class=FailureClass.DATA_UNAVAILABLE,
        evidence={},
        repair="record",
        rerun_scope="none",
    )
    assert second.result.startswith("ESCALATED")
