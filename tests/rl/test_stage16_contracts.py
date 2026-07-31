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
from toporetarget.rl.resampling import resample_reference_20hz, shortest_arc_slerp
from toporetarget.rl.rewards import paper_literal_reward
from toporetarget.rl.termination import (
    TerminationInput,
    TerminationType,
    classify_termination,
)


def _clip() -> Stage16ReferenceClip:
    timestamps = np.array([0.0, 0.1, 0.2])
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, 0, 3] = [0.0, 0.1, 0.2]
    return Stage16ReferenceClip(
        timestamps=timestamps,
        q_finger_ref=np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]),
        object_pose_base_ref=poses,
        object_axis_points_base_ref=object_axis_points_from_poses(poses),
        tracked_link_positions_base_ref=np.zeros((3, 2, 3)),
        joint_order=("joint_0", "joint_1"),
        tracked_link_names=("palm", "finger"),
        provenance={"dataset": "synthetic_test"},
    )


def test_reference_contract_and_20_hz_resampling_round_trip(tmp_path) -> None:
    source = _clip()
    output = resample_reference_20hz(source)
    assert output.timestamps.tolist() == pytest.approx([0.0, 0.05, 0.1, 0.15, 0.2])
    assert output.q_finger_ref[1].tolist() == pytest.approx([0.05, 0.15])
    path = output.to_npz(tmp_path / "reference.npz")
    loaded = Stage16ReferenceClip.from_npz(path)
    assert loaded.content_hash() == output.content_hash()
    assert loaded.validate(expected_hz=20.0)["valid"]


def test_static_reference_fails_closed() -> None:
    clip = _clip()
    clip.timestamps = clip.timestamps[:1]
    clip.q_finger_ref = clip.q_finger_ref[:1]
    clip.object_pose_base_ref = clip.object_pose_base_ref[:1]
    clip.object_axis_points_base_ref = clip.object_axis_points_base_ref[:1]
    clip.tracked_link_positions_base_ref = clip.tracked_link_positions_base_ref[:1]
    clip.reference_indices = clip.reference_indices[:1]
    with pytest.raises(Stage16ReferenceValidationError, match="STATIC_REFERENCE_NOT_RL_ELIGIBLE"):
        clip.validate()


def test_shortest_arc_slerp_normalizes_and_uses_short_path() -> None:
    result = shortest_arc_slerp(
        np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, -1.0]), 0.5
    )
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert abs(result[3]) == pytest.approx(1.0)


def test_resampling_recomputes_axes_and_base_frame_angular_velocity() -> None:
    source = _clip()
    source.object_pose_base_ref[2, :3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    source.object_axis_points_base_ref = object_axis_points_from_poses(source.object_pose_base_ref)
    output = resample_reference_20hz(source)
    assert output.metadata["axis_resampling"] == "recomputed_from_resampled_pose"
    assert np.linalg.norm(output.object_velocity_ref[:, 3:], axis=1).max() > 0.0
    assert output.object_axis_points_base_ref[-1, 0].tolist() == pytest.approx([0.2, 0.05, 0.0])


def test_resampling_180_degree_rotation_has_finite_angular_velocity() -> None:
    source = _clip()
    source.object_pose_base_ref[-1, :3, :3] = np.diag([-1.0, -1.0, 1.0])
    source.object_axis_points_base_ref = object_axis_points_from_poses(source.object_pose_base_ref)
    output = resample_reference_20hz(source)
    assert np.isfinite(output.object_velocity_ref).all()


def test_observation_order_dimension_and_delay() -> None:
    clip = resample_reference_20hz(_clip())
    contract = ObservationContract(dof_count=2, link_count=2)
    observation = build_observation(
        q=np.array([1.0, 2.0]),
        qdot=np.array([3.0, 4.0]),
        previous_action=np.array([5.0, 6.0]),
        current_object_axis_points=np.zeros((6, 3)),
        reference=clip,
        reference_index=clip.frame_count - 1,
        contract=contract,
    )
    assert observation.shape == (contract.dimension,)
    assert observation[:6].tolist() == pytest.approx([1, 2, 3, 4, 5, 6])
    delayed = ObservationDelayBuffer(2)
    first = delayed.push(np.array([1]), np.array([1]), np.zeros((6, 3)))
    delayed.push(np.array([2]), np.array([2]), np.ones((6, 3)))
    third = delayed.push(np.array([3]), np.array([3]), np.full((6, 3), 2.0))
    assert first[0].tolist() == [1]
    assert third[0].tolist() == [1]


def test_literal_reward_zero_error_is_maximum_and_smoothness_is_signed() -> None:
    values = dict(
        object_axis_points=np.zeros((6, 3)),
        object_axis_points_ref=np.zeros((6, 3)),
        link_positions=np.zeros((2, 3)),
        link_positions_ref=np.zeros((2, 3)),
        q=np.zeros(2),
        q_ref=np.zeros(2),
        joint_lower=-np.ones(2),
        joint_upper=np.ones(2),
        action=np.zeros(2),
        previous_action=np.zeros(2),
        second_previous_action=np.zeros(2),
    )
    zero = paper_literal_reward(**values)
    error = paper_literal_reward(**(values | {"object_axis_points": np.full((6, 3), 0.04)}))
    noisy = paper_literal_reward(**(values | {"action": np.ones(2)}))
    assert zero["object"] == pytest.approx(1.0)
    assert error["object"] < zero["object"]
    assert noisy["weighted_smoothness"] < 0.0


def test_termination_boundaries_are_strict_as_in_table_four() -> None:
    baseline = dict(
        step=0,
        reference_index=0,
        reference_frame_count=2,
        object_height_m=0.06,
        object_linear_velocity_mps=10.0,
        object_angular_velocity_radps=500.0,
        object_position_error_m=0.05,
        object_orientation_error_rad=np.deg2rad(45.0),
        max_axis_point_error_m=0.05,
    )
    assert classify_termination(TerminationInput(**baseline)) is None
    assert (
        classify_termination(TerminationInput(**(baseline | {"object_height_m": 0.06 - 1e-9})))
        == TerminationType.FAILURE_OBJECT_HEIGHT
    )
    assert (
        classify_termination(
            TerminationInput(**(baseline | {"object_position_error_m": 0.05 + 1e-9}))
        )
        == TerminationType.FAILURE_OBJECT_POSITION
    )
    assert (
        classify_termination(TerminationInput(**(baseline | {"reference_index": 1})))
        == TerminationType.SUCCESS_REFERENCE_COMPLETE
    )
