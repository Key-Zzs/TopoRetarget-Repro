from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.evaluation.human_object_interaction_profile import (
    HumanObjectCouplingContactProfileContractV1,
    build_human_object_interaction_profile,
)


def _pose(translation: np.ndarray, yaw_rad: np.ndarray | None = None) -> np.ndarray:
    xyz = np.asarray(translation, dtype=np.float64)
    yaw = np.zeros(len(xyz)) if yaw_rad is None else np.asarray(yaw_rad, dtype=np.float64)
    quat_xyzw = Rotation.from_euler("z", yaw[:, None]).as_quat()
    return np.concatenate((xyz, quat_xyzw[:, 3:4], quat_xyzw[:, :3]), axis=1)


def test_no_contact_and_single_or_multi_region_layers_are_preserved() -> None:
    timestamps = np.arange(6, dtype=np.float64) * 0.05
    still = np.zeros((6, 3), dtype=np.float64)
    region = np.zeros((6, 3), dtype=bool)
    region[2, 0] = True
    region[3:, :2] = True
    result = build_human_object_interaction_profile(
        hand_pose_world_wxyz=_pose(still),
        object_pose_world_wxyz=_pose(still + np.array([0.1, 0.0, 0.0])),
        timestamps_s=timestamps,
        region_contact=region,
        any_hand_surface_contact=region.any(axis=1),
        multi_region_contact=region.sum(axis=1) >= 2,
        opposing_contact_topology=np.array([False, False, False, True, True, True]),
    )
    assert result["number_of_active_regions"].tolist() == [0, 0, 1, 2, 2, 2]
    assert not result["any_hand_surface_contact"][0]
    assert result["multi_region_contact"][3]
    assert result["opposing_contact_topology"][3]


def test_rigidly_coupled_common_motion_has_zero_relative_rates() -> None:
    timestamps = np.arange(21, dtype=np.float64) * 0.05
    yaw = 0.4 * timestamps
    hand_xyz = np.stack((0.2 * timestamps, -0.1 * timestamps, 0.05 * timestamps), axis=1)
    rotation = Rotation.from_euler("z", yaw[:, None]).as_matrix()
    offset_hand = np.array([0.08, -0.03, 0.02])
    object_xyz = hand_xyz + np.einsum("tij,j->ti", rotation, offset_hand)
    result = build_human_object_interaction_profile(
        hand_pose_world_wxyz=_pose(hand_xyz, yaw),
        object_pose_world_wxyz=_pose(object_xyz, yaw + 0.25),
        timestamps_s=timestamps,
    )
    np.testing.assert_allclose(
        result["relative_translation_hand_m"], np.tile(offset_hand, (21, 1)), atol=1e-10
    )
    np.testing.assert_allclose(result["relative_linear_speed_mps"], 0.0, atol=1e-9)
    np.testing.assert_allclose(result["relative_angular_speed_radps"], 0.0, atol=1e-9)
    np.testing.assert_allclose(result["linear_coupling_ratio"], 0.0, atol=1e-8)
    np.testing.assert_allclose(result["angular_coupling_ratio"], 0.0, atol=1e-8)


def test_sliding_relative_to_object_is_detected() -> None:
    timestamps = np.arange(11, dtype=np.float64) * 0.05
    hand_xyz = np.zeros((11, 3), dtype=np.float64)
    object_xyz = np.stack((0.04 * timestamps, np.zeros(11), np.zeros(11)), axis=1)
    result = build_human_object_interaction_profile(
        hand_pose_world_wxyz=_pose(hand_xyz),
        object_pose_world_wxyz=_pose(object_xyz),
        timestamps_s=timestamps,
    )
    np.testing.assert_allclose(result["relative_linear_speed_mps"], 0.04, atol=1e-10)
    assert np.all(result["linear_coupling_ratio"] > 0.999999)


def test_absolute_speed_scale_does_not_change_dimensionless_coupling() -> None:
    timestamps = np.arange(11, dtype=np.float64) * 0.05

    def profile(scale: float) -> dict[str, np.ndarray]:
        hand = np.stack((scale * timestamps, np.zeros(11), np.zeros(11)), axis=1)
        obj = np.stack((1.5 * scale * timestamps, np.zeros(11), np.zeros(11)), axis=1)
        return build_human_object_interaction_profile(
            hand_pose_world_wxyz=_pose(hand),
            object_pose_world_wxyz=_pose(obj),
            timestamps_s=timestamps,
        )

    slow = profile(0.1)
    fast = profile(1.0)
    np.testing.assert_allclose(
        slow["linear_coupling_ratio"], fast["linear_coupling_ratio"], atol=5e-8
    )


def test_world_hand_and_object_local_transform_consistency() -> None:
    timestamps = np.arange(5, dtype=np.float64) * 0.05
    yaw = np.full(5, np.pi / 2.0)
    hand_xyz = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    object_xyz = hand_xyz + np.array([-0.2, 0.1, 0.3])
    result = build_human_object_interaction_profile(
        hand_pose_world_wxyz=_pose(hand_xyz, yaw),
        object_pose_world_wxyz=_pose(object_xyz, yaw + 0.3),
        timestamps_s=timestamps,
    )
    expected_hand = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        result["relative_translation_hand_m"], np.tile(expected_hand, (5, 1)), atol=1e-12
    )
    assert np.isclose(np.linalg.norm(result["relative_rotation_vector_hand_rad"][0]), 0.3)


def test_30hz_and_retimed_20hz_preserve_constant_velocity() -> None:
    def speed(rate_hz: float, count: int) -> np.ndarray:
        timestamps = np.arange(count, dtype=np.float64) / rate_hz
        zero = np.zeros((count, 3), dtype=np.float64)
        obj = np.stack((0.12 * timestamps, np.zeros(count), np.zeros(count)), axis=1)
        return build_human_object_interaction_profile(
            hand_pose_world_wxyz=_pose(zero),
            object_pose_world_wxyz=_pose(obj),
            timestamps_s=timestamps,
        )["relative_linear_speed_mps"]

    np.testing.assert_allclose(speed(30.0, 31), 0.12, atol=1e-10)
    np.testing.assert_allclose(speed(20.0, 21), 0.12, atol=1e-10)


def test_contract_forbids_binary_or_outcome_tuned_profile() -> None:
    for field in (
        "raw_functional_grasp_binary_required",
        "hard_coupling_threshold_defined",
        "outcome_tuned",
        "force_closure_claimed",
    ):
        try:
            HumanObjectCouplingContactProfileContractV1(**{field: True})
        except ValueError:
            pass
        else:
            raise AssertionError(f"{field} must fail closed")
