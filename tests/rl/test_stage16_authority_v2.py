"""Pure offline tests for Stage16 measurement-authority V2 contracts."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.evaluation.source_contact_semantics import ManoSurfaceRegionMap
from toporetarget.rl.stage16_authority_v2 import (
    RawHumanGraspReadinessProfileV1,
    Stage16ActualAngularVelocityAuthorityV2,
    angular_velocity_semantic_alignment,
    opposing_contact_topology,
    raw_human_grasp_profile,
    timing_layer_profile,
)


def _pose(rotation: Rotation, count: int) -> np.ndarray:
    xyzw = rotation.as_quat()
    if xyzw.ndim == 1:
        xyzw = np.broadcast_to(xyzw, (count, 4))
    return np.concatenate((np.zeros((count, 3)), xyzw[:, 3:4], xyzw[:, :3]), axis=-1)


def test_angular_authority_static_and_constant_world_axis() -> None:
    timestamps = np.arange(101, dtype=np.float64) * 0.01
    static = _pose(Rotation.identity(), len(timestamps))
    zero = np.zeros((len(timestamps), 3), dtype=np.float64)
    static_audit = angular_velocity_semantic_alignment(
        object_pose_wxyz=static,
        trace_angular_velocity=zero,
        timestamps_s=timestamps,
    )
    np.testing.assert_allclose(static_audit["authority_omega_world"], 0.0, atol=1.0e-12)

    omega = np.array([0.0, 0.0, 0.7])
    rotation = Rotation.from_rotvec(timestamps[:, None] * omega[None, :])
    audit = angular_velocity_semantic_alignment(
        object_pose_wxyz=_pose(rotation, len(timestamps)),
        trace_angular_velocity=np.broadcast_to(omega, (len(timestamps), 3)),
        timestamps_s=timestamps,
    )
    np.testing.assert_allclose(
        audit["authority_omega_world"], np.broadcast_to(omega, (len(timestamps), 3)), atol=1.0e-10
    )
    assert audit["offset_diagnostics"]["t"]["max"] < 1.0e-10


def test_angular_authority_world_axis_with_nonidentity_body_orientation() -> None:
    timestamps = np.arange(101, dtype=np.float64) * 0.01
    omega = np.array([0.0, 0.0, 0.4])
    initial = Rotation.from_euler("xy", [37.0, -21.0], degrees=True)
    rotations = Rotation.from_rotvec(timestamps[:, None] * omega[None, :]) * initial
    audit = angular_velocity_semantic_alignment(
        object_pose_wxyz=_pose(rotations, len(timestamps)),
        trace_angular_velocity=np.broadcast_to(omega, (len(timestamps), 3)),
        timestamps_s=timestamps,
    )
    np.testing.assert_allclose(
        audit["authority_omega_world"], np.broadcast_to(omega, (len(timestamps), 3)), atol=1.0e-10
    )
    assert audit["frame_diagnostics"]["documented_world"]["max"] < 1.0e-10
    assert audit["frame_diagnostics"]["hypothetical_local_to_world"]["mean"] > 0.1


def test_angular_authority_so3_wraparound_and_provenance() -> None:
    timestamps = np.arange(21, dtype=np.float64) * 0.05
    angles = np.deg2rad(np.linspace(170.0, 190.0, len(timestamps)))
    omega = np.array([0.0, 0.0, np.deg2rad(20.0)])
    rotations = Rotation.from_rotvec(
        np.stack((np.zeros_like(angles), np.zeros_like(angles), angles), axis=-1)
    )
    audit = angular_velocity_semantic_alignment(
        object_pose_wxyz=_pose(rotations, len(timestamps)),
        trace_angular_velocity=np.broadcast_to(omega, (len(timestamps), 3)),
        timestamps_s=timestamps,
    )
    np.testing.assert_allclose(
        audit["authority_omega_world"], np.broadcast_to(omega, (len(timestamps), 3)), atol=1.0e-10
    )
    contract = Stage16ActualAngularVelocityAuthorityV2()
    assert "root_state_w[10:13]" in contract.legacy_trace_source
    assert contract.frame == contract.legacy_trace_frame == "WORLD"
    assert contract.timestamp.startswith("same post-physics")


def _region_fixture(frame_count: int = 4) -> tuple[np.ndarray, ManoSurfaceRegionMap, np.ndarray]:
    # One connected triangle per thumb/index/middle/ring/pinky/palm region.
    region_id = np.repeat(np.arange(6, dtype=np.int16), 3)
    segment_id = np.repeat(np.arange(6, dtype=np.int16), 3)
    soft = np.eye(6, dtype=np.float64)[region_id]
    faces = np.arange(18, dtype=np.int64).reshape(6, 3)
    distances = np.full((frame_count, 18), 0.02, dtype=np.float64)
    return distances, ManoSurfaceRegionMap(region_id, segment_id, soft), faces


def test_raw_profile_single_tip_palm_only_and_persistence_duration() -> None:
    distances, regions, faces = _region_fixture()
    distances[1:3, :3] = 0.001
    result = raw_human_grasp_profile(
        distances_m=distances,
        region_map=regions,
        mano_faces=faces,
    )
    assert result["any_robust_region_contact"].tolist() == [False, True, True, False]
    assert not result["multi_region_contact"].any()

    palm, regions, faces = _region_fixture()
    palm[:, 15:18] = 0.001
    palm_result = raw_human_grasp_profile(
        distances_m=palm,
        region_map=regions,
        mano_faces=faces,
    )
    assert palm_result["any_robust_region_contact"].all()
    assert not palm_result["multi_region_contact"].any()
    contract = RawHumanGraspReadinessProfileV1()
    assert contract.runtime_persistence_frames == 2
    assert np.isclose(contract.persistence_duration_s, 2.0 / 30.0)
    assert not contract.binary_functional_authority_validated
    assert not contract.outcome_tuned
    np.testing.assert_allclose(
        result["minimum_surface_distance_m"], distances.min(axis=1), atol=0.0
    )
    assert np.asarray(result["region_order"]).tolist() == [
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "palm",
    ]


def test_raw_profile_multi_region_and_opposing_geometry() -> None:
    distances, regions, faces = _region_fixture()
    distances[:, :6] = 0.001
    result = raw_human_grasp_profile(
        distances_m=distances,
        region_map=regions,
        mano_faces=faces,
    )
    assert result["multi_region_contact"].all()
    assert result["thumb_non_thumb_contact"].all()

    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    same_side = opposing_contact_topology(
        closest_points_object=points,
        contact_normals_object=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        contact_region_ids=np.array([0, 1]),
        minimum_separation_m=0.004,
        minimum_angle_deg=90.0,
    )
    opposing = opposing_contact_topology(
        closest_points_object=points,
        contact_normals_object=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        contact_region_ids=np.array([0, 1]),
        minimum_separation_m=0.004,
        minimum_angle_deg=90.0,
    )
    assert not same_side["opposing"]
    assert opposing["opposing"]
    assert opposing["maximum_normal_angle_deg"] == 180.0


def test_timing_profile_order_late_actual_sign_and_seconds() -> None:
    profile = timing_layer_profile(
        raw_frame=170,
        retarget_frame=180,
        actual_frame=190,
        lift_frame=184,
        control_dt_s=0.05,
    )
    assert profile["raw_to_retarget_frames"] == 10
    assert profile["retarget_to_actual_frames"] == 10
    assert profile["raw_margin_frames"] == 14
    assert profile["retarget_margin_frames"] == 4
    assert profile["actual_margin_frames"] == -6
    assert np.isclose(profile["actual_margin_s"], -0.3)
    assert profile["margin_sign_convention"] == "positive_before_lift_negative_after_lift"


def test_timing_profile_unavailable_raw_is_fail_closed() -> None:
    profile = timing_layer_profile(
        raw_frame=None,
        retarget_frame=181,
        actual_frame=198,
        lift_frame=184,
        control_dt_s=0.05,
    )
    assert profile["raw_frame"] is None
    assert profile["raw_margin_frames"] is None
    assert profile["raw_to_retarget_frames"] == "NOT_IDENTIFIABLE"
    assert profile["retarget_to_actual_frames"] == 17
    assert np.isclose(profile["retarget_to_actual_s"], 0.85)
