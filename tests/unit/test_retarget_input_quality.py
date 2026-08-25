from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from toporetarget.retarget.bones import load_bone_profile
from toporetarget.retarget.final_refinement import _make_source_features
from toporetarget.retarget.frames import FrameDegeneracyError, load_frame_profile
from toporetarget.retarget.input_quality import (
    RetargetInputQualityContractV1,
    RetargetInputQualityError,
    bone_quality,
    keypoint_frame_diagnostics,
    repair_mano_pose,
    repair_object_pose_qxyzw,
    select_mano_primary_wrist_frames,
)


def test_mano_global_orientation_is_primary_over_degenerate_keypoint_frame() -> None:
    keypoints = np.zeros((3, 21, 3), dtype=np.float64)
    diagnostic = keypoint_frame_diagnostics(keypoints)
    frames, authority = select_mano_primary_wrist_frames(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]),
        np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
    )

    assert not diagnostic["valid"].any()
    assert authority.tolist() == ["MANO_GLOBAL_WRIST_ORIENTATION"] * 3
    assert np.allclose(frames[:, :3, 3], [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])


def test_zero_length_tracked_bone_recovers_from_valid_mano_skeleton() -> None:
    tracked = np.zeros((2, 3, 3), dtype=np.float64)
    mano = tracked.copy()
    mano[:, 1, 0] = 0.02
    mano[:, 2, 0] = 0.04
    result = bone_quality(tracked, mano, np.asarray([0, 1]), np.asarray([1, 2]))

    assert result["mano_parametric_recovery"].tolist() == [True, True]
    assert not result["unrecoverable"].any()


def test_mano_skeleton_degeneration_is_unrecoverable() -> None:
    points = np.zeros((1, 3, 3), dtype=np.float64)
    result = bone_quality(points, points, np.asarray([0, 1]), np.asarray([1, 2]))
    assert result["unrecoverable"].tolist() == [True]


def test_short_mano_gap_uses_geodesic_orientation_and_linear_tracks() -> None:
    contract = RetargetInputQualityContractV1(maximum_repair_gap_seconds=0.11)
    timestamps = np.asarray([0.0, 0.05, 0.10])
    pose = np.zeros((3, 51), dtype=np.float64)
    pose[1] = np.nan
    pose[2, :3] = [0.0, 0.0, np.pi / 2]
    pose[2, 3:48] = 2.0
    pose[2, 48:51] = [0.2, 0.0, 0.0]

    repaired, receipt = repair_mano_pose(pose, timestamps, contract)

    assert Rotation.from_rotvec(repaired[1, :3]).magnitude() == pytest.approx(np.pi / 4)
    assert np.allclose(repaired[1, 3:48], 1.0)
    assert np.allclose(repaired[1, 48:51], [0.1, 0.0, 0.0])
    assert receipt["short_invalid_gaps"] == [[1, 2]]


def test_long_or_boundary_gap_fails_closed() -> None:
    timestamps = np.asarray([0.0, 0.1, 0.2, 0.3])
    pose = np.zeros((4, 51), dtype=np.float64)
    pose[1:3] = np.nan
    with pytest.raises(RetargetInputQualityError, match="UNRECOVERABLE_TRACKING_GAP"):
        repair_mano_pose(pose, timestamps)

    pose[:] = 0.0
    pose[0] = np.nan
    with pytest.raises(RetargetInputQualityError, match="UNRECOVERABLE_TRACKING_GAP"):
        repair_mano_pose(pose, timestamps)


def test_short_object_gap_preserves_so3_and_translation() -> None:
    timestamps = np.asarray([0.0, 0.05, 0.10])
    pose = np.zeros((3, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    pose[1] = np.nan
    pose[2, :4] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    pose[2, 4:] = [0.2, 0.0, 0.0]

    repaired, receipt = repair_object_pose_qxyzw(pose, timestamps)

    assert Rotation.from_quat(repaired[1, :4]).magnitude() == pytest.approx(np.pi / 4)
    assert np.allclose(repaired[1, 4:], [0.1, 0.0, 0.0])
    assert receipt["objects"][0]["repaired_frames"] == [1]


def test_wrist_temporal_propagation_precedes_keypoint_diagnostic_fallback() -> None:
    orient = np.asarray([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 0.2]])
    translation = np.asarray([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.2, 0.0, 0.0]])
    keypoint = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()

    frames, authority = select_mano_primary_wrist_frames(
        orient,
        translation,
        timestamps=np.asarray([0.0, 0.05, 0.10]),
        keypoint_wrist_pose=keypoint,
    )

    assert authority[1] == "TEMPORALLY_PROPAGATED_VALID_WRIST_FRAME"
    assert np.allclose(frames[1, :3, 3], [0.1, 0.0, 0.0])


def test_long_wrist_gap_is_not_temporally_propagated() -> None:
    orient = np.asarray([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 0.2]])
    translation = np.asarray([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.2, 0.0, 0.0]])
    keypoint = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()

    _, authority = select_mano_primary_wrist_frames(
        orient,
        translation,
        timestamps=np.asarray([0.0, 0.10, 0.20]),
        keypoint_wrist_pose=keypoint,
    )

    assert authority[1] == "KEYPOINT_DERIVED_WRIST_FRAME_DIAGNOSTIC_ONLY"


def test_final_refinement_source_features_bypass_keypoint_axis_with_mano_frame() -> None:
    points = np.arange(63, dtype=np.float64).reshape(21, 3) * 0.001
    points[0] = [0.0, 0.0, 0.0]
    points[9] = [0.0, 0.10, 0.0]
    points[5] = [0.0, 0.08, 0.0]
    points[17] = [0.0, 0.04, 0.0]
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    with pytest.raises(FrameDegeneracyError):
        frame_profile.frame_transform(points, side="right")

    mano_frame = np.eye(4, dtype=np.float64)
    features = _make_source_features(points, frame_profile, bone_profile, "right", mano_frame)
    assert np.array_equal(features.frame_transform, mano_frame)
    assert np.all(features.valid_bones)
