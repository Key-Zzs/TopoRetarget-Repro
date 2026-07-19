from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from toporetarget.data.schema import KeypointTrack
from toporetarget.data.synthetic import make_synthetic_sequence
from toporetarget.geometry.se3 import wrist_to_scene
from toporetarget.keypoints import ManoToMediaPipe21Converter, MappingError
from toporetarget.keypoints.registry import get_layout, load_profiles

SOURCE_NAMES = list(get_layout("mano16_smplx").semantic_names)
TIP_VERTICES = (744, 320, 443, 554, 671)


def make_mano_sequence(*, reordered: bool = False, invalid_source: bool = False):
    sequence = make_synthetic_sequence(num_frames=4)
    hand = sequence.hands[0]
    local_joints = np.zeros((sequence.num_frames, 16, 3), dtype=np.float64)
    for index in range(16):
        local_joints[:, index] = [0.006 * index, 0.002 * (index % 3), 0.001 * (index % 2)]
    joint_scene = wrist_to_scene(hand.wrist_pose_scene.pose_scene, local_joints)
    vertices_local = np.zeros((sequence.num_frames, 778, 3), dtype=np.float64)
    for tip_order, vertex_index in enumerate(TIP_VERTICES):
        vertices_local[:, vertex_index] = [0.02 + 0.01 * tip_order, 0.015, 0.01]
    vertices_scene = wrist_to_scene(hand.wrist_pose_scene.pose_scene, vertices_local)
    names = SOURCE_NAMES.copy()
    if reordered:
        order = list(reversed(range(16)))
        joint_scene = joint_scene[:, order]
        names = [names[index] for index in order]
    valid = np.ones((sequence.num_frames, 16), dtype=bool)
    if invalid_source:
        valid[1, SOURCE_NAMES.index("thumb_mcp")] = False
        joint_scene[1, names.index("thumb_mcp")] = np.nan
    hand.keypoint_tracks = {
        "mano16": KeypointTrack(
            joint_scene,
            layout_name="mano16",
            valid=valid,
            semantic_names=names,
        )
    }
    hand.vertices_scene = vertices_scene
    return sequence


def test_named_semantics_are_used_when_source_array_order_changes() -> None:
    sequence = make_mano_sequence(reordered=True)
    before = copy.deepcopy(sequence)
    converted = ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_r")
    target = converted.hands[0].keypoint_tracks["mediapipe21"]
    assert target.positions_scene.shape == (4, 21, 3)
    np.testing.assert_array_equal(
        target.positions_scene[:, 0],
        before.hands[0]
        .keypoint_tracks["mano16"]
        .positions_scene[
            :, before.hands[0].keypoint_tracks["mano16"].semantic_names.index("wrist")
        ],
    )
    assert target.semantic_names == list(get_layout("mediapipe21").semantic_names)
    assert "mediapipe21" not in sequence.hands[0].keypoint_tracks
    np.testing.assert_array_equal(sequence.hands[0].vertices_scene, before.hands[0].vertices_scene)


def test_invalid_source_point_does_not_create_a_fingertip_or_joint() -> None:
    sequence = make_mano_sequence(invalid_source=True)
    converted = ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_r")
    target = converted.hands[0].keypoint_tracks["mediapipe21"]
    thumb_mcp = target.semantic_names.index("thumb_mcp")
    assert not target.valid[1, thumb_mcp]
    assert np.isnan(target.positions_scene[1, thumb_mcp]).all()
    assert target.valid[1, target.semantic_names.index("thumb_tip")]


def test_existing_target_requires_explicit_overwrite() -> None:
    sequence = make_mano_sequence()
    sequence.hands[0].keypoint_tracks["mediapipe21"] = KeypointTrack(
        np.zeros((4, 21, 3)), "mediapipe21", valid=np.ones((4, 21), dtype=bool)
    )
    with pytest.raises(MappingError, match="already contains mediapipe21"):
        ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_r")
    converted = ManoToMediaPipe21Converter().convert_sequence(
        sequence, hand_id="hand_r", overwrite=True
    )
    assert converted.hands[0].keypoint_tracks["mediapipe21"].provenance["overwrite"] is True


def test_unknown_source_layout_is_not_shape_guessed() -> None:
    sequence = make_mano_sequence()
    track = sequence.hands[0].keypoint_tracks.pop("mano16")
    track.layout_name = "unknown_16"
    sequence.hands[0].keypoint_tracks["unknown_16"] = track
    with pytest.raises(MappingError, match="no verified source layout"):
        ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_r")


def test_vertex_count_mismatch_fails_fast() -> None:
    sequence = make_mano_sequence()
    sequence.hands[0].vertices_scene = sequence.hands[0].vertices_scene[:, :-1]
    with pytest.raises(MappingError, match="expects 778 vertices"):
        ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_r")


def test_validated_mano21_reorder_requires_named_complete_mapping() -> None:
    sequence = make_mano_sequence()
    target_names = list(get_layout("mediapipe21").semantic_names)
    positions = np.arange(sequence.num_frames * 21 * 3, dtype=np.float64).reshape(
        sequence.num_frames, 21, 3
    )
    sequence.hands[0].keypoint_tracks = {
        "mano21_named": KeypointTrack(
            positions,
            layout_name="mano21_named",
            valid=np.ones((sequence.num_frames, 21), dtype=bool),
            semantic_names=target_names,
        )
    }
    base = load_profiles()["mano_v1_2_smplx_to_mediapipe21"]
    profile = replace(
        base,
        profile_id="test_validated_mano21_reorder",
        source_joint_layout="mano21_named",
        mapping_mode="validated_mano21_reorder",
        joint_mapping={name: name for name in target_names},
        fingertip_mapping={},
    )
    converted = ManoToMediaPipe21Converter(profile).convert_sequence(sequence, hand_id="hand_r")
    np.testing.assert_array_equal(
        converted.hands[0].keypoint_tracks["mediapipe21"].positions_scene, positions
    )
