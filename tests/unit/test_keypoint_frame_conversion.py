from __future__ import annotations

import copy

import numpy as np

from tests.unit.test_mano_mediapipe_mapping import make_mano_sequence
from toporetarget.keypoints import (
    ManoToMediaPipe21Converter,
    mediapipe21_scene_to_wrist,
    mediapipe21_wrist_to_scene,
)


def test_scene_wrist_scene_roundtrip_and_wrist_pose_integrity() -> None:
    sequence = make_mano_sequence()
    original = copy.deepcopy(sequence)
    converted = ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_r")
    wrist = mediapipe21_scene_to_wrist(converted, "hand_r")
    reconstructed = mediapipe21_wrist_to_scene(converted, "hand_r", wrist)
    np.testing.assert_allclose(
        reconstructed, converted.hands[0].keypoint_tracks["mediapipe21"].positions_scene
    )
    np.testing.assert_allclose(wrist[:, 0], 0.0, atol=1e-12)
    np.testing.assert_array_equal(
        converted.hands[0].wrist_pose_scene.pose_scene,
        original.hands[0].wrist_pose_scene.pose_scene,
    )
    np.testing.assert_array_equal(
        converted.rigid_objects[0].pose_scene.pose_scene,
        original.rigid_objects[0].pose_scene.pose_scene,
    )


def test_left_hand_is_not_mirrored() -> None:
    sequence = make_mano_sequence()
    right = sequence.hands[0]
    expected_right = ManoToMediaPipe21Converter().convert_sequence(
        copy.deepcopy(sequence), hand_id="hand_r"
    )
    left = copy.deepcopy(right)
    left.hand_id = "hand_l"
    left.side = "left"
    sequence.hands.append(left)
    converted = ManoToMediaPipe21Converter().convert_sequence(sequence, hand_id="hand_l")
    np.testing.assert_array_equal(
        converted.hand("hand_l").keypoint_tracks["mediapipe21"].positions_scene,
        expected_right.hand("hand_r").keypoint_tracks["mediapipe21"].positions_scene,
    )
