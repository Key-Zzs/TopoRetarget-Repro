"""Scene-frame primary data and wrist-frame derived views for MediaPipe21."""

from __future__ import annotations

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.se3 import scene_to_wrist, wrist_to_scene


def mediapipe21_scene_to_wrist(
    sequence: HOISequence,
    hand_id: str,
    *,
    layout_name: str = "mediapipe21",
) -> np.ndarray:
    track = sequence.hand(hand_id).keypoint_tracks[layout_name]
    return scene_to_wrist(sequence.hand(hand_id).wrist_pose_scene.pose_scene, track.positions_scene)


def mediapipe21_wrist_to_scene(
    sequence: HOISequence,
    hand_id: str,
    positions_wrist: np.ndarray,
) -> np.ndarray:
    return wrist_to_scene(
        sequence.hand(hand_id).wrist_pose_scene.pose_scene,
        np.asarray(positions_wrist, dtype=np.float64),
    )


__all__ = ["mediapipe21_scene_to_wrist", "mediapipe21_wrist_to_scene"]
