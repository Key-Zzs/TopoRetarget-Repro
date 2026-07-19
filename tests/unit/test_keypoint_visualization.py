from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tests.unit.test_mano_mediapipe_mapping import make_mano_sequence
from toporetarget.keypoints import ManoToMediaPipe21Converter
from toporetarget.keypoints.visualization import launch_interactive_keypoint_viewer


def test_interactive_viewer_is_read_only_and_constructs_local_controls(monkeypatch) -> None:
    source = make_mano_sequence()
    converted = ManoToMediaPipe21Converter().convert_sequence(source, hand_id="hand_r")
    before = copy.deepcopy(converted)
    monkeypatch.setattr(plt, "show", lambda: None)
    launch_interactive_keypoint_viewer(
        converted,
        hand_id="hand_r",
        start_frame=0,
        end_frame=4,
        view="scene",
        show_source_layout=True,
        show_mesh=True,
        show_target=True,
        show_skeleton=True,
        show_labels=True,
        show_object_mesh=True,
        show_axes=True,
    )
    assert converted.hands[0].keypoint_tracks["mediapipe21"].positions_scene is not None
    assert (
        converted.hands[0].keypoint_tracks["mediapipe21"].positions_scene
        == before.hands[0].keypoint_tracks["mediapipe21"].positions_scene
    ).all()
    plt.close("all")
