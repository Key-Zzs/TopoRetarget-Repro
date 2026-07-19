"""Opt-in real GRAB Stage 3 validation; public CI skips without local assets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.adapters.grab_inspect import GrabInspectionAdapter
from toporetarget.keypoints import ManoToMediaPipe21Converter, validate_mapping
from toporetarget.keypoints.registry import load_profiles

pytestmark = pytest.mark.licensed_data

sequence_path = os.environ.get("GRAB_SEQUENCE")
mano_root = os.environ.get("MANO_MODEL_ROOT")


@pytest.mark.skipif(
    not sequence_path or not mano_root,
    reason="GRAB_SEQUENCE and MANO_MODEL_ROOT are not configured",
)
@pytest.mark.parametrize(("hand", "frame_end"), (("right", 60), ("left", 10)))
def test_real_grab_mano_to_mediapipe21(hand: str, frame_end: int) -> None:
    source = GrabInspectionAdapter(
        sequence_path=Path(sequence_path or ""),
        mano_model_root=Path(mano_root or ""),
        hand=hand,
    ).load_sequence(frame_range=FrameRange(0, frame_end))
    converted = ManoToMediaPipe21Converter().convert_sequence(
        source,
        hand_id=f"hand_{'r' if hand == 'right' else 'l'}",
        mano_model_root=Path(mano_root or ""),
    )
    report = validate_mapping(
        source,
        converted,
        hand_id=f"hand_{'r' if hand == 'right' else 'l'}",
        profile=load_profiles()["mano_v1_2_smplx_to_mediapipe21"],
    )
    assert converted.num_frames == frame_end
    assert converted.hand(f"hand_{'r' if hand == 'right' else 'l'}").keypoint_tracks[
        "mediapipe21"
    ].positions_scene.shape == (frame_end, 21, 3)
    assert report.metrics["non_tip_joint_copy_rmse_m"] == 0.0
    assert report.metrics["fingertip_anchor_rmse_m"] == 0.0
    assert report.metrics["timestamp_max_abs_error_s"] == 0.0
    assert report.metrics["frame_count_match"] is True
    assert report.metrics["native_fps_match"] is True
    assert report.metrics["object_pose_unchanged"] is True
    assert report.metrics["source_tracks_preserved"] is True
