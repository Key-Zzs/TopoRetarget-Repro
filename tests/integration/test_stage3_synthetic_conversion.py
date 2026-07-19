from __future__ import annotations

from tests.unit.test_mano_mediapipe_mapping import make_mano_sequence
from toporetarget.keypoints import ManoToMediaPipe21Converter, validate_mapping
from toporetarget.keypoints.registry import load_profiles


def test_stage3_synthetic_conversion_and_consistency_report() -> None:
    source = make_mano_sequence(reordered=True)
    converted = ManoToMediaPipe21Converter().convert_sequence(source, hand_id="hand_r")
    report = validate_mapping(
        source,
        converted,
        hand_id="hand_r",
        profile=load_profiles()["mano_v1_2_smplx_to_mediapipe21"],
    )
    assert report.metrics["non_tip_joint_copy_rmse_m"] == 0.0
    assert report.metrics["fingertip_anchor_rmse_m"] == 0.0
    assert report.metrics["scene_wrist_roundtrip_max_m"] < 1e-12
    assert report.metrics["object_pose_unchanged"] is True
    assert report.metrics["source_tracks_preserved"] is True
