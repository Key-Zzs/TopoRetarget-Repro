from dataclasses import replace

import numpy as np
import pytest

from toporetarget.data.schema import PoseTrack
from toporetarget.data.synthetic import make_synthetic_sequence
from toporetarget.viz.comparison import ComparisonMetrics


def test_known_one_mm_vertex_perturbation_and_one_degree_rotation() -> None:
    raw = make_synthetic_sequence(num_frames=2)
    canonical = make_synthetic_sequence(num_frames=2)
    canonical.hands[0].vertices_scene = canonical.hands[0].vertices_scene.copy()
    canonical.hands[0].vertices_scene[:, :, 0] += 0.001
    pose = canonical.hands[0].wrist_pose_scene.pose_scene.copy()
    rotation = np.eye(4)
    angle = np.deg2rad(1.0)
    rotation[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[0] = pose[0] @ rotation
    canonical.hands[0].wrist_pose_scene = PoseTrack(pose, child_frame_name="H")
    result = ComparisonMetrics.compute(raw, canonical).as_dict()
    assert result["metrics"]["hand_vertex_rmse_m"]["mean"] == pytest.approx(0.001)
    assert result["per_frame"]["wrist_rotation_geodesic_deg"][0] == pytest.approx(1.0, abs=1e-6)


def test_identical_sequences_are_zero_and_unavailable_is_explicit() -> None:
    sequence = make_synthetic_sequence()
    result = ComparisonMetrics.compute(sequence, sequence).as_dict()
    for name in ("hand_vertex_rmse_m", "wrist_translation_error_m", "object_world_vertex_rmse_m"):
        assert result["metrics"][name]["available"]
        assert result["metrics"][name]["max"] == pytest.approx(0.0)
    without_objects = replace(sequence, rigid_objects=[])
    missing = ComparisonMetrics.compute(sequence, without_objects).as_dict()
    assert missing["metrics"]["object_pose_translation_error_m"]["available"] is False
