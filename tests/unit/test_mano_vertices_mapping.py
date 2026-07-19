from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

from toporetarget.data.synthetic import make_synthetic_sequence
from toporetarget.geometry.se3 import wrist_to_scene
from toporetarget.keypoints import ManoToMediaPipe21Converter, MappingError
from toporetarget.keypoints.registry import load_profiles


@pytest.mark.parametrize("sparse", [False, True])
def test_vertices_plus_dense_or_sparse_regressor(tmp_path, sparse: bool) -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")
    sequence = make_synthetic_sequence(num_frames=2)
    hand = sequence.hands[0]
    hand.keypoint_tracks = {}
    vertices_local = np.zeros((2, 778, 3), dtype=np.float64)
    regressor = np.zeros((16, 778), dtype=np.float64)
    for joint_index in range(16):
        vertex_index = 100 + joint_index
        vertices_local[:, vertex_index] = [0.004 * joint_index, 0.002, 0.001]
        regressor[joint_index, vertex_index] = 1.0
    for vertex_index, value in zip(
        (744, 320, 443, 554, 671),
        (
            [0.1, 0.0, 0.0],
            [0.11, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.13, 0.0, 0.0],
            [0.14, 0.0, 0.0],
        ),
        strict=True,
    ):
        vertices_local[:, vertex_index] = value
    hand.vertices_scene = wrist_to_scene(hand.wrist_pose_scene.pose_scene, vertices_local)
    model_root = tmp_path / "mano"
    model_root.mkdir()
    regressor_value = scipy_sparse.csc_matrix(regressor) if sparse else regressor
    with (model_root / "MANO_RIGHT.pkl").open("wb") as handle:
        pickle.dump({"v_template": np.zeros((778, 3)), "J_regressor": regressor_value}, handle)
    profile = replace(
        load_profiles()["mano_v1_2_smplx_to_mediapipe21"],
        mapping_mode="vertices_with_joint_regressor",
    )
    converted = ManoToMediaPipe21Converter(profile).convert_sequence(
        sequence, hand_id="hand_r", mano_model_root=model_root
    )
    target = converted.hands[0].keypoint_tracks["mediapipe21"]
    assert target.valid.all()
    np.testing.assert_allclose(target.positions_scene[:, 5], hand.vertices_scene[:, 101])
    np.testing.assert_allclose(target.positions_scene[:, 4], hand.vertices_scene[:, 744])


def test_vertices_regressor_requires_model_root(tmp_path) -> None:
    sequence = make_synthetic_sequence(num_frames=1)
    sequence.hands[0].keypoint_tracks = {}
    sequence.hands[0].vertices_scene = np.zeros((1, 778, 3), dtype=np.float64)
    profile = replace(
        load_profiles()["mano_v1_2_smplx_to_mediapipe21"],
        mapping_mode="vertices_with_joint_regressor",
    )
    with pytest.raises(MappingError, match="requires --mano-model-root"):
        ManoToMediaPipe21Converter(profile).convert_sequence(sequence, hand_id="hand_r")


def test_missing_model_file_is_actionable(tmp_path) -> None:
    sequence = make_synthetic_sequence(num_frames=1)
    sequence.hands[0].keypoint_tracks = {}
    sequence.hands[0].vertices_scene = np.zeros((1, 778, 3), dtype=np.float64)
    profile = replace(
        load_profiles()["mano_v1_2_smplx_to_mediapipe21"],
        mapping_mode="vertices_with_joint_regressor",
    )
    with pytest.raises(MappingError, match="MANO model file MANO_RIGHT.pkl"):
        ManoToMediaPipe21Converter(profile).convert_sequence(
            sequence, hand_id="hand_r", mano_model_root=tmp_path
        )
