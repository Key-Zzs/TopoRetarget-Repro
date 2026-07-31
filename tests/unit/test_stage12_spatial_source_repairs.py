"""Regression coverage for the Stage 12 spatial-source repair."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from toporetarget.adapters.datasets.contactpose import static_mano_to_object_transform
from toporetarget.adapters.datasets.hocap import HOCapAdapterV1
from toporetarget.adapters.datasets.stage12_base import (
    Stage12AdapterError,
    contactpose_annotation_mano21_track,
    contactpose_official_mano21_track,
)
from toporetarget.data.schema import (
    HOISequence,
    MeshDefinition,
    PoseTrack,
    RigidObjectTrack,
    SequenceMetadata,
)
from toporetarget.keypoints.registry import get_layout


def _object(object_id: str, *, role: str | None = None) -> RigidObjectTrack:
    return RigidObjectTrack(
        object_id=object_id,
        mesh=MeshDefinition(
            vertices_local=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        ),
        pose_scene=PoseTrack(np.eye(4, dtype=np.float64)[None]),
        metadata={} if role is None else {"role": role},
    )


def test_multi_object_primary_requires_explicit_semantic_role() -> None:
    sequence = HOISequence(
        metadata=SequenceMetadata(timestamps=np.asarray([0.0]), num_frames=1),
        rigid_objects=[_object("context"), _object("target")],
    )
    with pytest.raises(KeyError, match="requires exactly one"):
        sequence.primary_rigid_object()

    sequence.rigid_objects[1].metadata["role"] = "primary_manipulation_object"
    assert sequence.primary_rigid_object().object_id == "target"


def test_hocap_primary_object_must_be_a_declared_part() -> None:
    ids = ["G10_1", "G10_2", "G10_3", "G10_4"]
    assert HOCapAdapterV1._primary_object_id(ids, "G10_2") == "G10_2"
    with pytest.raises(Stage12AdapterError, match="absent from this sequence"):
        HOCapAdapterV1._primary_object_id(ids, "G10_5")


def test_contactpose_static_transform_accepts_no_observation_h_to() -> None:
    transform = static_mano_to_object_transform(
        {
            "mTc": {
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.10, -0.20, 0.30],
            }
        }
    )
    assert np.allclose(transform[:3, :3], np.eye(3))
    assert np.allclose(transform[:3, 3], [-0.10, 0.20, -0.30])
    assert "hTo" not in inspect.signature(static_mano_to_object_transform).parameters


def test_contactpose_official_mano21_keeps_middle_ring_and_pinky_semantics() -> None:
    mano_names = list(get_layout("mano16_smplx").semantic_names)
    target_names = list(get_layout("mediapipe21").semantic_names)
    joints = np.empty((1, 16, 3), dtype=np.float64)
    for index in range(16):
        joints[0, index] = [index, 100 + index, 200 + index]
    vertices = np.empty((1, 778, 3), dtype=np.float64)
    for index in range(778):
        vertices[0, index] = [index, 1000 + index, 2000 + index]

    track = contactpose_official_mano21_track(
        joints, vertices, valid=np.asarray([True], dtype=bool)
    )
    output = track.positions_scene[0]
    source_index = {name: index for index, name in enumerate(mano_names)}
    target_index = {name: index for index, name in enumerate(target_names)}
    for name in (
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
    ):
        assert np.array_equal(output[target_index[name]], joints[0, source_index[name]])
    for name, vertex_index in {"middle_tip": 444, "ring_tip": 555, "pinky_tip": 672}.items():
        assert np.array_equal(output[target_index[name]], vertices[0, vertex_index])


def test_contactpose_annotation_openpose21_is_the_identity_semantic_source() -> None:
    positions = np.arange(63, dtype=np.float64).reshape(1, 21, 3)
    track = contactpose_annotation_mano21_track(
        positions, valid=np.asarray([True]), source_path="annotations.json"
    )
    assert track.layout_name == "mano21_named"
    assert track.semantic_names == list(get_layout("mediapipe21").semantic_names)
    assert np.array_equal(track.positions_scene, positions)
    assert track.provenance["mapping_mode"] == (
        "contactpose_annotation_openpose21_identity_semantics"
    )
    assert track.provenance["fitted_mano_used_for_keypoints"] is False
