from __future__ import annotations

import copy

import numpy as np
import pytest

from toporetarget.adapters.datasets.hocap import hocap_mano_storage_index
from toporetarget.adapters.datasets.hocap_episode import (
    EpisodeType,
    HandObjectSignals,
    classify_sequence_interactions,
    segment_hand_object_signals,
)


def _signals(
    *,
    side: str = "right",
    object_id: str = "G01_1",
    offset: int = 0,
    frames: int = 100,
) -> HandObjectSignals:
    distance = np.full(frames, 0.12, dtype=np.float64)
    contact_start = 20 + offset
    contact_end = 56 + offset
    distance[12 + offset : contact_start] = np.linspace(
        0.09, 0.02, contact_start - (12 + offset), endpoint=True
    )
    distance[contact_start:contact_end] = 0.003
    distance[contact_end : 60 + offset] = np.linspace(
        0.02, 0.08, 60 + offset - contact_end, endpoint=True
    )
    near = np.where(distance <= 0.01, 8, 0)
    semantic = np.zeros((frames, 6), dtype=bool)
    semantic[contact_start:contact_end, 1] = True
    tips = np.broadcast_to(distance[:, None], (frames, 5)).copy()
    translation = np.zeros((frames, 3), dtype=np.float64)
    translation[25 + offset : 30 + offset, 2] = np.linspace(0.0, 0.03, 5)
    translation[30 + offset : 44 + offset, 2] = 0.03
    translation[44 + offset : 49 + offset, 2] = np.linspace(0.03, 0.0, 5)
    rotation = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
    linear = np.zeros(frames, dtype=np.float64)
    linear[22 + offset : 49 + offset] = 0.05
    angular = np.zeros(frames, dtype=np.float64)
    bottom = translation[:, 2].copy()
    wrist = np.zeros((frames, 3), dtype=np.float64)
    wrist[:, 0] = np.linspace(0.0, 0.02, frames)
    return HandObjectSignals(
        subject="subject_test",
        raw_sequence="subject_test/20260101_000000",
        side=side,
        object_id=object_id,
        fps=30.0,
        min_surface_distance_m=distance,
        near_surface_vertex_count=near,
        semantic_contact_region_mask=semantic,
        fingertip_distance_m=tips,
        object_translation_world_m=translation,
        object_rotation_world=rotation,
        object_linear_speed_mps=linear,
        object_angular_speed_radps=angular,
        object_bottom_height_m=bottom,
        relative_translation_rate_mps=linear,
        relative_angular_rate_radps=angular,
        wrist_translation_world_m=wrist,
        source_support_metadata={"source_explicit_support_present": False},
        provenance={"fixture": "synthetic"},
    )


def test_hocap_mano_storage_slots_are_side_fixed() -> None:
    assert hocap_mano_storage_index("right") == 0
    assert hocap_mano_storage_index("left") == 1


@pytest.mark.parametrize("side", ["left", "right"])
def test_complete_single_hand_pick_place_lifecycle(side: str) -> None:
    rows = segment_hand_object_signals(_signals(side=side))

    assert len(rows) == 1
    row = rows[0]
    assert row["active_hand"] == side
    assert row["episode_type"] == EpisodeType.SINGLE_HAND_PICK_PLACE.value
    assert row["physicalization_v1_eligible"] is True
    assert row["start_frame"] == 8
    assert row["approach_frame"] == 16
    assert row["contact_frame"] == 20
    assert row["pickup_frame"] == 27
    assert row["place_frame"] == 49
    assert row["release_frame"] == 56
    assert row["retreat_frame"] == 58
    assert row["end_frame"] == 66


def test_boundaries_follow_events_not_fixed_padding() -> None:
    first = segment_hand_object_signals(_signals(offset=0))[0]
    second = segment_hand_object_signals(_signals(offset=7))[0]

    for field in (
        "approach_frame",
        "contact_frame",
        "pickup_frame",
        "place_frame",
        "release_frame",
        "retreat_frame",
        "end_frame",
    ):
        assert int(second[field]) - int(first[field]) == 7
    assert second["duration_frames"] == first["duration_frames"]


def test_different_object_left_right_overlap_remains_two_episodes() -> None:
    left = segment_hand_object_signals(_signals(side="left", object_id="G01_1"))[0]
    right = segment_hand_object_signals(_signals(side="right", object_id="G01_2", offset=5))[0]

    rows = classify_sequence_interactions([left, right])

    assert len(rows) == 2
    assert all(row["physicalization_v1_eligible"] for row in rows)
    assert all(row["overlapping_other_hand_other_object"] for row in rows)


def test_same_object_bimanual_is_not_split_into_single_hand_episodes() -> None:
    left = segment_hand_object_signals(_signals(side="left"))[0]
    right = segment_hand_object_signals(_signals(side="right", offset=2))[0]

    rows = classify_sequence_interactions([left, right])

    assert len(rows) == 1
    assert rows[0]["episode_type"] == EpisodeType.BIMANUAL_SAME_OBJECT.value
    assert rows[0]["active_hand"] == "both"
    assert rows[0]["physicalization_v1_eligible"] is False
    assert rows[0]["other_hand_same_target"] is True


def test_handover_is_not_split_into_single_hand_episodes() -> None:
    left = segment_hand_object_signals(_signals(side="left"))[0]
    right = copy.deepcopy(segment_hand_object_signals(_signals(side="right"))[0])
    left["release_frame"] = 40
    left["end_frame"] = 43
    right["contact_frame"] = 46
    right["start_frame"] = 43

    rows = classify_sequence_interactions([left, right])

    assert len(rows) == 1
    assert rows[0]["episode_type"] == EpisodeType.HANDOVER.value
    assert rows[0]["physicalization_v1_eligible"] is False


def test_segmentation_is_deterministic() -> None:
    signals = _signals()
    assert segment_hand_object_signals(signals) == segment_hand_object_signals(signals)
