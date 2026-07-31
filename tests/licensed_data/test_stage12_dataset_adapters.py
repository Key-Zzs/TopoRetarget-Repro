"""Opt-in, sequence-scoped smoke checks for the Stage 12 NAS adapters."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from toporetarget.adapters.datasets import get_dataset_adapter_registry
from toporetarget.data.adapters.base import FrameRange

pytestmark = pytest.mark.licensed_data

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_CONFIG = REPO_ROOT / "configs" / "benchmarks" / "stage12_selection.yaml"


def _selections() -> list[dict[str, Any]]:
    values = yaml.safe_load(SELECTION_CONFIG.read_text(encoding="utf-8")) or {}
    return list(values["selections"])


@pytest.mark.parametrize("selection", _selections())
def test_stage12_adapter_loads_canonical_provenance_and_viewer_handle(
    selection: dict[str, Any],
) -> None:
    if os.environ.get("STAGE12_RUN_NAS_TESTS") != "1":
        pytest.skip("set STAGE12_RUN_NAS_TESTS=1 for sequence-scoped NAS adapter checks")
    dataset = str(selection["dataset"])
    sequence = str(selection["sequence"])
    frame_start, frame_stop = (int(value) for value in selection["frame_range"])
    adapter = get_dataset_adapter_registry().create(dataset)
    description = adapter.describe(sequence)
    assert description["lazy"] is True
    source = adapter.load_sequence(
        sequence,
        frame_range=FrameRange(frame_start, min(frame_start + 3, frame_stop)),
    )
    canonical = adapter.convert_to_canonical(source)
    assert canonical.metadata.schema_version == "toporetarget.hoi.v2"
    expected_frames = 1 if dataset == "contactpose" else min(3, frame_stop - frame_start)
    assert canonical.hands[0].keypoint_tracks["mediapipe21"].positions_scene.shape == (
        expected_frames,
        21,
        3,
    )
    assert canonical.metadata.provenance.source_sequence
    assert canonical.metadata.provenance.source_hash
    assert adapter.validate(canonical)["status"] == "ok"
    assert adapter.visualize(canonical)["status"] == "ready"
    hand = canonical.hands[0]
    if dataset == "dexycb":
        assert hand.keypoint_tracks["mano21_named"].provenance["source"] == "dataset_native"
        assert hand.metadata["mano_representation"] == "pca"
        assert hand.metadata["num_pca_components"] == 45
        native = hand.keypoint_tracks["mano21_named"]
        name_to_index = {name: index for index, name in enumerate(native.semantic_names or [])}
        tip_vertices = {
            "thumb_tip": 744,
            "index_tip": 320,
            "middle_tip": 443,
            "ring_tip": 554,
            "pinky_tip": 671,
        }
        assert hand.vertices_scene is not None
        for name, vertex_index in tip_vertices.items():
            native_tip = native.positions_scene[:, name_to_index[name]]
            error = np.linalg.norm(
                hand.vertices_scene[:, vertex_index] - native_tip,
                axis=-1,
            )
            assert float(np.max(error)) <= 1e-2
    elif dataset == "hocap":
        assert hand.keypoint_tracks["mano16_smplx"].provenance["source"] == "backend_posed"
        assert hand.metadata["mano_representation"] == "pca"
        assert hand.metadata["num_pca_components"] == 45
        reconstruction = hand.metadata["mano_reconstruction"]
        assert reconstruction["execution"] == ("pca45_explicit_basis_expansion_single_mean")
        assert reconstruction["execution_flat_hand_mean"] is True
    elif dataset == "contactpose":
        assert canonical.num_frames == 1
        assert canonical.metadata.metadata["temporal_metrics"] == "NOT_APPLICABLE"
        assert hand.keypoint_tracks["mano21_named"].provenance["source"] == "contactpose_official"
    elif dataset == "oakink":
        assert hand.keypoint_tracks["mano21_named"].provenance["source"] == "dataset_native"
        assert hand.wrist_pose_scene.orientation_available is False


def test_stage12_registry_has_four_new_adapters() -> None:
    names = get_dataset_adapter_registry().names()
    assert {"dexycb", "oakink", "hocap", "contactpose"}.issubset(names)


def test_stage12_freezes_two_right_hand_trajectories_per_dataset() -> None:
    selections = _selections()
    assert len(selections) == 8
    assert Counter(str(row["dataset"]) for row in selections) == {
        "dexycb": 2,
        "oakink": 2,
        "hocap": 2,
        "contactpose": 2,
    }
    assert all(row["hand"] == "right" for row in selections)
    for row in selections:
        frame_count = int(row["frame_range"][1]) - int(row["frame_range"][0])
        if row["dataset"] == "contactpose":
            assert frame_count == 1
            assert row["sample_type"] == "static_contact_evaluation_only"
            assert row["articulated_frame_count"] == 1
            assert row["temporal_metrics_applicable"] is False
        else:
            assert frame_count == 60
