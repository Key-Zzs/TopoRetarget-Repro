"""Opt-in, sequence-scoped smoke checks for the Stage 12 NAS adapters."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

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
    assert canonical.hands[0].keypoint_tracks["mediapipe21"].positions_scene.shape == (3, 21, 3)
    assert canonical.metadata.provenance.source_sequence
    assert canonical.metadata.provenance.source_hash
    assert adapter.validate(canonical)["status"] == "ok"
    assert adapter.visualize(canonical)["status"] == "ready"


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
    assert all(int(row["frame_range"][1]) - int(row["frame_range"][0]) == 60 for row in selections)
