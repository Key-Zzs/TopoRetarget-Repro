"""Opt-in local real-data smoke test; public CI skips without explicit resources."""

import os
from pathlib import Path

import pytest

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.adapters.grab_inspect import GrabInspectionAdapter

pytestmark = pytest.mark.licensed_data

sequence_path = os.environ.get("GRAB_SEQUENCE")
mano_root = os.environ.get("MANO_MODEL_ROOT")


@pytest.mark.skipif(
    not sequence_path or not mano_root,
    reason="GRAB_SEQUENCE and MANO_MODEL_ROOT are not configured",
)
def test_one_real_grab_clip() -> None:
    adapter = GrabInspectionAdapter(
        sequence_path=Path(sequence_path or ""),
        mano_model_root=Path(mano_root or ""),
        hand="right",
    )
    sequence = adapter.load_sequence(frame_range=FrameRange(0, 60))
    assert sequence.num_frames == 60
    assert sequence.metadata.native_fps == 120.0
