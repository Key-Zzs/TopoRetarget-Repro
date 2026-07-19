from pathlib import Path

import pytest

from tests.unit.test_grab_adapter_with_fake_mano import FakeManoBackend
from tests.unit.test_grab_parser import _write_fixture
from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.adapters.grab import GrabAdapterError, GrabDatasetAdapter, GrabLoadOptions
from toporetarget.data.storage import load_hoi_sequence


def test_formal_adapter_supports_auto_and_both_without_second_parser(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    before = source.read_bytes()
    adapter = GrabDatasetAdapter(
        sequence_path=source,
        backend=FakeManoBackend(),
        options=GrabLoadOptions(
            hands="auto", include_table=False, contact_mode="none", include_mediapipe21=False
        ),
    )
    sequence = adapter.load_sequence(frame_range=FrameRange(1, 4))
    assert [hand.side for hand in sequence.hands] == ["right", "left"]
    assert sequence.timestamps.tolist() == [1 / 120, 2 / 120, 3 / 120]
    assert sequence.metadata.metadata["no_temporal_resampling"] is True
    assert source.read_bytes() == before


def test_formal_adapter_missing_explicit_hand_fails(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "GRAB", include_left=False)
    adapter = GrabDatasetAdapter(
        sequence_path=source,
        backend=FakeManoBackend(),
        options=GrabLoadOptions(
            hands="both", include_table=False, contact_mode="none", include_mediapipe21=False
        ),
    )
    with pytest.raises(GrabAdapterError, match="missing"):
        adapter.load_sequence()


def test_formal_adapter_cache_is_explicit_and_does_not_overwrite(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    before = source.read_bytes()
    adapter = GrabDatasetAdapter(sequence_path=source, backend=FakeManoBackend())
    output = tmp_path / "cache" / "clip.zarr"
    options = GrabLoadOptions(
        hands="right",
        start_frame=0,
        end_frame=3,
        include_table=False,
        contact_mode="none",
        include_mediapipe21=False,
    )
    adapter.create_cache(output=output, options=options)
    assert load_hoi_sequence(output).num_frames == 3
    assert source.read_bytes() == before
    with pytest.raises(GrabAdapterError, match="already exists"):
        adapter.create_cache(output=output, options=options)
