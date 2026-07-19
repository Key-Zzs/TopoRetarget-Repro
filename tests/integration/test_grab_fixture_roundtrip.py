import pytest

from tests.unit.test_grab_adapter_with_fake_mano import FakeManoBackend
from tests.unit.test_grab_parser import _write_fixture
from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.adapters.grab_inspect import GrabInspectionAdapter
from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence

pytest.importorskip("zarr")


def test_grab_like_fixture_round_trip(tmp_path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    sequence = GrabInspectionAdapter(
        sequence_path=source,
        backend=FakeManoBackend(),
    ).load_sequence(frame_range=FrameRange(0, 2))
    destination = tmp_path / "clip.zarr"
    save_hoi_sequence(sequence, destination)
    assert load_hoi_sequence(destination).num_frames == 2
