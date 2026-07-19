import numpy as np

from tests.unit.test_grab_adapter_with_fake_mano import FakeManoBackend
from tests.unit.test_grab_parser import _write_fixture
from toporetarget.data.adapters.grab import GrabDatasetAdapter, GrabLoadOptions


def test_formal_adapter_selects_one_clip_without_source_mutation(tmp_path) -> None:
    root = tmp_path / "GRAB"
    source = _write_fixture(root, include_left=True)
    before = source.read_bytes()
    adapter = GrabDatasetAdapter(
        sequence_path=source,
        grab_root=root,
        backend=FakeManoBackend(),
        options=GrabLoadOptions(
            hands="both", include_table=False, contact_mode="none", include_mediapipe21=False
        ),
    )
    sequence = adapter.load_sequence(
        options=GrabLoadOptions(
            hands="both",
            start_frame=1,
            end_frame=4,
            include_table=False,
            contact_mode="none",
            include_mediapipe21=False,
        )
    )
    assert sequence.metadata.sequence_id == "s1/demo"
    assert sequence.num_frames == 3
    assert sequence.metadata.native_fps == 120
    assert [hand.side for hand in sequence.hands] == ["right", "left"]
    assert sequence.hands[0].vertices_scene is not sequence.hands[1].vertices_scene
    np.testing.assert_allclose(sequence.timestamps, np.array([1, 2, 3]) / 120)
    assert source.read_bytes() == before
