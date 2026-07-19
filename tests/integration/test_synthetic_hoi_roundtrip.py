import numpy as np
import pytest

from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence
from toporetarget.data.synthetic import SyntheticAdapter
from toporetarget.viz.comparison import ComparisonMetrics

pytest.importorskip("zarr")


def test_synthetic_adapter_clip_round_trip(tmp_path) -> None:
    adapter = SyntheticAdapter()
    clip = adapter.load_sequence(
        "demo",
        frame_range=__import__(
            "toporetarget.data.adapters.base", fromlist=["FrameRange"]
        ).FrameRange(1, 5),
    )
    destination = tmp_path / "synthetic.zarr"
    save_hoi_sequence(clip, destination)
    loaded = load_hoi_sequence(destination)
    result = ComparisonMetrics.compute(clip, loaded).as_dict()
    assert loaded.num_frames == 4
    np.testing.assert_array_equal(loaded.timestamps, clip.timestamps)
    assert result["frame_count_match"] is True
    assert result["metrics"]["timestamp_max_abs_error_s"]["max"] == 0.0
