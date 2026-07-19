import pytest

from toporetarget.data.synthetic import make_synthetic_sequence
from toporetarget.viz.comparison import ComparisonMetrics
from toporetarget.viz.matplotlib_viewer import ViewerOptions, render_comparison

pytest.importorskip("matplotlib")


def test_headless_side_by_side_and_overlay_png(tmp_path) -> None:
    sequence = make_synthetic_sequence(num_frames=2)
    result = ComparisonMetrics.compute(sequence, sequence)
    side = tmp_path / "side.png"
    overlay = tmp_path / "overlay.png"
    render_comparison(
        sequence, sequence, result, options=ViewerOptions(layout="side-by-side"), output=side
    )
    render_comparison(
        sequence, sequence, result, options=ViewerOptions(layout="overlay"), output=overlay
    )
    assert side.stat().st_size > 0
    assert overlay.stat().st_size > 0
