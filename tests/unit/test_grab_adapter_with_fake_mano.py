import numpy as np

from tests.unit.test_grab_parser import _write_fixture
from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.adapters.grab_inspect import GrabInspectionAdapter
from toporetarget.data.mano_backends.base import ManoRenderResult, axis_angle_to_matrix
from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence
from toporetarget.viz.comparison import ComparisonMetrics
from toporetarget.viz.matplotlib_viewer import ViewerOptions, render_comparison


class FakeManoBackend:
    def render(self, *, params, v_template, side, frame_count):
        vertices = np.broadcast_to(v_template, (frame_count, *v_template.shape)).copy()
        vertices += np.asarray(params["transl"])[:, None, :]
        rotation = axis_angle_to_matrix(params["global_orient"])
        vertices = np.einsum("bij,bvj->bvi", rotation, vertices)
        pose = np.repeat(np.eye(4, dtype=np.float64)[None, ...], frame_count, axis=0)
        pose[:, :3, :3] = rotation
        pose[:, :3, 3] = params["transl"]
        joints = vertices.copy()
        return ManoRenderResult(
            vertices_scene=vertices,
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            wrist_pose_scene=pose,
            joints_scene=joints,
            keypoint_layout="mano_native",
            model_profile="fake",
        )


def test_fake_backend_enters_canonical_schema_and_zarr_round_trip(tmp_path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    adapter = GrabInspectionAdapter(
        sequence_path=source,
        hand="right",
        backend=FakeManoBackend(),
    )
    sequence = adapter.load_sequence(frame_range=FrameRange(1, 4))
    assert sequence.metadata.native_fps == 120.0
    np.testing.assert_array_equal(sequence.timestamps, np.array([1, 2, 3]) / 120.0)
    assert sequence.hands[0].metadata["selected_hand"] == "right"
    destination = tmp_path / "canonical.zarr"
    save_hoi_sequence(sequence, destination)
    loaded = load_hoi_sequence(destination)
    result = ComparisonMetrics.compute(sequence, loaded).as_dict()
    assert result["frame_count_match"] is True
    assert result["metrics"]["hand_vertex_rmse_m"]["max"] == 0.0


def test_fake_backend_can_render_both_layouts(tmp_path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    adapter = GrabInspectionAdapter(sequence_path=source, backend=FakeManoBackend())
    sequence = adapter.load_sequence(frame_range=FrameRange(0, 4))
    result = ComparisonMetrics.compute(sequence, sequence)
    side = tmp_path / "side.png"
    overlay = tmp_path / "overlay.png"
    render_comparison(
        sequence,
        sequence,
        result,
        options=ViewerOptions(layout="side-by-side"),
        output=side,
    )
    render_comparison(
        sequence,
        sequence,
        result,
        options=ViewerOptions(layout="overlay"),
        output=overlay,
    )
    assert side.is_file() and overlay.is_file()


def test_source_path_is_not_modified(tmp_path) -> None:
    source = _write_fixture(tmp_path / "GRAB")
    before = source.read_bytes()
    GrabInspectionAdapter(sequence_path=source, backend=FakeManoBackend()).load_sequence()
    assert source.read_bytes() == before
