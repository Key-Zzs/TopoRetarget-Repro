import numpy as np
import pytest

from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence
from toporetarget.data.synthetic import make_synthetic_sequence

zarr = pytest.importorskip("zarr")


def test_zarr_lossless_semantic_round_trip(tmp_path) -> None:
    sequence = make_synthetic_sequence(irregular_timestamps=True)
    destination = tmp_path / "sequence.zarr"
    save_hoi_sequence(sequence, destination)
    loaded = load_hoi_sequence(destination)
    np.testing.assert_array_equal(loaded.timestamps, sequence.timestamps)
    np.testing.assert_array_equal(loaded.hands[0].vertices_scene, sequence.hands[0].vertices_scene)
    np.testing.assert_array_equal(
        loaded.rigid_objects[0].mesh.faces, sequence.rigid_objects[0].mesh.faces
    )
    assert loaded.metadata.provenance.no_temporal_resampling
    assert (destination / "metadata.json").is_file()
