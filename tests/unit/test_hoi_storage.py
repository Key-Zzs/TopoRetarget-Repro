import numpy as np
import pytest

from toporetarget.data.contacts.grab import build_grab_contacts
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


def test_zarr_preserves_grab_semantic_contact_round_trip(tmp_path) -> None:
    sequence = make_synthetic_sequence(num_frames=2)
    vertex_count = sequence.rigid_objects[0].mesh.vertices_local.shape[0]
    labels = np.zeros((2, vertex_count), dtype=np.int16)
    labels[:, 0] = 21
    labels[:, 1] = 41
    sequence.contacts = build_grab_contacts(
        {"contact": {"object": labels}},
        hand_ids=[sequence.hands[0].hand_id],
        object_id=sequence.rigid_objects[0].object_id,
        object_vertex_count=vertex_count,
        frame_count=sequence.num_frames,
        mode="semantic",
    )
    destination = tmp_path / "semantic.zarr"
    save_hoi_sequence(sequence, destination)
    loaded = load_hoi_sequence(destination)
    np.testing.assert_array_equal(loaded.contacts[0].labels, labels)
    np.testing.assert_array_equal(loaded.contacts[0].binary, labels != 0)
    np.testing.assert_array_equal(loaded.contacts[0].semantic_ids, labels)
    assert loaded.contacts[0].semantic_mapping[21]["category"] == "left_hand"
