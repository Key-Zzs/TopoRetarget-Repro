import numpy as np
import pytest

from toporetarget.data.contacts.grab import ContactLoadError, build_grab_contacts


def test_grab_contact_modes_preserve_numeric_labels() -> None:
    labels = np.array([[0, 21, 0], [55, 0, 0]], dtype=np.int8)
    auxiliary = {"contact": {"object": labels, "body": np.zeros((2, 4), dtype=np.int8)}}
    source = build_grab_contacts(
        auxiliary,
        hand_ids=["right_hand", "left_hand"],
        object_id="cube",
        object_vertex_count=3,
        frame_count=2,
        mode="source",
    )
    assert len(source) == 2
    np.testing.assert_array_equal(source[0].labels, labels)
    np.testing.assert_array_equal(source[0].binary, labels != 0)
    assert not np.shares_memory(source[0].labels, source[1].labels)
    binary = build_grab_contacts(
        auxiliary,
        hand_ids=["right_hand"],
        object_id="cube",
        object_vertex_count=3,
        frame_count=2,
        mode="binary",
    )[0]
    np.testing.assert_array_equal(binary.labels, labels)
    with pytest.raises(ContactLoadError, match="semantic"):
        build_grab_contacts(
            auxiliary,
            hand_ids=["right_hand"],
            object_id="cube",
            object_vertex_count=3,
            frame_count=2,
            mode="semantic",
        )
