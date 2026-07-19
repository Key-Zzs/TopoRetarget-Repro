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
    semantic = build_grab_contacts(
        auxiliary,
        hand_ids=["right_hand"],
        object_id="cube",
        object_vertex_count=3,
        frame_count=2,
        mode="semantic",
    )[0]
    np.testing.assert_array_equal(semantic.semantic_ids, labels)
    assert semantic.semantic_mapping[21]["category"] == "left_hand"
    assert semantic.semantic_mapping[55]["category"] == "right_hand"
    assert semantic.metadata["fully_mapped"] is True


def test_grab_semantic_contact_strict_and_non_strict_unknown_labels() -> None:
    auxiliary = {"contact": {"object": np.array([[0, 56]], dtype=np.int64)}}
    with pytest.raises(ContactLoadError, match="unmapped labels: 56"):
        build_grab_contacts(
            auxiliary,
            hand_ids=["right_hand"],
            object_id="cube",
            object_vertex_count=2,
            frame_count=1,
            mode="semantic",
        )
    non_strict = build_grab_contacts(
        auxiliary,
        hand_ids=["right_hand"],
        object_id="cube",
        object_vertex_count=2,
        frame_count=1,
        mode="semantic",
        strict=False,
    )[0]
    assert non_strict.semantic_ids[0, 1] == 56
    assert non_strict.metadata["unmapped_labels"] == [56]
    assert non_strict.metadata["fully_mapped"] is False
