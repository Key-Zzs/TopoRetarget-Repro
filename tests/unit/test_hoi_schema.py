from dataclasses import replace

import numpy as np
import pytest

from toporetarget.data.schema import (
    ArticulatedObjectTrack,
    ArticulatedPartTrack,
    ContactTrack,
    HandTrack,
    HOIValidationError,
)
from toporetarget.data.synthetic import make_synthetic_sequence


def test_single_right_hand_schema_and_optional_contact_absence() -> None:
    sequence = make_synthetic_sequence()
    assert sequence.hands[0].side == "right"
    assert sequence.contacts == []
    assert sequence.validate() == []
    assert not any("robot" in name.lower() for name in sequence.__dataclass_fields__)


def test_left_hand_bimanual_and_side_is_not_mirrored() -> None:
    sequence = make_synthetic_sequence()
    right = sequence.hands[0]
    left = HandTrack(
        hand_id="hand_l",
        side="left",
        wrist_pose_scene=right.wrist_pose_scene,
        valid=right.valid.copy(),
        keypoint_tracks=right.keypoint_tracks,
        mesh=right.mesh,
        vertices_scene=right.vertices_scene.copy(),
    )
    sequence.hands.append(left)
    sequence.validate()
    assert sequence.hand("hand_l").side == "left"
    np.testing.assert_array_equal(sequence.hand("hand_l").vertices_scene, right.vertices_scene)


def test_multiple_rigid_objects_articulated_object_and_contacts() -> None:
    sequence = make_synthetic_sequence(num_frames=3)
    first = sequence.rigid_objects[0]
    second = replace(first, object_id="object_1")
    sequence.rigid_objects.append(second)
    part = ArticulatedPartTrack("part_0", first.mesh, first.pose_scene, parent_id=None)
    sequence.articulated_objects.append(
        ArticulatedObjectTrack(
            "articulated_0", parts=[part], parent_child_structure={"part_0": None}
        )
    )
    sequence.contacts.append(
        ContactTrack("hand_r", "object_0", "source", np.ones(sequence.num_frames, dtype=bool))
    )
    sequence.validate()
    assert len(sequence.rigid_objects) == 2
    assert len(sequence.articulated_objects[0].parts) == 1


def test_irregular_timestamps_are_retained_and_invalid_timestamps_fail() -> None:
    sequence = make_synthetic_sequence(irregular_timestamps=True)
    assert not np.allclose(np.diff(sequence.timestamps), 1.0 / sequence.metadata.native_fps)
    broken = replace(sequence.metadata, timestamps=np.array([0.0, 0.0] + [0.1] * 6))
    sequence.metadata = broken
    with pytest.raises(HOIValidationError, match="strictly increasing"):
        sequence.validate()


def test_native_fps_does_not_change_frame_count() -> None:
    sequence = make_synthetic_sequence(num_frames=5, native_fps=20.0)
    assert sequence.num_frames == 5
    assert sequence.metadata.native_fps == 20.0
