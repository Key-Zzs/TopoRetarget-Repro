from __future__ import annotations

import inspect

import numpy as np

from toporetarget.rl.physics_retargeting.contact_topology import (
    body_contact_group,
    collapse_contact_records,
    extract_persistent_contact_topology,
)
from toporetarget.rl.physics_retargeting.task_semantics import extract_task_semantics


def _reference() -> dict[str, np.ndarray]:
    frames = 41
    object_position = np.zeros((frames, 3), dtype=np.float64)
    object_position[:, 0] = np.linspace(0.0, 0.20, frames)
    quaternion = np.zeros((frames, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    twist = np.zeros((frames, 6), dtype=np.float64)
    wrist_object = np.repeat(np.eye(4)[None], frames, axis=0)
    return {
        "object_pose_translation_world_ref": object_position,
        "object_pose_quaternion_world_ref_wxyz": quaternion,
        "object_twist_world_ref": twist,
        "T_wrist_object_ref": wrist_object,
    }


def _contact(step: int, body: str, force: float = 1.0) -> dict[str, object]:
    return {
        "control_step": step,
        "net_contact_force_world_on_object_n": [force, 0.0, 0.0],
        "present_hand_body_names": [body],
    }


def test_sparse_trace_uses_audited_ambiguous_fallback() -> None:
    contract = extract_task_semantics(
        clip="synthetic",
        reference=_reference(),
        contact_records=[_contact(100, "r_index_finger_distal")],
    )
    assert contract.task_class == "generic_contact_preserving_motion"
    assert contract.classification_status == "TASK_SEMANTIC_CLASSIFICATION_AMBIGUOUS"
    assert contract.source_motion_class == "transport"


def test_contact_groups_and_force_filter_are_shared() -> None:
    rows = [
        _contact(10, "r_thumb_distal"),
        _contact(11, "r_thumb_distal"),
        _contact(20, "r_index_finger_distal", force=1.0e-8),
    ]
    collapsed = collapse_contact_records(rows)
    assert sorted(collapsed) == [10, 11]
    topology = extract_persistent_contact_topology(
        clip="synthetic", contact_records=rows, retimed_frame_count=321
    )
    assert topology.required_body_groups == ("thumb",)
    assert topology.minimum_persistence_control_steps == 1
    assert body_contact_group("r_wrist") == "palm"


def test_semantic_extractor_has_no_clip_identity_conditionals() -> None:
    source = inspect.getsource(extract_task_semantics)
    assert "170105" not in source
    assert "170650" not in source
    assert "if clip" not in source
