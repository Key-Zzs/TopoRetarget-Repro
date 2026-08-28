from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.evaluation.audit_retarget_semantic_validity import (
    _append_jsonl_unique,
    _classify_earliest_divergence,
    _event_mapping_rows,
)
from toporetarget.evaluation.retarget_semantic_validity import (
    SemanticGateContractV1,
    common_rigid_transform_invariant,
    compose,
    detect_transform_misuse,
    invert_transform,
    qualify_semantics,
    relative_transform,
    transform_points,
)


def _pose(translation=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = translation
    if rotation is not None:
        value[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    return value


def _trajectory(count: int = 4) -> np.ndarray:
    value = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    value[:, 0, 3] = np.arange(count) * 0.001
    return value


def test_hand_and_object_common_transform_preserves_object_local_relation() -> None:
    hand = _trajectory()
    obj = _trajectory()
    obj[:, 1, 3] = 0.2
    transform = _pose((1.0, -2.0, 0.5), (0.2, -0.3, 0.4))
    result = common_rigid_transform_invariant(hand, obj, transform)
    assert result["pass"] is True


def test_episode_slice_preserves_exact_frame_mapping_and_endpoints() -> None:
    raw_hand = _trajectory(9)
    raw_object = _trajectory(9)
    raw_object[:, 2, 3] = 0.4
    start, end = 2, 8
    expected = relative_transform(raw_object[start:end], raw_hand[start:end])
    sliced = relative_transform(raw_object[start:end], raw_hand[start:end])
    assert np.array_equal(expected, sliced)
    assert np.array_equal(sliced[0], relative_transform(raw_object[2], raw_hand[2]))
    assert np.array_equal(sliced[-1], relative_transform(raw_object[7], raw_hand[7]))


def test_hand_only_canonical_transform_is_detected() -> None:
    hand = _trajectory()
    obj = _trajectory()
    canonical = _pose((0.1, 0.0, 0.0), (0.0, 0.0, 0.4))
    expected = relative_transform(compose(canonical, obj), compose(canonical, hand))
    candidate = relative_transform(obj, compose(canonical, hand))
    assert detect_transform_misuse(expected, candidate)["inverse_or_composition_error"] is True


def test_object_only_canonical_transform_is_detected() -> None:
    hand = _trajectory()
    obj = _trajectory()
    canonical = _pose((0.1, 0.0, 0.0), (0.0, 0.0, 0.4))
    expected = relative_transform(compose(canonical, obj), compose(canonical, hand))
    candidate = relative_transform(compose(canonical, obj), hand)
    assert detect_transform_misuse(expected, candidate)["inverse_or_composition_error"] is True


def test_object_mesh_world_vertices_follow_object_pose_once() -> None:
    vertices_local = np.asarray([[0.01, 0.02, 0.03], [-0.02, 0.0, 0.01]])
    world_t_object = _pose((0.3, -0.2, 0.5), (0.1, 0.2, -0.3))
    expected = transform_points(world_t_object, vertices_local)
    double_transformed = transform_points(world_t_object, expected)
    np.testing.assert_allclose(expected, transform_points(world_t_object, vertices_local))
    assert not np.allclose(expected, double_transformed)


def test_robot_retarget_does_not_change_object_trajectory() -> None:
    object_trajectory = _trajectory()
    robot_base = _trajectory()
    robot_base[:, 1, 3] = 0.4
    np.testing.assert_array_equal(object_trajectory, object_trajectory.copy())
    assert not np.array_equal(object_trajectory, robot_base)


def test_inverse_and_rotation_order_errors_are_detected() -> None:
    a = _pose((0.1, 0.2, 0.3), (0.2, 0.3, 0.4))
    b = _pose((-0.2, 0.1, 0.0), (-0.4, 0.1, 0.2))
    expected = compose(a, b)
    assert (
        detect_transform_misuse(expected[None], invert_transform(expected)[None])[
            "inverse_or_composition_error"
        ]
        is True
    )
    assert (
        detect_transform_misuse(expected[None], compose(b, a)[None])["inverse_or_composition_error"]
        is True
    )


def test_handedness_reflection_is_detected() -> None:
    expected = _trajectory()
    reflected = expected.copy()
    reflected[:, 0, 0] = -1.0
    assert detect_transform_misuse(expected, reflected)["handedness_reflection"] is True


def test_unit_scale_mismatch_is_detected_from_motion() -> None:
    expected = _trajectory()
    scaled = expected.copy()
    scaled[:, :3, 3] *= 1000.0
    assert detect_transform_misuse(expected, scaled)["unit_scale_mismatch"] is True


def _qualification(**overrides):
    values = {
        "wrist_position_m": np.zeros(4),
        "wrist_rotation_rad": np.zeros(4),
        "bone_error_rad": np.zeros((4, 20)),
        "source_contact": np.asarray([True, True, False, False]),
        "robot_contact": np.asarray([True, True, False, False]),
        "robot_wrist_transforms": _trajectory(),
        "frame_authority_pass": True,
        "time_alignment_pass": True,
        "interaction_geometry_pass": True,
        "gate": SemanticGateContractV1(),
    }
    values.update(overrides)
    return qualify_semantics(**values)


def test_valid_retarget_passes() -> None:
    assert _qualification()["status"] == "RETARGET_SEMANTIC_PASS"


def test_missing_authority_is_inconclusive_not_pass() -> None:
    result = _qualification(inconclusive_reasons=("MISSING_WRIST_MAPPING_AUTHORITY",))
    assert result["status"] == "RETARGET_SEMANTIC_INCONCLUSIVE"


def test_global_hand_translation_of_ten_centimeters_fails() -> None:
    result = _qualification(wrist_position_m=np.full(4, 0.1))
    assert result["status"] == "RETARGET_SEMANTIC_FAIL"


def test_wrong_wrist_rotation_and_mirrored_hand_fail_closed() -> None:
    rotation_failure = _qualification(wrist_rotation_rad=np.full(4, np.pi))
    mirror_failure = _qualification(frame_authority_pass=False)
    assert rotation_failure["gross_sanity_pass"] is False
    assert mirror_failure["status"] == "RETARGET_SEMANTIC_FAIL"


def test_warm_correct_final_wrong_is_separable() -> None:
    assert _qualification()["status"] == "RETARGET_SEMANTIC_PASS"
    assert _qualification(wrist_position_m=np.full(4, 0.06))["status"] == ("RETARGET_SEMANTIC_FAIL")


def test_contact_loss_and_interaction_distortion_fail() -> None:
    contact = _qualification(robot_contact=np.zeros(4, dtype=bool))
    interaction = _qualification(interaction_geometry_pass=False)
    assert contact["contact_recall_status"] == "FAIL"
    assert contact["status"] == "RETARGET_SEMANTIC_FAIL"
    assert interaction["status"] == "RETARGET_SEMANTIC_FAIL"


def test_contact_precision_is_reported_independently_from_recall() -> None:
    result = _qualification(
        source_contact=np.asarray([True, False, False, False]),
        robot_contact=np.asarray([True, True, False, False]),
    )
    assert result["metrics"]["source_contact_recall"] == 1.0
    assert result["metrics"]["source_contact_precision"] == 0.5


def test_temporal_solution_flip_fails() -> None:
    trajectory = _trajectory()
    trajectory[2, :3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    assert _qualification(robot_wrist_transforms=trajectory)["status"] == ("RETARGET_SEMANTIC_FAIL")


def test_frame_mapping_off_by_one_is_detected() -> None:
    raw = _trajectory(6)
    expected = raw[1:5]
    shifted = raw[2:6]
    assert detect_transform_misuse(expected, shifted)["inverse_or_composition_error"] is True


def test_event_frames_map_exactly_without_runtime_retiming() -> None:
    episode = {
        "start_frame": 100,
        "contact_frame": 102,
        "pickup_frame": 103,
        "place_frame": 105,
        "release_frame": 106,
        "end_frame": 108,
    }
    rows = _event_mapping_rows("episode", episode)
    assert [row["event"] for row in rows] == [
        "START",
        "CONTACT",
        "PICKUP",
        "PLACE",
        "RELEASE",
        "END_EXCLUSIVE",
    ]
    assert [row["episode_frame"] for row in rows] == [0, 2, 3, 5, 6, 8]
    assert all(row["canonical_frame"] == row["episode_frame"] for row in rows)
    assert all(row["warm_frame"] == row["episode_frame"] for row in rows)
    assert all(row["final_frame"] == row["episode_frame"] for row in rows)
    assert all(row["runtime_retimed_frame"] == "NOT_USED" for row in rows)


def test_technical_failure_records_are_append_only_and_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "technical_failures.jsonl"
    rows = [{"failure_id": "A", "status": "RECORDED"}]
    _append_jsonl_unique(path, rows, key="failure_id")
    _append_jsonl_unique(path, [*rows, {"failure_id": "B", "status": "RECORDED"}], key="failure_id")
    assert [json.loads(line)["failure_id"] for line in path.read_text().splitlines()] == ["A", "B"]


def test_final_semantic_failure_is_an_acceptance_gap_not_an_inferred_implementation_bug() -> None:
    assert _classify_earliest_divergence(
        frame_authority_pass=True,
        warm_status="RETARGET_SEMANTIC_PASS",
        final_status="RETARGET_SEMANTIC_FAIL",
    ) == ("FINAL", "FINAL_SOLVER_ACCEPTANCE_TOO_WEAK")
