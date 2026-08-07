from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.geometry_audit.simulation_trace_replay import (
    infer_object_id,
    load_factor8_hocap_reference_object_pose,
    load_stage16d_simulation_trace,
)


def _write_trace(path: Path, *, frames: int = 3, replicas: int = 2, bodies: int = 2) -> None:
    object_pose = np.zeros((frames, replicas, 7), dtype=np.float32)
    hand_pose = np.zeros((frames, replicas, bodies, 7), dtype=np.float32)
    object_pose[..., 3] = 1.0
    hand_pose[..., 3] = 1.0
    contact = np.zeros((frames, replicas, bodies), dtype=bool)
    contact[1, 0, 1] = True
    groups = np.zeros((frames, replicas, 2), dtype=bool)
    groups[1, 0, 1] = True
    forces = np.zeros((frames, replicas, bodies, 3), dtype=np.float32)
    forces[1, 0, 1, 0] = 2.5
    twist = np.zeros((frames, replicas, 6), dtype=np.float32)
    twist[1, 0, :3] = (3.0, 4.0, 0.0)
    np.savez_compressed(
        path,
        object_pose=object_pose,
        hand_collision_body_pose=hand_pose,
        hand_collision_body_names=np.asarray(["r_wrist", "r_thumb_distal"]),
        contact_force_world=forces,
        contact_pair_presence=contact,
        contact_group_presence=groups,
        contact_group_names=np.asarray(["palm", "thumb"]),
        object_twist=twist,
        mean_absolute_effort=np.full((frames, replicas), 0.25, dtype=np.float32),
        finite=np.ones((frames, replicas), dtype=bool),
        reason_code=np.zeros((frames, replicas), dtype=np.int64),
        actions=np.zeros((frames, 26), dtype=np.float32),
        initialization_static_pass=np.asarray(True),
    )


def _write_geometry(path: Path, *, frames: int = 3, replicas: int = 2, bodies: int = 2) -> None:
    penetration = np.zeros((frames, replicas, bodies), dtype=np.float64)
    penetration[1, 0, bodies - 1] = 0.004
    worst = penetration.max(axis=-1)
    pair = penetration.argmax(axis=-1)
    np.savez_compressed(
        path,
        penetration_depth_m=penetration,
        frame_worst_penetration_m=worst,
        frame_worst_pair_index=pair,
    )


def _write_corrected_trace(path: Path, *, frames: int = 3, bodies: int = 2) -> None:
    object_pose = np.zeros((frames, 7), dtype=np.float32)
    hand_pose = np.zeros((frames, bodies, 7), dtype=np.float32)
    object_pose[:, 3] = 1.0
    hand_pose[..., 3] = 1.0
    contact = np.zeros((frames, bodies), dtype=bool)
    contact[1, 1] = True
    force = np.zeros((frames, 3), dtype=np.float32)
    force[1, 0] = 3.5
    np.savez_compressed(
        path,
        object_pose=object_pose,
        object_twist=np.zeros((frames, 6), dtype=np.float32),
        hand_collision_body_pose=hand_pose,
        hand_collision_body_names=np.asarray(["r_wrist", "r_thumb_distal"]),
        contact_force_world=force,
        contact_pair_presence=contact,
        actuator_effort=np.full((frames, 26), -0.5, dtype=np.float32),
        reason_code=np.zeros(frames, dtype=np.int64),
        action=np.zeros((frames, 26), dtype=np.float32),
    )


def test_load_trace_and_frame_diagnostics(tmp_path: Path) -> None:
    trace_path = tmp_path / "calibration_dev_hocap_170105_test.npz"
    geometry_path = tmp_path / "calibration_dev_hocap_170105_test_geometry.npz"
    _write_trace(trace_path)
    _write_geometry(geometry_path)

    trace = load_stage16d_simulation_trace(
        trace_path,
        geometry_path=geometry_path,
        expected_body_names=("r_wrist", "r_thumb_distal"),
    )

    assert trace.frame_count == 3
    assert trace.replica_count == 2
    assert list(trace.frame_indices(1, None)) == [1, 2]
    row = trace.diagnostics(1, 0)
    assert row.contact_body_count == 1
    assert row.contact_groups == ("thumb",)
    assert row.contact_force_norm_n == pytest.approx(2.5)
    assert row.object_linear_speed_mps == pytest.approx(5.0)
    assert row.worst_penetration_m == pytest.approx(0.004)
    assert row.worst_pair_index == 1


def test_loader_fails_closed_on_manifest_order_mismatch(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    _write_trace(trace_path)

    with pytest.raises(ValueError, match="body order"):
        load_stage16d_simulation_trace(
            trace_path,
            expected_body_names=("r_thumb_distal", "r_wrist"),
        )


def test_loader_rejects_invalid_geometry_shape(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    geometry_path = tmp_path / "geometry.npz"
    _write_trace(trace_path)
    _write_geometry(geometry_path, bodies=1)

    with pytest.raises(ValueError, match="penetration_depth_m"):
        load_stage16d_simulation_trace(trace_path, geometry_path=geometry_path)


def test_replica_and_frame_ranges_are_bounded(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.npz"
    _write_trace(trace_path)
    trace = load_stage16d_simulation_trace(trace_path)

    with pytest.raises(ValueError, match="replica"):
        trace.validate_replica(2)
    with pytest.raises(ValueError, match="start frame"):
        trace.frame_indices(3, None)
    with pytest.raises(ValueError, match="end frame"):
        trace.frame_indices(2, 2)


def test_object_id_inference_is_explicit_and_unambiguous() -> None:
    assert infer_object_id(Path("candidate_hocap_170650_trace.npz")) == "hocap_170650"
    with pytest.raises(ValueError, match="cannot infer"):
        infer_object_id(Path("candidate_trace.npz"))


def test_load_corrected_nominal_trace_and_qualification(tmp_path: Path) -> None:
    trace_path = tmp_path / "trajectory_trace_hocap_170105_v3.npz"
    qualification = tmp_path / "trajectory_qualification_hocap_170105_v3.json"
    _write_corrected_trace(trace_path)
    qualification.write_text(
        '{"status":"STAGE16D_TRAJECTORY_QUALIFICATION_BLOCKED",'
        '"success_rate":0.75,"semantic_reach_rate":1.0}',
        encoding="utf-8",
    )

    trace = load_stage16d_simulation_trace(
        trace_path,
        expected_body_names=("r_wrist", "r_thumb_distal"),
    )

    assert trace.trace_kind == "physics_consistent_corrected_nominal"
    assert trace.frame_count == 3
    assert trace.replica_count == 1
    assert trace.qualification_status == "STAGE16D_TRAJECTORY_QUALIFICATION_BLOCKED"
    assert trace.qualification_metrics == {"success_rate": 0.75, "semantic_reach_rate": 1.0}
    row = trace.diagnostics(1, 0)
    assert row.contact_groups == ("thumb",)
    assert row.contact_force_norm_n == pytest.approx(3.5)
    assert row.mean_absolute_effort == pytest.approx(0.5)


def test_corrected_trace_rejects_replica_without_frame_telemetry(tmp_path: Path) -> None:
    trace_path = tmp_path / "trajectory_trace_hocap_170105_v3.npz"
    _write_corrected_trace(trace_path)
    trace = load_stage16d_simulation_trace(trace_path)

    with pytest.raises(ValueError, match="replica"):
        trace.validate_replica(1)


def _write_hocap_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(41, dtype=np.float64) * 0.05
    position = np.stack((timestamps, np.square(timestamps), -timestamps), axis=-1)
    quaternion = np.zeros((41, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    quaternion[20:, 0] = -1.0
    twist = np.zeros((41, 6), dtype=np.float64)
    twist[:, 0] = 1.0
    twist[:, 1] = 2.0 * timestamps
    twist[:, 2] = -1.0
    np.savez_compressed(
        path,
        timestamps=timestamps,
        object_pose_translation_world_ref=position,
        object_pose_quaternion_world_ref_wxyz=quaternion,
        object_twist_world_ref=twist,
    )
    return position, quaternion


def test_factor8_hocap_reference_preserves_all_source_keys(tmp_path: Path) -> None:
    reference = tmp_path / "hocap_170105.world_wrist.stage16.npz"
    position, quaternion = _write_hocap_reference(reference)

    pose = load_factor8_hocap_reference_object_pose(reference, expected_frames=321)

    assert pose.shape == (321, 7)
    np.testing.assert_allclose(pose[::8, :3], position, atol=1.0e-7)
    np.testing.assert_allclose(np.abs(pose[::8, 3:]), np.abs(quaternion), atol=1.0e-7)


def test_factor8_hocap_reference_rejects_physx_source_trace(tmp_path: Path) -> None:
    source = tmp_path / "source_trace_170105.npz"
    pose = np.zeros((321, 7), dtype=np.float32)
    pose[:, 3] = 1.0
    np.savez_compressed(source, object_pose=pose)

    with pytest.raises(ValueError, match="PhysX source trace"):
        load_factor8_hocap_reference_object_pose(source, expected_frames=321)


def test_factor8_hocap_reference_requires_matching_runtime_length(tmp_path: Path) -> None:
    reference = tmp_path / "hocap_170105.world_wrist.stage16.npz"
    _write_hocap_reference(reference)

    with pytest.raises(ValueError, match="produces 321 frames"):
        load_factor8_hocap_reference_object_pose(reference, expected_frames=41)
