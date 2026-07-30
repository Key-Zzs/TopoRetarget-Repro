from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.final_refinement import FinalRetargetTrajectory
from toporetarget.retarget.refinement_checkpoint import (
    CheckpointError,
    CheckpointStore,
    _assemble_arrays,
)
from toporetarget.retarget.refinement_performance import (
    RefinementEvaluationCache,
    RefinementExecutionProfile,
    TimerBook,
)


def test_exact_x_cache_hits_and_query_change_invalidates() -> None:
    cache = RefinementEvaluationCache(3, "ctx")
    value = np.array([1.0, 2.0], dtype=np.float64)
    cache.prepare(value, "q0")
    cache.put("candidate_points", np.ones((2, 3)))
    assert cache.get("candidate_points") is not None

    cache.prepare(value.copy(), "q0")
    assert cache.get("candidate_points") is not None
    cache.prepare(value.copy(), "q1")
    assert cache.get("candidate_points") is None

    changed = value.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    cache.prepare(changed, "q1")
    assert cache.as_dict()["unique_x"] == 2
    assert cache.as_dict()["query_set_invalidations"] == 1


def test_timer_book_and_execution_profile() -> None:
    timers = TimerBook()
    with timers.measure("forward"):
        _ = 1 + 1
    assert timers.as_dict()["counts"]["forward"] == 1

    profile = RefinementExecutionProfile.load()
    assert profile.device == "cpu"
    assert profile.dtype == "float64"
    assert profile.as_dict()["profile_id"] == "cached_checkpoint_cpu_float64_v1"
    v4 = RefinementExecutionProfile.load("wuji_continuous_sequential_fast_exact_v4_compiled_sign")
    assert v4.ambiguity_fd_backend == "compiled_spatial_central_fd_winding_v1"
    assert not v4.recommended
    assert not v4.stage12_default


def test_checkpoint_store_atomic_chain_and_orphan_detection(tmp_path) -> None:
    manifest = {
        "run_id": "unit",
        "input_signature": "input",
        "solver_profile_hash": "solver",
        "execution_profile_hash": "execution",
        "query_profile_hash": "query",
        "frame_range": [0, 3],
    }
    store = CheckpointStore.open(tmp_path / "run", manifest=manifest)

    def row(local: int, previous: str | None) -> tuple[dict, dict[str, np.ndarray]]:
        arrays = {
            "full_signed_distance": np.ones(4, dtype=np.float64),
            "hard_residual": np.ones(1, dtype=np.float64),
            "soft_residual": np.ones(1, dtype=np.float64),
        }
        metadata = {
            "schema_version": "toporetarget.final_retarget_checkpoint.v1",
            "local_frame_index": local,
            "global_frame_index": local,
            "timestamp": float(local),
            "optimizer_converged": True,
            "optimizer_status_code": 0,
            "qpos_bounds_pass": True,
            "slack_bounds_pass": True,
            "active_constraints_feasible": True,
            "full_surface_hard_audit_pass": True,
            "full_surface_soft_audit_pass": True,
            "active_set_converged": True,
            "all_values_finite": True,
            "strict_accepted": True,
            "solver_success": True,
            "previous_checkpoint_hash": previous,
            "acceptance_reason": "strict contract passed",
            "per_frame_checkpoint_hash": "",
        }
        from toporetarget.retarget.refinement_checkpoint import _checkpoint_hash

        metadata["per_frame_checkpoint_hash"] = _checkpoint_hash(metadata, arrays)
        return metadata, arrays

    first, first_arrays = row(0, None)
    first_hash = store.save_frame(first, first_arrays)
    orphan, orphan_arrays = row(2, first_hash)
    store.save_frame(orphan, orphan_arrays)
    status = store.validate_chain()
    assert status["contiguous_frames"] == [0]
    assert status["orphan_frames"] == [2]

    with pytest.raises(CheckpointError):
        store.load_frame(1)

    with pytest.raises(CheckpointError):
        CheckpointStore.open(
            tmp_path / "run",
            manifest={**manifest, "input_signature": "changed"},
            resume=True,
        )


def test_checkpoint_store_rejects_non_strict_and_corrupt_frames(tmp_path) -> None:
    manifest = {
        "run_id": "corrupt",
        "input_signature": "input",
        "solver_profile_hash": "solver",
        "execution_profile_hash": "execution",
        "query_profile_hash": "query",
        "frame_range": [0, 1],
    }
    store = CheckpointStore.open(tmp_path / "run", manifest=manifest)
    arrays = {
        "full_signed_distance": np.ones(2, dtype=np.float64),
        "hard_residual": np.ones(1, dtype=np.float64),
        "soft_residual": np.ones(1, dtype=np.float64),
    }
    metadata = {
        "schema_version": "toporetarget.final_retarget_checkpoint.v1",
        "local_frame_index": 0,
        "global_frame_index": 0,
        "timestamp": 0.0,
        "optimizer_converged": False,
        "optimizer_status_code": 9,
        "qpos_bounds_pass": True,
        "slack_bounds_pass": True,
        "active_constraints_feasible": True,
        "full_surface_hard_audit_pass": True,
        "full_surface_soft_audit_pass": True,
        "active_set_converged": True,
        "all_values_finite": True,
        "strict_accepted": False,
        "solver_success": False,
        "previous_checkpoint_hash": None,
        "acceptance_reason": "status 9",
    }
    from toporetarget.retarget.refinement_checkpoint import _checkpoint_hash

    metadata["per_frame_checkpoint_hash"] = _checkpoint_hash(metadata, arrays)
    with pytest.raises(CheckpointError):
        store.save_frame(metadata, arrays)

    accepted_metadata = dict(metadata)
    accepted_metadata.update(
        {
            "optimizer_converged": True,
            "optimizer_status_code": 0,
            "strict_accepted": True,
            "solver_success": True,
            "acceptance_reason": "strict contract passed",
        }
    )
    accepted_metadata["per_frame_checkpoint_hash"] = _checkpoint_hash(accepted_metadata, arrays)
    store.save_frame(accepted_metadata, arrays)
    path = store.frames_dir / "frame_000000.npz"
    payload = bytearray(path.read_bytes())
    payload[:4] = b"bad!"
    path.write_bytes(payload)
    scan = store.scan()
    assert scan["invalid_frames"] == [0]


def test_checkpoint_assembly_preserves_final_artifact_shapes() -> None:
    metadata = {
        "timestamp": 0.0,
        "global_frame_index": 0,
        "full_soft_violation_count": 0,
        "unqueried_soft_violation_count": 0,
        "strict_accepted": True,
        "solver_success": True,
        "optimizer_status_code": 0,
        "iterations": 1,
        "function_evaluations": 1,
        "jacobian_evaluations": 1,
        "solve_time_s": 1.0,
        "active_set_rounds": 1,
        "active_set_converged": True,
        "optimizer_converged": True,
        "optimizer_message": "ok",
        "optimizer_iterations": 1,
        "optimizer_function_evaluations": 1,
        "optimizer_jacobian_evaluations": 1,
        "qpos_bounds_pass": True,
        "slack_bounds_pass": True,
        "active_constraints_feasible": True,
        "full_surface_hard_audit_pass": True,
        "full_surface_soft_audit_pass": True,
        "all_values_finite": True,
        "stationarity_checked": False,
        "stationarity_residual": 0.0,
        "acceptance_reason": "strict contract passed",
        "initial_objective": 1.0,
        "final_objective": 0.5,
        "final_objective_change": 0.5,
        "final_step_norm": 0.1,
    }
    row = {
        "qpos": np.zeros(22),
        "base_pose_scene": np.eye(4),
        "base_correction": np.zeros(6),
        "joint_limit_margins": np.ones(22),
        "robot_keypoints_base": np.zeros((21, 3)),
        "robot_keypoints_scene": np.zeros((21, 3)),
        "robot_link_poses": np.zeros((16, 4, 4)),
        "collision_points_scene": np.zeros((512, 3)),
        "query_ids": np.arange(512),
        "query_active_round": np.zeros(512, dtype=np.int64),
        "query_inclusion_reason": np.full(512, b"initial"),
        "slack": np.zeros(512),
        "signed_distance": np.ones(512),
        "hard_residual": np.ones(512),
        "soft_residual": np.ones(512),
        "full_signed_distance": np.ones(512),
        "full_closest_points": np.zeros((512, 3)),
        "full_surface_normals": np.zeros((512, 3)),
        "full_hard_residual": np.ones(512),
        "full_soft_violation_count": np.asarray(0),
        "unqueried_soft_violation_count": np.asarray(0),
        "objective_components": np.arange(12, dtype=np.float64),
    }
    arrays = _assemble_arrays([metadata], [row])
    FinalRetargetTrajectory({"schema_version": "toporetarget.final_retarget.v2"}, arrays).validate()
    assert arrays["joint_limit_margins"].shape == (1, 22)
    assert arrays["collision_points_scene"].shape == (1, 512, 3)
