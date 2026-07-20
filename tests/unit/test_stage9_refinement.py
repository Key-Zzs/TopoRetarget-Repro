from __future__ import annotations

import numpy as np

from toporetarget.retarget.final_refinement import (
    CollisionQueryProfile,
    CollisionQuerySet,
    FinalRetargetTrajectory,
    build_query_set,
    load_final_trajectory,
    map_previous_state_to_seed,
    save_final_trajectory,
    so3_exp,
    so3_log,
)


def test_so3_exp_log_and_zero_gradient() -> None:
    import torch

    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    so3_exp(zero).sum().backward()
    assert torch.all(torch.isfinite(zero.grad))
    value = np.asarray([0.01, -0.02, 0.03])
    rotation = so3_exp(torch.as_tensor(value, dtype=torch.float64)).detach().numpy()
    assert np.linalg.norm(so3_log(rotation) - value) <= 1e-10


def test_previous_state_is_remapped_to_current_seed() -> None:
    previous = np.eye(4)
    previous[:3, 3] = [0.2, -0.1, 0.3]
    current_seed = np.eye(4)
    current_seed[:3, 3] = [0.1, 0.1, 0.1]
    mapped = map_previous_state_to_seed(previous, np.arange(22), current_seed)
    assert np.allclose(mapped[:3], [0.1, -0.2, 0.2])
    assert np.allclose(mapped[3:6], 0.0)
    assert np.array_equal(mapped[6:], np.arange(22))


def test_adaptive_query_set_is_deterministic_and_complete() -> None:
    distances = np.asarray([-0.002, 0.004, 0.02, 0.03, 0.006, 0.02])
    geometries = np.asarray(["g0", "g0", "g0", "g1", "g1", "g1"])
    profile = CollisionQueryProfile(
        "adaptive", "1", "adaptive", 0.01, 5, "not_paper_specified", (), "hash"
    )
    result = build_query_set(distances, geometries, profile)
    assert np.array_equal(result.sample_ids, [0, 1, 4])
    assert "initial_penetration" in result.inclusion_reasons[0]
    assert "nearest_per_geometry" in result.inclusion_reasons[2]
    assert result.query_hash == build_query_set(distances, geometries, profile).query_hash


def test_final_artifact_ragged_round_trip(tmp_path) -> None:
    arrays = {
        "timestamps": np.asarray([0.0]),
        "qpos": np.zeros((1, 22)),
        "base_pose_scene": np.eye(4)[None],
        "base_corrections": np.zeros((1, 6)),
        "robot_keypoints_base": np.zeros((1, 21, 3)),
        "robot_keypoints_scene": np.zeros((1, 21, 3)),
        "collision_points_scene": np.zeros((1, 512, 3)),
        "slack_concat": np.asarray([0.0, 0.001]),
        "query_offsets": np.asarray([0, 2]),
        "full_signed_distance": np.ones((1, 512)),
        "full_closest_points": np.zeros((1, 512, 3)),
        "full_surface_normals": np.zeros((1, 512, 3)),
        "full_hard_residual": np.ones((1, 512)),
        "full_soft_violation_count": np.zeros(1, dtype=np.int64),
        "unqueried_soft_violation_count": np.zeros(1, dtype=np.int64),
        "active_set_converged": np.asarray([True]),
        "robot_link_poses": np.eye(4)[None, None],
        "valid_mask": np.asarray([True]),
    }
    trajectory = FinalRetargetTrajectory(
        {"schema_version": "toporetarget.final_retarget.v1"}, arrays
    )
    path = save_final_trajectory(trajectory, tmp_path / "final.zarr")
    loaded = load_final_trajectory(path)
    assert np.array_equal(loaded.arrays["query_offsets"], [0, 2])
    assert loaded.schema_version == "toporetarget.final_retarget.v1"


def test_query_set_validation_rejects_duplicates() -> None:
    query = CollisionQuerySet(
        np.asarray([1, 1]), ("a", "b"), np.zeros(2, dtype=int), np.zeros(2), "h"
    )
    try:
        query.validate(4)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate query IDs must fail")
