from __future__ import annotations

import numpy as np

from toporetarget.geometry.signed_distance.reference import ReferenceSignedDistanceBackend
from toporetarget.geometry.signed_distance.signed_grid import (
    OriginalMeshSignedGridSDFBackend,
    grid_resolution_from_profile,
)
from toporetarget.geometry.signed_distance.validation import make_synthetic_mesh
from toporetarget.retarget.final_refinement import regularization_profile_for_solver


def _grid(shape: str, tmp_path) -> tuple[OriginalMeshSignedGridSDFBackend, np.ndarray, np.ndarray]:
    vertices, faces = make_synthetic_mesh(shape)
    return (
        OriginalMeshSignedGridSDFBackend.build(
            vertices,
            faces,
            resolution=24,
            profile_id="original_mesh_signed_grid_24_v1",
            cache_root=tmp_path,
            query_batch_size=2048,
        ),
        vertices,
        faces,
    )


def test_original_mesh_grid_uses_positive_outside_and_cache(tmp_path) -> None:
    grid, vertices, faces = _grid("cube", tmp_path)
    first = grid.query_local(np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
    assert first.signed_distance[0] < 0.0
    assert first.signed_distance[1] > 0.0
    assert first.sign_valid.all()
    assert grid.describe()["source_geometry"] == "original_strict_watertight_mesh"
    assert grid.describe()["sign_source"] == "reference_triangle_winding"
    repeat = OriginalMeshSignedGridSDFBackend.build(
        vertices,
        faces,
        resolution=24,
        profile_id="original_mesh_signed_grid_24_v1",
        cache_root=tmp_path,
        query_batch_size=2048,
    )
    np.testing.assert_array_equal(repeat.signed_distance_grid, grid.signed_distance_grid)
    assert repeat.metadata["cache_key"] == grid.metadata["cache_key"]


def test_original_mesh_grid_interpolation_and_scene_equivariance(tmp_path) -> None:
    grid, vertices, faces = _grid("sphere", tmp_path)
    reference = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.6], [1.3, 0.2, -0.1], [-1.1, 0.1, 0.2]],
        dtype=np.float64,
    )
    result = grid.query_local(points)
    expected = reference.query_local(points)
    # This is a deliberately coarse unit grid; production acceptance has the
    # tighter 192/256 gate on object-local, active-region probes.
    assert np.max(np.abs(result.signed_distance - expected.signed_distance)) < 0.15
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = [0.2, -0.3, 0.5]
    scene = grid.query_scene(points @ pose[:3, :3].T + pose[:3, 3], pose)
    np.testing.assert_allclose(scene.signed_distance, result.signed_distance, atol=1e-12)
    np.testing.assert_allclose(
        scene.surface_normals, result.surface_normals @ pose[:3, :3].T, atol=1e-12
    )


def test_original_mesh_grid_outside_policy_is_positive_exact_and_audited(tmp_path) -> None:
    grid, vertices, faces = _grid("cube", tmp_path)
    point = grid.grid_max[None, :] + np.asarray([[0.1, 0.0, 0.0]])
    result = grid.query_local(point)
    assert result.signed_distance[0] > 0.0
    reference = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
    expected = reference.query_local(point)
    np.testing.assert_allclose(result.signed_distance, expected.signed_distance, atol=1e-12)
    np.testing.assert_allclose(result.closest_points, expected.closest_points, atol=1e-12)
    assert grid.audit()["out_of_grid_query_count"] == 1
    assert not result.gradient_valid[0]


def test_signed_grid_profile_parser() -> None:
    assert grid_resolution_from_profile("original_mesh_signed_grid_192_v1") == 192
    assert grid_resolution_from_profile("original_mesh_signed_grid_256_v1") == 256
    assert grid_resolution_from_profile("convex_hull_exact_solver_only") is None
    assert grid_resolution_from_profile("original_mesh_signed_grid_x_v1") is None


def test_grid_solver_profiles_preserve_the_fixed_temporal_semantics() -> None:
    for profile in (
        "scipy_slsqp_active_set_contact_rich_v3_original_mesh_grid_192",
        "scipy_slsqp_active_set_contact_rich_v3_original_mesh_grid_256",
    ):
        assert regularization_profile_for_solver(profile) == "faithful_regularization_fix_v1"
