from __future__ import annotations

import numpy as np

from toporetarget.geometry.signed_distance.batched_exact import (
    BatchedOriginalMeshProximityBackend,
)
from toporetarget.geometry.signed_distance.reference import ReferenceSignedDistanceBackend
from toporetarget.geometry.signed_distance.validation import make_synthetic_mesh
from toporetarget.retarget.final_refinement import (
    RefinementSolverProfile,
    choose_solver_sdf_backend,
)


def test_batched_exact_backend_matches_reference_and_is_persistent() -> None:
    vertices, faces = make_synthetic_mesh("cube")
    reference = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
    candidate = BatchedOriginalMeshProximityBackend.build(vertices, faces)
    points = np.asarray([[0.0, 0.0, 0.0], [0.8, 0.1, -0.2], [1.2, 0.0, 0.0]], dtype=np.float64)
    expected = reference.query_local(points)
    actual = candidate.query_local(points)
    np.testing.assert_allclose(actual.signed_distance, expected.signed_distance, atol=0.0)
    np.testing.assert_allclose(actual.closest_points, expected.closest_points, atol=0.0)
    np.testing.assert_allclose(actual.surface_normals, expected.surface_normals, atol=0.0)
    np.testing.assert_array_equal(actual.closest_face_indices, expected.closest_face_indices)
    counters = candidate.describe()["counters"]
    assert counters["mesh_load_count"] == 1
    assert counters["bvh_build_count"] == 1
    assert counters["proximity_context_build_count"] == 1
    assert counters["query_batch_count"] == 1
    assert counters["query_point_count"] == len(points)


def test_batched_exact_scene_transform_and_determinism() -> None:
    vertices, faces = make_synthetic_mesh("sphere")
    candidate = BatchedOriginalMeshProximityBackend.build(vertices, faces)
    local = np.asarray([[0.1, 0.0, 0.0], [1.2, -0.1, 0.0]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = [0.2, -0.3, 0.5]
    scene = local @ pose[:3, :3].T + pose[:3, 3]
    expected = candidate.query_local(local)
    actual = candidate.query_scene(scene, pose)
    repeated = candidate.query_scene(scene, pose)
    np.testing.assert_allclose(actual.signed_distance, expected.signed_distance, atol=1e-12)
    np.testing.assert_allclose(actual.closest_points, repeated.closest_points, atol=0.0)
    np.testing.assert_array_equal(actual.closest_face_indices, repeated.closest_face_indices)


def test_exact_bvh_solver_profile_selects_the_batched_original_mesh_backend() -> None:
    vertices, faces = make_synthetic_mesh("cube")
    profile = RefinementSolverProfile.load(
        "scipy_slsqp_active_set_contact_rich_v3_original_mesh_batched_exact_bvh"
    )
    reference = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
    backend, report = choose_solver_sdf_backend(
        vertices,
        faces,
        reference,
        profile,
        object_pose_scene=np.eye(4, dtype=np.float64),
    )
    assert backend.backend_id == "original_mesh_batched_exact_bvh_v1"
    assert report["cross_validation"]["status"] == "reference_faithful_by_construction"
