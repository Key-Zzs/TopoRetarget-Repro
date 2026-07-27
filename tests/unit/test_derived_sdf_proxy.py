from __future__ import annotations

import numpy as np
import pytest

from toporetarget.geometry.signed_distance.derived_proxy import (
    ObjectSDFGeometryPolicy,
    _voxel_fallback,
    build_hybrid_signed_distance_backend,
)
from toporetarget.geometry.signed_distance.reference import ReferenceSignedDistanceBackend


def _subdivided_cube(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    lookup: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    specs = (
        ((1.0, -1.0, -1.0), (0.0, 2.0 / n, 0.0), (0.0, 0.0, 2.0 / n)),
        ((-1.0, -1.0, -1.0), (0.0, 0.0, 2.0 / n), (0.0, 2.0 / n, 0.0)),
        ((-1.0, 1.0, -1.0), (0.0, 0.0, 2.0 / n), (2.0 / n, 0.0, 0.0)),
        ((-1.0, -1.0, -1.0), (2.0 / n, 0.0, 0.0), (0.0, 0.0, 2.0 / n)),
        ((-1.0, -1.0, 1.0), (2.0 / n, 0.0, 0.0), (0.0, 2.0 / n, 0.0)),
        ((-1.0, -1.0, -1.0), (0.0, 2.0 / n, 0.0), (2.0 / n, 0.0, 0.0)),
    )
    for origin, u, v in specs:
        face_ids: list[list[int]] = []
        for row in range(n + 1):
            current: list[int] = []
            for col in range(n + 1):
                point = tuple(
                    round(origin[index] + row * u[index] + col * v[index], 9) for index in range(3)
                )
                if point not in lookup:
                    lookup[point] = len(vertices)
                    vertices.append(point)
                current.append(lookup[point])
            face_ids.append(current)
        for row in range(n):
            for col in range(n):
                if origin[2] == 1.0 and row == n // 2 and col == n // 2:
                    continue
                a, b = face_ids[row][col], face_ids[row + 1][col]
                c, d = face_ids[row + 1][col + 1], face_ids[row][col + 1]
                faces.extend(((a, b, c), (a, c, d)))
    return (
        np.asarray(vertices, dtype=np.float64) * 0.02,
        np.asarray(faces, dtype=np.int64),
    )


def test_local_repair_is_deterministic_and_does_not_mutate_source() -> None:
    vertices, faces = _subdivided_cube()
    original_vertices = vertices.copy()
    original_faces = faces.copy()
    backend, geometry = build_hybrid_signed_distance_backend(vertices, faces)
    assert geometry.candidate_id == "candidate_1_local_repair"
    assert geometry.proxy_audit.watertight
    assert geometry.proxy_audit.boundary_edge_count == 0
    assert geometry.proxy_audit.non_manifold_edge_count == 0
    assert len(geometry.boundary_loops) == 1
    assert len(geometry.synthetic_face_ids) == 4
    assert geometry.patch_area_m2 / float(geometry.source_audit.surface_area) <= 0.05
    assert np.array_equal(vertices, original_vertices)
    assert np.array_equal(faces, original_faces)
    _, repeat = build_hybrid_signed_distance_backend(vertices, faces)
    assert repeat.cache_signature == geometry.cache_signature
    assert np.array_equal(repeat.proxy_faces, geometry.proxy_faces)
    assert np.array_equal(repeat.synthetic_face_ids, geometry.synthetic_face_ids)
    boundary_result = backend.query_local(
        geometry.source_distance_vertices[geometry.boundary_loops[0]]
    )
    assert np.all(boundary_result.near_original_boundary)


def test_hybrid_magnitude_and_original_closest_face_are_preserved() -> None:
    vertices, faces = _subdivided_cube()
    backend, geometry = build_hybrid_signed_distance_backend(vertices, faces)
    points = np.asarray([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.0, 0.01001]], dtype=np.float64)
    result = backend.query_local(points)
    assert np.allclose(np.abs(result.signed_distance), result.unsigned_distance, atol=0.0, rtol=0.0)
    assert np.all(np.isfinite(result.sign_valid))
    assert np.all(result.closest_face_indices >= 0)
    assert np.all(result.closest_face_indices < len(faces))
    assert np.all(result.proxy_closest_face_indices >= 0)
    assert result.geometry_metadata["proxy_mesh_hash"] == geometry.proxy_mesh_hash
    assert bool(result.inside[0])
    assert not bool(result.inside[1])


def test_identity_matches_strict_reference_and_transform_round_trip() -> None:
    # The identity gate is exercised on a closed cube without the test hole.
    closed_vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    closed_faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    backend, geometry = build_hybrid_signed_distance_backend(closed_vertices, closed_faces)
    reference = ReferenceSignedDistanceBackend(closed_vertices, closed_faces, sign_mode="strict")
    points = np.asarray([[0, 0, 0], [2, 0, 0], [0.3, -0.2, 0.9]], dtype=np.float64)
    hybrid = backend.query_local(points)
    strict = reference.query_local(points)
    assert geometry.candidate_id == "candidate_0_identity"
    assert np.max(np.abs(hybrid.signed_distance - strict.signed_distance)) <= 1e-10
    assert np.array_equal(hybrid.inside, strict.inside)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray([0.3, -0.4, 0.2])
    scene_points = points + pose[:3, 3]
    scene = backend.query_scene(scene_points, pose)
    local = backend.query_local(points)
    assert np.allclose(scene.signed_distance, local.signed_distance)
    assert np.allclose(scene.closest_points, local.closest_points @ pose[:3, :3].T + pose[:3, 3])
    assert np.allclose(scene.unsigned_distance, local.unsigned_distance)


def test_hybrid_finite_difference_outside_gradient() -> None:
    vertices, faces = _subdivided_cube()
    backend, _ = build_hybrid_signed_distance_backend(vertices, faces)
    point = np.asarray([[0.03, 0.0, 0.0]], dtype=np.float64)
    epsilon = 1e-6
    plus = backend.query_local(point + [epsilon, 0.0, 0.0]).signed_distance[0]
    minus = backend.query_local(point - [epsilon, 0.0, 0.0]).signed_distance[0]
    assert abs(float((plus - minus) / (2 * epsilon)) - 1.0) <= 1e-5


def test_voxel_fallback_is_fixed_resolution_and_watertight() -> None:
    pytest.importorskip("skimage")
    vertices, faces = _subdivided_cube(n=4)
    policy = ObjectSDFGeometryPolicy.load()
    proxy_vertices, proxy_faces = _voxel_fallback(vertices, faces, policy)
    from toporetarget.geometry.mesh_audit import audit_mesh

    report = audit_mesh(proxy_vertices, proxy_faces)
    assert report.watertight
    assert report.boundary_edge_count == 0
    assert np.allclose(np.ptp(proxy_vertices, axis=0), np.ptp(vertices, axis=0), rtol=0.1)
