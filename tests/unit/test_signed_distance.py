import numpy as np
import pytest

from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles
from toporetarget.geometry.signed_distance.reference import (
    ReferenceSignedDistanceBackend,
    SignedDistanceError,
)
from toporetarget.geometry.signed_distance.validation import (
    AnalyticShapeBackend,
    make_synthetic_mesh,
)


def test_point_triangle_closest_handles_face_edge_and_vertex() -> None:
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    points = np.array([[0.2, 0.2, 0.4], [0.5, -0.2, 0.0], [-1.0, -1.0, 0.0]], dtype=np.float64)
    closest, faces, barycentric, distance = closest_points_on_triangles(points, triangles)
    np.testing.assert_allclose(closest[0], [0.2, 0.2, 0.0])
    np.testing.assert_allclose(closest[1], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(closest[2], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(faces, 0)
    np.testing.assert_allclose(barycentric.sum(axis=1), 1.0)
    np.testing.assert_allclose(distance, [0.4, 0.2, np.sqrt(2.0)])


def test_positive_outside_and_scene_equivariance() -> None:
    vertices, faces = make_synthetic_mesh("cube")
    backend = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
    points = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=np.float64)
    result = backend.query_local(points)
    assert result.signed_distance[0] < 0
    assert result.signed_distance[1] > 0
    assert result.on_surface[2]
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    pose[:3, 3] = [0.2, -0.3, 0.5]
    scene = backend.query_scene(points @ pose[:3, :3].T + pose[:3, 3], pose)
    np.testing.assert_allclose(scene.signed_distance, result.signed_distance, atol=1e-8)
    np.testing.assert_allclose(
        scene.closest_points, result.closest_points @ pose[:3, :3].T + pose[:3, 3], atol=1e-8
    )


def test_open_mesh_strict_fails_but_unsigned_is_explicit() -> None:
    vertices, faces = make_synthetic_mesh("open_cube")
    with pytest.raises(SignedDistanceError):
        ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
    backend = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="unsigned_only")
    result = backend.query_local(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
    assert np.all(np.isnan(result.signed_distance))
    assert result.inside is None
    assert not np.any(result.sign_valid)
    assert np.all(result.valid)


def test_analytic_oracle_meets_exact_validation() -> None:
    for shape in ("sphere", "cube"):
        backend = AnalyticShapeBackend(shape)
        result = backend.query_local(np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=np.float64))
        assert result.sign_valid.all()
        assert result.signed_distance[0] < 0
        assert result.signed_distance[1] > 0
