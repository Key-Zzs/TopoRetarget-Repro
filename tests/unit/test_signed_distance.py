import numpy as np
import pytest

from toporetarget.geometry.signed_distance.closest_point import (
    ObjectLocalBVH,
    closest_points_on_triangles,
)
from toporetarget.geometry.signed_distance.derived_proxy import build_hybrid_signed_distance_backend
from toporetarget.geometry.signed_distance.gradient import analytic_spatial_gradient
from toporetarget.geometry.signed_distance.reference import (
    ReferenceSignedDistanceBackend,
    SignedDistanceError,
)
from toporetarget.geometry.signed_distance.sign_cache import LipschitzSignCache
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


def test_object_local_bvh_matches_bruteforce_exactly() -> None:
    vertices, faces = make_synthetic_mesh("cube")
    triangles = vertices[faces]
    points = np.asarray(
        [[2.0, 0.3, 0.2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.5, 0.2, 0.1]],
        dtype=np.float64,
    )
    bvh = ObjectLocalBVH(triangles, leaf_size=2)
    actual = closest_points_on_triangles(points, triangles, tree=bvh)
    expected = closest_points_on_triangles(points, triangles, tree=None)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(actual[3], expected[3], atol=1e-12)
    assert bvh.stats()["candidate_triangle_evaluations"] > 0


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


def test_spatial_gradient_uses_signed_closest_feature_direction() -> None:
    backend = AnalyticShapeBackend("sphere")
    points = np.asarray([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    result = backend.query_local(points)
    gradient = analytic_spatial_gradient(points, result)
    assert gradient.analytic_mask.all()
    np.testing.assert_allclose(gradient.spatial_gradient_scene, [[1.0, 0.0, 0.0]] * 2)
    np.testing.assert_allclose(gradient.gradient_norm, 1.0)


def test_lipschitz_sign_cache_reuses_only_certified_signs() -> None:
    cache = LipschitzSignCache("mesh", "sign-profile")
    initial = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64)
    cache.record_exact(initial, np.asarray([7]), np.asarray([1.0]), np.asarray([True]), lineage="a")
    signs, hits = cache.lookup(
        np.asarray([[2.1, 0.0, 0.0]], dtype=np.float64), np.asarray([7]), lineage="b"
    )
    assert hits.tolist() == [True]
    assert signs.tolist() == [1]
    _signs, crossing_hits = cache.lookup(
        np.asarray([[0.9, 0.0, 0.0]], dtype=np.float64), np.asarray([7]), lineage="c"
    )
    assert crossing_hits.tolist() == [False]
    stats = cache.as_dict()
    assert stats["certified_reuse_count"] == 1
    assert stats["sign_cache_misses"] == 1


def test_hybrid_backend_cache_preserves_exact_signed_distance() -> None:
    vertices, faces = make_synthetic_mesh("cube")
    backend, geometry = build_hybrid_signed_distance_backend(vertices, faces)
    cache = LipschitzSignCache(geometry.source_mesh_hash, "unit-sign-profile")
    points = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64)
    first = backend.query_local(points, sample_ids=np.asarray([3]), sign_cache=cache)
    second = backend.query_local(
        np.asarray([[2.01, 0.0, 0.0]], dtype=np.float64),
        sample_ids=np.asarray([3]),
        sign_cache=cache,
    )
    assert first.sign_source is not None and first.sign_source[0] == "EXACT_GENERALIZED_WINDING"
    assert second.sign_source is not None and second.sign_source[0] == "LIPSCHITZ_CERTIFIED_REUSE"
    assert second.signed_distance[0] == pytest.approx(1.01)
