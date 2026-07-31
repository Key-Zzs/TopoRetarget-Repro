from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles
from toporetarget.geometry.signed_distance.compiled_sdf_cpu import (
    CompiledBVHHandle,
    CompiledGeneralizedWindingHandle,
    CompiledSpatialFDBackend,
    compiled_available,
    compiled_exact_query,
)
from toporetarget.geometry.signed_distance.derived_proxy import build_hybrid_signed_distance_backend
from toporetarget.geometry.signed_distance.sign_cache import LipschitzSignCache
from toporetarget.geometry.signed_distance.validation import make_synthetic_mesh
from toporetarget.geometry.signed_distance.winding import generalized_winding_number

pytestmark = pytest.mark.skipif(
    not compiled_available(), reason="compiled SDF CPU extension not built"
)


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = make_synthetic_mesh("cube")
    return (
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )


def test_compiled_exact_bvh_matches_vectorized_reference_at_features() -> None:
    vertices, faces = _mesh()
    points = np.ascontiguousarray(
        [[0.2, 0.2, 1.0], [0.5, -1.0, 0.0], [1.0, 1.0, 1.0], [0.1, -0.2, 1.4]],
        dtype=np.float64,
    )
    handle = CompiledBVHHandle(vertices, faces)
    actual = compiled_exact_query(handle, points)
    expected = closest_points_on_triangles(points, vertices[faces], tree=None)
    np.testing.assert_allclose(actual["closest_point"], expected[0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(actual["unsigned_distance"], expected[3], atol=1e-10, rtol=0.0)
    assert np.all(np.isfinite(actual["barycentric"]))
    assert handle.stats()["query_count"] == 1


def test_compiled_exact_bvh_rejects_invalid_input_and_handles_empty_batch() -> None:
    vertices, faces = _mesh()
    handle = CompiledBVHHandle(vertices, faces)
    empty = np.empty((0, 3), dtype=np.float64)
    result = compiled_exact_query(handle, empty)
    assert result["closest_point"].shape == (0, 3)
    with pytest.raises(TypeError, match="float64"):
        compiled_exact_query(handle, np.ones((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        compiled_exact_query(handle, np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        compiled_exact_query(handle, np.ones((3, 3), dtype=np.float64).T)


def test_compiled_spatial_fd_matches_reference_hybrid_without_cache_pollution() -> None:
    vertices, faces = _mesh()
    reference, _geometry = build_hybrid_signed_distance_backend(vertices, faces)
    compiled = CompiledSpatialFDBackend(reference)
    points = np.ascontiguousarray(
        [[1.0, 0.0, 0.1], [0.2, 0.3, 1.0], [1.1, -0.2, 0.4]], dtype=np.float64
    )
    step = 1e-5
    axes = np.eye(3, dtype=np.float64)
    probes = np.concatenate(
        [points + step * axis for axis in axes] + [points - step * axis for axis in axes]
    )
    expected_phi = reference.query_local(probes, cache_update=False).signed_distance.reshape(6, -1)
    expected = ((expected_phi[:3] - expected_phi[3:]) / (2.0 * step)).T
    actual = compiled.spatial_fd_gradient_scene(points, np.eye(4), step)
    np.testing.assert_allclose(actual.gradient_scene, expected, atol=1e-8, rtol=0.0)
    assert actual.probe_count == 6 * len(points)
    assert np.all(actual.probe_result.sign_source == "EXACT_GENERALIZED_WINDING")


def test_compiled_solver_query_matches_hybrid_and_preserves_certified_cache_contract() -> None:
    vertices, faces = _mesh()
    reference, _geometry = build_hybrid_signed_distance_backend(vertices, faces)
    compiled = CompiledSpatialFDBackend(reference, compiled_winding=True)
    points = np.ascontiguousarray(
        [[1.0, 0.2, 0.1], [0.2, 0.3, 1.0], [1.1, -0.2, 0.4]], dtype=np.float64
    )
    ids = np.asarray([3, 7, 11], dtype=np.int64)
    reference_cache = LipschitzSignCache(reference.mesh_hash, "test-policy")
    compiled_cache = LipschitzSignCache(reference.mesh_hash, "test-policy")
    for pass_index in range(2):
        expected = reference.query_local(
            points,
            sample_ids=ids,
            sign_cache=reference_cache,
            evaluation_lineage="compiled-solver-test",
        )
        actual = compiled.query_local(
            points,
            sample_ids=ids,
            sign_cache=compiled_cache,
            evaluation_lineage="compiled-solver-test",
        )
        np.testing.assert_allclose(actual.signed_distance, expected.signed_distance, atol=1e-10)
        np.testing.assert_allclose(actual.unsigned_distance, expected.unsigned_distance, atol=1e-10)
        np.testing.assert_allclose(actual.closest_points, expected.closest_points, atol=1e-10)
        np.testing.assert_array_equal(actual.inside, expected.inside)
        np.testing.assert_array_equal(actual.sign_valid, expected.sign_valid)
        if pass_index:
            # Both implementations must reuse the same cache certificate on
            # the repeated query; exact calls retain their backend identity.
            np.testing.assert_array_equal(
                actual.sign_source == "LIPSCHITZ_CERTIFIED_REUSE",
                expected.sign_source == "LIPSCHITZ_CERTIFIED_REUSE",
            )
        else:
            assert np.all(actual.sign_source == "COMPILED_GENERALIZED_WINDING")


def test_compiled_generalized_winding_matches_reference_and_is_deterministic() -> None:
    vertices, faces = _mesh()
    points = np.ascontiguousarray(
        [[0.0, 0.0, 0.0], [1.2, 0.1, -0.2], [0.5, -0.4, 0.7], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    handle = CompiledGeneralizedWindingHandle(vertices, faces)
    actual = handle.query(points)
    expected = generalized_winding_number(points, vertices[faces])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(handle.query(points), actual)
    assert handle.stats()["queried_point_count"] == 2 * len(points)


@pytest.mark.parametrize("kind", ["sphere", "thin", "concave_components", "nested_components"])
def test_compiled_winding_matches_reference_on_topological_edge_cases(kind: str) -> None:
    trimesh = pytest.importorskip("trimesh")
    if kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    elif kind == "thin":
        mesh = trimesh.creation.box(extents=(4.0, 0.02, 2.0))
    elif kind == "concave_components":
        mesh = trimesh.util.concatenate(
            (
                trimesh.creation.box(extents=(2.0, 0.8, 0.8)),
                trimesh.creation.box(extents=(0.8, 2.0, 0.8)),
            )
        )
    else:
        mesh = trimesh.util.concatenate(
            (
                trimesh.creation.box(extents=(4.0, 4.0, 4.0)),
                trimesh.creation.box(extents=(1.0, 1.0, 1.0)),
            )
        )
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    rng = np.random.default_rng(20260730)
    points = np.ascontiguousarray(
        np.vstack((rng.uniform(-2.5, 2.5, size=(128, 3)), vertices[: min(8, len(vertices))])),
        dtype=np.float64,
    )
    handle = CompiledGeneralizedWindingHandle(vertices, faces)
    expected = generalized_winding_number(points, vertices[faces])
    actual = handle.query(points)
    expected_inside = np.abs(expected) >= 0.5
    actual_inside = np.abs(actual) >= 0.5
    np.testing.assert_allclose(actual, expected, atol=5e-11, rtol=0.0)
    np.testing.assert_array_equal(actual_inside, expected_inside)


def test_compiled_sign_reuses_only_lipschitz_certified_fd_probes() -> None:
    vertices, faces = _mesh()
    reference, _geometry = build_hybrid_signed_distance_backend(vertices, faces)
    compiled = CompiledSpatialFDBackend(reference, compiled_winding=True)
    points = np.ascontiguousarray(
        [[2.0, 0.0, 0.0], [-2.0, 0.1, -0.1], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    step = 1e-5
    axes = np.eye(3, dtype=np.float64)
    probes = np.ascontiguousarray(
        (points[:, None, :] + step * np.concatenate((axes, -axes))[None, :, :]).reshape(-1, 3)
    )
    expected = reference.query_local(probes, cache_update=False)
    actual = compiled.spatial_fd_gradient_scene(points, np.eye(4), step)
    np.testing.assert_allclose(
        actual.probe_result.signed_distance, expected.signed_distance, atol=1e-10, rtol=0.0
    )
    assert compiled.probe_sign_stats["certified_probe_reuse"] == 12
    assert compiled.probe_sign_stats["exact_probe_sign_calls"] == 6
    assert np.count_nonzero(actual.probe_result.sign_source == "CERTIFIED_FD_PROBE_REUSE") == 12


def test_compiled_handle_supports_concurrent_read_only_queries() -> None:
    vertices, faces = _mesh()
    handle = CompiledBVHHandle(vertices, faces)
    points = np.ascontiguousarray(np.linspace(-1.0, 1.0, 96).reshape(-1, 3), dtype=np.float64)

    def query() -> np.ndarray:
        return compiled_exact_query(handle, points)["unsigned_distance"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _index: query(), range(8)))
    for value in values[1:]:
        np.testing.assert_array_equal(value, values[0])
