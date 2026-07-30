from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles
from toporetarget.geometry.signed_distance.compiled_sdf_cpu import (
    CompiledBVHHandle,
    CompiledSpatialFDBackend,
    compiled_available,
    compiled_exact_query,
)
from toporetarget.geometry.signed_distance.derived_proxy import build_hybrid_signed_distance_backend
from toporetarget.geometry.signed_distance.validation import make_synthetic_mesh

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
