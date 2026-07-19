import numpy as np
import pytest

from toporetarget.geometry.surface_sampling import (
    SurfaceSamplingProfile,
    load_surface_profile,
    sample_mesh_surface,
)


def _two_triangles() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [4, 0, 0], [4, 1, 0], [5, 0, 0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return vertices, faces


def test_paper_profile_resolves_count_from_paper_config() -> None:
    profile = load_surface_profile("paper_strict_area_uniform")
    assert profile.count == 50
    assert profile.source.endswith("num_object_surface_samples")
    assert profile.paper_status["method"] == "not_provided"


def test_sampling_is_deterministic_and_does_not_touch_global_rng() -> None:
    vertices, faces = _two_triangles()
    profile = SurfaceSamplingProfile("test", "1", "area_uniform_triangles", 50, 4)
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    first = sample_mesh_surface(vertices, faces, profile)
    observed = np.random.random()
    second = sample_mesh_surface(vertices, faces, profile)
    assert observed == expected
    np.testing.assert_array_equal(first.face_indices, second.face_indices)
    np.testing.assert_allclose(first.barycentric, second.barycentric)
    np.testing.assert_allclose(first.points_local, second.points_local)
    assert first.validate(vertices, faces)["point_reconstruction_max_error"] <= 1e-12


def test_scale_and_hash_validation(tmp_path) -> None:
    vertices, faces = _two_triangles()
    profile = SurfaceSamplingProfile("test", "1", "area_uniform_triangles", 8, 4)
    samples = sample_mesh_surface(vertices, faces, profile, scale=np.array([2.0, 1.0, 1.0]))
    path = tmp_path / "samples.npz"
    samples.save(path)
    loaded = type(samples).load(path, vertices=vertices, faces=faces)
    assert loaded.count == 8
    with pytest.raises(ValueError, match="hash mismatch"):
        type(samples).load(path, vertices=vertices * 2.0, faces=faces)


def test_all_degenerate_mesh_fails() -> None:
    vertices = np.zeros((3, 3))
    faces = np.array([[0, 1, 2]])
    profile = SurfaceSamplingProfile("test", "1", "area_uniform_triangles", 4, 0)
    with pytest.raises(ValueError, match="non-degenerate"):
        sample_mesh_surface(vertices, faces, profile)
