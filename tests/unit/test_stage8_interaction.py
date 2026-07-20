from __future__ import annotations

import numpy as np
import pytest
import torch

from toporetarget.retarget.delaunay import (
    DelaunayValidationError,
    edge_category,
    extract_unique_edges,
    load_delaunay_profile,
    tetrahedralize,
)
from toporetarget.retarget.graph_weights import build_source_weights, direct_source_weights
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_graph,
    save_interaction_graph,
)
from toporetarget.retarget.interaction_graph import (
    INTERACTION_GRAPH_SCHEMA_VERSION,
    InteractionGraphTrajectory,
)
from toporetarget.retarget.interaction_objective import (
    InteractionMeshObjective,
    InteractionMeshResidual,
)
from toporetarget.retarget.laplacian import (
    dense_weighted_laplacian,
    laplacian_numpy,
    sparse_weighted_laplacian,
)


def _ring_vertices(count: int = 71) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    return np.stack([0.001 * index, 0.002 * np.sin(index), 0.003 * np.cos(index)], axis=-1)


def _ring_edges(count: int = 71) -> np.ndarray:
    return np.stack([np.arange(count), (np.arange(count) + 1) % count], axis=-1)


def test_strict_delaunay_and_edge_extraction_are_deterministic() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    profile = load_delaunay_profile()
    first = tetrahedralize(points, profile, expected_count=4)
    second = tetrahedralize(points, profile, expected_count=4)
    np.testing.assert_array_equal(first.simplices, second.simplices)
    assert first.simplices.shape == (1, 4)
    assert np.all(first.simplex_volumes > 0)
    edges = extract_unique_edges(first.simplices, vertex_count=4)
    assert edges.shape == (6, 2)
    np.testing.assert_array_equal(edges, np.sort(edges, axis=1))
    duplicate = points.copy()
    duplicate[3] = duplicate[0]
    with pytest.raises(DelaunayValidationError, match="exact duplicate"):
        tetrahedralize(duplicate, profile, expected_count=4)


def test_delaunay_preserves_vertex_set_under_rigid_transform() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    angle = 0.37
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = points @ rotation.T + np.asarray([4.0, -2.0, 3.0])
    profile = load_delaunay_profile()
    first = tetrahedralize(points, profile, expected_count=4)
    second = tetrahedralize(transformed, profile, expected_count=4)
    np.testing.assert_array_equal(
        np.sort(first.simplices, axis=1), np.sort(second.simplices, axis=1)
    )
    np.testing.assert_array_equal(
        extract_unique_edges(first.simplices, vertex_count=4),
        extract_unique_edges(second.simplices, vertex_count=4),
    )
    assert not profile.jitter
    assert edge_category((50, 0)) == "hand-object"


def test_strict_delaunay_rejects_coplanar_source_without_jitter() -> None:
    points = np.asarray([[i, j, 0.0] for i, j in ((0, 0), (1, 0), (0, 1), (1, 1))], dtype=float)
    with pytest.raises(DelaunayValidationError, match="affine rank"):
        tetrahedralize(points, load_delaunay_profile(), expected_count=4)


def test_source_weights_are_directed_and_row_normalized() -> None:
    vertices = _ring_vertices()
    edges = _ring_edges()
    stable = build_source_weights(vertices, edges, 30.0)
    direct = direct_source_weights(vertices, edges, 30.0)
    stable.validate()
    assert stable.directed_count == 2 * len(edges)
    np.testing.assert_allclose(stable.row_sums, 1.0, atol=1e-14)
    np.testing.assert_allclose(stable.weights, direct.weights, atol=1e-14)
    assert not np.allclose(stable.weights[: len(edges)], stable.weights[len(edges) :])


def test_sparse_laplacian_matches_dense_and_preserves_autograd() -> None:
    vertices = _ring_vertices()
    directed = build_source_weights(vertices, _ring_edges(), 30.0)
    value = torch.tensor(vertices, dtype=torch.float64, requires_grad=True)
    sparse = sparse_weighted_laplacian(
        value, directed.source_index, directed.destination_index, directed.weights
    )
    dense = dense_weighted_laplacian(
        value, directed.source_index, directed.destination_index, directed.weights
    )
    torch.testing.assert_close(sparse, dense)
    batch = sparse_weighted_laplacian(
        value.unsqueeze(0), directed.source_index, directed.destination_index, directed.weights
    )
    assert batch.shape == (1, 71, 3)
    sparse.square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_eq7_identity_zero_and_exact_mean_squared_scaling() -> None:
    vertices = _ring_vertices()
    directed = build_source_weights(vertices, _ring_edges(), 30.0)
    model = InteractionMeshResidual(
        vertices, directed.source_index, directed.destination_index, directed.weights
    )
    objective = InteractionMeshObjective(model)
    identity = torch.tensor(vertices, dtype=torch.float64)
    assert float(objective(identity)) == pytest.approx(0.0, abs=1e-28)
    robot = identity.clone()
    robot[0, 0] += 0.01
    residual = objective.residual_tensor(robot)
    expected = residual.square().sum() / 71.0
    assert float(objective.loss_tensor(robot)) == pytest.approx(float(expected), rel=1e-14)
    assert objective.scaled_residual_tensor(robot).shape == (213,)
    batched = objective.loss_tensor(torch.stack([identity, robot]))
    assert batched.shape == (2,)


def test_ragged_graph_artifact_round_trip(tmp_path) -> None:
    source = _ring_vertices()
    directed = build_source_weights(source, _ring_edges(), 30.0)
    trajectory = InteractionGraphTrajectory(
        metadata={
            "schema_version": INTERACTION_GRAPH_SCHEMA_VERSION,
            "frame_status": ["valid"],
            "frame_statistics": [{}],
            "graph_hashes": ["synthetic"],
            "source_vertex_metadata": [],
        },
        timestamps=np.asarray([0.0]),
        source_vertices=source[None],
        source_laplacian=laplacian_numpy(
            source, directed.source_index, directed.destination_index, directed.weights
        )[None],
        simplex_frames=[np.asarray([[0, 1, 2, 3]], dtype=np.int64)],
        edge_frames=[_ring_edges()],
        directed_frames=[directed],
        frame_statistics=[{}],
        frame_valid=np.asarray([True]),
        frame_status=["valid"],
        frame_indices=np.asarray([0]),
        object_face_indices=np.arange(50, dtype=np.int64),
        object_barycentric=np.full((50, 3), 1.0 / 3.0),
        graph_hashes=["synthetic"],
        source_vertex_metadata=[],
    )
    path = tmp_path / "graph.zarr"
    save_interaction_graph(trajectory, path)
    loaded = load_interaction_graph(path)
    assert interaction_artifact_hash(path) == loaded.artifact_hash
    np.testing.assert_array_equal(loaded.simplex_frames[0], trajectory.simplex_frames[0])
    np.testing.assert_array_equal(loaded.edge_frames[0], trajectory.edge_frames[0])
    np.testing.assert_allclose(loaded.directed_frames[0].weights, directed.weights)
    np.testing.assert_array_equal(loaded.directed_frames[0].row_offsets, directed.row_offsets)
