"""Source-derived directed distance weights for the Stage 8 graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GraphWeightError(ValueError):
    """Raised when graph weights cannot be constructed."""


@dataclass(frozen=True)
class DirectedGraphWeights:
    source_index: np.ndarray
    destination_index: np.ndarray
    weights: np.ndarray
    log_unnormalized: np.ndarray
    distance_squared: np.ndarray
    row_offsets: np.ndarray
    row_sums: np.ndarray

    @property
    def directed_count(self) -> int:
        return int(self.weights.shape[0])

    def validate(self, vertex_count: int = 71, *, atol: float = 1e-12) -> DirectedGraphWeights:
        n = self.directed_count
        if self.source_index.shape != (n,) or self.destination_index.shape != (n,):
            raise GraphWeightError("directed index arrays have inconsistent shapes")
        if self.weights.shape != (n,) or self.distance_squared.shape != (n,):
            raise GraphWeightError("directed weight arrays have inconsistent shapes")
        if self.row_offsets.shape != (vertex_count + 1,):
            raise GraphWeightError("row_offsets must have one entry per vertex plus a sentinel")
        if self.row_offsets[0] != 0 or self.row_offsets[-1] != n:
            raise GraphWeightError("row_offsets do not cover all directed edges")
        if np.any(self.source_index < 0) or np.any(self.source_index >= vertex_count):
            raise GraphWeightError("directed source index out of range")
        if np.any(self.destination_index < 0) or np.any(self.destination_index >= vertex_count):
            raise GraphWeightError("directed destination index out of range")
        if np.any(self.source_index == self.destination_index):
            raise GraphWeightError("self weights are not allowed")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise GraphWeightError("weights must be finite and non-negative")
        if not np.allclose(self.row_sums, 1.0, atol=atol, rtol=0):
            raise GraphWeightError(
                f"row sums do not equal one: max error {np.max(np.abs(self.row_sums - 1))}"
            )
        return self


def _directed_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(edges, dtype=np.int64)
    if value.ndim != 2 or value.shape[1:] != (2,):
        raise GraphWeightError(f"edges must have shape [E,2], got {value.shape}")
    forward = value
    reverse = value[:, ::-1]
    directed = np.concatenate([forward, reverse], axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]
    if len(directed) and np.any(directed[:, 0] == directed[:, 1]):
        raise GraphWeightError("self edges cannot form an adjacency")
    return directed[:, 0], directed[:, 1]


def build_source_weights(
    source_vertices: np.ndarray,
    edges: np.ndarray,
    kappa: float,
    *,
    vertex_count: int = 71,
) -> DirectedGraphWeights:
    """Compute Eq. (5) once from source vertices and row-normalize it."""

    vertices = np.asarray(source_vertices, dtype=np.float64)
    if vertices.shape != (vertex_count, 3):
        raise GraphWeightError(f"source vertices must have shape [{vertex_count},3]")
    if not np.isfinite(kappa) or kappa < 0:
        raise GraphWeightError("kappa must be finite and non-negative")
    source_index, destination_index = _directed_edges(edges)
    difference = vertices[source_index] - vertices[destination_index]
    distance_squared = np.einsum("ij,ij->i", difference, difference)
    log_unnormalized = -float(kappa) * distance_squared
    weights = np.zeros_like(log_unnormalized)
    row_offsets = np.zeros(vertex_count + 1, dtype=np.int64)
    row_sums = np.zeros(vertex_count, dtype=np.float64)
    for row in range(vertex_count):
        mask = source_index == row
        row_offsets[row + 1] = row_offsets[row] + int(np.count_nonzero(mask))
        if not np.any(mask):
            raise GraphWeightError(f"vertex {row} has no neighbors")
        values = log_unnormalized[mask]
        shifted = np.exp(values - np.max(values))
        total = float(np.sum(shifted))
        if not np.isfinite(total) or total <= 0:
            raise GraphWeightError(f"weight normalization failed for vertex {row}")
        weights[mask] = shifted / total
        row_sums[row] = float(np.sum(weights[mask]))
    return DirectedGraphWeights(
        source_index=source_index,
        destination_index=destination_index,
        weights=weights,
        log_unnormalized=log_unnormalized,
        distance_squared=distance_squared,
        row_offsets=row_offsets,
        row_sums=row_sums,
    ).validate(vertex_count)


def direct_source_weights(
    source_vertices: np.ndarray, edges: np.ndarray, kappa: float
) -> DirectedGraphWeights:
    """Reference implementation using the direct exponential formula."""

    result = build_source_weights(source_vertices, edges, kappa)
    direct = np.exp(result.log_unnormalized)
    weights = np.zeros_like(direct)
    for row in range(result.row_offsets.shape[0] - 1):
        start, stop = result.row_offsets[row : row + 2]
        values = direct[start:stop]
        total = np.sum(values)
        if total == 0:
            # The reference path deliberately reports underflow rather than
            # changing Eq. (5); the stable implementation remains usable.
            continue
        weights[start:stop] = values / total
    return DirectedGraphWeights(
        source_index=result.source_index.copy(),
        destination_index=result.destination_index.copy(),
        weights=weights,
        log_unnormalized=result.log_unnormalized.copy(),
        distance_squared=result.distance_squared.copy(),
        row_offsets=result.row_offsets.copy(),
        row_sums=np.add.reduceat(weights, result.row_offsets[:-1]),
    )


__all__ = [
    "DirectedGraphWeights",
    "GraphWeightError",
    "build_source_weights",
    "direct_source_weights",
]
