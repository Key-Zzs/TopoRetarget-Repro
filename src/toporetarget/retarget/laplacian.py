"""Torch differentiable directed weighted Laplacian coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_torch(value: Any, *, dtype: Any | None = None, device: Any | None = None) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype or value.dtype, device=device or value.device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def sparse_weighted_laplacian(
    vertices: Any,
    source_index: Any,
    destination_index: Any,
    weights: Any,
) -> Any:
    """Compute ``Delta(V) = V - W V`` from directed COO edges.

    ``vertices`` may be ``[N,3]`` or ``[B,N,3]``.  The scatter operation is
    differentiable with respect to vertices and does not detach robot FK.
    """

    import torch

    value = _as_torch(vertices)
    if value.ndim not in {2, 3} or value.shape[-1] != 3:
        raise ValueError(f"vertices must have shape [N,3] or [B,N,3], got {tuple(value.shape)}")
    src = _as_torch(source_index, dtype=torch.long, device=value.device).reshape(-1)
    dst = _as_torch(destination_index, dtype=torch.long, device=value.device).reshape(-1)
    weight = _as_torch(weights, dtype=value.dtype, device=value.device).reshape(-1)
    if src.shape != dst.shape or src.shape != weight.shape:
        raise ValueError("directed index and weight arrays must have equal lengths")
    neighbour = value[..., dst, :] * weight[..., None]
    if value.ndim == 2:
        neighbour_sum = torch.zeros_like(value).index_add(0, src, neighbour)
    else:
        neighbour_sum = torch.zeros_like(value).index_add(1, src, neighbour)
    return value - neighbour_sum


def dense_weighted_laplacian(
    vertices: Any,
    source_index: Any,
    destination_index: Any,
    weights: Any,
    *,
    vertex_count: int | None = None,
) -> Any:
    """Dense reference implementation used by numerical tests only."""

    import torch

    value = _as_torch(vertices)
    n = int(vertex_count or value.shape[-2])
    src = _as_torch(source_index, dtype=torch.long, device=value.device).reshape(-1)
    dst = _as_torch(destination_index, dtype=torch.long, device=value.device).reshape(-1)
    weight = _as_torch(weights, dtype=value.dtype, device=value.device).reshape(-1)
    matrix = torch.zeros((n, n), dtype=value.dtype, device=value.device)
    matrix[src, dst] = weight
    identity = torch.eye(n, dtype=value.dtype, device=value.device)
    return torch.matmul(identity - matrix, value)


def laplacian_numpy(
    vertices: np.ndarray,
    source_index: np.ndarray,
    destination_index: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Small NumPy reference helper for artifact/report generation."""

    value = np.asarray(vertices, dtype=np.float64)
    result = value.copy()
    np.subtract.at(
        result,
        np.asarray(source_index),
        np.asarray(weights)[:, None] * value[np.asarray(destination_index)],
    )
    return result


__all__ = [
    "dense_weighted_laplacian",
    "laplacian_numpy",
    "sparse_weighted_laplacian",
]
