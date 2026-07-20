"""Eq. (6)-(7) weighted interaction residuals and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .laplacian import sparse_weighted_laplacian

GRAPH_VERTEX_COUNT = 71


@dataclass
class InteractionMeshResidual:
    """Torch-native Eq. (6) residual with source graph data held constant."""

    source_vertices: np.ndarray
    source_index: np.ndarray
    destination_index: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_vertices, dtype=np.float64)
        if source.shape != (GRAPH_VERTEX_COUNT, 3):
            raise ValueError(f"source_vertices must have shape [{GRAPH_VERTEX_COUNT},3]")
        self.source_vertices = source
        self.source_index = np.asarray(self.source_index, dtype=np.int64).reshape(-1)
        self.destination_index = np.asarray(self.destination_index, dtype=np.int64).reshape(-1)
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if not (self.source_index.shape == self.destination_index.shape == self.weights.shape):
            raise ValueError("directed graph arrays must have equal lengths")
        self._source_laplacian_cache: Any | None = None

    def source_laplacian(self) -> Any:
        if self._source_laplacian_cache is None:
            self._source_laplacian_cache = sparse_weighted_laplacian(
                self.source_vertices,
                self.source_index,
                self.destination_index,
                self.weights,
            )
        return self._source_laplacian_cache

    def __call__(self, robot_vertices: Any) -> Any:
        import torch

        value = robot_vertices
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float64)
        if value.shape[-2:] != (GRAPH_VERTEX_COUNT, 3):
            raise ValueError(
                f"robot_vertices must end in [{GRAPH_VERTEX_COUNT},3], got {tuple(value.shape)}"
            )
        source = torch.as_tensor(self.source_vertices, dtype=value.dtype, device=value.device)
        source_laplacian = sparse_weighted_laplacian(
            source,
            self.source_index,
            self.destination_index,
            self.weights,
        )
        robot_laplacian = sparse_weighted_laplacian(
            value,
            self.source_index,
            self.destination_index,
            self.weights,
        )
        return robot_laplacian - source_laplacian

    def source_laplacian_tensor(self, *, dtype: Any, device: Any) -> Any:
        import torch

        source = torch.as_tensor(self.source_vertices, dtype=dtype, device=device)
        return sparse_weighted_laplacian(
            source, self.source_index, self.destination_index, self.weights
        )


@dataclass
class InteractionMeshObjective:
    """Eq. (7), intentionally containing no other retargeting term."""

    residual_model: InteractionMeshResidual

    def residual_tensor(self, robot_vertices: Any) -> Any:
        return self.residual_model(robot_vertices)

    def scaled_residual_tensor(self, robot_vertices: Any) -> Any:
        import torch

        residual = self.residual_tensor(robot_vertices)
        return residual.reshape(*residual.shape[:-2], -1) / torch.sqrt(
            torch.as_tensor(71.0, dtype=residual.dtype, device=residual.device)
        )

    def loss_tensor(self, robot_vertices: Any) -> Any:
        residual = self.residual_tensor(robot_vertices)
        return (residual.square().sum(dim=(-2, -1))) / float(GRAPH_VERTEX_COUNT)

    def __call__(self, robot_vertices: Any) -> Any:
        return self.loss_tensor(robot_vertices)

    def diagnostics(
        self,
        robot_vertices: Any,
        *,
        vertex_names: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        import torch

        residual = self.residual_tensor(robot_vertices)
        loss = self.loss_tensor(robot_vertices)
        per_vertex = residual.square().sum(dim=-1)  # [..., 71]
        per_kind = {
            "hand": per_vertex[..., :21].sum(dim=-1) / float(GRAPH_VERTEX_COUNT),
            "object": per_vertex[..., 21:].sum(dim=-1) / float(GRAPH_VERTEX_COUNT),
        }
        per_vertex_contribution = per_vertex / float(GRAPH_VERTEX_COUNT)
        flat = per_vertex_contribution.reshape(-1, GRAPH_VERTEX_COUNT)
        max_indices = torch.argmax(flat, dim=-1)
        names = list(vertex_names or [f"vertex_{i}" for i in range(GRAPH_VERTEX_COUNT)])
        if len(names) != GRAPH_VERTEX_COUNT:
            raise ValueError("vertex_names must contain exactly 71 names")
        return {
            "loss": loss,
            "residual": residual,
            "source_laplacian_norm": torch.linalg.vector_norm(
                self.residual_model.source_laplacian_tensor(
                    dtype=residual.dtype, device=residual.device
                ),
                dim=(-2, -1),
            ),
            "robot_laplacian_norm": torch.linalg.vector_norm(
                residual
                + self.residual_model.source_laplacian_tensor(
                    dtype=residual.dtype, device=residual.device
                ),
                dim=(-2, -1),
            ),
            "per_vertex_contribution": per_vertex_contribution,
            "per_kind_contribution": per_kind,
            "per_hand_point_contribution": per_vertex_contribution[..., :21],
            "per_object_point_contribution": per_vertex_contribution[..., 21:],
            "per_coordinate_contribution": residual.square() / float(GRAPH_VERTEX_COUNT),
            "max_residual_vertex": max_indices,
            "max_residual_vertex_name": [names[int(i)] for i in max_indices.reshape(-1)],
        }


def interaction_loss_numpy(
    source_vertices: np.ndarray,
    robot_vertices: np.ndarray,
    source_index: np.ndarray,
    destination_index: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convenience NumPy boundary for reports; Torch remains the canonical path."""

    residual_model = InteractionMeshResidual(
        np.asarray(source_vertices), source_index, destination_index, weights
    )
    import torch

    residual = residual_model(torch.as_tensor(robot_vertices, dtype=torch.float64))
    return residual.detach().cpu().numpy(), float((residual.square().sum() / 71.0).detach().cpu())


__all__ = [
    "GRAPH_VERTEX_COUNT",
    "InteractionMeshObjective",
    "InteractionMeshResidual",
    "interaction_loss_numpy",
]
