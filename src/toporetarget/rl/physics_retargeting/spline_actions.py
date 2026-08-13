"""Bounded continuous time parameterization for Stage 16-D actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PiecewiseSplineResidualV1:
    frame_count: int = 321
    action_dimension: int = 26
    knot_count: int = 16
    action_lower: float = -1.0
    action_upper: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_count != 321:
            raise ValueError("Stage16D spline must produce exactly 321 actions")
        if self.action_dimension != 26:
            raise ValueError("Stage16D spline action dimension is frozen at 26")
        if self.knot_count not in {16, 32, 64}:
            raise ValueError("Stage16D knot count must be 16, 32, or 64")
        if self.action_lower != -1.0 or self.action_upper != 1.0:
            raise ValueError("Stage16D action range is frozen at [-1,1]")

    def materialize(self, knots: torch.Tensor) -> torch.Tensor:
        if knots.shape[-2:] != (self.knot_count, self.action_dimension):
            raise ValueError("spline knot tensor has incompatible shape")
        if not bool(torch.isfinite(knots).all()):
            raise ValueError("spline knots must be finite")
        bounded = knots.clamp(self.action_lower, self.action_upper)
        coordinate = torch.linspace(
            0.0,
            self.knot_count - 1,
            self.frame_count,
            dtype=bounded.dtype,
            device=bounded.device,
        )
        left = torch.floor(coordinate).long().clamp_max(self.knot_count - 1)
        right = (left + 1).clamp_max(self.knot_count - 1)
        alpha = coordinate - left.to(coordinate.dtype)
        prefix = (1,) * (bounded.ndim - 2)
        weight = alpha.reshape(prefix + (self.frame_count, 1))
        return ((1.0 - weight) * bounded[..., left, :] + weight * bounded[..., right, :]).clamp(
            self.action_lower, self.action_upper
        )

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "PiecewiseSplineResidualV1", **asdict(self)}


__all__ = ["PiecewiseSplineResidualV1"]
