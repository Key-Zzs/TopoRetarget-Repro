"""Frozen G1/G2 budgets for geometry-aware Stage 16-D CEM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryAwareOptimizerConfigV2:
    level: str
    knots: int
    population: int
    replicas: int
    iterations: int
    elites: int = 12
    selection_holdout_replicas: int = 8
    formal_holdout_replicas: int = 20
    window_margin: int = 10

    def __post_init__(self) -> None:
        expected = {
            "G1": (16, 96, 4, 8, 12),
            "G2": (32, 96, 8, 8, 12),
        }
        signature = (self.knots, self.population, self.replicas, self.iterations, self.elites)
        if self.level not in expected or signature != expected[self.level]:
            raise ValueError("geometry-aware optimizer must use the frozen G1/G2 budget")
        if self.selection_holdout_replicas != 8 or self.formal_holdout_replicas != 20:
            raise ValueError("geometry-aware optimizer holdout counts are frozen at 8/20")
        if self.window_margin != 10:
            raise ValueError("geometry-aware optimizer window margin is frozen at 10")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "GeometryAwareOptimizerConfigV2", **asdict(self)}


G1_CONFIG = GeometryAwareOptimizerConfigV2("G1", 16, 96, 4, 8)
G2_CONFIG = GeometryAwareOptimizerConfigV2("G2", 32, 96, 8, 8)


__all__ = ["G1_CONFIG", "G2_CONFIG", "GeometryAwareOptimizerConfigV2"]
