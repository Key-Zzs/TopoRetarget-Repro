"""Bounded phase-wise robust spline CEM and semantic candidate ranking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

import torch

from .spline_actions import PiecewiseSplineResidualV1


def upper_cvar(values: tuple[float, ...], alpha: float = 0.8) -> float:
    if not values or not 0.0 < alpha < 1.0:
        raise ValueError("CVaR requires values and alpha in (0,1)")
    ordered = sorted(float(value) for value in values)
    count = max(1, math.ceil((1.0 - alpha) * len(ordered)))
    return fmean(ordered[-count:])


@dataclass(frozen=True)
class PhysicsCandidateReplicaV1:
    catastrophic_failure: bool
    semantic_failure: bool
    contact_topology_failure: bool
    penetration_m: float
    safety_violation: float
    semantic_progress: float
    contact_recall: float
    contact_persistence: float
    terminal_stability: float
    robot_fidelity_error: float
    source_object_soft_prior_error: float
    action_smoothness: float
    effort: float

    def __post_init__(self) -> None:
        values = tuple(
            float(getattr(self, name))
            for name in (
                "penetration_m",
                "safety_violation",
                "semantic_progress",
                "contact_recall",
                "contact_persistence",
                "terminal_stability",
                "robot_fidelity_error",
                "source_object_soft_prior_error",
                "action_smoothness",
                "effort",
            )
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("candidate replica metrics must be finite and nonnegative")


@dataclass(frozen=True)
class PhysicsCandidateEvaluationV1:
    candidate_id: int
    replicas: tuple[PhysicsCandidateReplicaV1, ...]

    def lexical_key(self) -> tuple[float | int, ...]:
        rows = self.replicas
        if not rows:
            raise ValueError("robust evaluation requires replicas")
        return (
            fmean(float(row.catastrophic_failure) for row in rows),
            fmean(float(row.semantic_failure) for row in rows),
            fmean(float(row.contact_topology_failure) for row in rows),
            upper_cvar(tuple(row.penetration_m for row in rows)),
            upper_cvar(tuple(row.safety_violation for row in rows)),
            -fmean(row.semantic_progress for row in rows),
            -fmean(row.contact_recall for row in rows),
            -fmean(row.contact_persistence for row in rows),
            -fmean(row.terminal_stability for row in rows),
            fmean(row.robot_fidelity_error for row in rows),
            fmean(row.source_object_soft_prior_error for row in rows),
            fmean(row.action_smoothness for row in rows),
            fmean(row.effort for row in rows),
            self.candidate_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "PhysicsCandidateEvaluationV1",
            "candidate_id": self.candidate_id,
            "replicas": [asdict(row) for row in self.replicas],
            "lexical_key": list(self.lexical_key()),
        }


@dataclass(frozen=True)
class PhaseWiseSplineCEMConfigV1:
    knot_count: int = 16
    population: int = 64
    replicas: int = 4
    iterations: int = 5
    elites: int = 12
    initial_std: float = 0.25
    minimum_std: float = 0.03
    seed: int = 20260806

    def __post_init__(self) -> None:
        valid = (
            (16, 64, 4, 5, 12),
            (32, 96, 4, 8, 12),
            (32, 96, 8, 8, 12),
            (32, 128, 4, 8, 12),
            (64, 96, 8, 8, 12),
            (64, 128, 4, 8, 12),
        )
        signature = (
            self.knot_count,
            self.population,
            self.replicas,
            self.iterations,
            self.elites,
        )
        if signature not in valid:
            raise ValueError("Stage16D CEM config exceeds the frozen decision tree")
        if self.initial_std != 0.25 or self.minimum_std != 0.03:
            raise ValueError("Stage16D CEM std contract is frozen")
        if self.replicas == 8 and self.population == 128:
            raise ValueError("replica and population final upgrades are mutually exclusive")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "PhaseWiseSplineCEMConfigV1", **asdict(self)}


class PhaseWiseRobustSplineCEMV1:
    """Distribution update core; the object is never an optimization variable."""

    def __init__(self, config: PhaseWiseSplineCEMConfigV1, *, device: str = "cpu") -> None:
        self.config = config
        self.parameterization = PiecewiseSplineResidualV1(knot_count=config.knot_count)
        self.device = torch.device(device)
        self.mean = torch.zeros((config.knot_count, 26), device=self.device)
        self.std = torch.full_like(self.mean, config.initial_std)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.records: list[dict[str, Any]] = []

    def ask(self) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(
            (self.config.population, self.config.knot_count, 26),
            generator=self.generator,
            device=self.device,
        )
        knots = (self.mean[None] + self.std[None] * noise).clamp(-1.0, 1.0)
        return knots, self.parameterization.materialize(knots)

    def tell(
        self,
        *,
        iteration: int,
        knots: torch.Tensor,
        evaluations: tuple[PhysicsCandidateEvaluationV1, ...],
    ) -> None:
        if iteration < 0 or iteration >= self.config.iterations:
            raise ValueError("CEM iteration is outside the frozen budget")
        if knots.shape != (self.config.population, self.config.knot_count, 26):
            raise ValueError("CEM knot population shape mismatch")
        if len(evaluations) != self.config.population:
            raise ValueError("CEM needs one evaluation per candidate")
        ranked = sorted(evaluations, key=PhysicsCandidateEvaluationV1.lexical_key)
        elite_ids = torch.tensor(
            [row.candidate_id for row in ranked[: self.config.elites]],
            dtype=torch.long,
            device=self.device,
        )
        elite = knots.index_select(0, elite_ids)
        previous = self.mean.clone()
        self.mean = elite.mean(dim=0)
        self.std = elite.std(dim=0, correction=0).clamp_min(self.config.minimum_std)
        self.records.append(
            {
                "iteration": iteration,
                "elite_candidate_ids": elite_ids.cpu().tolist(),
                "best_lexical_key": list(ranked[0].lexical_key()),
                "mean_shift_l2": float(torch.linalg.vector_norm(self.mean - previous).cpu()),
                "mean_std": float(self.std.mean().cpu()),
                "minimum_std": float(self.std.min().cpu()),
            }
        )

    def best_action_trace(self) -> torch.Tensor:
        return self.parameterization.materialize(self.mean)


__all__ = [
    "PhaseWiseRobustSplineCEMV1",
    "PhaseWiseSplineCEMConfigV1",
    "PhysicsCandidateEvaluationV1",
    "PhysicsCandidateReplicaV1",
    "upper_cvar",
]
