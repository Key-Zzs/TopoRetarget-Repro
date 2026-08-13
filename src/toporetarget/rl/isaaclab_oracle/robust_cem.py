"""Actual multi-horizon robust CEM core for Stage 16-C.5B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .robust_oracle import RobustCandidateEvaluationV2


@dataclass(frozen=True)
class RobustCEMConfigV1:
    horizons: tuple[int, ...] = (1, 5, 10)
    population: int = 32
    iterations: int = 3
    elites: int = 8
    replicas: int = 4
    action_dimension: int = 26
    initial_std: float = 0.35
    minimum_std: float = 0.05
    action_lower: float = -1.0
    action_upper: float = 1.0
    seed: int = 20260806

    def __post_init__(self) -> None:
        if self.horizons != (1, 5, 10):
            raise ValueError("C5B freezes horizons at [1, 5, 10]")
        if self.population not in {32, 48} or self.replicas not in {4, 8}:
            raise ValueError("C5B supports population 32/48 and replicas 4/8 only")
        if self.population == 48 and self.replicas == 8:
            raise ValueError("C5B permits only one upgrade: population=48 or replicas=8")
        if self.iterations != 3 or self.elites != 8:
            raise ValueError("C5B freezes three iterations and eight elites")
        if self.action_dimension != 26:
            raise ValueError("C5B requires the frozen 26D action")
        if self.initial_std != 0.35 or self.minimum_std != 0.05:
            raise ValueError("C5B freezes initial_std=0.35 and minimum_std=0.05")
        if self.action_lower != -1.0 or self.action_upper != 1.0:
            raise ValueError("C5B freezes normalized action bounds at [-1, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "RobustCEMConfigV1",
            "horizons": list(self.horizons),
            "population": self.population,
            "iterations": self.iterations,
            "elites": self.elites,
            "replicas": self.replicas,
            "action_dimension": self.action_dimension,
            "initial_std": self.initial_std,
            "minimum_std": self.minimum_std,
            "action_bounds": [self.action_lower, self.action_upper],
            "seed": self.seed,
            "upgrade": (
                "population_48"
                if self.population == 48
                else "replicas_8"
                if self.replicas == 8
                else "none"
            ),
        }


@dataclass
class _HorizonDistribution:
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class RobustCEMIterationRecordV1:
    iteration: int
    horizon: int
    elite_candidate_ids: list[int]
    best_lexical_key: list[float | int]
    mean_std: float
    minimum_std: float
    mean_shift_l2: float

    def as_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "horizon": self.horizon,
            "elite_candidate_ids": self.elite_candidate_ids,
            "best_lexical_key": self.best_lexical_key,
            "mean_std": self.mean_std,
            "minimum_std": self.minimum_std,
            "mean_shift_l2": self.mean_shift_l2,
        }


class RobustMultiHorizonCEMV1:
    """Maintain and update one real action-sequence distribution per horizon."""

    def __init__(
        self,
        config: RobustCEMConfigV1 | None = None,
        *,
        device: torch.device | str = "cpu",
        initial_means: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        self.config = config or RobustCEMConfigV1()
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.config.seed)
        self._distributions: dict[int, _HorizonDistribution] = {}
        for horizon in self.config.horizons:
            if initial_means is None or horizon not in initial_means:
                mean = torch.zeros(
                    (horizon, self.config.action_dimension),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                mean = initial_means[horizon].to(self.device, dtype=torch.float32).clone()
                if mean.shape != (horizon, self.config.action_dimension):
                    raise ValueError("CEM initial mean shape does not match its horizon")
                mean.clamp_(self.config.action_lower, self.config.action_upper)
            std = torch.full_like(mean, self.config.initial_std)
            self._distributions[horizon] = _HorizonDistribution(mean=mean, std=std)
        self.records: list[RobustCEMIterationRecordV1] = []
        self._last_samples: dict[int, torch.Tensor] = {}

    def ask(self, active_horizons: Sequence[int]) -> dict[int, torch.Tensor]:
        active = tuple(int(value) for value in active_horizons)
        if not active or any(value not in self.config.horizons for value in active):
            raise ValueError("CEM ask needs nonempty frozen active horizons")
        samples: dict[int, torch.Tensor] = {}
        for horizon in active:
            distribution = self._distributions[horizon]
            noise = torch.randn(
                (self.config.population, horizon, self.config.action_dimension),
                generator=self._generator,
                dtype=torch.float32,
                device=self.device,
            )
            values = distribution.mean[None] + distribution.std[None] * noise
            samples[horizon] = values.clamp(self.config.action_lower, self.config.action_upper)
        self._last_samples = samples
        return {horizon: values.clone() for horizon, values in samples.items()}

    def tell(
        self,
        iteration: int,
        evaluations: Mapping[int, Sequence[RobustCandidateEvaluationV2]],
    ) -> None:
        if iteration < 0 or iteration >= self.config.iterations:
            raise ValueError("CEM iteration outside frozen range")
        if set(evaluations) != set(self._last_samples):
            raise ValueError("CEM evaluations must match the most recent active horizons")
        for horizon, rows in evaluations.items():
            if len(rows) != self.config.population:
                raise ValueError("CEM needs one robust evaluation per sampled candidate")
            if {row.candidate_id for row in rows} != set(range(self.config.population)):
                raise ValueError("CEM evaluation candidate IDs are incomplete")
            if any(row.horizon != horizon for row in rows):
                raise ValueError("CEM evaluation horizon mismatch")
            ranked = sorted(rows, key=RobustCandidateEvaluationV2.lexical_key)
            elites = ranked[: self.config.elites]
            elite_ids = torch.tensor(
                [row.candidate_id for row in elites], dtype=torch.long, device=self.device
            )
            elite_actions = self._last_samples[horizon].index_select(0, elite_ids)
            distribution = self._distributions[horizon]
            old_mean = distribution.mean.clone()
            distribution.mean.copy_(elite_actions.mean(dim=0))
            distribution.std.copy_(
                elite_actions.std(dim=0, correction=0).clamp_min(self.config.minimum_std)
            )
            self.records.append(
                RobustCEMIterationRecordV1(
                    iteration=iteration,
                    horizon=horizon,
                    elite_candidate_ids=[row.candidate_id for row in elites],
                    best_lexical_key=list(elites[0].lexical_key()),
                    mean_std=float(distribution.std.mean().detach().cpu()),
                    minimum_std=float(distribution.std.min().detach().cpu()),
                    mean_shift_l2=float(
                        torch.linalg.vector_norm(distribution.mean - old_mean).detach().cpu()
                    ),
                )
            )

    def distribution(self, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        if horizon not in self._distributions:
            raise ValueError("unknown CEM horizon")
        value = self._distributions[horizon]
        return value.mean.clone(), value.std.clone()

    def warm_start_next_step(self) -> None:
        """Shift each horizon mean by one control step; never pad active rollouts."""

        for distribution in self._distributions.values():
            if distribution.mean.shape[0] > 1:
                distribution.mean[:-1].copy_(distribution.mean[1:].clone())
                distribution.mean[-1].copy_(distribution.mean[-2])
                distribution.std[:-1].copy_(distribution.std[1:].clone())
                distribution.std[-1].copy_(distribution.std[-2])

    def reset_step_records(self) -> None:
        self.records.clear()

    def convergence_report(self) -> dict[str, object]:
        return {
            "version": "RobustMultiHorizonCEMV1Convergence",
            "config": self.config.as_dict(),
            "records": [row.as_dict() for row in self.records],
        }


__all__ = [
    "RobustCEMConfigV1",
    "RobustCEMIterationRecordV1",
    "RobustMultiHorizonCEMV1",
]
