"""Distributional candidate-state qualification for Stage 16-C.5A-R4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .robust_metrics import (
    DistributionThresholdsV1,
    confidence_interval_distance,
    distribution_distances,
    freeze_natural_envelope,
    summarize_samples,
    termination_distribution_divergence,
    wilson_confidence_interval,
)

R4_PHASES = ("pre-contact", "contact-onset", "sustained-contact", "post-contact")
R4_DISTRIBUTION_FIELDS = (
    "object_pose",
    "object_velocity",
    "wrist_state",
    "finger_state",
    "tracked_links",
    "reward_components",
    "contact",
)
R4_METRICS = (
    "mean_difference",
    "variance_difference",
    "p95_difference",
    "wasserstein_distance",
    "mmd",
    "termination_distribution_divergence",
    "success_probability_confidence_interval",
)


def _array(value: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("R4 distribution field must be a finite [replica, feature] array")
    return result


@dataclass(frozen=True)
class DistributionalReplicationContractV1:
    """Metric and sampling choices frozen before collecting candidate results."""

    natural_replicas: int = 20
    candidate_replicas: int = 20
    confidence: float = 0.95
    envelope_multiplier: float = 2.0
    envelope_split_count: int = 20
    phases: tuple[str, ...] = R4_PHASES
    fields: tuple[str, ...] = R4_DISTRIBUTION_FIELDS
    metrics: tuple[str, ...] = R4_METRICS

    def __post_init__(self) -> None:
        if self.natural_replicas != 20 or self.candidate_replicas != 20:
            raise ValueError("R4 freezes natural and candidate qualification at 20 replicas")
        if self.confidence != 0.95 or self.envelope_multiplier != 2.0:
            raise ValueError("R4 freezes 95% confidence and a 2x natural envelope")
        if self.phases != R4_PHASES or self.fields != R4_DISTRIBUTION_FIELDS:
            raise ValueError("R4 phases and distribution fields are immutable")
        if self.metrics != R4_METRICS:
            raise ValueError("R4 metrics are immutable")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "DistributionalReplicationContractV1",
            "natural_replicas": self.natural_replicas,
            "candidate_replicas": self.candidate_replicas,
            "confidence": self.confidence,
            "envelope_multiplier": self.envelope_multiplier,
            "envelope_split_count": self.envelope_split_count,
            "phases": list(self.phases),
            "fields": list(self.fields),
            "metrics": list(self.metrics),
            "threshold_freeze_order": "natural_baseline_before_candidate_replication",
        }


@dataclass(frozen=True)
class DistributionPopulationV1:
    """One clip/phase replica population with the complete R4 field inventory."""

    clip: str
    phase: str
    reference_index: int
    fields: Mapping[str, np.ndarray]
    terminations: tuple[str, ...]
    successes: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.clip or self.phase not in R4_PHASES or self.reference_index < 0:
            raise ValueError("invalid R4 clip/phase identity")
        if tuple(self.fields) != R4_DISTRIBUTION_FIELDS:
            raise ValueError(
                "R4 population fields must be in the frozen order: "
                f"expected={R4_DISTRIBUTION_FIELDS} actual={tuple(self.fields)}"
            )
        arrays = [_array(value) for value in self.fields.values()]
        counts = {value.shape[0] for value in arrays}
        if len(counts) != 1:
            raise ValueError("R4 population fields must have one replica count")
        count = counts.pop()
        if len(self.terminations) != count or len(self.successes) != count:
            raise ValueError("R4 task distributions must align with state replicas")

    @property
    def replica_count(self) -> int:
        return next(iter(self.fields.values())).shape[0]

    def as_summary(self) -> dict[str, object]:
        interval = wilson_confidence_interval(sum(self.successes), len(self.successes))
        return {
            "clip": self.clip,
            "phase": self.phase,
            "reference_index": self.reference_index,
            "replica_count": self.replica_count,
            "fields": {name: summarize_samples(value) for name, value in self.fields.items()},
            "termination_distribution": {
                reason: self.terminations.count(reason) for reason in sorted(set(self.terminations))
            },
            "success_probability": sum(self.successes) / len(self.successes),
            "success_probability_confidence_interval_95": list(interval),
        }


@dataclass(frozen=True)
class NaturalPhysicsDistributionV1:
    """Natural PhysX population and candidate-independent frozen thresholds."""

    population: DistributionPopulationV1
    thresholds: DistributionThresholdsV1
    contract: DistributionalReplicationContractV1

    @classmethod
    def freeze(
        cls,
        population: DistributionPopulationV1,
        contract: DistributionalReplicationContractV1 | None = None,
    ) -> NaturalPhysicsDistributionV1:
        selected = contract or DistributionalReplicationContractV1()
        if population.replica_count != selected.natural_replicas:
            raise ValueError("natural population does not match the frozen R4 replica count")
        thresholds = freeze_natural_envelope(
            population.fields,
            population.terminations,
            population.successes,
            multiplier=selected.envelope_multiplier,
            split_count=selected.envelope_split_count,
        )
        return cls(population=population, thresholds=thresholds, contract=selected)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "NaturalPhysicsDistributionV1",
            "contract": self.contract.as_dict(),
            "population": self.population.as_summary(),
            "thresholds": self.thresholds.as_dict(),
        }


@dataclass(frozen=True)
class DistributionalReplicationGateV1:
    clip: str
    phase: str
    reference_index: int
    field_results: Mapping[str, Mapping[str, object]]
    termination_distribution_divergence: float
    termination_distribution_limit: float
    success_probability_confidence_interval_natural: tuple[float, float]
    success_probability_confidence_interval_candidate: tuple[float, float]
    success_interval_distance: float
    success_interval_distance_limit: float
    passes: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "DistributionalReplicationGateV1",
            "clip": self.clip,
            "phase": self.phase,
            "reference_index": self.reference_index,
            "field_results": dict(self.field_results),
            "termination_distribution_divergence": self.termination_distribution_divergence,
            "termination_distribution_limit": self.termination_distribution_limit,
            "success_probability_confidence_interval_natural": list(
                self.success_probability_confidence_interval_natural
            ),
            "success_probability_confidence_interval_candidate": list(
                self.success_probability_confidence_interval_candidate
            ),
            "success_interval_distance": self.success_interval_distance,
            "success_interval_distance_limit": self.success_interval_distance_limit,
            "passes": self.passes,
            "gate_rule": "all frozen distances <= 2x natural half-split p95 envelope",
        }


class DistributionalCandidateReplicatorV1:
    """Qualify already collected candidate replicas against a frozen baseline.

    Simulator state restoration remains in :mod:`candidate_state`; keeping the
    numerical gate separate makes it impossible for comparison code to write
    object, wrist, or controller state.
    """

    def __init__(self, contract: DistributionalReplicationContractV1 | None = None) -> None:
        self.contract = contract or DistributionalReplicationContractV1()

    def qualify(
        self,
        baseline: NaturalPhysicsDistributionV1,
        candidate: DistributionPopulationV1,
    ) -> DistributionalReplicationGateV1:
        natural = baseline.population
        if baseline.contract != self.contract:
            raise ValueError("candidate qualifier contract differs from frozen baseline")
        if (candidate.clip, candidate.phase, candidate.reference_index) != (
            natural.clip,
            natural.phase,
            natural.reference_index,
        ):
            raise ValueError("candidate and natural populations must identify the same boundary")
        if candidate.replica_count != self.contract.candidate_replicas:
            raise ValueError("candidate population does not match the frozen R4 replica count")
        results: dict[str, Mapping[str, object]] = {}
        all_pass = True
        for name in self.contract.fields:
            distances = distribution_distances(
                natural.fields[name], candidate.fields[name], scale_reference=natural.fields[name]
            ).as_dict()
            checks = {
                metric: value <= baseline.thresholds.metric_limits[metric]
                for metric, value in distances.items()
            }
            field_pass = all(checks.values())
            all_pass = all_pass and field_pass
            results[name] = {
                "distances": distances,
                "limits": dict(baseline.thresholds.metric_limits),
                "checks": checks,
                "passes": field_pass,
            }
        termination = termination_distribution_divergence(
            natural.terminations, candidate.terminations
        )
        natural_interval = wilson_confidence_interval(
            sum(natural.successes), len(natural.successes)
        )
        candidate_interval = wilson_confidence_interval(
            sum(candidate.successes), len(candidate.successes)
        )
        interval_distance = confidence_interval_distance(natural_interval, candidate_interval)
        all_pass = (
            all_pass
            and termination <= baseline.thresholds.termination_divergence_limit
            and interval_distance <= baseline.thresholds.success_interval_distance_limit
        )
        return DistributionalReplicationGateV1(
            clip=candidate.clip,
            phase=candidate.phase,
            reference_index=candidate.reference_index,
            field_results=results,
            termination_distribution_divergence=termination,
            termination_distribution_limit=baseline.thresholds.termination_divergence_limit,
            success_probability_confidence_interval_natural=natural_interval,
            success_probability_confidence_interval_candidate=candidate_interval,
            success_interval_distance=interval_distance,
            success_interval_distance_limit=baseline.thresholds.success_interval_distance_limit,
            passes=all_pass,
        )


__all__ = [
    "DistributionPopulationV1",
    "DistributionalCandidateReplicatorV1",
    "DistributionalReplicationContractV1",
    "DistributionalReplicationGateV1",
    "NaturalPhysicsDistributionV1",
    "R4_DISTRIBUTION_FIELDS",
    "R4_METRICS",
    "R4_PHASES",
]
