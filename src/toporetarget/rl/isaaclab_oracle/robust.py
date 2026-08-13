"""Frozen statistical contract for a stochastic Stage 16-C.5 Oracle.

These structures score independently produced PhysX replicas.  They neither
alter a rollout nor write simulator state; the runtime launcher remains
responsible for making every replica a fresh process/scene from frame zero.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

OBJECT_POSITION_GATE_M = 0.02
OBJECT_ROTATION_GATE_DEG = 10.0
OBJECT_AXIS_GATE_M = 0.03
SUCCESS_RATE_GATE = 0.90
FINAL_REACH_GATE = 0.90


def percentile(values: Sequence[float], probability: float) -> float:
    """Deterministic linear percentile for finite nonempty scalar samples."""

    if not values:
        raise ValueError("robust statistic needs at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("robust statistic values must be finite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def upper_cvar(values: Sequence[float], alpha: float) -> float:
    """Mean cost in the inclusive upper ``1-alpha`` tail."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("CVaR alpha must be strictly between zero and one")
    if not values:
        raise ValueError("CVaR needs at least one sample")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("CVaR values must be finite")
    tail_start = max(0, math.ceil(alpha * len(ordered)) - 1)
    return fmean(ordered[tail_start:])


@dataclass(frozen=True)
class RobustOracleContractV1:
    """R3 robust objective frozen before a candidate is evaluated."""

    replica_count: int = 4
    cvar_alpha: float = 0.8
    std_lambda: float = 1.0
    objective: str = "mean_plus_lambda_std_v1"

    def __post_init__(self) -> None:
        if self.replica_count not in {1, 4, 8}:
            raise ValueError("R3 benchmark and qualification permit replicas 1, 4, or 8")
        if self.cvar_alpha != 0.8:
            raise ValueError("R3 freezes CVaR alpha at 0.8")
        if self.std_lambda < 0.0 or not math.isfinite(self.std_lambda):
            raise ValueError("robust standard-deviation lambda must be finite and nonnegative")
        if self.objective not in {"mean_plus_lambda_std_v1", "cvar_alpha_v1"}:
            raise ValueError("unknown robust objective")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "RobustOracleContractV1",
            "replica_count": self.replica_count,
            "cvar_alpha": self.cvar_alpha,
            "std_lambda": self.std_lambda,
            "objective": self.objective,
            "physical_gates": {
                "object_position_m": OBJECT_POSITION_GATE_M,
                "object_rotation_deg": OBJECT_ROTATION_GATE_DEG,
                "object_axis_m": OBJECT_AXIS_GATE_M,
                "success_rate": SUCCESS_RATE_GATE,
                "final_reach_rate": FINAL_REACH_GATE,
            },
        }


@dataclass(frozen=True)
class RobustReplicaResultV1:
    """One independently executed candidate rollout, already measured."""

    cost: float
    object_position_error_m: float
    object_rotation_error_deg: float
    object_axis_error_m: float
    success: bool
    final_reach: bool
    contact_stability_penalty: float
    action_smoothness: float
    effort: float
    termination_reason: str

    def __post_init__(self) -> None:
        values = (
            self.cost,
            self.object_position_error_m,
            self.object_rotation_error_deg,
            self.object_axis_error_m,
            self.contact_stability_penalty,
            self.action_smoothness,
            self.effort,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("robust replica fields must be finite")
        if (
            min(
                self.object_position_error_m,
                self.object_rotation_error_deg,
                self.object_axis_error_m,
            )
            < 0
        ):
            raise ValueError("physical error metrics cannot be negative")
        if min(self.contact_stability_penalty, self.action_smoothness, self.effort) < 0.0:
            raise ValueError("robust selector penalties cannot be negative")
        if not self.termination_reason:
            raise ValueError("replica termination reason is required")

    @property
    def normalized_gate_margin(self) -> float:
        """Largest normalized error; <= 1 means all continuous gates pass."""

        return max(
            self.object_position_error_m / OBJECT_POSITION_GATE_M,
            self.object_rotation_error_deg / OBJECT_ROTATION_GATE_DEG,
            self.object_axis_error_m / OBJECT_AXIS_GATE_M,
        )

    @property
    def formal_gate_violation(self) -> float:
        continuous = max(0.0, self.normalized_gate_margin - 1.0)
        discrete = 0.0 if self.success and self.final_reach else 1.0
        return max(continuous, discrete)

    @property
    def failed(self) -> bool:
        return self.formal_gate_violation > 0.0


@dataclass(frozen=True)
class RobustCandidateEvaluationV1:
    """Replica aggregate and the frozen lexical comparison key."""

    candidate_id: str
    contract: RobustOracleContractV1
    replicas: tuple[RobustReplicaResultV1, ...]
    failure_probability: float
    cvar_formal_gate_violation: float
    worst_normalized_gate_margin: float
    mean_object_error_m: float
    mean_rotation_error_deg: float
    mean_contact_stability_penalty: float
    mean_action_smoothness: float
    mean_effort: float
    mean_cost: float
    std_cost: float
    worst_cost: float
    cvar_cost: float
    robust_cost: float

    def lexical_key(self) -> tuple[float, float, float, float, float, float, float, float, str]:
        """The R3-required deterministic lexicographic selector order."""

        return (
            self.failure_probability,
            self.cvar_formal_gate_violation,
            self.worst_normalized_gate_margin,
            self.mean_object_error_m,
            self.mean_rotation_error_deg,
            self.mean_contact_stability_penalty,
            self.mean_action_smoothness,
            self.mean_effort,
            self.candidate_id,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "RobustCandidateEvaluationV1",
            "candidate_id": self.candidate_id,
            "contract": self.contract.as_dict(),
            "replica_count": len(self.replicas),
            "failure_probability": self.failure_probability,
            "cvar_formal_gate_violation": self.cvar_formal_gate_violation,
            "worst_normalized_gate_margin": self.worst_normalized_gate_margin,
            "mean_object_error_m": self.mean_object_error_m,
            "mean_rotation_error_deg": self.mean_rotation_error_deg,
            "mean_contact_stability_penalty": self.mean_contact_stability_penalty,
            "mean_action_smoothness": self.mean_action_smoothness,
            "mean_effort": self.mean_effort,
            "mean_cost": self.mean_cost,
            "std_cost": self.std_cost,
            "worst_cost": self.worst_cost,
            "cvar_cost": self.cvar_cost,
            "robust_cost": self.robust_cost,
            "lexical_key": list(self.lexical_key()),
        }


class RobustCandidateEvaluatorV1:
    """Aggregate an already independent replica set under the frozen R3 rule."""

    def __init__(self, contract: RobustOracleContractV1 | None = None) -> None:
        self.contract = contract or RobustOracleContractV1()

    def evaluate(
        self, candidate_id: str, replicas: Iterable[RobustReplicaResultV1]
    ) -> RobustCandidateEvaluationV1:
        if not candidate_id:
            raise ValueError("robust candidate ID must be nonempty")
        values = tuple(replicas)
        if len(values) != self.contract.replica_count:
            raise ValueError(
                "robust candidate replica count does not match frozen contract: "
                f"expected={self.contract.replica_count} actual={len(values)}"
            )
        costs = [row.cost for row in values]
        mean_cost = fmean(costs)
        std_cost = math.sqrt(fmean([(cost - mean_cost) ** 2 for cost in costs]))
        cvar_cost = upper_cvar(costs, self.contract.cvar_alpha)
        robust_cost = (
            mean_cost + self.contract.std_lambda * std_cost
            if self.contract.objective == "mean_plus_lambda_std_v1"
            else cvar_cost
        )
        return RobustCandidateEvaluationV1(
            candidate_id=candidate_id,
            contract=self.contract,
            replicas=values,
            failure_probability=fmean([float(row.failed) for row in values]),
            cvar_formal_gate_violation=upper_cvar(
                [row.formal_gate_violation for row in values], self.contract.cvar_alpha
            ),
            worst_normalized_gate_margin=max(row.normalized_gate_margin for row in values),
            mean_object_error_m=fmean(row.object_position_error_m for row in values),
            mean_rotation_error_deg=fmean(row.object_rotation_error_deg for row in values),
            mean_contact_stability_penalty=fmean(row.contact_stability_penalty for row in values),
            mean_action_smoothness=fmean(row.action_smoothness for row in values),
            mean_effort=fmean(row.effort for row in values),
            mean_cost=mean_cost,
            std_cost=std_cost,
            worst_cost=max(costs),
            cvar_cost=cvar_cost,
            robust_cost=robust_cost,
        )


class RobustCandidateSelector:
    """Select the lexicographically safest evaluation with a stable ID tiebreak."""

    def select(
        self, evaluations: Iterable[RobustCandidateEvaluationV1]
    ) -> RobustCandidateEvaluationV1:
        values = tuple(evaluations)
        if not values:
            raise ValueError("robust selector requires at least one candidate")
        return min(values, key=RobustCandidateEvaluationV1.lexical_key)


def qualify_c5c_independent_replicas(
    replicas: Sequence[RobustReplicaResultV1],
) -> dict[str, object]:
    """Evaluate the frozen C5C contract: one selected trace plus 20 replicas.

    The caller records the one optimized trace separately.  This function
    intentionally accepts only its independently reset/replayed replicas, so
    a successful single trace can never hide poor replica success, reach, or
    formal-gate statistics.
    """

    values = tuple(replicas)
    if len(values) != 20:
        raise ValueError("C5C robust qualification requires exactly 20 independent replicas")
    position = [row.object_position_error_m for row in values]
    rotation = [row.object_rotation_error_deg for row in values]
    axis = [row.object_axis_error_m for row in values]
    success_rate = fmean(float(row.success) for row in values)
    final_reach_rate = fmean(float(row.final_reach) for row in values)
    formal_gate_pass_rate = fmean(float(not row.failed) for row in values)
    return {
        "version": "stage16c5c_robust_replica_qualification_v1",
        "replica_count": len(values),
        "success_rate": success_rate,
        "final_reach_rate": final_reach_rate,
        "formal_gate_pass_rate": formal_gate_pass_rate,
        "object_position_error_m": {
            "mean": fmean(position),
            "p95": percentile(position, 0.95),
            "worst": max(position),
        },
        "object_rotation_error_deg": {
            "mean": fmean(rotation),
            "p95": percentile(rotation, 0.95),
            "worst": max(rotation),
        },
        "object_axis_error_m": {
            "mean": fmean(axis),
            "p95": percentile(axis, 0.95),
            "worst": max(axis),
        },
        "termination_distribution": dict(
            sorted(Counter(row.termination_reason for row in values).items())
        ),
        "passes_frozen_gate": (
            success_rate >= SUCCESS_RATE_GATE
            and final_reach_rate >= FINAL_REACH_GATE
            and formal_gate_pass_rate >= SUCCESS_RATE_GATE
        ),
    }


__all__ = [
    "FINAL_REACH_GATE",
    "OBJECT_AXIS_GATE_M",
    "OBJECT_POSITION_GATE_M",
    "OBJECT_ROTATION_GATE_DEG",
    "RobustCandidateEvaluationV1",
    "RobustCandidateEvaluatorV1",
    "RobustCandidateSelector",
    "RobustOracleContractV1",
    "RobustReplicaResultV1",
    "SUCCESS_RATE_GATE",
    "percentile",
    "qualify_c5c_independent_replicas",
    "upper_cvar",
]
