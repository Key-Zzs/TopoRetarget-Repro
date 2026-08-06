"""Frozen lexicographic robust candidate evaluation for Stage 16-C.5B/C5C."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

from .robust import (
    FINAL_REACH_GATE,
    OBJECT_AXIS_GATE_M,
    OBJECT_POSITION_GATE_M,
    OBJECT_ROTATION_GATE_DEG,
    SUCCESS_RATE_GATE,
    percentile,
    upper_cvar,
)


@dataclass(frozen=True)
class RobustCandidateReplicaV2:
    """One stochastic candidate rollout measured at its horizon boundary."""

    object_position_error_m: float
    object_rotation_error_deg: float
    object_axis_error_m: float
    tracking_error: float
    contact_stability: float
    smoothness: float
    effort: float
    termination_reason: str = "NONE"
    success: bool = False
    final_reach: bool = False
    terminal_required: bool = False

    def __post_init__(self) -> None:
        values = (
            self.object_position_error_m,
            self.object_rotation_error_deg,
            self.object_axis_error_m,
            self.tracking_error,
            self.contact_stability,
            self.smoothness,
            self.effort,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("robust candidate replica metrics must be finite and nonnegative")
        if not self.termination_reason:
            raise ValueError("robust candidate termination reason is required")

    @property
    def normalized_gate_margin(self) -> float:
        return max(
            self.object_position_error_m / OBJECT_POSITION_GATE_M,
            self.object_rotation_error_deg / OBJECT_ROTATION_GATE_DEG,
            self.object_axis_error_m / OBJECT_AXIS_GATE_M,
        )

    @property
    def gate_violation(self) -> float:
        continuous = max(0.0, self.normalized_gate_margin - 1.0)
        task_failure = self.termination_reason.startswith("FAILURE_")
        terminal_failure = self.terminal_required and not (self.success and self.final_reach)
        return max(continuous, float(task_failure or terminal_failure))

    @property
    def formal_failure(self) -> bool:
        return self.gate_violation > 0.0


@dataclass(frozen=True)
class RobustCandidateEvaluationV2:
    candidate_id: int
    horizon: int
    replicas: tuple[RobustCandidateReplicaV2, ...]
    failure_probability: float
    cvar_gate_violation: float
    worst_normalized_gate_margin: float
    p95_axis_error_m: float
    p95_object_position_error_m: float
    p95_object_rotation_error_deg: float
    mean_tracking_error: float
    mean_contact_stability: float
    mean_smoothness: float
    mean_effort: float

    def lexical_key(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float, int, int]:
        """Exact frozen C5B ordering followed only by stable identity ties."""

        return (
            self.failure_probability,
            self.cvar_gate_violation,
            self.worst_normalized_gate_margin,
            self.p95_axis_error_m,
            self.p95_object_position_error_m,
            self.p95_object_rotation_error_deg,
            self.mean_tracking_error,
            self.mean_contact_stability,
            self.mean_smoothness,
            self.mean_effort,
            self.horizon,
            self.candidate_id,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "RobustCandidateEvaluationV2",
            "candidate_id": self.candidate_id,
            "horizon": self.horizon,
            "replica_count": len(self.replicas),
            "failure_probability": self.failure_probability,
            "cvar_gate_violation": self.cvar_gate_violation,
            "worst_normalized_gate_margin": self.worst_normalized_gate_margin,
            "p95_axis_error_m": self.p95_axis_error_m,
            "p95_object_position_error_m": self.p95_object_position_error_m,
            "p95_object_rotation_error_deg": self.p95_object_rotation_error_deg,
            "mean_tracking_error": self.mean_tracking_error,
            "mean_contact_stability": self.mean_contact_stability,
            "mean_smoothness": self.mean_smoothness,
            "mean_effort": self.mean_effort,
            "lexical_key": list(self.lexical_key()),
        }


class RobustCandidateEvaluatorV2:
    def __init__(self, *, replica_count: int = 4, cvar_alpha: float = 0.8) -> None:
        if replica_count not in {4, 8}:
            raise ValueError("C5B permits the default four or one-time upgrade to eight replicas")
        if cvar_alpha != 0.8:
            raise ValueError("C5B freezes CVaR alpha at 0.8")
        self.replica_count = replica_count
        self.cvar_alpha = cvar_alpha

    def evaluate(
        self,
        *,
        candidate_id: int,
        horizon: int,
        replicas: Iterable[RobustCandidateReplicaV2],
    ) -> RobustCandidateEvaluationV2:
        values = tuple(replicas)
        if candidate_id < 0 or horizon not in {1, 5, 10}:
            raise ValueError("invalid robust candidate identity")
        if len(values) != self.replica_count:
            raise ValueError("robust candidate replica count differs from frozen evaluator")
        return RobustCandidateEvaluationV2(
            candidate_id=candidate_id,
            horizon=horizon,
            replicas=values,
            failure_probability=fmean(float(row.formal_failure) for row in values),
            cvar_gate_violation=upper_cvar([row.gate_violation for row in values], self.cvar_alpha),
            worst_normalized_gate_margin=max(row.normalized_gate_margin for row in values),
            p95_axis_error_m=percentile([row.object_axis_error_m for row in values], 0.95),
            p95_object_position_error_m=percentile(
                [row.object_position_error_m for row in values], 0.95
            ),
            p95_object_rotation_error_deg=percentile(
                [row.object_rotation_error_deg for row in values], 0.95
            ),
            mean_tracking_error=fmean(row.tracking_error for row in values),
            mean_contact_stability=fmean(row.contact_stability for row in values),
            mean_smoothness=fmean(row.smoothness for row in values),
            mean_effort=fmean(row.effort for row in values),
        )


class RobustLexicographicSelectorV1:
    def select(
        self, evaluations: Iterable[RobustCandidateEvaluationV2]
    ) -> RobustCandidateEvaluationV2:
        values = tuple(evaluations)
        if not values:
            raise ValueError("robust selector requires at least one candidate")
        return min(values, key=RobustCandidateEvaluationV2.lexical_key)


def qualify_two_clip_c5c(
    clip_replicas: dict[str, Sequence[RobustCandidateReplicaV2]],
) -> dict[str, object]:
    """Formal 20-episode qualification of exactly the two frozen HO-Cap clips."""

    if tuple(sorted(clip_replicas)) != ("hocap_170105", "hocap_170650"):
        raise ValueError("C5C requires exactly hocap_170105 and hocap_170650")
    clips: dict[str, object] = {}
    overall = True
    for clip in sorted(clip_replicas):
        replicas = tuple(clip_replicas[clip])
        if len(replicas) != 20:
            raise ValueError("C5C requires exactly twenty independent episodes per clip")
        position = [row.object_position_error_m for row in replicas]
        rotation = [row.object_rotation_error_deg for row in replicas]
        axis = [row.object_axis_error_m for row in replicas]
        success_rate = fmean(float(row.success) for row in replicas)
        reach_rate = fmean(float(row.final_reach) for row in replicas)
        position_p95 = percentile(position, 0.95)
        rotation_p95 = percentile(rotation, 0.95)
        axis_p95 = percentile(axis, 0.95)
        passes = (
            success_rate >= SUCCESS_RATE_GATE
            and reach_rate >= FINAL_REACH_GATE
            and position_p95 <= OBJECT_POSITION_GATE_M
            and rotation_p95 <= OBJECT_ROTATION_GATE_DEG
            and axis_p95 <= OBJECT_AXIS_GATE_M
        )
        overall = overall and passes
        clips[clip] = {
            "episodes": 20,
            "fresh_reset_frame0": True,
            "independent_physx_rollouts": True,
            "success_rate": success_rate,
            "final_reach_rate": reach_rate,
            "object_position_error_m": {
                "mean": fmean(position),
                "p95": position_p95,
            },
            "object_rotation_error_deg": {
                "mean": fmean(rotation),
                "p95": rotation_p95,
            },
            "object_axis_error_m": {"mean": fmean(axis), "p95": axis_p95},
            "termination_distribution": dict(
                sorted(Counter(row.termination_reason for row in replicas).items())
            ),
            "passes": passes,
        }
    return {
        "version": "stage16c5c_two_clip_formal_qualification_v1",
        "clips": clips,
        "passes": overall,
        "status": (
            "STAGE16C5_PHYSX_ROBUST_ORACLE_VALIDATED"
            if overall
            else "STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL"
        ),
    }


__all__ = [
    "RobustCandidateEvaluationV2",
    "RobustCandidateEvaluatorV2",
    "RobustCandidateReplicaV2",
    "RobustLexicographicSelectorV1",
    "qualify_two_clip_c5c",
]
