"""Hard-gate-first ranking for geometry-aware Stage 16-D candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

from .robust_optimizer import upper_cvar


@dataclass(frozen=True)
class GeometryAwareCandidateReplicaV2:
    catastrophic_failure: bool
    absolute_geometry_failure: bool
    relative_geometry_failure: bool
    semantic_failure: bool
    contact_topology_failure: bool
    terminal_stability_failure: bool
    contact_causality_failure: bool
    max_penetration_m: float
    active_p95_penetration_m: float
    contact_coverage: float
    contact_persistence: float
    semantic_progress: float
    robot_fidelity_error: float
    source_object_soft_prior_error: float
    action_smoothness: float
    effort: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"geometry-aware replica metric is invalid: {name}")


@dataclass(frozen=True)
class GeometryAwareCandidateEvaluationV2:
    candidate_id: int
    replicas: tuple[GeometryAwareCandidateReplicaV2, ...]

    def lexical_key(self) -> tuple[float | int, ...]:
        if not self.replicas:
            raise ValueError("geometry-aware evaluation requires replicas")
        rows = self.replicas

        def probability(name: str) -> float:
            return fmean(float(getattr(row, name)) for row in rows)

        return (
            probability("catastrophic_failure"),
            probability("absolute_geometry_failure"),
            probability("relative_geometry_failure"),
            probability("semantic_failure"),
            probability("contact_topology_failure"),
            probability("terminal_stability_failure"),
            probability("contact_causality_failure"),
            upper_cvar(tuple(row.max_penetration_m for row in rows)),
            upper_cvar(tuple(row.active_p95_penetration_m for row in rows)),
            -fmean(row.contact_coverage for row in rows),
            -fmean(row.contact_persistence for row in rows),
            -fmean(row.semantic_progress for row in rows),
            fmean(row.robot_fidelity_error for row in rows),
            fmean(row.source_object_soft_prior_error for row in rows),
            fmean(row.action_smoothness for row in rows),
            fmean(row.effort for row in rows),
            self.candidate_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "GeometryAwareCandidateEvaluationV2",
            "candidate_id": self.candidate_id,
            "replicas": [asdict(row) for row in self.replicas],
            "lexical_key": list(self.lexical_key()),
        }


__all__ = ["GeometryAwareCandidateEvaluationV2", "GeometryAwareCandidateReplicaV2"]
