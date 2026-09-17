"""Frozen cold-start contributor schedules for ExecutionV4.

ExecutionV4 leaves ObjectiveV2 and REFINEMENT execution untouched.  It only
widens the contributor blocks considered after the immutable ExecutionV3
cold-start prefix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

V4Mode = Literal["SEQUENTIAL", "JOINT_BLOCK", "ADAPTIVE_SEQUENTIAL"]


@dataclass(frozen=True)
class ColdStartSearchV4Candidate:
    """Small, outcome-independent ExecutionV4 candidate declaration."""

    name: str
    mode: V4Mode
    top_k: int
    contributor_probe_max_nfev: int = 24
    selected_primary_maxiter: int = 8
    secondary_polish_maxiter: int = 8
    stop_at_interaction_target: bool = False
    schema_version: str = "ColdStartSearchV4CandidateV1"

    def validate(self) -> ColdStartSearchV4Candidate:
        if self.name not in {"V4_A_TOP2_SEQUENTIAL", "V4_B_TOP2_JOINT_BLOCK", "V4_C_ADAPTIVE_K3"}:
            raise ValueError("unsupported ExecutionV4 candidate")
        expected = {
            "V4_A_TOP2_SEQUENTIAL": ("SEQUENTIAL", 2, False),
            "V4_B_TOP2_JOINT_BLOCK": ("JOINT_BLOCK", 2, False),
            "V4_C_ADAPTIVE_K3": ("ADAPTIVE_SEQUENTIAL", 3, True),
        }[self.name]
        if (self.mode, self.top_k, self.stop_at_interaction_target) != expected:
            raise ValueError("ExecutionV4 candidate identity does not match its frozen schedule")
        if self.top_k > 3:
            raise ValueError("ExecutionV4 Kmax must not exceed three")
        if (
            self.contributor_probe_max_nfev != 24
            or self.selected_primary_maxiter != 8
            or self.secondary_polish_maxiter != 8
        ):
            raise ValueError("ExecutionV4 must retain the frozen ExecutionV3 solver budgets")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_cold_start_search_v4_candidates() -> tuple[ColdStartSearchV4Candidate, ...]:
    """Return candidates in their preregistered simplicity order."""

    return (
        ColdStartSearchV4Candidate(
            name="V4_A_TOP2_SEQUENTIAL", mode="SEQUENTIAL", top_k=2
        ).validate(),
        ColdStartSearchV4Candidate(
            name="V4_B_TOP2_JOINT_BLOCK", mode="JOINT_BLOCK", top_k=2
        ).validate(),
        ColdStartSearchV4Candidate(
            name="V4_C_ADAPTIVE_K3",
            mode="ADAPTIVE_SEQUENTIAL",
            top_k=3,
            stop_at_interaction_target=True,
        ).validate(),
    )


__all__ = [
    "ColdStartSearchV4Candidate",
    "default_cold_start_search_v4_candidates",
]
