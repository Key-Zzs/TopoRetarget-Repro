"""Dual-mode execution authority for the frozen Candidate-B2 objective.

This module is deliberately separate from :mod:`objective_v2_execution`.
ExecutionV2 remains the immutable refinement method; V3 makes the presence or
absence of an old-production trajectory an explicit mode choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from toporetarget.retarget.objective_v2_execution import (
    SearchExecutionContract,
    default_search_contracts,
    deterministic_block_seeds,
)

EXECUTION_V3_SCHEMA_VERSION = "RetargetObjectiveV2ExecutionContractV3"
EXECUTION_INPUT_AUTHORITY_SCHEMA_VERSION = "ExecutionInputAuthorityV1"
COLD_START_SEED_AUTHORITY_SCHEMA_VERSION = "ColdStartSeedAuthorityV1"
COLD_START_SEED_AUTHORITY_V2_SCHEMA_VERSION = "ColdStartSeedAuthorityV2"
COLD_START_BOOTSTRAP_SCHEMA_VERSION = "ColdStartBootstrapContractV1"
COLD_START_SEARCH_V2_SCHEMA_VERSION = "ColdStartSearchV2DevelopmentCandidate"


class RetargetMode(StrEnum):
    REFINEMENT = "REFINEMENT"
    COLD_START = "COLD_START"


class ExecutionBaselineAuthority(StrEnum):
    OLD_PRODUCTION_TRAJECTORY = "OLD_PRODUCTION_TRAJECTORY"
    PREVIOUS_ACCEPTED_RUNTIME_STATE = "PREVIOUS_ACCEPTED_RUNTIME_STATE"
    COLD_START_GENERIC = "COLD_START_GENERIC"


class ExecutionInputAuthorityError(ValueError):
    """Raised before optimizer construction when mode authority is invalid."""


@dataclass(frozen=True)
class ExecutionFrameInputsV3:
    """Mode-specific state authorities; arrays remain owned by the caller."""

    mode: RetargetMode
    runtime_step_index: int
    old_production_q: np.ndarray | None
    previous_accepted_q: np.ndarray | None
    previous_accepted_base: np.ndarray | None

    def validate(self) -> ExecutionFrameInputsV3:
        if self.runtime_step_index < 0:
            raise ExecutionInputAuthorityError("runtime step index must be nonnegative")
        previous = (self.previous_accepted_q, self.previous_accepted_base)
        if (previous[0] is None) != (previous[1] is None):
            raise ExecutionInputAuthorityError("previous accepted q/base authority is incomplete")
        if self.runtime_step_index == 0 and any(item is not None for item in previous):
            raise ExecutionInputAuthorityError("frame zero forbids previous accepted runtime state")
        if self.runtime_step_index > 0 and any(item is None for item in previous):
            raise ExecutionInputAuthorityError(
                "nonzero cold/refinement runtime step needs previous state"
            )
        if self.mode is RetargetMode.REFINEMENT:
            if self.old_production_q is None:
                raise ExecutionInputAuthorityError(
                    "REFINEMENT_REQUIRES_OLD_PRODUCTION_TRAJECTORY_AUTHORITY"
                )
        elif self.old_production_q is not None:
            raise ExecutionInputAuthorityError("COLD_START_FORBIDS_OLD_PRODUCTION_Q")
        return self


@dataclass(frozen=True)
class ExecutionV3Candidate:
    """Predeclared, episode-agnostic cold-start schedule."""

    name: str
    cold_seed_sources: tuple[str, ...]
    use_previous_accepted_after_frame0: bool
    source_geometric_seed: bool = False
    top_k: int = 1
    contributor_probe_max_nfev: int = 24
    selected_primary_maxiter: int = 8
    secondary_polish_maxiter: int = 8
    schema_version: str = EXECUTION_V3_SCHEMA_VERSION

    def validate(self) -> ExecutionV3Candidate:
        allowed = {
            "wuji_canonical_rest",
            "joint_range_midpoint",
            "source_geometric_multistart",
            "previous_accepted_runtime",
        }
        if not self.name.startswith("V3_") or not self.cold_seed_sources:
            raise ValueError("V3 candidate identity/seed declaration is incomplete")
        if set(self.cold_seed_sources) - allowed:
            raise ValueError("V3 candidate contains an unsupported seed authority")
        if "old_production" in self.cold_seed_sources:
            raise ValueError("cold-start candidate cannot contain old_production")
        observed_previous = "previous_accepted_runtime" in self.cold_seed_sources
        if observed_previous != self.use_previous_accepted_after_frame0:
            raise ValueError("previous-runtime seed declaration is inconsistent")
        observed_geometric = "source_geometric_multistart" in self.cold_seed_sources
        if observed_geometric != self.source_geometric_seed:
            raise ValueError("source-geometric seed declaration is inconsistent")
        if self.top_k != 1:
            raise ValueError("V3 candidates retain frozen top_k=1")
        if (
            min(
                self.contributor_probe_max_nfev,
                self.selected_primary_maxiter,
                self.secondary_polish_maxiter,
            )
            <= 0
        ):
            raise ValueError("V3 solver budgets must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColdStartSearchV2Candidate:
    """Outcome-independent whole-hand bootstrap schedule for cold start.

    The geometric solve is only a basin initializer.  Candidate-B2 remains the
    scientific objective, and contributor selection remains top-1.  Ranking is
    deliberately evaluated on each bootstrapped full-hand state rather than on
    an unrelated neutral carrier.
    """

    name: str = "CS2_A_FULL_HAND_GEOMETRIC_BOOTSTRAP_TOP1"
    bootstrap_seed_sources: tuple[str, ...] = (
        "wuji_canonical_rest",
        "joint_range_midpoint",
    )
    bootstrap_solver_profile: str = "paper_repro_scipy_trf"
    bootstrap_max_nfev: int = 250
    bootstrap_free_dofs: str = "ALL_20_FINGER_DOFS"
    bootstrap_base_authority: str = "SOURCE_ROBOT_WRIST_FRAME_ALIGNMENT"
    use_previous_accepted_after_frame0: bool = True
    post_bootstrap_contributor_ranking: bool = True
    top_k: int = 1
    contributor_probe_max_nfev: int = 24
    selected_primary_maxiter: int = 8
    secondary_polish_maxiter: int = 8
    numerical_epsilon: float = 1.0e-10
    schema_version: str = COLD_START_SEARCH_V2_SCHEMA_VERSION

    def validate(self) -> ColdStartSearchV2Candidate:
        allowed = {"wuji_canonical_rest", "joint_range_midpoint"}
        if self.name != "CS2_A_FULL_HAND_GEOMETRIC_BOOTSTRAP_TOP1":
            raise ValueError("unsupported cold-start Search V2 candidate")
        if not self.bootstrap_seed_sources or set(self.bootstrap_seed_sources) - allowed:
            raise ValueError("bootstrap contains an unsupported seed authority")
        if any("old" in source.lower() for source in self.bootstrap_seed_sources):
            raise ValueError("cold-start bootstrap cannot consume q_old")
        if self.bootstrap_solver_profile != "paper_repro_scipy_trf":
            raise ValueError("bootstrap solver must reuse the predeclared geometric primitive")
        if self.bootstrap_max_nfev != 250:
            raise ValueError("bootstrap budget is frozen before development")
        if self.bootstrap_free_dofs != "ALL_20_FINGER_DOFS":
            raise ValueError("bootstrap must define the full hand")
        if not self.post_bootstrap_contributor_ranking or self.top_k != 1:
            raise ValueError("Search V2 freezes post-bootstrap top-1 contributor ranking")
        if (
            self.contributor_probe_max_nfev != 24
            or self.selected_primary_maxiter != 8
            or self.secondary_polish_maxiter != 8
        ):
            raise ValueError("Candidate-B2 budgets must remain frozen ExecutionV2 values")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_cold_start_search_v2_candidates() -> tuple[ColdStartSearchV2Candidate, ...]:
    """Return the sole evidence-authorized Search V2 development candidate."""

    return (ColdStartSearchV2Candidate().validate(),)


def screen_whole_hand_bootstrap_states(
    *,
    states: Sequence[tuple[str, np.ndarray]],
    lower_q: np.ndarray,
    upper_q: np.ndarray,
    expected_dofs: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Fail closed on undefined, non-finite, or out-of-bounds bootstrap states."""

    lower = np.asarray(lower_q, dtype=np.float64)
    upper = np.asarray(upper_q, dtype=np.float64)
    if lower.shape != (expected_dofs,) or upper.shape != (expected_dofs,):
        raise ValueError("whole-hand bounds do not match the declared DOF count")
    accepted: list[tuple[str, np.ndarray]] = []
    for name, raw in states:
        value = np.asarray(raw, dtype=np.float64)
        if value.shape != (expected_dofs,):
            raise ValueError(f"bootstrap state does not define the full hand: {name}")
        if not np.all(np.isfinite(value)):
            continue
        if np.any(value < lower - 1.0e-12) or np.any(value > upper + 1.0e-12):
            continue
        if any(np.array_equal(value, prior) for _prior_name, prior in accepted):
            continue
        accepted.append((str(name), value.copy()))
    if not accepted:
        raise ExecutionInputAuthorityError("TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE")
    return tuple(accepted)


def default_execution_v3_candidates() -> tuple[ExecutionV3Candidate, ...]:
    """Return the two unconditional designs and one conditional generic design."""

    return (
        ExecutionV3Candidate(
            name="V3_A_GENERIC_ASSET_SOURCE_COLD_START",
            cold_seed_sources=("wuji_canonical_rest", "joint_range_midpoint"),
            use_previous_accepted_after_frame0=False,
        ).validate(),
        ExecutionV3Candidate(
            name="V3_B_GENERIC_WITH_PREVIOUS_ACCEPTED_RUNTIME",
            cold_seed_sources=(
                "wuji_canonical_rest",
                "joint_range_midpoint",
                "previous_accepted_runtime",
            ),
            use_previous_accepted_after_frame0=True,
        ).validate(),
        ExecutionV3Candidate(
            name="V3_C_SOURCE_GEOMETRIC_WITH_PREVIOUS_ACCEPTED_RUNTIME",
            cold_seed_sources=(
                "wuji_canonical_rest",
                "joint_range_midpoint",
                "source_geometric_multistart",
                "previous_accepted_runtime",
            ),
            use_previous_accepted_after_frame0=True,
            source_geometric_seed=True,
        ).validate(),
    )


def refinement_seeds_v3(
    *,
    frame_inputs: ExecutionFrameInputsV3,
    neutral_q: np.ndarray,
    lower_q: np.ndarray,
    upper_q: np.ndarray,
    block: Sequence[int],
) -> tuple[tuple[str, np.ndarray], ...]:
    """Delegate refinement seed construction byte-for-byte to frozen S1/V2."""

    frame_inputs.validate()
    if frame_inputs.mode is not RetargetMode.REFINEMENT:
        raise ExecutionInputAuthorityError("refinement seed builder received cold-start inputs")
    s1: SearchExecutionContract = default_search_contracts()[0]
    return deterministic_block_seeds(
        old_q=np.asarray(frame_inputs.old_production_q, dtype=np.float64),
        neutral_q=neutral_q,
        lower_q=lower_q,
        upper_q=upper_q,
        block=block,
        transported_q=None,
        contract=s1,
    )


def cold_start_seeds_v3(
    *,
    candidate: ExecutionV3Candidate,
    frame_inputs: ExecutionFrameInputsV3,
    neutral_q: np.ndarray,
    lower_q: np.ndarray,
    upper_q: np.ndarray,
    source_geometric_q: np.ndarray | None = None,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build full-state cold-start seeds without an old-q carrier or alias."""

    candidate.validate()
    frame_inputs.validate()
    if frame_inputs.mode is not RetargetMode.COLD_START:
        raise ExecutionInputAuthorityError("cold-start seed builder received refinement inputs")
    neutral = np.asarray(neutral_q, dtype=np.float64)
    lower = np.asarray(lower_q, dtype=np.float64)
    upper = np.asarray(upper_q, dtype=np.float64)
    if not (neutral.shape == lower.shape == upper.shape):
        raise ValueError("robot seed authorities must have identical shapes")
    authorities: dict[str, np.ndarray | None] = {
        "wuji_canonical_rest": neutral,
        "joint_range_midpoint": lower + 0.5 * (upper - lower),
        "source_geometric_multistart": source_geometric_q,
        "previous_accepted_runtime": frame_inputs.previous_accepted_q,
    }
    result: list[tuple[str, np.ndarray]] = []
    for source in candidate.cold_seed_sources:
        value = authorities[source]
        if value is None:
            if source == "previous_accepted_runtime" and frame_inputs.runtime_step_index == 0:
                continue
            raise ExecutionInputAuthorityError(f"cold-start seed authority unavailable: {source}")
        seed = np.clip(np.asarray(value, dtype=np.float64), lower, upper)
        if seed.shape != neutral.shape:
            raise ValueError(f"cold-start seed has wrong shape: {source}")
        if any(np.array_equal(seed, prior) for _name, prior in result):
            continue
        result.append((source, seed.copy()))
    if not result:
        raise ExecutionInputAuthorityError("TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE")
    return tuple(result)


__all__ = [
    "COLD_START_BOOTSTRAP_SCHEMA_VERSION",
    "COLD_START_SEED_AUTHORITY_SCHEMA_VERSION",
    "COLD_START_SEED_AUTHORITY_V2_SCHEMA_VERSION",
    "COLD_START_SEARCH_V2_SCHEMA_VERSION",
    "ColdStartSearchV2Candidate",
    "EXECUTION_INPUT_AUTHORITY_SCHEMA_VERSION",
    "EXECUTION_V3_SCHEMA_VERSION",
    "ExecutionBaselineAuthority",
    "ExecutionFrameInputsV3",
    "ExecutionInputAuthorityError",
    "ExecutionV3Candidate",
    "RetargetMode",
    "cold_start_seeds_v3",
    "default_cold_start_search_v2_candidates",
    "default_execution_v3_candidates",
    "refinement_seeds_v3",
    "screen_whole_hand_bootstrap_states",
]
