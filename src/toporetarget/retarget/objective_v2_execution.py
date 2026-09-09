"""Deterministic execution helpers for the frozen Candidate-B2 objective.

The helpers in this module contain no OakInk2 episode or finger special cases.
They turn robot asset names and semantic keypoint profiles into deterministic
search blocks, build an outcome-independent seed pool, and apply the
feasibility-first Candidate-B2 retention rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from toporetarget.retarget.objective_v2 import NUMERICAL_RETENTION_EPSILON

EXECUTION_CONTRACT_SCHEMA_VERSION = "RetargetObjectiveV2ExecutionContractV2"
CONTRIBUTOR_SEARCH_SCHEMA_VERSION = "InteractionContributorSearchV2"

SEMANTIC_FINGER_KEYPOINTS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}
_FINGER_ALIASES: dict[str, tuple[str, ...]] = {
    "thumb": ("thumb",),
    "index": ("index",),
    "middle": ("middle",),
    "ring": ("ring",),
    "little": ("little", "pinky"),
}

SearchName = Literal["S1_DETERMINISTIC_MULTI_START", "S2_SEQUENTIAL_MULTI_START"]


@dataclass(frozen=True)
class SearchExecutionContract:
    """Outcome-independent Candidate-B2 search schedule."""

    name: SearchName
    top_k: int
    seed_sources: tuple[str, ...]
    contributor_probe_max_nfev: int
    selected_primary_maxiter: int
    secondary_polish_maxiter: int
    sequential_warm_start: bool
    old_production_baseline: bool = True
    terminal_state_screening: str = "independent_feasibility_not_optimizer_success_flag"
    primary_retention: str = "retain best feasible Candidate-B2 primary state"
    polish_fallback: str = "reject interaction regression and return retained primary"
    deterministic_tie_break: str = "seed ordinal then seed id"
    numerical_epsilon: float = NUMERICAL_RETENTION_EPSILON
    schema_version: str = EXECUTION_CONTRACT_SCHEMA_VERSION

    def validate(self) -> SearchExecutionContract:
        if self.top_k != 1:
            raise ValueError("D2C contracts freeze top_k=1; top_k=2 requires separate evidence")
        if not self.old_production_baseline:
            raise ValueError("old production q must remain in every candidate pool")
        if self.contributor_probe_max_nfev <= 0:
            raise ValueError("contributor probe budget must be positive")
        if self.selected_primary_maxiter <= 0 or self.secondary_polish_maxiter <= 0:
            raise ValueError("primary and secondary budgets must be positive")
        expected = self.sequential_warm_start
        observed = "previous_refined_transported" in self.seed_sources
        if expected != observed:
            raise ValueError("sequential seed declaration is inconsistent")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScreenedCandidate:
    """Minimal data needed for deterministic lexicographic screening."""

    candidate_id: str
    feasible: bool
    finite: bool
    primary_hinge: float
    secondary_objective: float
    interaction_e_im: float
    ordinal: int
    optimizer_converged: bool | None = None

    @property
    def usable(self) -> bool:
        return bool(self.feasible and self.finite)


def default_search_contracts() -> tuple[SearchExecutionContract, ...]:
    """Return the two predeclared D2C contracts; no result-driven S3 exists."""

    return (
        SearchExecutionContract(
            name="S1_DETERMINISTIC_MULTI_START",
            top_k=1,
            seed_sources=("old_production", "wuji_canonical_rest", "joint_range_midpoint"),
            contributor_probe_max_nfev=24,
            selected_primary_maxiter=8,
            secondary_polish_maxiter=8,
            sequential_warm_start=False,
        ).validate(),
        SearchExecutionContract(
            name="S2_SEQUENTIAL_MULTI_START",
            top_k=1,
            seed_sources=(
                "old_production",
                "previous_refined_transported",
                "wuji_canonical_rest",
                "joint_range_midpoint",
            ),
            contributor_probe_max_nfev=24,
            selected_primary_maxiter=8,
            secondary_polish_maxiter=8,
            sequential_warm_start=True,
        ).validate(),
    )


def asset_derived_dof_blocks(dof_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Map every robot DOF to one semantic finger using asset-provided names."""

    blocks: dict[str, list[int]] = {name: [] for name in SEMANTIC_FINGER_KEYPOINTS}
    for index, raw_name in enumerate(dof_names):
        name = str(raw_name).lower()
        matches = [
            finger
            for finger, aliases in _FINGER_ALIASES.items()
            if any(alias in name for alias in aliases)
        ]
        if len(matches) != 1:
            raise ValueError(f"asset DOF does not map to exactly one semantic finger: {raw_name}")
        blocks[matches[0]].append(index)
    missing = [finger for finger, indices in blocks.items() if not indices]
    if missing:
        raise ValueError(f"asset has empty semantic finger blocks: {missing}")
    flattened = [index for indices in blocks.values() for index in indices]
    if sorted(flattened) != list(range(len(dof_names))):
        raise ValueError("asset-derived semantic blocks do not cover every DOF exactly once")
    return {finger: tuple(indices) for finger, indices in blocks.items()}


def contributor_scores(per_keypoint_mass: Sequence[float]) -> dict[str, float]:
    """Aggregate exact per-keypoint Eq.7 mass into semantic finger blocks."""

    mass = np.asarray(per_keypoint_mass, dtype=np.float64)
    if mass.ndim != 1 or len(mass) < 21 or not np.all(np.isfinite(mass)):
        raise ValueError("contributor mass must contain at least 21 finite semantic vertices")
    return {
        finger: float(np.sum(mass[np.asarray(indices, dtype=np.int64)]))
        for finger, indices in SEMANTIC_FINGER_KEYPOINTS.items()
    }


def rank_contributors(scores: dict[str, float]) -> tuple[str, ...]:
    """Rank contributors by descending mass, then semantic name."""

    expected = set(SEMANTIC_FINGER_KEYPOINTS)
    if set(scores) != expected:
        raise ValueError("contributor scores do not match the semantic finger authority")
    if any(not np.isfinite(value) or value < 0.0 for value in scores.values()):
        raise ValueError("contributor scores must be finite and nonnegative")
    return tuple(sorted(scores, key=lambda name: (-float(scores[name]), name)))


def deterministic_block_seeds(
    *,
    old_q: np.ndarray,
    neutral_q: np.ndarray,
    lower_q: np.ndarray,
    upper_q: np.ndarray,
    block: Sequence[int],
    transported_q: np.ndarray | None,
    contract: SearchExecutionContract,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build the frozen seed pool without episode/frame-specific values."""

    contract.validate()
    old = np.asarray(old_q, dtype=np.float64)
    neutral = np.asarray(neutral_q, dtype=np.float64)
    lower = np.asarray(lower_q, dtype=np.float64)
    upper = np.asarray(upper_q, dtype=np.float64)
    if not (old.shape == neutral.shape == lower.shape == upper.shape):
        raise ValueError("seed authorities must have identical shapes")
    indices = np.asarray(tuple(block), dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("contributor block must contain unique DOF indices")
    if np.any(indices < 0) or np.any(indices >= len(old)):
        raise ValueError("contributor block contains an out-of-range DOF")
    authorities: dict[str, np.ndarray] = {
        "old_production": old,
        "wuji_canonical_rest": neutral,
        "joint_range_midpoint": lower + 0.5 * (upper - lower),
    }
    if transported_q is not None:
        transported = np.asarray(transported_q, dtype=np.float64)
        if transported.shape != old.shape:
            raise ValueError("transported seed has the wrong shape")
        authorities["previous_refined_transported"] = transported
    seeds: list[tuple[str, np.ndarray]] = []
    for source in contract.seed_sources:
        if source not in authorities:
            raise ValueError(f"seed authority unavailable: {source}")
        candidate = old.copy()
        candidate[indices] = np.clip(authorities[source][indices], lower[indices], upper[indices])
        if any(np.array_equal(candidate, prior) for _name, prior in seeds):
            continue
        seeds.append((source, candidate))
    return tuple(seeds)


def select_candidate(candidates: Sequence[ScreenedCandidate]) -> ScreenedCandidate | None:
    """Apply hard validity, Candidate-B2 primary, secondary, and stable tie-break."""

    usable = [candidate for candidate in candidates if candidate.usable]
    if not usable:
        return None
    return min(
        usable,
        key=lambda candidate: (
            float(candidate.primary_hinge),
            float(candidate.secondary_objective),
            int(candidate.ordinal),
            str(candidate.candidate_id),
        ),
    )


def retain_after_polish(
    primary: ScreenedCandidate,
    polished: ScreenedCandidate | None,
    *,
    interaction_target: float,
    numerical_epsilon: float = NUMERICAL_RETENTION_EPSILON,
) -> tuple[ScreenedCandidate, str]:
    """Keep a valid primary whenever polish regresses Candidate-B2 interaction."""

    if not primary.usable:
        raise ValueError("the retained primary candidate must be independently usable")
    if polished is None or not polished.usable:
        return primary, "PRIMARY_RETAINED_POLISH_INVALID"
    if primary.interaction_e_im <= interaction_target:
        limit = interaction_target + numerical_epsilon
    else:
        limit = primary.interaction_e_im + numerical_epsilon
    if polished.interaction_e_im > limit:
        return primary, "PRIMARY_RETAINED_INTERACTION_REGRESSION"
    polished_key = (polished.primary_hinge, polished.secondary_objective)
    primary_key = (primary.primary_hinge, primary.secondary_objective)
    if polished_key <= primary_key:
        return polished, "POLISHED_RETAINED"
    return primary, "PRIMARY_RETAINED_LEXICOGRAPHIC_REGRESSION"


__all__ = [
    "CONTRIBUTOR_SEARCH_SCHEMA_VERSION",
    "EXECUTION_CONTRACT_SCHEMA_VERSION",
    "SEMANTIC_FINGER_KEYPOINTS",
    "ScreenedCandidate",
    "SearchExecutionContract",
    "asset_derived_dof_blocks",
    "contributor_scores",
    "default_search_contracts",
    "deterministic_block_seeds",
    "rank_contributors",
    "retain_after_polish",
    "select_candidate",
]
