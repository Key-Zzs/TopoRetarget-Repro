"""Bounded Stage-16B adaptive-oracle and single-clip PPO recovery ledgers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class AdaptiveOracleFailure(StrEnum):
    ADAPTIVE_SELECTOR_WRONG_HORIZON = "ADAPTIVE_SELECTOR_WRONG_HORIZON"
    TERMINAL_HORIZON_HANDLING_FAILURE = "TERMINAL_HORIZON_HANDLING_FAILURE"
    MPC_PREDICTION_EXECUTION_MISMATCH = "MPC_PREDICTION_EXECUTION_MISMATCH"
    CEM_CONTACT_MODE_MISS = "CEM_CONTACT_MODE_MISS"
    GATE_BARRIER_UNDERWEIGHTED = "GATE_BARRIER_UNDERWEIGHTED"
    PHYSICS_OR_REFERENCE_DRIFT = "PHYSICS_OR_REFERENCE_DRIFT"


class SingleClipPPOFailure(StrEnum):
    PPO_NUMERICAL_FAILURE = "PPO_NUMERICAL_FAILURE"
    PPO_NO_LEARNING = "PPO_NO_LEARNING"
    PPO_ACTION_SATURATION = "PPO_ACTION_SATURATION"
    PPO_PREFIX_BIAS = "PPO_PREFIX_BIAS"
    PPO_CRITIC_COLLAPSE = "PPO_CRITIC_COLLAPSE"
    PPO_STALL_AFTER_ORACLE_SUCCESS = "PPO_STALL_AFTER_ORACLE_SUCCESS"


@dataclass(frozen=True)
class Stage16BTransition:
    failure: str
    attempt: int
    evidence: dict[str, Any]
    fallback: str
    repair: str
    rerun: str
    result: str
    remaining_class_repairs: int
    remaining_formal_reruns: int
    remaining_major_transitions: int
    remaining_budget_upgrades: int | None = None


class _BoundedStateMachine:
    def __init__(
        self,
        *,
        repairs_per_class: int,
        formal_reruns: int,
        major_transitions: int,
        budget_upgrades: int | None,
    ) -> None:
        self.repairs_per_class = repairs_per_class
        self.formal_reruns = formal_reruns
        self.major_transitions = major_transitions
        self.budget_upgrades = budget_upgrades
        self.class_attempts: dict[str, int] = {}
        self.rerun_count = 0
        self.major_count = 0
        self.upgrade_count = 0
        self.transitions: list[Stage16BTransition] = []

    def record(
        self,
        *,
        failure: StrEnum,
        evidence: dict[str, Any],
        fallback: str,
        repair: str,
        rerun: str,
        result: str,
        budget_upgrade: bool = False,
    ) -> Stage16BTransition:
        key = str(failure)
        attempt = self.class_attempts.get(key, 0) + 1
        self.class_attempts[key] = attempt
        self.rerun_count += 1
        self.major_count += 1
        if budget_upgrade:
            self.upgrade_count += 1
        if attempt > self.repairs_per_class:
            result = "BLOCKED_CLASS_REPAIR_BUDGET_EXHAUSTED"
        elif self.rerun_count > self.formal_reruns:
            result = "BLOCKED_FORMAL_RERUN_BUDGET_EXHAUSTED"
        elif self.major_count > self.major_transitions:
            result = "BLOCKED_MAJOR_TRANSITION_BUDGET_EXHAUSTED"
        elif self.budget_upgrades is not None and self.upgrade_count > self.budget_upgrades:
            result = "BLOCKED_BUDGET_UPGRADE_EXHAUSTED"
        transition = Stage16BTransition(
            failure=key,
            attempt=attempt,
            evidence=dict(evidence),
            fallback=fallback,
            repair=repair,
            rerun=rerun,
            result=result,
            remaining_class_repairs=max(self.repairs_per_class - attempt, 0),
            remaining_formal_reruns=max(self.formal_reruns - self.rerun_count, 0),
            remaining_major_transitions=max(self.major_transitions - self.major_count, 0),
            remaining_budget_upgrades=(
                None
                if self.budget_upgrades is None
                else max(self.budget_upgrades - self.upgrade_count, 0)
            ),
        )
        self.transitions.append(transition)
        return transition

    def as_dict(self) -> dict[str, Any]:
        return {
            "bounded": self.major_count <= self.major_transitions
            and self.rerun_count <= self.formal_reruns
            and all(value <= self.repairs_per_class for value in self.class_attempts.values())
            and (self.budget_upgrades is None or self.upgrade_count <= self.budget_upgrades),
            "repairs_per_class": self.repairs_per_class,
            "formal_reruns": self.formal_reruns,
            "major_transitions": self.major_transitions,
            "budget_upgrades": self.budget_upgrades,
            "class_attempts": dict(self.class_attempts),
            "rerun_count": self.rerun_count,
            "major_count": self.major_count,
            "upgrade_count": self.upgrade_count,
            "transitions": [asdict(value) for value in self.transitions],
        }


class Stage16BAdaptiveOracleStateMachine(_BoundedStateMachine):
    def __init__(self) -> None:
        super().__init__(
            repairs_per_class=3,
            formal_reruns=5,
            major_transitions=12,
            budget_upgrades=1,
        )


class Stage16BSingleClipPPOStateMachine(_BoundedStateMachine):
    def __init__(self) -> None:
        super().__init__(
            repairs_per_class=3,
            formal_reruns=5,
            major_transitions=16,
            budget_upgrades=None,
        )


__all__ = [
    "AdaptiveOracleFailure",
    "SingleClipPPOFailure",
    "Stage16BAdaptiveOracleStateMachine",
    "Stage16BSingleClipPPOStateMachine",
    "Stage16BTransition",
]
