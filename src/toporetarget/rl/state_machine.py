"""Persistent bounded failure-classification / repair / rerun state machine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .failure_classifier import FALLBACKS, FailureClass


@dataclass(frozen=True)
class RecoveryBudget:
    repairs_per_class: int = 3
    reruns_per_phase: int = 5
    backend_switches: int = 3
    major_repairs: int = 20


@dataclass
class FailureTransition:
    phase: str
    failure_class: FailureClass
    evidence: dict[str, Any]
    attempt: int
    fallback: str
    repair: str
    rerun_scope: str
    result: str
    remaining_budget: dict[str, int]


class Stage16RecoveryStateMachine:
    """Rejects unbounded retries and records every classified failure."""

    def __init__(self, budget: RecoveryBudget = RecoveryBudget()) -> None:
        self.budget = budget
        self._class_attempts: dict[FailureClass, int] = {}
        self._phase_reruns: dict[str, int] = {}
        self.backend_switch_count = 0
        self.major_repair_count = 0
        self.transitions: list[FailureTransition] = []
        self._written_count = 0

    @classmethod
    def from_jsonl(
        cls, path: str | Path, budget: RecoveryBudget = RecoveryBudget()
    ) -> Stage16RecoveryStateMachine:
        """Resume budgets from an append-only transition log without replaying rows."""

        machine = cls(budget)
        source = Path(path)
        if not source.is_file():
            return machine
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                failure_class = FailureClass(record["failure_class"])
                transition = FailureTransition(
                    phase=str(record["phase"]),
                    failure_class=failure_class,
                    evidence=dict(record["evidence"]),
                    attempt=int(record["attempt"]),
                    fallback=str(record["fallback"]),
                    repair=str(record["repair"]),
                    rerun_scope=str(record["rerun_scope"]),
                    result=str(record["result"]),
                    remaining_budget={
                        str(key): int(value)
                        for key, value in dict(record["remaining_budget"]).items()
                    },
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Stage16 recovery log at line {line_number}") from exc
            machine.transitions.append(transition)
            machine._class_attempts[failure_class] = max(
                machine._class_attempts.get(failure_class, 0), transition.attempt
            )
            machine._phase_reruns[transition.phase] = (
                machine._phase_reruns.get(transition.phase, 0) + 1
            )
            machine.major_repair_count += 1
            machine.backend_switch_count = max(
                machine.backend_switch_count,
                int(transition.evidence.get("backend_switch_count", 0)),
            )
        machine._written_count = len(machine.transitions)
        return machine

    def record(
        self,
        *,
        phase: str,
        failure_class: FailureClass,
        evidence: dict[str, Any],
        repair: str,
        rerun_scope: str,
        result: str = "RECORDED",
    ) -> FailureTransition:
        attempt = self._class_attempts.get(failure_class, 0) + 1
        phase_reruns = self._phase_reruns.get(phase, 0) + 1
        if attempt > self.budget.repairs_per_class:
            result = "ESCALATED_REPAIR_BUDGET_EXHAUSTED"
        elif phase_reruns > self.budget.reruns_per_phase:
            result = "ESCALATED_PHASE_RERUN_BUDGET_EXHAUSTED"
        self._class_attempts[failure_class] = attempt
        self._phase_reruns[phase] = phase_reruns
        self.major_repair_count += 1
        if self.major_repair_count > self.budget.major_repairs:
            result = "ESCALATED_MAJOR_REPAIR_BUDGET_EXHAUSTED"
        transition = FailureTransition(
            phase=phase,
            failure_class=failure_class,
            evidence=dict(evidence),
            attempt=attempt,
            fallback=FALLBACKS[failure_class],
            repair=repair,
            rerun_scope=rerun_scope,
            result=result,
            remaining_budget={
                "class_repairs": max(self.budget.repairs_per_class - attempt, 0),
                "phase_reruns": max(self.budget.reruns_per_phase - phase_reruns, 0),
                "backend_switches": max(
                    self.budget.backend_switches - self.backend_switch_count, 0
                ),
                "major_repairs": max(self.budget.major_repairs - self.major_repair_count, 0),
            },
        )
        self.transitions.append(transition)
        return transition

    def switch_backend(self, *, phase: str, evidence: dict[str, Any]) -> FailureTransition:
        self.backend_switch_count += 1
        result = (
            "RECORDED"
            if self.backend_switch_count <= self.budget.backend_switches
            else "ESCALATED_BACKEND_SWITCH_BUDGET_EXHAUSTED"
        )
        return self.record(
            phase=phase,
            failure_class=FailureClass.DEPENDENCY_FAILURE,
            evidence={**evidence, "backend_switch_count": self.backend_switch_count},
            repair="select_next_backend_in_fixed_decision_order",
            rerun_scope="backend_smoke",
            result=result,
        )

    def write_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            for transition in self.transitions[self._written_count :]:
                handle.write(json.dumps(asdict(transition), default=str, sort_keys=True) + "\n")
        self._written_count = len(self.transitions)
        return destination

    def summary(self) -> dict[str, Any]:
        return {
            "budget": asdict(self.budget),
            "transition_count": len(self.transitions),
            "class_attempts": {key.value: value for key, value in self._class_attempts.items()},
            "phase_reruns": dict(self._phase_reruns),
            "backend_switch_count": self.backend_switch_count,
            "major_repair_count": self.major_repair_count,
            "bounded": self.backend_switch_count <= self.budget.backend_switches
            and self.major_repair_count <= self.budget.major_repairs,
        }


__all__ = ["FailureTransition", "RecoveryBudget", "Stage16RecoveryStateMachine"]
