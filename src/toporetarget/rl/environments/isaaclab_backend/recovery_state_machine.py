"""Fail-closed bounded recovery ledger for Stage 16-C.3R2 through C.5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RecoveryStage(StrEnum):
    REPORT_SEMANTICS = "REPORT_SEMANTICS"
    CONTACT_API_ISOLATION = "CONTACT_API_ISOLATION"
    CONTACT_READOUT = "CONTACT_READOUT"
    FREE_ROOT_FINAL_ATTEMPT = "FREE_ROOT_FINAL_ATTEMPT"
    WRIST_ARCHITECTURE_SWITCH = "WRIST_ARCHITECTURE_SWITCH"
    D6_WRAPPER_IMPORT = "D6_WRAPPER_IMPORT"
    WRIST_QUALIFICATION = "WRIST_QUALIFICATION"
    CONTACT_CAUSALITY = "CONTACT_CAUSALITY"
    C3_REQUALIFICATION = "C3_REQUALIFICATION"
    C4_BENCHMARK = "C4_BENCHMARK"
    C5_STATE_REPLICATION = "C5_STATE_REPLICATION"
    C5_ORACLE = "C5_ORACLE"
    CLOSEOUT = "CLOSEOUT"


@dataclass(frozen=True)
class RecoveryTransition:
    source: RecoveryStage
    target: RecoveryStage
    reason: str


class Stage16C3R2C5RecoveryStateMachine:
    """Fail-closed bounded ledger for the C.3R2 through C.5 decision tree."""

    _MAX_FAILURE_CLASS_REPAIRS = 3
    _MAX_PHASE_RERUNS = 5
    _MAX_FREE_ROOT_CONTROLLER_IMPLEMENTATIONS = 1
    _MAX_FREE_ROOT_QUALIFICATION_RUNS = 2
    _MAX_WRIST_ARCHITECTURE_SWITCHES = 1
    _MAX_CONTACT_API_STRATEGIES = 3
    _MAX_CEM_UPGRADES = 1
    _MAX_REPLICATION_SWITCHES = 1
    _MAX_MAJOR_TRANSITIONS = 36
    _MAX_WRIST_PROFILE_RUNS = 3

    def __init__(self) -> None:
        self.stage = RecoveryStage.REPORT_SEMANTICS
        self.transitions: list[RecoveryTransition] = []
        self.free_root_qualification_runs = 0
        self.free_root_controller_implementations = 0
        self.wrist_profile_runs = 0
        self.wrist_architecture_switches = 0
        self.contact_api_strategies = 0
        self.cem_upgrades = 0
        self.replication_switches = 0
        self.major_transitions = 0
        self.failure_class_repairs: dict[str, int] = {}
        self.phase_reruns: dict[str, int] = {}
        self.c3_blocked = False
        self.c3_validated = False
        self.c4_validated = False

    def transition(self, target: RecoveryStage, *, reason: str) -> None:
        if self.stage == RecoveryStage.CLOSEOUT:
            raise RuntimeError("C3R2_C5_RECOVERY_CLOSED")
        if self.major_transitions >= self._MAX_MAJOR_TRANSITIONS:
            raise RuntimeError("MAJOR_TRANSITION_BUDGET_EXHAUSTED")
        if self.c3_blocked and target in {
            RecoveryStage.C4_BENCHMARK,
            RecoveryStage.C5_STATE_REPLICATION,
            RecoveryStage.C5_ORACLE,
        }:
            raise RuntimeError("C3_GATE_BLOCKS_C4_C5")
        if target == RecoveryStage.C4_BENCHMARK and not self.c3_validated:
            raise RuntimeError("C3_VALIDATION_REQUIRED_FOR_C4")
        if (
            target
            in {
                RecoveryStage.C5_STATE_REPLICATION,
                RecoveryStage.C5_ORACLE,
            }
            and not self.c4_validated
        ):
            raise RuntimeError("C4_VALIDATION_REQUIRED_FOR_C5")
        self.transitions.append(RecoveryTransition(self.stage, target, reason))
        self.stage = target
        self.major_transitions += 1

    def record_failure_class_repair(self, failure_class: str) -> None:
        """Consume one bounded repair for a named, evidence-backed failure class."""

        repairs = self.failure_class_repairs.get(failure_class, 0)
        if repairs >= self._MAX_FAILURE_CLASS_REPAIRS:
            raise RuntimeError("FAILURE_CLASS_REPAIR_BUDGET_EXHAUSTED")
        self.failure_class_repairs[failure_class] = repairs + 1

    def record_phase_rerun(self, phase: str) -> None:
        """Consume one bounded rerun for a named phase."""

        reruns = self.phase_reruns.get(phase, 0)
        if reruns >= self._MAX_PHASE_RERUNS:
            raise RuntimeError("PHASE_RERUN_BUDGET_EXHAUSTED")
        self.phase_reruns[phase] = reruns + 1

    def record_contact_api_strategy(self) -> None:
        if self.contact_api_strategies >= self._MAX_CONTACT_API_STRATEGIES:
            raise RuntimeError("CONTACT_API_STRATEGY_BUDGET_EXHAUSTED")
        self.contact_api_strategies += 1

    def record_free_root_controller_implementation(self) -> None:
        if (
            self.free_root_controller_implementations
            >= self._MAX_FREE_ROOT_CONTROLLER_IMPLEMENTATIONS
        ):
            raise RuntimeError("FREE_ROOT_CONTROLLER_IMPLEMENTATION_BUDGET_EXHAUSTED")
        self.free_root_controller_implementations += 1

    def record_free_root_run(self) -> None:
        if self.free_root_qualification_runs >= self._MAX_FREE_ROOT_QUALIFICATION_RUNS:
            raise RuntimeError("FREE_ROOT_QUALIFICATION_BUDGET_EXHAUSTED")
        self.free_root_qualification_runs += 1

    def record_wrist_architecture_switch(self) -> None:
        if self.wrist_architecture_switches >= self._MAX_WRIST_ARCHITECTURE_SWITCHES:
            raise RuntimeError("WRIST_ARCHITECTURE_SWITCH_BUDGET_EXHAUSTED")
        self.wrist_architecture_switches += 1

    def record_wrist_profile_run(self) -> None:
        if self.wrist_profile_runs >= self._MAX_WRIST_PROFILE_RUNS:
            raise RuntimeError("WRIST_PROFILE_BUDGET_EXHAUSTED")
        self.wrist_profile_runs += 1

    def record_cem_upgrade(self) -> None:
        if self.cem_upgrades >= self._MAX_CEM_UPGRADES:
            raise RuntimeError("CEM_UPGRADE_BUDGET_EXHAUSTED")
        self.cem_upgrades += 1

    def record_replication_switch(self) -> None:
        if self.replication_switches >= self._MAX_REPLICATION_SWITCHES:
            raise RuntimeError("REPLICATION_SWITCH_BUDGET_EXHAUSTED")
        self.replication_switches += 1

    def validate_c3(self) -> None:
        if self.c3_blocked:
            raise RuntimeError("C3_CANNOT_VALIDATE_AFTER_BLOCK")
        self.c3_validated = True

    def validate_c4(self) -> None:
        if not self.c3_validated:
            raise RuntimeError("C3_VALIDATION_REQUIRED_FOR_C4")
        self.c4_validated = True

    def block_c3(self, *, reason: str) -> None:
        self.c3_blocked = True
        self.transition(RecoveryStage.CLOSEOUT, reason=reason)

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": "Stage16C3R2C5RecoveryStateMachine",
            "stage": self.stage.value,
            "c3_blocked": self.c3_blocked,
            "c3_validated": self.c3_validated,
            "c4_validated": self.c4_validated,
            "budgets": {
                "failure_class_repairs": self.failure_class_repairs,
                "failure_class_repairs_max": self._MAX_FAILURE_CLASS_REPAIRS,
                "phase_reruns": self.phase_reruns,
                "phase_reruns_max": self._MAX_PHASE_RERUNS,
                "free_root_controller_implementations": self.free_root_controller_implementations,
                "free_root_controller_implementations_max": (
                    self._MAX_FREE_ROOT_CONTROLLER_IMPLEMENTATIONS
                ),
                "free_root_qualification_runs": self.free_root_qualification_runs,
                "free_root_qualification_runs_max": self._MAX_FREE_ROOT_QUALIFICATION_RUNS,
                "wrist_profile_runs": self.wrist_profile_runs,
                "wrist_profile_runs_max": self._MAX_WRIST_PROFILE_RUNS,
                "wrist_architecture_switches": self.wrist_architecture_switches,
                "wrist_architecture_switches_max": self._MAX_WRIST_ARCHITECTURE_SWITCHES,
                "contact_api_strategies": self.contact_api_strategies,
                "contact_api_strategies_max": self._MAX_CONTACT_API_STRATEGIES,
                "cem_upgrades": self.cem_upgrades,
                "cem_upgrades_max": self._MAX_CEM_UPGRADES,
                "replication_switches": self.replication_switches,
                "replication_switches_max": self._MAX_REPLICATION_SWITCHES,
                "major_transitions": self.major_transitions,
                "major_transitions_max": self._MAX_MAJOR_TRANSITIONS,
            },
            "transitions": [
                {
                    "source": transition.source.value,
                    "target": transition.target.value,
                    "reason": transition.reason,
                }
                for transition in self.transitions
            ],
        }
