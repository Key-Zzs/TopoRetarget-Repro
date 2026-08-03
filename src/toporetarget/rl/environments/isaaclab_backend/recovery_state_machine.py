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
    """Track allowed transitions and close C.4/C.5 after a wrist-gate failure."""

    def __init__(self) -> None:
        self.stage = RecoveryStage.REPORT_SEMANTICS
        self.transitions: list[RecoveryTransition] = []
        self.free_root_qualification_runs = 0
        self.wrist_profile_runs = 0
        self.wrist_architecture_switches = 0
        self.c3_blocked = False

    def transition(self, target: RecoveryStage, *, reason: str) -> None:
        if self.stage == RecoveryStage.CLOSEOUT:
            raise RuntimeError("C3R2_C5_RECOVERY_CLOSED")
        if self.c3_blocked and target in {
            RecoveryStage.C4_BENCHMARK,
            RecoveryStage.C5_STATE_REPLICATION,
            RecoveryStage.C5_ORACLE,
        }:
            raise RuntimeError("C3_GATE_BLOCKS_C4_C5")
        self.transitions.append(RecoveryTransition(self.stage, target, reason))
        self.stage = target

    def record_free_root_run(self) -> None:
        self.free_root_qualification_runs += 1
        if self.free_root_qualification_runs > 2:
            raise RuntimeError("FREE_ROOT_QUALIFICATION_BUDGET_EXHAUSTED")

    def record_wrist_architecture_switch(self) -> None:
        self.wrist_architecture_switches += 1
        if self.wrist_architecture_switches > 1:
            raise RuntimeError("WRIST_ARCHITECTURE_SWITCH_BUDGET_EXHAUSTED")

    def record_wrist_profile_run(self) -> None:
        self.wrist_profile_runs += 1
        if self.wrist_profile_runs > 3:
            raise RuntimeError("WRIST_PROFILE_BUDGET_EXHAUSTED")

    def block_c3(self, *, reason: str) -> None:
        self.c3_blocked = True
        self.transition(RecoveryStage.CLOSEOUT, reason=reason)

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": "Stage16C3R2C5RecoveryStateMachine",
            "stage": self.stage.value,
            "c3_blocked": self.c3_blocked,
            "budgets": {
                "free_root_qualification_runs": self.free_root_qualification_runs,
                "free_root_qualification_runs_max": 2,
                "wrist_profile_runs": self.wrist_profile_runs,
                "wrist_profile_runs_max": 3,
                "wrist_architecture_switches": self.wrist_architecture_switches,
                "wrist_architecture_switches_max": 1,
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
