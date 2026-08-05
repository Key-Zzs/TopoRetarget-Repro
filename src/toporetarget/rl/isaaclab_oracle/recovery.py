"""Bounded recovery transition ledger for Stage 16-C.5A."""

from __future__ import annotations

from dataclasses import dataclass, field

_PHASES = (
    "INPUT_FREEZE",
    "STATE_FIELD_AUDIT",
    "STATE_CONTRACT",
    "CANDIDATE_POOL",
    "NOISE_FLOOR",
    "TENSOR_CLONE",
    "HISTORY_REPLAY",
    "INDEPENDENCE",
    "RESOURCE_BENCHMARK",
    "CLOSEOUT",
)

_FAIL_CLOSED_FAILURES = {
    "SOURCE_HASH_DRIFT",
    "REFERENCE_HASH_DRIFT",
    "CANDIDATE_SETUP_WRITE_OUTSIDE_CANDIDATE_IDS",
    "EXECUTION_ROLLOUT_DIRECT_STATE_WRITE",
    "PHYSX_REPLICATION_BASELINE_NONDETERMINISM",
}

_R2_PHASES = (
    "INPUT_FREEZE",
    "P0_CURRENT_CONTRACT_AUDIT",
    "P1_API_AUDIT",
    "P2_CANDIDATE_MATRIX_FREEZE",
    "S0_SMOKE",
    "S1_PRECONTACT",
    "S2_CONTACT_ONSET",
    "S3_SUSTAINED_CONTACT",
    "S4_POSTCONTACT",
    "S5_FULL_REPLICATION",
    "CPU_DIAGNOSTIC",
    "CONTRACT_SELECTION",
    "C3P_SEMANTIC_REGRESSION",
    "C4P_RESOURCE_BENCHMARK",
    "O1_REQUALIFICATION",
    "CLOSEOUT",
)

_R2_FAIL_CLOSED_FAILURES = {
    "SOURCE_HASH_DRIFT",
    "REFERENCE_HASH_DRIFT",
    "CONTROLLER_HASH_DRIFT",
    "PHYSICS_BASELINE_HASH_DRIFT",
    "CANDIDATE_ADDED_AFTER_FREEZE",
    "RUNTIME_CONFIG_FALLBACK",
    "DIRECT_OBJECT_OR_WRIST_EXECUTION_WRITE",
    "SELECTED_CONTRACT_C3P_FAILURE",
    "SELECTED_CONTRACT_C4P_FAILURE",
}


@dataclass
class Stage16C5ARecoveryStateMachine:
    """Records bounded repairs and allows exactly one replication-method switch."""

    phase: str = "INPUT_FREEZE"
    transitions: list[dict[str, object]] = field(default_factory=list)
    failure_counts: dict[str, int] = field(default_factory=dict)
    phase_reruns: dict[str, int] = field(default_factory=dict)
    replication_method_switches: int = 0
    terminal_failure: str | None = None

    def transition(self, target: str, *, reason: str) -> None:
        if self.terminal_failure is not None:
            raise RuntimeError(f"STAGE16C5A_FAIL_CLOSED:{self.terminal_failure}")
        if target not in _PHASES:
            raise ValueError(f"unknown C.5A phase: {target}")
        if len(self.transitions) >= 24:
            raise RuntimeError("STAGE16C5A_MAJOR_TRANSITION_BUDGET_EXHAUSTED")
        self.transitions.append({"from": self.phase, "to": target, "reason": reason})
        self.phase = target

    def record_failure(self, failure_class: str, *, rerun_phase: bool = True) -> None:
        count = self.failure_counts.get(failure_class, 0) + 1
        if count > 3:
            raise RuntimeError(f"STAGE16C5A_REPAIR_BUDGET_EXHAUSTED:{failure_class}")
        self.failure_counts[failure_class] = count
        if rerun_phase:
            reruns = self.phase_reruns.get(self.phase, 0) + 1
            if reruns > 5:
                raise RuntimeError(f"STAGE16C5A_PHASE_RERUN_BUDGET_EXHAUSTED:{self.phase}")
            self.phase_reruns[self.phase] = reruns

    def fail_closed(self, failure_class: str, *, reason: str) -> None:
        """Terminate C.5A without consuming a repair or fallback budget."""

        if failure_class not in _FAIL_CLOSED_FAILURES:
            raise ValueError(f"failure class is not fail-closed: {failure_class}")
        self.terminal_failure = failure_class
        self.transitions.append(
            {
                "from": self.phase,
                "to": "CLOSEOUT",
                "reason": reason,
                "failure_class": failure_class,
                "status": "FAIL_CLOSED",
            }
        )
        self.phase = "CLOSEOUT"

    def switch_to_history_replay(self, *, reason: str) -> None:
        if self.replication_method_switches >= 1:
            raise RuntimeError("STAGE16C5A_REPLICATION_SWITCH_BUDGET_EXHAUSTED")
        self.replication_method_switches += 1
        self.transition("HISTORY_REPLAY", reason=reason)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "Stage16C5ARecoveryStateMachine",
            "phase": self.phase,
            "transitions": self.transitions,
            "failure_counts": self.failure_counts,
            "phase_reruns": self.phase_reruns,
            "replication_method_switches": self.replication_method_switches,
            "terminal_failure": self.terminal_failure,
            "limits": {
                "per_failure_class_repairs": 3,
                "phase_reruns": 5,
                "replication_method_switches": 1,
                "major_transitions": 24,
            },
        }


@dataclass
class Stage16C5AR2RecoveryStateMachine:
    """Fail-closed ledger for frozen PhysX-contract qualification only."""

    phase: str = "INPUT_FREEZE"
    transitions: list[dict[str, object]] = field(default_factory=list)
    failure_counts: dict[str, int] = field(default_factory=dict)
    stage_retries: dict[str, int] = field(default_factory=dict)
    gpu_candidates_started: set[str] = field(default_factory=set)
    cpu_candidates_started: set[str] = field(default_factory=set)
    replication_method_switches: int = 0
    terminal_failure: str | None = None

    def transition(self, target: str, *, reason: str) -> None:
        if self.terminal_failure is not None:
            raise RuntimeError(f"STAGE16C5A_R2_FAIL_CLOSED:{self.terminal_failure}")
        if target not in _R2_PHASES:
            raise ValueError(f"unknown C.5A-R2 phase: {target}")
        if len(self.transitions) >= 32:
            raise RuntimeError("STAGE16C5A_R2_MAJOR_TRANSITION_BUDGET_EXHAUSTED")
        self.transitions.append({"from": self.phase, "to": target, "reason": reason})
        self.phase = target

    def start_candidate(self, candidate_id: str, *, device_kind: str) -> None:
        if device_kind == "gpu":
            if (
                candidate_id not in self.gpu_candidates_started
                and len(self.gpu_candidates_started) >= 6
            ):
                raise RuntimeError("STAGE16C5A_R2_GPU_CANDIDATE_BUDGET_EXHAUSTED")
            self.gpu_candidates_started.add(candidate_id)
        elif device_kind == "cpu":
            if (
                candidate_id not in self.cpu_candidates_started
                and len(self.cpu_candidates_started) >= 1
            ):
                raise RuntimeError("STAGE16C5A_R2_CPU_CANDIDATE_BUDGET_EXHAUSTED")
            self.cpu_candidates_started.add(candidate_id)
        else:
            raise ValueError(f"unknown device kind: {device_kind}")

    def record_failure(self, failure_class: str, *, retry_stage: bool = True) -> None:
        count = self.failure_counts.get(failure_class, 0) + 1
        if count > 3:
            raise RuntimeError(f"STAGE16C5A_R2_REPAIR_BUDGET_EXHAUSTED:{failure_class}")
        self.failure_counts[failure_class] = count
        if retry_stage:
            retries = self.stage_retries.get(self.phase, 0) + 1
            if retries > 2:
                raise RuntimeError(f"STAGE16C5A_R2_STAGE_RETRY_BUDGET_EXHAUSTED:{self.phase}")
            self.stage_retries[self.phase] = retries

    def fail_closed(self, failure_class: str, *, reason: str) -> None:
        if failure_class not in _R2_FAIL_CLOSED_FAILURES:
            raise ValueError(f"failure class is not fail-closed: {failure_class}")
        self.terminal_failure = failure_class
        self.transitions.append(
            {
                "from": self.phase,
                "to": "CLOSEOUT",
                "reason": reason,
                "failure_class": failure_class,
                "status": "FAIL_CLOSED",
            }
        )
        self.phase = "CLOSEOUT"

    def switch_replication_method(self, *, reason: str) -> None:
        if self.replication_method_switches >= 1:
            raise RuntimeError("STAGE16C5A_R2_REPLICATION_SWITCH_BUDGET_EXHAUSTED")
        self.replication_method_switches += 1
        self.transition("S5_FULL_REPLICATION", reason=reason)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "Stage16C5AR2RecoveryStateMachine",
            "phase": self.phase,
            "transitions": self.transitions,
            "failure_counts": self.failure_counts,
            "stage_retries": self.stage_retries,
            "gpu_candidates_started": sorted(self.gpu_candidates_started),
            "cpu_candidates_started": sorted(self.cpu_candidates_started),
            "replication_method_switches": self.replication_method_switches,
            "terminal_failure": self.terminal_failure,
            "limits": {
                "gpu_candidates": 6,
                "cpu_diagnostic_candidates": 1,
                "retries_per_candidate_stage": 2,
                "per_failure_class_repairs": 3,
                "replication_method_switches": 1,
                "major_transitions": 32,
            },
        }


__all__ = ["Stage16C5ARecoveryStateMachine", "Stage16C5AR2RecoveryStateMachine"]
