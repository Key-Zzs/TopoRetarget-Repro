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

_R4_PHASES = (
    "INPUT_FREEZE",
    "NATURAL_DISTRIBUTION",
    "REPLICATION_GATE",
    "POOL_BUILD",
    "CEM_SMOKE",
    "SHORT_ROLLOUT",
    "FULL_ORACLE",
    "FORMAL_EVAL",
    "CLOSEOUT",
)

_R4_FAILURES = {
    "POOL_OOM",
    "REPLICATION_DISTRIBUTION_FAIL",
    "CEM_COLLAPSE",
    "HIGH_VARIANCE",
    "FORMAL_GATE_FAIL",
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


@dataclass
class Stage16C5R4RecoveryStateMachine:
    """Exact bounded recovery policy for distributional R4/C5B/C5C."""

    phase: str = "INPUT_FREEZE"
    transitions: list[dict[str, object]] = field(default_factory=list)
    failures: list[dict[str, object]] = field(default_factory=list)
    candidate_scale_reductions: int = 0
    replica_upgrades: int = 0

    def transition(self, target: str, *, reason: str) -> None:
        if target not in _R4_PHASES:
            raise ValueError(f"unknown R4 recovery phase: {target}")
        source_index = _R4_PHASES.index(self.phase)
        target_index = _R4_PHASES.index(target)
        if target_index < source_index and target != self.phase:
            raise RuntimeError("R4 recovery transitions cannot move backward")
        self.transitions.append({"from": self.phase, "to": target, "reason": reason})
        self.phase = target

    def record_failure(self, failure_class: str, *, evidence: str) -> str:
        if failure_class not in _R4_FAILURES:
            raise ValueError(f"unknown R4 failure class: {failure_class}")
        action: str
        if failure_class == "POOL_OOM":
            if self.candidate_scale_reductions >= 1:
                action = "BLOCKED_POOL_OOM_AT_384"
            else:
                self.candidate_scale_reductions += 1
                action = "REDUCE_CANDIDATE_SCALE_TO_384"
        elif failure_class == "REPLICATION_DISTRIBUTION_FAIL":
            action = "AUDIT_SNAPSHOT_STATE_FIELDS_AND_METRICS_GATE_IMMUTABLE"
        elif failure_class == "CEM_COLLAPSE":
            action = "AUDIT_STD_FLOOR_ELITES_AND_SEED_DIVERSITY"
        elif failure_class == "HIGH_VARIANCE":
            if self.replica_upgrades >= 1:
                action = "RETAIN_HIGH_VARIANCE_FAILURE"
            else:
                self.replica_upgrades += 1
                action = "UPGRADE_REPLICAS_4_TO_8_ONCE"
        else:
            action = "RETAIN_FORMAL_FAILURE_ANALYZE_CONTACT_ORIENTATION_ACTION_NO_RELAXATION"
        self.failures.append(
            {
                "phase": self.phase,
                "failure_class": failure_class,
                "evidence": evidence,
                "action": action,
            }
        )
        return action

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "Stage16C5R4RecoveryStateMachine",
            "phase": self.phase,
            "phases": list(_R4_PHASES),
            "transitions": self.transitions,
            "failures": self.failures,
            "candidate_scale_reductions": self.candidate_scale_reductions,
            "replica_upgrades": self.replica_upgrades,
            "immutable_recovery_rules": {
                "POOL_OOM": "reduce once to 384 then block",
                "REPLICATION_DISTRIBUTION_FAIL": "audit state/metrics; never change gate",
                "CEM_COLLAPSE": "audit std floor/elites/seed diversity",
                "HIGH_VARIANCE": "replicas 4 to 8 once",
                "FORMAL_GATE_FAIL": "retain failure; never relax gates",
            },
        }


__all__ = [
    "Stage16C5ARecoveryStateMachine",
    "Stage16C5AR2RecoveryStateMachine",
    "Stage16C5R4RecoveryStateMachine",
]
