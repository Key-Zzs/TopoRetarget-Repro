"""Fail-closed recovery ledger for reference-kinematics V2 and Phase 3.

The state machine intentionally has no Isaac Lab dependency.  Scripts can
record an auditable ordered transition before launching a simulator process,
and tests can validate the same recovery contract in the base environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage16DReferenceKinematicsPhase3StateMachine(str, Enum):
    """The ordered, bounded states authorized by the Phase 2.5/3 plan."""

    INPUT_FREEZE = "INPUT_FREEZE"
    V1_REFERENCE_AUDIT = "V1_REFERENCE_AUDIT"
    POSE_AUDIT = "POSE_AUDIT"
    TIMESTAMP_REPAIR = "TIMESTAMP_REPAIR"
    LINEAR_VELOCITY_REPAIR = "LINEAR_VELOCITY_REPAIR"
    ANGULAR_VELOCITY_REPAIR = "ANGULAR_VELOCITY_REPAIR"
    KINEMATICS_QUALIFICATION = "KINEMATICS_QUALIFICATION"
    PHASE1_RERUN = "PHASE1_RERUN"
    PHASE3_ENTRY = "PHASE3_ENTRY"
    OBSERVABILITY_AUDIT = "OBSERVABILITY_AUDIT"
    REWARD_V2_FREEZE = "REWARD_V2_FREEZE"
    GPU_PROBE = "GPU_PROBE"
    PHASE3_P1 = "PHASE3_P1"
    PHASE3_4M = "PHASE3_4M"
    PHASE3_16M = "PHASE3_16M"
    CHECKPOINT_SELECTION = "CHECKPOINT_SELECTION"
    FORMAL_EVALUATION = "FORMAL_EVALUATION"
    TRACE_EXPORT = "TRACE_EXPORT"
    DOCS = "DOCS"
    CLOSEOUT = "CLOSEOUT"


@dataclass
class ReferenceKinematicsPhase3Transitions:
    """Record only forward progress; failed gates must stop rather than rewind."""

    state: Stage16DReferenceKinematicsPhase3StateMachine = (
        Stage16DReferenceKinematicsPhase3StateMachine.INPUT_FREEZE
    )
    transitions: list[dict[str, str]] = field(default_factory=list)

    def transition(
        self,
        target: Stage16DReferenceKinematicsPhase3StateMachine,
        *,
        reason: str,
    ) -> None:
        if not reason:
            raise ValueError("Phase 3 transition requires an evidence reason")
        states = tuple(Stage16DReferenceKinematicsPhase3StateMachine)
        if states.index(target) <= states.index(self.state):
            raise ValueError("Phase 3 state must advance monotonically")
        self.transitions.append({"from": self.state.value, "to": target.value, "reason": reason})
        self.state = target


__all__ = [
    "ReferenceKinematicsPhase3Transitions",
    "Stage16DReferenceKinematicsPhase3StateMachine",
]
