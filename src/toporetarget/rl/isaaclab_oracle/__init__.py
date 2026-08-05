"""Stage 16-C.5A candidate-state replication primitives.

This package intentionally does not import Isaac Sim or Isaac Lab.  Runtime
scripts create the Isaac application first, then pass a live DirectRLEnv to
the capture/restore helpers below.
"""

from .action_history import CandidateActionHistoryV1
from .candidate_pool import PhysXOracleCandidatePoolV1
from .candidate_state import (
    capture_candidate_state,
    hash_candidate_state,
    replicate_candidate_state,
    restore_candidate_state,
    validate_candidate_state,
)
from .contracts import Stage16C5CandidateStateV1, Stage16C5WriteAuditV1
from .recovery import Stage16C5AR2RecoveryStateMachine, Stage16C5ARecoveryStateMachine

__all__ = [
    "CandidateActionHistoryV1",
    "PhysXOracleCandidatePoolV1",
    "Stage16C5ARecoveryStateMachine",
    "Stage16C5AR2RecoveryStateMachine",
    "Stage16C5CandidateStateV1",
    "Stage16C5WriteAuditV1",
    "capture_candidate_state",
    "hash_candidate_state",
    "replicate_candidate_state",
    "restore_candidate_state",
    "validate_candidate_state",
]
