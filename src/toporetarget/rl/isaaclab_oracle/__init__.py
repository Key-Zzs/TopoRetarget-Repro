"""Stage 16-C.5A candidate-state, topology, and robust-oracle primitives.

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
from .robust import (
    RobustCandidateEvaluationV1,
    RobustCandidateEvaluatorV1,
    RobustCandidateSelector,
    RobustOracleContractV1,
    RobustReplicaResultV1,
    qualify_c5c_independent_replicas,
)
from .sharded_pool import CandidateShardV1, ShardDispatchRecordV1, ShardedCandidatePoolV1
from .topology import (
    ContactTopologyExperimentV1,
    balanced_shard_sizes,
    classify_contact_topology,
    r3_topology_matrix,
)

__all__ = [
    "CandidateActionHistoryV1",
    "PhysXOracleCandidatePoolV1",
    "Stage16C5ARecoveryStateMachine",
    "Stage16C5AR2RecoveryStateMachine",
    "Stage16C5CandidateStateV1",
    "Stage16C5WriteAuditV1",
    "CandidateShardV1",
    "ContactTopologyExperimentV1",
    "RobustCandidateEvaluationV1",
    "RobustCandidateEvaluatorV1",
    "RobustCandidateSelector",
    "RobustOracleContractV1",
    "RobustReplicaResultV1",
    "qualify_c5c_independent_replicas",
    "ShardDispatchRecordV1",
    "ShardedCandidatePoolV1",
    "balanced_shard_sizes",
    "capture_candidate_state",
    "classify_contact_topology",
    "hash_candidate_state",
    "replicate_candidate_state",
    "r3_topology_matrix",
    "restore_candidate_state",
    "validate_candidate_state",
]
