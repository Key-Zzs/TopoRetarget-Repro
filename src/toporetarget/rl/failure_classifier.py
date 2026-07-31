"""Bounded, evidence-carrying Stage-16 failure categories."""

from __future__ import annotations

from enum import StrEnum


class FailureClass(StrEnum):
    GIT_OR_BRANCH_FAILURE = "GIT_OR_BRANCH_FAILURE"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    ASSET_MAPPING_FAILURE = "ASSET_MAPPING_FAILURE"
    REFERENCE_FRAME_FAILURE = "REFERENCE_FRAME_FAILURE"
    INITIAL_PENETRATION_OR_EXPLOSION = "INITIAL_PENETRATION_OR_EXPLOSION"
    ACTUATOR_OR_PD_FAILURE = "ACTUATOR_OR_PD_FAILURE"
    OBJECT_DYNAMICS_FAILURE = "OBJECT_DYNAMICS_FAILURE"
    OBSERVATION_CONTRACT_FAILURE = "OBSERVATION_CONTRACT_FAILURE"
    REWARD_OR_TERMINATION_FAILURE = "REWARD_OR_TERMINATION_FAILURE"
    PPO_NUMERICAL_FAILURE = "PPO_NUMERICAL_FAILURE"
    GPU_OOM = "GPU_OOM"
    SIM_THROUGHPUT_FAILURE = "SIM_THROUGHPUT_FAILURE"
    LEARNING_STALL = "LEARNING_STALL"
    MULTICLIP_INTERFERENCE = "MULTICLIP_INTERFERENCE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    VISUALIZATION_FAILURE = "VISUALIZATION_FAILURE"
    ARTIFACT_OR_CHECKPOINT_CORRUPTION = "ARTIFACT_OR_CHECKPOINT_CORRUPTION"


FALLBACKS: dict[FailureClass, str] = {
    FailureClass.GIT_OR_BRANCH_FAILURE: "fail_closed_no_git_write",
    FailureClass.DEPENDENCY_FAILURE: "audit_then_isolated_environment",
    FailureClass.ASSET_MAPPING_FAILURE: "validate_asset_names_and_hashes",
    FailureClass.REFERENCE_FRAME_FAILURE: "kinematic_replay_and_transform_audit",
    FailureClass.INITIAL_PENETRATION_OR_EXPLOSION: "reset_preview_then_asset_or_pd_repair",
    FailureClass.ACTUATOR_OR_PD_FAILURE: "fixed_global_pd_qualification_grid",
    FailureClass.OBJECT_DYNAMICS_FAILURE: "free_object_reset_and_inertia_audit",
    FailureClass.OBSERVATION_CONTRACT_FAILURE: "contract_dimension_and_delay_test",
    FailureClass.REWARD_OR_TERMINATION_FAILURE: "literal_table4_boundary_test",
    FailureClass.PPO_NUMERICAL_FAILURE: "rollback_atomic_checkpoint_then_validate_logprob",
    FailureClass.GPU_OOM: "env_count_fallback_with_rollout_shards",
    FailureClass.SIM_THROUGHPUT_FAILURE: "correctness_backend_or_smaller_vector_batch",
    FailureClass.LEARNING_STALL: "zero_oracle_pd_and_normalization_diagnostics",
    FailureClass.MULTICLIP_INTERFERENCE: "fixed_manifest_then_bounded_two_clip_diagnosis",
    FailureClass.DATA_UNAVAILABLE: "preserve_inventory_and_skip_only_affected_protocol",
    FailureClass.VISUALIZATION_FAILURE: "numerical_dashboard_fallback",
    FailureClass.ARTIFACT_OR_CHECKPOINT_CORRUPTION: (
        "hash_validate_and_rollback_last_atomic_checkpoint"
    ),
}


__all__ = ["FALLBACKS", "FailureClass"]
