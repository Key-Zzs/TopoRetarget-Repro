"""Fail-closed contracts for the table-supported full-trajectory P3 restart."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .physical_p3 import physical_stage_budget, preceding_stage
from .reference_tracking.contact_reward_mode import ContactRewardMode

FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA = "Stage16FullTrajectoryP3CheckpointV1"
FULL_TRAJECTORY_P3_RESULT_SCHEMA = "Stage16FullTrajectoryP3ResultV1"


def checkpoint_metadata(
    *,
    stage: str,
    stage_samples: int,
    cumulative_samples: int,
    policy_training_samples: int,
    mode: str | ContactRewardMode,
    episode_start: Mapping[str, Any],
    support_contract_hash: str,
    reference_hash: str,
    training_reset: str = "frame0",
) -> dict[str, object]:
    """Build immutable lineage metadata for a formal C0--C4 checkpoint."""

    selected_mode = ContactRewardMode.parse(mode)
    budget = physical_stage_budget(stage)
    if stage not in {"C0", "C1", "C2", "C3", "C4"}:
        raise ValueError("FULL_TRAJECTORY_P3_STAGE_FORBIDDEN")
    if not 0 <= stage_samples <= budget.additional_samples:
        raise ValueError("FULL_TRAJECTORY_P3_STAGE_SAMPLES_INVALID")
    if training_reset not in {"frame0", "uniform_rsi"}:
        raise ValueError("FULL_TRAJECTORY_P3_TRAINING_RESET_INVALID")
    if cumulative_samples < stage_samples or policy_training_samples < 0:
        raise ValueError("FULL_TRAJECTORY_P3_CUMULATIVE_SAMPLES_INVALID")
    start_index = episode_start.get("start_index")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("FULL_TRAJECTORY_P3_EPISODE_START_INVALID")
    if not isinstance(support_contract_hash, str) or len(support_contract_hash) != 64:
        raise ValueError("FULL_TRAJECTORY_P3_SUPPORT_HASH_INVALID")
    if not isinstance(reference_hash, str) or len(reference_hash) != 64:
        raise ValueError("FULL_TRAJECTORY_P3_REFERENCE_HASH_INVALID")
    return {
        "schema_version": FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA,
        "curriculum_stage": stage,
        "stage_samples": stage_samples,
        "cumulative_samples": cumulative_samples,
        "policy_training_samples": policy_training_samples,
        "contact_mode": selected_mode.value,
        "episode_start": dict(episode_start),
        "support_contract_hash": support_contract_hash,
        "reference_hash": reference_hash,
        "table_support": "finite_inferred_table_proxy_v1",
        "mid_trajectory_rsi": "uniform[0,320]" if training_reset == "uniform_rsi" else "disabled",
    }


def validate_resume_metadata(
    payload: Mapping[str, Any],
    *,
    clip: str,
    mode: str | ContactRewardMode,
    stage: str,
    num_envs: int,
    episode_start: Mapping[str, Any],
    support_contract_hash: str,
    reference_hash: str,
    training_reset: str = "frame0",
) -> dict[str, int | str]:
    """Accept only the direct predecessor with the identical start contract."""

    selected_mode = ContactRewardMode.parse(mode)
    if stage not in {"C1", "C2", "C3", "C4"}:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_TARGET_INVALID")
    if training_reset not in {"frame0", "uniform_rsi"}:
        raise ValueError("FULL_TRAJECTORY_P3_TRAINING_RESET_INVALID")
    if payload.get("schema_version") != FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_SCHEMA_INVALID")
    if payload.get("clip") != clip or int(payload.get("selected_num_envs", -1)) != num_envs:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_IDENTITY_INVALID")
    if payload.get("contact_mode") != selected_mode.value:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_MODE_INVALID")
    if payload.get("curriculum_stage") != preceding_stage(stage):
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_STAGE_ORDER_INVALID")
    if payload.get("episode_start") != dict(episode_start):
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_EPISODE_START_DRIFT")
    if payload.get("support_contract_hash") != support_contract_hash:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_SUPPORT_DRIFT")
    if payload.get("reference_hash") != reference_hash:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_REFERENCE_DRIFT")
    if payload.get("table_support") != "finite_inferred_table_proxy_v1":
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_TABLE_SUPPORT_INVALID")
    expected_rsi = "uniform[0,320]" if training_reset == "uniform_rsi" else "disabled"
    if payload.get("mid_trajectory_rsi") != expected_rsi:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_RSI_INVALID")
    cumulative = int(payload.get("cumulative_samples", -1))
    policy_samples = int(payload.get("policy_training_samples", -1))
    if cumulative < 0 or policy_samples < 0:
        raise ValueError("FULL_TRAJECTORY_P3_RESUME_SAMPLE_COUNTER_INVALID")
    return {
        "source_stage": str(payload["curriculum_stage"]),
        "cumulative_samples": cumulative,
        "policy_training_samples": policy_samples,
    }


__all__ = [
    "FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA",
    "FULL_TRAJECTORY_P3_RESULT_SCHEMA",
    "checkpoint_metadata",
    "validate_resume_metadata",
]
