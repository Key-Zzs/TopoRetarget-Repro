"""Pure contracts for the Stage 16 P3 gravity/friction PPO continuation.

The Isaac runner intentionally keeps its simulator lifecycle in a script.  This
module contains the small, fail-closed portion that must be testable without
starting Kit: stage budgets, legal promotions, and checkpoint metadata checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .gravity_friction_curriculum import CURRICULUM_STAGES, INITIAL_SAFE_BANKS
from .reference_tracking.contact_reward_mode import ContactRewardMode

PHYSICAL_PPO_CHECKPOINT_SCHEMA = "Stage16P3GravityFrictionPPOCheckpointV1"
PHYSICAL_PPO_RESULT_SCHEMA = "Stage16P3GravityFrictionPPOResultV1"


@dataclass(frozen=True)
class PhysicalStageBudgetV1:
    """Pre-registered additional PPO samples for one curriculum stage."""

    stage: str
    additional_samples: int
    checkpoint_stage_samples: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.stage not in CURRICULUM_STAGES:
            raise ValueError("PHYSICAL_PPO_STAGE_UNKNOWN")
        if self.additional_samples <= 0:
            raise ValueError("PHYSICAL_PPO_STAGE_BUDGET_INVALID")
        if (
            not self.checkpoint_stage_samples
            or self.checkpoint_stage_samples[-1] != self.additional_samples
        ):
            raise ValueError("PHYSICAL_PPO_STAGE_CHECKPOINT_ENDPOINT_MISSING")
        if tuple(sorted(set(self.checkpoint_stage_samples))) != self.checkpoint_stage_samples:
            raise ValueError("PHYSICAL_PPO_STAGE_CHECKPOINTS_NOT_STRICT")
        if any(
            value <= 0 or value > self.additional_samples for value in self.checkpoint_stage_samples
        ):
            raise ValueError("PHYSICAL_PPO_STAGE_CHECKPOINT_RANGE_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


PHYSICAL_STAGE_BUDGETS: dict[str, PhysicalStageBudgetV1] = {
    "C0": PhysicalStageBudgetV1("C0", 1_048_576, (1_048_576,)),
    "C1": PhysicalStageBudgetV1("C1", 1_048_576, (1_048_576,)),
    "C2": PhysicalStageBudgetV1("C2", 2_097_152, (2_097_152,)),
    "C3": PhysicalStageBudgetV1("C3", 4_194_304, (2_097_152, 4_194_304)),
    "C4": PhysicalStageBudgetV1("C4", 4_194_304, (1_048_576, 2_097_152, 4_194_304)),
}


def physical_stage_budget(stage: str) -> PhysicalStageBudgetV1:
    """Return a frozen budget; callers may only reduce it for a discardable smoke."""

    try:
        return PHYSICAL_STAGE_BUDGETS[stage]
    except KeyError as exc:
        raise ValueError("PHYSICAL_PPO_STAGE_UNKNOWN") from exc


def preceding_stage(stage: str) -> str | None:
    """Return the only legal predecessor in the training-progress curriculum."""

    try:
        position = CURRICULUM_STAGES.index(stage)
    except ValueError as exc:
        raise ValueError("PHYSICAL_PPO_STAGE_UNKNOWN") from exc
    return None if position == 0 else CURRICULUM_STAGES[position - 1]


def checkpoint_state(
    *,
    stage: str,
    physical_stage_samples: int,
    physical_cumulative_samples: int,
    policy_training_samples: int,
    selected_contact_mode: str | ContactRewardMode,
    allowed_reset_banks: tuple[str, ...],
    curriculum_state: Mapping[str, object],
) -> dict[str, object]:
    """Build exact P3 checkpoint counters after enforcing immutable semantics."""

    mode = ContactRewardMode.parse(selected_contact_mode)
    budget = physical_stage_budget(stage)
    if not 0 <= physical_stage_samples <= budget.additional_samples:
        raise ValueError("PHYSICAL_PPO_STAGE_SAMPLE_COUNTER_INVALID")
    if physical_cumulative_samples < physical_stage_samples or policy_training_samples < 0:
        raise ValueError("PHYSICAL_PPO_CUMULATIVE_SAMPLE_COUNTER_INVALID")
    if tuple(allowed_reset_banks) != INITIAL_SAFE_BANKS:
        raise ValueError("PHYSICAL_PPO_RESET_BANKS_DRIFT")
    if curriculum_state.get("curriculum_stage") != stage:
        raise ValueError("PHYSICAL_PPO_CHECKPOINT_STAGE_PHYSICS_MISMATCH")
    if curriculum_state.get("selected_contact_mode") != mode.value:
        raise ValueError("PHYSICAL_PPO_CHECKPOINT_CONTACT_MODE_MISMATCH")
    checkpoint_banks = curriculum_state.get("allowed_reset_banks")
    if (
        not isinstance(checkpoint_banks, (list, tuple))
        or not all(isinstance(value, str) for value in checkpoint_banks)
        or tuple(checkpoint_banks) != INITIAL_SAFE_BANKS
    ):
        raise ValueError("PHYSICAL_PPO_CHECKPOINT_RESET_BANKS_MISMATCH")
    return {
        "physical_checkpoint_schema": PHYSICAL_PPO_CHECKPOINT_SCHEMA,
        "curriculum_stage": stage,
        "physical_stage_samples": physical_stage_samples,
        "physical_cumulative_samples": physical_cumulative_samples,
        "policy_training_samples": policy_training_samples,
        "selected_contact_mode": mode.value,
        "allowed_reset_banks": list(allowed_reset_banks),
        "curriculum_state": dict(curriculum_state),
    }


def validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_clip: str,
    expected_num_envs: int,
    expected_contact_mode: str | ContactRewardMode,
    target_stage: str,
) -> dict[str, object]:
    """Reject any resume that changes clip, parallelism, contact mode or physics order."""

    mode = ContactRewardMode.parse(expected_contact_mode)
    if payload.get("schema_version") != PHYSICAL_PPO_CHECKPOINT_SCHEMA:
        raise ValueError("PHYSICAL_PPO_RESUME_SCHEMA_INVALID")
    if payload.get("clip") != expected_clip:
        raise ValueError("PHYSICAL_PPO_RESUME_CLIP_MISMATCH")
    if int(payload.get("selected_num_envs", -1)) != expected_num_envs:
        raise ValueError("PHYSICAL_PPO_RESUME_ENV_COUNT_MISMATCH")
    if payload.get("selected_contact_mode") != mode.value:
        raise ValueError("PHYSICAL_PPO_RESUME_CONTACT_MODE_MISMATCH")
    predecessor = preceding_stage(target_stage)
    source_stage = payload.get("curriculum_stage")
    if source_stage not in {target_stage, predecessor}:
        raise ValueError("PHYSICAL_PPO_RESUME_STAGE_ORDER_INVALID")
    state = payload.get("curriculum_state")
    if not isinstance(state, Mapping) or state.get("curriculum_stage") != source_stage:
        raise ValueError("PHYSICAL_PPO_RESUME_CURRICULUM_STATE_INVALID")
    if state.get("selected_contact_mode") != mode.value:
        raise ValueError("PHYSICAL_PPO_RESUME_CURRICULUM_CONTACT_MODE_INVALID")
    if tuple(state.get("allowed_reset_banks", ())) != INITIAL_SAFE_BANKS:
        raise ValueError("PHYSICAL_PPO_RESUME_RESET_BANKS_INVALID")
    stage_samples = int(payload.get("physical_stage_samples", -1))
    cumulative = int(payload.get("physical_cumulative_samples", -1))
    policy_samples = int(payload.get("policy_training_samples", -1))
    if stage_samples < 0 or cumulative < stage_samples or policy_samples < 0:
        raise ValueError("PHYSICAL_PPO_RESUME_SAMPLE_COUNTER_INVALID")
    if (
        source_stage == target_stage
        and stage_samples > physical_stage_budget(target_stage).additional_samples
    ):
        raise ValueError("PHYSICAL_PPO_RESUME_STAGE_SAMPLE_COUNTER_EXCEEDS_BUDGET")
    return {
        "source_stage": source_stage,
        "physical_stage_samples": stage_samples if source_stage == target_stage else 0,
        "physical_cumulative_samples": cumulative,
        "policy_training_samples": policy_samples,
    }


__all__ = [
    "PHYSICAL_PPO_CHECKPOINT_SCHEMA",
    "PHYSICAL_PPO_RESULT_SCHEMA",
    "PHYSICAL_STAGE_BUDGETS",
    "PhysicalStageBudgetV1",
    "checkpoint_state",
    "physical_stage_budget",
    "preceding_stage",
    "validate_resume_payload",
]
