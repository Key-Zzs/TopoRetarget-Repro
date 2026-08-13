"""Pure tests for the P3 PPO promotion and checkpoint rules."""

from __future__ import annotations

import pytest

from toporetarget.rl.gravity_friction_curriculum import INITIAL_SAFE_BANKS
from toporetarget.rl.physical_p3 import (
    PHYSICAL_PPO_CHECKPOINT_SCHEMA,
    checkpoint_state,
    physical_stage_budget,
    preceding_stage,
    validate_resume_payload,
)


def _state(stage: str) -> dict[str, object]:
    return {
        "curriculum_stage": stage,
        "selected_contact_mode": "aggregate_v3",
        "allowed_reset_banks": list(INITIAL_SAFE_BANKS),
    }


def _payload(stage: str = "C0") -> dict[str, object]:
    return {
        "schema_version": PHYSICAL_PPO_CHECKPOINT_SCHEMA,
        "clip": "hocap_170105",
        "selected_num_envs": 1024,
        "selected_contact_mode": "aggregate_v3",
        "curriculum_stage": stage,
        "curriculum_state": _state(stage),
        "physical_stage_samples": 1_048_576 if stage == "C0" else 0,
        "physical_cumulative_samples": 1_048_576,
        "policy_training_samples": 3_178_496,
    }


def test_pre_registered_budgets_and_stage_order() -> None:
    assert physical_stage_budget("C0").additional_samples == 1_048_576
    assert physical_stage_budget("C2").additional_samples == 2_097_152
    assert physical_stage_budget("C3").checkpoint_stage_samples == (2_097_152, 4_194_304)
    assert physical_stage_budget("C4").checkpoint_stage_samples == (
        1_048_576,
        2_097_152,
        4_194_304,
    )
    assert preceding_stage("C0") is None
    assert preceding_stage("C3") == "C2"


def test_checkpoint_state_carries_exact_physics_and_counters() -> None:
    state = checkpoint_state(
        stage="C2",
        physical_stage_samples=2_097_152,
        physical_cumulative_samples=4_194_304,
        policy_training_samples=6_324_224,
        selected_contact_mode="aggregate_v3",
        allowed_reset_banks=INITIAL_SAFE_BANKS,
        curriculum_state=_state("C2"),
    )
    assert state["physical_checkpoint_schema"] == PHYSICAL_PPO_CHECKPOINT_SCHEMA
    assert state["curriculum_stage"] == "C2"
    assert state["physical_cumulative_samples"] == 4_194_304
    with pytest.raises(ValueError, match="STAGE_PHYSICS_MISMATCH"):
        checkpoint_state(
            stage="C2",
            physical_stage_samples=0,
            physical_cumulative_samples=0,
            policy_training_samples=1,
            selected_contact_mode="aggregate_v3",
            allowed_reset_banks=INITIAL_SAFE_BANKS,
            curriculum_state=_state("C1"),
        )


def test_resume_only_allows_same_or_immediate_predecessor_stage() -> None:
    resumed = validate_resume_payload(
        _payload("C0"),
        expected_clip="hocap_170105",
        expected_num_envs=1024,
        expected_contact_mode="aggregate_v3",
        target_stage="C1",
    )
    assert resumed["physical_stage_samples"] == 0
    with pytest.raises(ValueError, match="STAGE_ORDER"):
        validate_resume_payload(
            _payload("C0"),
            expected_clip="hocap_170105",
            expected_num_envs=1024,
            expected_contact_mode="aggregate_v3",
            target_stage="C2",
        )
    with pytest.raises(ValueError, match="CONTACT_MODE"):
        validate_resume_payload(
            _payload("C0"),
            expected_clip="hocap_170105",
            expected_num_envs=1024,
            expected_contact_mode="strict_per_finger_v4",
            target_stage="C1",
        )
