from __future__ import annotations

import pytest

from toporetarget.rl.full_trajectory_p3 import (
    checkpoint_metadata,
    validate_resume_metadata,
)
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode


def _start() -> dict[str, object]:
    return {"schema_version": "Stage16FullTrajectoryEpisodeStartV1", "start_index": 0}


def _checkpoint(stage: str = "C0") -> dict[str, object]:
    stage_samples = 1_048_576 if stage != "C2" else 2_097_152
    payload = checkpoint_metadata(
        stage=stage,
        stage_samples=stage_samples,
        cumulative_samples=stage_samples,
        policy_training_samples=123,
        mode=ContactRewardMode.AGGREGATE_V3,
        episode_start=_start(),
        support_contract_hash="a" * 64,
        reference_hash="b" * 64,
    )
    payload.update({"clip": "hocap_170105", "selected_num_envs": 1024})
    return payload


def test_resume_accepts_direct_predecessor_with_identical_contract() -> None:
    resumed = validate_resume_metadata(
        _checkpoint(),
        clip="hocap_170105",
        mode="aggregate_v3",
        stage="C1",
        num_envs=1024,
        episode_start=_start(),
        support_contract_hash="a" * 64,
        reference_hash="b" * 64,
    )
    assert resumed["source_stage"] == "C0"


def test_uniform_rsi_checkpoint_metadata_continues_to_c1() -> None:
    payload = _checkpoint()
    payload["mid_trajectory_rsi"] = "uniform[0,320]"
    resumed = validate_resume_metadata(
        payload,
        clip="hocap_170105",
        mode="aggregate_v3",
        stage="C1",
        num_envs=1024,
        episode_start=_start(),
        support_contract_hash="a" * 64,
        reference_hash="b" * 64,
        training_reset="uniform_rsi",
    )
    assert resumed["source_stage"] == "C0"


def test_resume_rejects_episode_start_drift() -> None:
    with pytest.raises(ValueError, match="EPISODE_START_DRIFT"):
        validate_resume_metadata(
            _checkpoint(),
            clip="hocap_170105",
            mode="aggregate_v3",
            stage="C1",
            num_envs=1024,
            episode_start={**_start(), "start_index": 1},
            support_contract_hash="a" * 64,
            reference_hash="b" * 64,
        )


def test_c3_and_c4_continue_only_from_the_direct_predecessor() -> None:
    c2 = _checkpoint("C2")
    c3 = validate_resume_metadata(
        c2,
        clip="hocap_170105",
        mode="aggregate_v3",
        stage="C3",
        num_envs=1024,
        episode_start=_start(),
        support_contract_hash="a" * 64,
        reference_hash="b" * 64,
    )
    assert c3["source_stage"] == "C2"
    c3_checkpoint = _checkpoint("C3")
    c4 = validate_resume_metadata(
        c3_checkpoint,
        clip="hocap_170105",
        mode="aggregate_v3",
        stage="C4",
        num_envs=1024,
        episode_start=_start(),
        support_contract_hash="a" * 64,
        reference_hash="b" * 64,
    )
    assert c4["source_stage"] == "C3"
