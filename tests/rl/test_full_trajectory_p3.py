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
    payload = checkpoint_metadata(
        stage=stage,
        stage_samples=1_048_576 if stage != "C2" else 2_097_152,
        cumulative_samples=1_048_576,
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
