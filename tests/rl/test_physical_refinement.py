from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.rl.isaaclab.run_physical_refinement import (
    _requires_refinement,
    _source,
    assert_symmetric_static_contracts,
)


def _contract() -> dict[str, object]:
    return {
        "runtime_static": {
            "reward_aggregation": {"mode": "grouped_multiplicative_v1"},
            "rse": {"enabled": True, "uniform_rsi_preserved": True},
            "physics": {"stage": "C4", "gravity_scale": 1.0, "friction_scale": 1.0},
            "controller": {"identifier": "finite_virtual_6d_wrist_actuator_v1"},
        },
        "runtime_static_sha256": "static-contract",
        "ppo_hyperparameters": {"rollout_steps": 40, "learning_rate": 0.0003},
        "ppo_hyperparameters_sha256": "ppo-contract",
        "reward_rse_mode": "grouped_multiplicative_v1_with_rse_v1",
        "max_new_updates": 10,
        "samples_per_update": 40960,
    }


def test_symmetric_contract_allows_only_source_and_clip_differences() -> None:
    first = _contract()
    second = copy.deepcopy(first)
    first.update({"clip": "hocap_170105", "source": {"checkpoint": "u10.pt"}})
    second.update({"clip": "hocap_170650", "source": {"checkpoint": "v4.pt"}})

    result = assert_symmetric_static_contracts(first, second)

    assert result["passed"] is True
    assert result["static_sha256"] == "static-contract"


def test_symmetric_contract_rejects_reward_rse_drift() -> None:
    first = _contract()
    second = _contract()
    second["reward_rse_mode"] = "other"

    with pytest.raises(RuntimeError, match="PF_V2_SYMMETRIC_STATIC_CONTRACT_DRIFT"):
        assert_symmetric_static_contracts(first, second)


def test_confirmed_acceptance_skips_ppo() -> None:
    assert _requires_refinement({"accepted": True}) is False


def test_unconfirmed_evaluation_requires_bounded_ppo() -> None:
    assert _requires_refinement({"accepted": False}) is True


def test_independent_l0_is_valid_source_for_physical_grouped_rse(tmp_path: Path) -> None:
    checkpoint = tmp_path / "l0.pt"
    torch.save(
        {
            "schema_version": "Stage16DPPO26DCheckpointV1",
            "clip": "hocap_heldout",
            "cumulative_samples": 1_024_000,
            "environment_contract": {
                "ppo26d": {"reward": {"identifier": "TopoRetargetReferenceTrackingReward26DV1"}}
            },
        },
        checkpoint,
    )
    result = tmp_path / "l0_training.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": "Stage16DPPO26DL0TrainingV1",
                "status": "STAGE16D_PPO26D_L0_COMPLETE_NOT_YET_QUALIFIED",
                "clip": "hocap_heldout",
                "cumulative_samples": 1_024_000,
                "target_l0_samples": 1_024_000,
                "l0_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    source = _source("hocap_heldout", independent={"source_training_result": result.resolve()})

    assert source["kind"] == "independent_l0_before_physical_grouped_rse"
    assert source["strict_v4_samples"] == 0
    assert source["checkpoint_sha256"]
