from __future__ import annotations

import copy

import pytest

from scripts.rl.isaaclab.run_stage16_pf_v2_symmetric_ppo import (
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
