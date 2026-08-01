from __future__ import annotations

from copy import deepcopy

import pytest

from toporetarget.rl.single_clip_ppo import (
    ADAPTIVE_ORACLE_VALIDATED,
    PPO_ENTRY_AUTHORIZED,
    SINGLE_CLIP_SAMPLE_LADDER,
    AdaptiveOracleBehaviorCloningV1,
    FrameZeroSingleClipCheckpointSelector,
    WorldWristPrefixCurriculumV1,
    deterministic_rollout_seed,
    frozen_single_clip_ppo_contract,
    oracle_authorizes_single_clip_ppo,
)


def test_single_clip_contract_is_bounded_and_26d() -> None:
    contract = frozen_single_clip_ppo_contract()
    assert contract["action_dim"] == 26
    assert tuple(contract["sample_ladder"]) == SINGLE_CLIP_SAMPLE_LADDER
    assert contract["sample_ladder"][-1] == 8_388_608
    assert not contract["clip_identity_in_observation"]


def test_frame_zero_selector_uses_formal_order() -> None:
    selector = FrameZeroSingleClipCheckpointSelector()
    base = {
        "success_rate": 0.9,
        "final_frame_reach_rate": 0.9,
        "progress_ratio": 0.95,
        "object_position_error_m": 0.01,
        "max_axis_point_error_m": 0.02,
        "object_rotation_error_deg": 5.0,
        "wrist_tracking_error": 0.01,
        "action_saturation_fraction": 0.1,
        "action_smoothness": 0.2,
    }
    lower = deepcopy(base)
    lower["success_rate"] = 0.85
    lower["object_position_error_m"] = 0.001
    assert selector.select([lower, base]) is base


def test_curriculum_bc_and_parallel_seeds_are_bounded() -> None:
    curriculum = WorldWristPrefixCurriculumV1()
    assert [curriculum.maximum_start_index(stage, 40) for stage in ("C0", "C1", "C2", "C3")] == [
        9,
        19,
        29,
        39,
    ]
    AdaptiveOracleBehaviorCloningV1().validate()
    with pytest.raises(ValueError, match="max_epochs"):
        AdaptiveOracleBehaviorCloningV1(max_epochs=51).validate()
    seeds = [deterministic_rollout_seed(20260801, index) for index in range(16)]
    assert len(set(seeds)) == 16
    assert seeds == [deterministic_rollout_seed(20260801, index) for index in range(16)]


def test_oracle_gate_requires_explicit_shared_pass() -> None:
    report = {
        "status": ADAPTIVE_ORACLE_VALIDATED,
        "ppo_entry": PPO_ENTRY_AUTHORIZED,
        "clips": [{"passes_gate": True}, {"passes_gate": True}],
    }
    assert oracle_authorizes_single_clip_ppo(report)
    report["clips"] = [{"passes_gate": True}]
    assert not oracle_authorizes_single_clip_ppo(report)
