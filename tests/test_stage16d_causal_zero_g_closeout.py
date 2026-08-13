"""Regression coverage for the Stage16-D causal zero-g milestone interface."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from toporetarget.rl.reference_tracking.contact_reward_mode import (
    ContactRewardMode,
    Stage16DContactRewardConfigV1,
    build_contact_reward,
    legacy_contact_mode,
    resolve_contact_mode,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV3,
    TopoRetargetReferenceTrackingReward26DV4,
)
from toporetarget.rl.reference_tracking.reference_gated_contact import (
    reference_gated_contact_reward,
)
from toporetarget.rl.reference_tracking.strict_per_finger_contact import (
    strict_per_finger_contact_reward,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_new_contact_config_defaults_to_stable_aggregate_v3() -> None:
    config = Stage16DContactRewardConfigV1()
    assert config.mode is ContactRewardMode.AGGREGATE_V3
    assert config.as_dict()["reward"] == {"contact": {"mode": "aggregate_v3"}}


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("aggregate_v3", ContactRewardMode.AGGREGATE_V3),
        ("strict_per_finger_v4", ContactRewardMode.STRICT_PER_FINGER_V4),
    ],
)
def test_contact_config_parses_each_supported_mode(mode: str, expected: ContactRewardMode) -> None:
    parsed = Stage16DContactRewardConfigV1.from_mapping({"reward": {"contact": {"mode": mode}}})
    assert parsed.mode is expected


def test_invalid_contact_mode_fails_fast_without_fallback() -> None:
    with pytest.raises(ValueError, match="STAGE16D_CONTACT_MODE_INVALID"):
        Stage16DContactRewardConfigV1.from_mapping({"reward": {"contact": {"mode": "foo"}}})


def test_legacy_v3_and_v4_contracts_migrate_deterministically() -> None:
    v3 = "TopoRetargetReferenceTrackingReward26DV3"
    v4 = "TopoRetargetReferenceTrackingReward26DV4"
    assert legacy_contact_mode(v3) is ContactRewardMode.AGGREGATE_V3
    assert legacy_contact_mode(v4) is ContactRewardMode.STRICT_PER_FINGER_V4
    assert resolve_contact_mode(configured_mode=None, reward_contract_identifier=v3) is (
        ContactRewardMode.AGGREGATE_V3
    )
    assert resolve_contact_mode(configured_mode=None, reward_contract_identifier=v4) is (
        ContactRewardMode.STRICT_PER_FINGER_V4
    )
    assert (
        resolve_contact_mode(
            configured_mode=None,
            reward_contract_identifier="TopoRetargetReferenceTrackingReward26DV2",
        )
        is None
    )


def test_contact_reward_factory_selects_real_v3_and_v4_implementations() -> None:
    v3 = build_contact_reward("aggregate_v3", frozen_parameters={"lambda_c_n": 1.0})
    v4 = build_contact_reward(
        "strict_per_finger_v4",
        frozen_parameters={"lambda_tip_n": 1.0, "numerical_floor_n": 1.0e-4},
    )
    assert isinstance(v3, TopoRetargetReferenceTrackingReward26DV3)
    assert isinstance(v4, TopoRetargetReferenceTrackingReward26DV4)


def test_v3_and_v4_contact_smokes_are_finite_and_distinct() -> None:
    v3 = build_contact_reward(ContactRewardMode.AGGREGATE_V3, frozen_parameters={"lambda_c_n": 1.0})
    v3_result = reference_gated_contact_reward(
        reference_expected_mask=torch.tensor([[True, False, False, False, False]]),
        fingertip_object_pair_force_world=torch.tensor(
            [[[1.0, 0.0, 0.0]] * 5], dtype=torch.float32
        ),
        lambda_c_n=v3.contact_force_scale_lambda_n,
        contract=v3.contact_contract(),
    )
    v4 = build_contact_reward(
        ContactRewardMode.STRICT_PER_FINGER_V4,
        frozen_parameters={"lambda_tip_n": 1.0, "numerical_floor_n": 1.0e-4},
    )
    v4_result = strict_per_finger_contact_reward(
        source_contact_mask=torch.tensor([[True, False, False, False, False]]),
        fingertip_object_pair_force_world=torch.tensor(
            [[[1.0, 0.0, 0.0]] * 5], dtype=torch.float32
        ),
        pair_presence=torch.tensor([[True, True, True, True, True]]),
        lambda_tip_n=v4.contact_force_scale_lambda_tip_n,
        contract=v4.contact_contract(),
    )
    assert bool(torch.isfinite(v3_result["r_contact"]).all())
    assert bool(torch.isfinite(v4_result["r_contact_v4"]).all())
    assert v3_result["r_contact"].shape == v4_result["r_contact_v4"].shape == (1,)


def test_tracked_contact_config_exposes_the_stable_default() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/rl/stage16/stage16d_ppo26d_reward.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    assert (
        Stage16DContactRewardConfigV1.from_mapping(payload).mode is ContactRewardMode.AGGREGATE_V3
    )
    assert payload["reward"]["contact"]["available"] == [
        "aggregate_v3",
        "strict_per_finger_v4",
    ]


def test_durable_milestone_contract_freezes_only_the_closeout_boundary() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/rl/stage16/stage16d_causal_zero_g_milestone.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload == {
        "schema_version": "Stage16DCausalZeroGravityMilestoneV1",
        "milestone": {"name": "stage16d_causal_zero_gravity", "version": 1},
        "physics": {
            "causal": True,
            "gravity_mode": "zero",
            "support": "absent",
            "external_guidance": False,
            "rollout_object_state_write": False,
            "rollout_wrist_root_write": False,
        },
        "reference": {"kinematics": "v2"},
        "action": {"ppo26d": True},
        "evaluation": {"suite": "v2"},
        "contact_reward": {
            "default": "aggregate_v3",
            "available": ["aggregate_v3", "strict_per_finger_v4"],
        },
        "method_status": {
            "aggregate_v3": "stable_baseline",
            "strict_per_finger_v4": "experimental_partial",
        },
    }


def test_local_artifacts_remain_untracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", ".local/**"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
