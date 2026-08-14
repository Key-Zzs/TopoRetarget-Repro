from __future__ import annotations

import torch
from pytest import approx

from toporetarget.rl.ppo_optimization_attribution import (
    ROOT_CAUSES,
    actor_mean_update_direction,
    classify_kl_dynamics,
    decision_contract,
    negative_pressure_fraction,
    normalize_advantages,
    ppo_surrogate,
)


def test_decision_contract_enum_is_frozen() -> None:
    contract = decision_contract()
    assert contract["schema_version"] == "P3C12OptimizationAttributionDecisionV1"
    assert tuple(contract["root_cause_enum"]) == ROOT_CAUSES


def test_ppo_surrogate_and_advantage_normalization_match_trainer_math() -> None:
    ratio = torch.tensor([0.5, 1.0, 1.5])
    advantages = torch.tensor([-2.0, 0.0, 2.0])
    expected = torch.minimum(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages)
    assert torch.equal(ppo_surrogate(ratio, advantages, 0.2), expected)
    normalized = normalize_advantages(advantages)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0))
    assert torch.allclose(normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_negative_update_direction_sign_is_not_reversed() -> None:
    policy_gradient = torch.tensor([-2.0, 0.0, 3.0])
    update_direction = actor_mean_update_direction(policy_gradient)
    assert torch.equal(update_direction, torch.tensor([2.0, 0.0, -3.0]))
    assert negative_pressure_fraction(update_direction) == approx(1.0 / 3.0)


def test_kl_summary_reports_receipt_level_evidence_only() -> None:
    result = classify_kl_dynamics(
        [
            {"kl": 0.01, "clip_fraction": 0.2},
            {"kl": 0.06, "clip_fraction": 0.0},
        ]
    )
    assert result["status"] == "RECEIPT_LEVEL_EVIDENCE"
    assert result["fraction_kl_above_target"] == 0.5
    assert result["causal_direction_available"] is False
