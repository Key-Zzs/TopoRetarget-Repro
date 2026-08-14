"""Pure PPO diagnostics used by the bounded C1.2 attribution pass.

The functions in this module mirror the checked-in PPO implementation but do
not own a model, optimizer, rollout, or simulator.  This keeps the attribution
pass suitable for stored receipts and makes the sign convention testable
without performing an optimizer step.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

OPTIMIZATION_ATTRIBUTION_DECISION_SCHEMA = "P3C12OptimizationAttributionDecisionV1"
ROOT_CAUSES = (
    "STATE_DISTRIBUTION_SHIFT_PRIMARY",
    "OBS_NORMALIZER_DRIFT_PRIMARY",
    "ADVANTAGE_SIGNAL_PRIMARY",
    "REWARD_SIGNAL_PRIMARY",
    "PPO_SURROGATE_OPTIMIZATION_PRIMARY",
    "PPO_CLIPPING_KL_DYNAMICS_PRIMARY",
    "POLICY_DISTRIBUTION_COLLAPSE_PRIMARY",
    "TANH_PARAMETERIZATION_PRIMARY",
    "CRITIC_ADVANTAGE_SCALING_PRIMARY",
    "RESIDUAL_AUTHORITY_EXHAUSTION_PRIMARY",
    "MULTI_FACTOR_PRIMARY",
    "INCONCLUSIVE",
)


def decision_contract() -> dict[str, object]:
    """Return the pre-registered C1.2 decision contract."""

    return {
        "schema_version": OPTIMIZATION_ATTRIBUTION_DECISION_SCHEMA,
        "root_cause_enum": list(ROOT_CAUSES),
        "fixed_probe": {
            "required": True,
            "minimum_observations": 1024,
            "raw_observation_contract": "Stage16DPPO26DObservationV2",
            "must_not_reconstruct": True,
        },
        "gradient_sign": {
            "definition": "update_direction = -d(actor_policy_loss)/d(pre_tanh_mean)",
            "negative_means_more_negative": True,
        },
        "reward_strong_contributor_gate": {
            "pressure_reduction_fraction": 0.30,
            "minimum_consistent_late_batches": 2,
            "is_engineering_attribution_threshold": True,
        },
        "confidence": {
            "high": "two independent causal evidence classes",
            "medium": "one strong causal evidence class",
            "low": "correlation or aggregate metrics only",
        },
        "forbidden_actions": [
            "authoritative PPO training",
            "reward or PPO hyperparameter changes",
            "action bound or mapping changes",
            "C1 retry",
            "C2/G3/C3/C4/P4/V4 formal training",
        ],
    }


def ppo_surrogate(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_epsilon: float
) -> torch.Tensor:
    """Compute the exact elementwise clipped PPO surrogate used by the trainer."""

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return torch.minimum(ratio * advantages, clipped_ratio * advantages)


def normalize_advantages(advantages: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Match the trainer's population-standard-deviation normalization."""

    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + epsilon)


def actor_mean_update_direction(policy_loss_gradient: torch.Tensor) -> torch.Tensor:
    """Return the direction of a unit learning-rate gradient-descent update.

    A negative value means the optimizer update decreases the pre-tanh mean.
    The function intentionally omits Adam state and gradient clipping; it is a
    sign convention helper, not an optimizer replay.
    """

    return -policy_loss_gradient


def negative_pressure_fraction(update_direction: torch.Tensor) -> float:
    """Fraction of finite entries whose update direction is negative."""

    finite = update_direction[torch.isfinite(update_direction)]
    if finite.numel() == 0:
        return 0.0
    return float((finite < 0.0).float().mean())


def classify_kl_dynamics(rows: list[Mapping[str, object]]) -> dict[str, object]:
    """Summarize the receipt-level KL/clip pattern without calling it causal."""

    kl_values = [
        float(value) for row in rows for value in [row.get("kl")] if isinstance(value, (int, float))
    ]
    clip_values = [
        float(value)
        for row in rows
        for value in [row.get("clip_fraction")]
        if isinstance(value, (int, float))
    ]
    if not kl_values or not clip_values:
        return {"status": "UNAVAILABLE", "reason": "missing PPO KL/clip metrics"}
    target = 0.03
    return {
        "status": "RECEIPT_LEVEL_EVIDENCE",
        "target_kl": target,
        "kl_mean": sum(kl_values) / len(kl_values),
        "kl_max": max(kl_values),
        "fraction_kl_above_target": sum(value > target for value in kl_values) / len(kl_values),
        "clip_fraction_mean": sum(clip_values) / len(clip_values),
        "fraction_clip_zero": sum(value == 0.0 for value in clip_values) / len(clip_values),
        "causal_direction_available": False,
    }


__all__ = [
    "OPTIMIZATION_ATTRIBUTION_DECISION_SCHEMA",
    "ROOT_CAUSES",
    "actor_mean_update_direction",
    "classify_kl_dynamics",
    "decision_contract",
    "negative_pressure_fraction",
    "normalize_advantages",
    "ppo_surrogate",
]
