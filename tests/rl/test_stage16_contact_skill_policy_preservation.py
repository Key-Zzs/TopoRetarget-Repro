from __future__ import annotations

import torch

from toporetarget.rl.ppo.policy_preservation import _configure_actor_lr_scale, validate_exact_batch
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer


def _batch() -> dict[str, object]:
    steps, environments = 2, 32
    return {
        "schema_version": "Stage16ContactCollapseExactPPOBatchV1",
        "rollout_steps": steps,
        "num_envs": environments,
        "observations": torch.zeros(steps, environments, 764),
        "actions": torch.zeros(steps, environments, 26),
        "old_log_probs": torch.zeros(steps, environments),
        "rewards": torch.zeros(steps, environments),
        "dones": torch.zeros(steps, environments),
        "values": torch.zeros(steps, environments),
        "returns": torch.zeros(steps, environments),
        "advantages": torch.zeros(steps, environments),
        "last_value": torch.zeros(environments),
        "reference_indices": torch.zeros(steps, environments, dtype=torch.long),
        "rng_before_optimizer_update": {},
    }


def test_exact_batch_validation_records_frozen_minibatch_semantics() -> None:
    result = validate_exact_batch(_batch())

    assert result["sample_count"] == 64
    assert result["observation_shape"] == [2, 32, 764]
    assert result["minibatch_order"] == "torch.Generator(device).manual_seed(0); randperm per epoch"


def test_exact_batch_validation_rejects_missing_rng_authority() -> None:
    batch = _batch()
    batch.pop("rng_before_optimizer_update")

    try:
        validate_exact_batch(batch)
    except ValueError as error:
        assert str(error) == "POLICY_PRESERVATION_RNG_MISSING"
    else:
        raise AssertionError("missing RNG authority was accepted")


def test_actor_lr_shadow_contract_keeps_critic_lr_at_baseline() -> None:
    trainer = PPO26DTrainer(observation_dim=764, device="cpu")

    result = _configure_actor_lr_scale(trainer, 0.5)

    assert result["actor_critic_shared_parameters"] is False
    assert result["effective_actor_lr"] == 5.0e-5
    assert result["critic_lr"] == 1.0e-4
    assert len(trainer.trainer.optimizer.param_groups) == 2
