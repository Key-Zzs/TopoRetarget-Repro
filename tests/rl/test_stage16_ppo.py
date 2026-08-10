from __future__ import annotations

from pathlib import Path

import torch

from toporetarget.rl.ppo.checkpoint import load_checkpoint, save_checkpoint
from toporetarget.rl.ppo.distribution import SoftplusGaussian
from toporetarget.rl.ppo.gae import generalized_advantage_estimate
from toporetarget.rl.ppo.networks import ACTOR_HIDDEN, CRITIC_HIDDEN, ActorCritic
from toporetarget.rl.ppo.storage import RolloutStorage
from toporetarget.rl.ppo.trainer import PPOConfig, PPOTrainer


def test_paper_networks_distribution_gae_and_rollout_shape() -> None:
    assert ACTOR_HIDDEN == (512, 256, 128)
    assert CRITIC_HIDDEN == (512, 512, 256, 128)
    model = ActorCritic(7, 3)
    distribution = SoftplusGaussian(
        model.action_location(torch.zeros(2, 7)), model.log_std_parameter
    )
    assert torch.all(distribution.std > 0)
    assert torch.all(distribution.sample().abs() <= 1.0)
    rewards = torch.ones(40, 4096)
    values = torch.zeros(40, 4096)
    dones = torch.zeros(40, 4096, dtype=torch.bool)
    advantages, returns = generalized_advantage_estimate(rewards, values, dones, torch.zeros(4096))
    assert advantages.shape == returns.shape == (40, 4096)
    storage = RolloutStorage(
        torch.zeros(40, 4096, 7),
        torch.zeros(40, 4096, 3),
        torch.zeros(40, 4096),
        rewards,
        dones,
        values,
    )
    assert storage.sample_count == 163840


def test_ppo_update_normalization_and_checkpoint_reload(tmp_path: Path) -> None:
    directory = tmp_path
    trainer = PPOTrainer(7, 3, config=PPOConfig(epochs=1, minibatches=1))
    observations = torch.randn(4, 1, 7)
    with torch.no_grad():
        actions, log_probs, values = trainer.act(observations.reshape(-1, 7))
    storage = RolloutStorage(
        observations,
        actions.reshape(4, 1, 3),
        log_probs.reshape(4, 1),
        torch.ones(4, 1),
        torch.zeros(4, 1, dtype=torch.bool),
        values.reshape(4, 1),
    )
    metrics = trainer.update(storage, torch.zeros(1))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    destination = directory / "resume.pt"
    save_checkpoint(
        destination,
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "normalizer": trainer.normalizer.state_dict(),
        },
    )
    reloaded = load_checkpoint(destination)
    resumed = PPOTrainer(7, 3, config=PPOConfig(epochs=1, minibatches=1))
    resumed.model.load_state_dict(reloaded["model"])
    resumed.optimizer.load_state_dict(reloaded["optimizer"])
    resumed.normalizer.load_state_dict(reloaded["normalizer"])
    assert resumed.normalizer.count.item() == trainer.normalizer.count.item()


def test_squashed_gaussian_log_prob_has_jacobian_and_is_finite_near_bounds() -> None:
    location = torch.zeros(2, 3)
    raw_std = torch.zeros(3)
    distribution = SoftplusGaussian(location, raw_std)
    action = torch.full((2, 3), 0.5)
    pre_tanh = torch.atanh(action)
    expected = (distribution.normal.log_prob(pre_tanh) - torch.log1p(-action.square())).sum(dim=-1)
    assert torch.allclose(distribution.log_prob(action), expected, atol=1.0e-6)
    boundary = torch.tensor([[-1.0, -0.999999, 0.999999, 1.0]])
    boundary_distribution = SoftplusGaussian(torch.zeros_like(boundary), torch.zeros(4))
    assert torch.isfinite(boundary_distribution.log_prob(boundary)).all()


def test_target_kl_stops_remaining_ppo_epochs() -> None:
    torch.manual_seed(4)
    config = PPOConfig(
        learning_rate=1.0e-2,
        epochs=4,
        minibatches=1,
        target_kl=1.0e-12,
    )
    trainer = PPOTrainer(3, 2, config=config)
    observations = torch.randn(8, 1, 3)
    with torch.no_grad():
        actions, log_probs, values = trainer.act(observations.reshape(-1, 3))
    storage = RolloutStorage(
        observations=observations,
        actions=actions.reshape(8, 1, 2),
        log_probs=log_probs.reshape(8, 1),
        rewards=torch.linspace(0.0, 1.0, 8).reshape(8, 1),
        dones=torch.zeros(8, 1, dtype=torch.bool),
        values=values.reshape(8, 1),
    )
    metrics = trainer.update(storage, torch.zeros(1))
    assert metrics["kl_early_stop"] is True
    assert metrics["updates"] < config.epochs * config.minibatches
