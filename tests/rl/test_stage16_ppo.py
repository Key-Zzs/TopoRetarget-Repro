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
    distribution = SoftplusGaussian(model.mean(torch.zeros(2, 7)), model.log_std_parameter)
    assert torch.all(distribution.std > 0)
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
