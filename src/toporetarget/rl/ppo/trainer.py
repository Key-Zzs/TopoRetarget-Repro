"""Explicit PPO update math for the Stage-16 contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .distribution import SoftplusGaussian
from .gae import generalized_advantage_estimate
from .networks import ActorCritic
from .normalization import RunningObservationNormalizer
from .storage import RolloutStorage


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 32
    entropy_coefficient: float = 0.001
    clip_epsilon: float = 0.2
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.03

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PPOTrainer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        config: PPOConfig = PPOConfig(),
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.normalizer = RunningObservationNormalizer(observation_dim)

    def distribution(self, observations: torch.Tensor) -> SoftplusGaussian:
        normalized = self.normalizer.normalize(observations)
        return SoftplusGaussian(
            self.model.action_location(normalized), self.model.log_std_parameter
        )

    def update_observation_normalizer(self, observations: torch.Tensor) -> None:
        """Update statistics outside a collection/update transaction."""

        self.normalizer.update(observations)

    def freeze_observation_normalizer(self) -> None:
        """Freeze accumulated normalization statistics for deterministic evaluation."""

        self.normalizer.training = False

    @torch.no_grad()
    def act(
        self, observations: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.model.value(self.normalizer.normalize(observations))
        distribution = self.distribution(observations)
        action = distribution.mean if deterministic else distribution.sample()
        return action, distribution.log_prob(action), values

    def update(self, storage: RolloutStorage, last_value: torch.Tensor) -> dict[str, float | bool]:
        advantages, returns = generalized_advantage_estimate(
            storage.rewards,
            storage.values,
            storage.dones,
            last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        if not bool(torch.isfinite(advantages).all()) or not bool(torch.isfinite(returns).all()):
            raise FloatingPointError("PPO GAE produced NaN or Inf")
        flat = storage.flatten(advantages, returns)
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (
            flat["advantages"].std(unbiased=False) + 1e-8
        )
        count = storage.sample_count
        if count % self.config.minibatches != 0:
            raise ValueError("PPO sample count must divide evenly into configured minibatches")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0)
        accumulators = {
            "actor_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "ratio": 0.0,
            "action_std": 0.0,
        }
        updates = 0
        kl_early_stop = False
        completed_epochs = 0
        for _ in range(self.config.epochs):
            order = torch.randperm(count, generator=generator, device=self.device)
            for indices in order.chunk(self.config.minibatches):
                observations = flat["observations"][indices].to(self.device)
                actions = flat["actions"][indices].to(self.device)
                old_log_probs = flat["log_probs"][indices].to(self.device)
                advantages_batch = flat["advantages"][indices].to(self.device)
                returns_batch = flat["returns"][indices].to(self.device)
                old_values = flat["values"][indices].to(self.device)
                distribution = self.distribution(observations)
                new_log_probs = distribution.log_prob(actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate = torch.minimum(
                    ratio * advantages_batch,
                    torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                    * advantages_batch,
                )
                actor_loss = -surrogate.mean()
                value_prediction = self.model.value(self.normalizer.normalize(observations))
                value_loss = torch.nn.functional.mse_loss(value_prediction, returns_batch)
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    post_distribution = self.distribution(observations)
                    post_log_probs = post_distribution.log_prob(actions)
                    log_ratio = post_log_probs - old_log_probs
                    approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                if not bool(torch.isfinite(approximate_kl)):
                    raise FloatingPointError("PPO approximate KL produced NaN or Inf")
                accumulators["actor_loss"] += float(actor_loss.detach())
                accumulators["value_loss"] += float(value_loss.detach())
                accumulators["entropy"] += float(entropy.detach())
                accumulators["kl"] += float(approximate_kl.detach())
                accumulators["clip_fraction"] += float(
                    ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().detach()
                )
                accumulators["grad_norm"] += float(grad_norm.detach())
                accumulators["ratio"] += float(ratio.mean().detach())
                accumulators["action_std"] += float(distribution.std.mean().detach())
                updates += 1
                if float(approximate_kl) > self.config.target_kl:
                    kl_early_stop = True
                    break
            if kl_early_stop:
                break
            completed_epochs += 1
        if updates == 0:
            raise RuntimeError("PPO update executed no minibatches")
        explained_variance = 1.0 - torch.var(returns - storage.values) / torch.var(
            returns
        ).clamp_min(1e-8)
        return {key: value / updates for key, value in accumulators.items()} | {
            "sample_count": float(count),
            "updates": float(updates),
            "completed_epochs": float(completed_epochs),
            "kl_early_stop": kl_early_stop,
            "target_kl": self.config.target_kl,
            "old_value_mean": float(old_values.mean()),
            "explained_variance": float(explained_variance.detach()),
        }


__all__ = ["PPOConfig", "PPOTrainer"]
