"""Fixed-shape PPO rollout storage with explicit paper sample accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutStorage:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor

    @property
    def rollout_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def sample_count(self) -> int:
        return self.rollout_steps * self.num_envs

    def flatten(self, advantages: torch.Tensor, returns: torch.Tensor) -> dict[str, torch.Tensor]:
        count = self.sample_count
        return {
            "observations": self.observations.reshape(count, -1),
            "actions": self.actions.reshape(count, -1),
            "log_probs": self.log_probs.reshape(count),
            "advantages": advantages.reshape(count),
            "returns": returns.reshape(count),
            "values": self.values.reshape(count),
        }


__all__ = ["RolloutStorage"]
