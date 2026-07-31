"""Paper Table-6 actor and critic MLP definitions."""

from __future__ import annotations

import torch
from torch import nn

ACTOR_HIDDEN = (512, 256, 128)
CRITIC_HIDDEN = (512, 512, 256, 128)


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = _mlp(observation_dim, ACTOR_HIDDEN, action_dim)
        self.critic = _mlp(observation_dim, CRITIC_HIDDEN, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(action_dim))

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def mean(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observations))


__all__ = ["ACTOR_HIDDEN", "CRITIC_HIDDEN", "ActorCritic"]
