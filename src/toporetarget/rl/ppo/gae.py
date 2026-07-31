"""Generalized advantage estimation with terminal bootstrap masking."""

from __future__ import annotations

import torch


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values and dones must have identical [T,N] shapes")
    advantage = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for index in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_value * not_done - values[index]
        running = delta + gamma * gae_lambda * not_done * running
        advantage[index] = running
        next_value = values[index]
    return advantage, advantage + values


__all__ = ["generalized_advantage_estimate"]
