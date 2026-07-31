"""Checkpointable running observation normalization with frozen evaluation mode."""

from __future__ import annotations

import torch


class RunningObservationNormalizer:
    def __init__(self, dimension: int, *, epsilon: float = 1e-8) -> None:
        self.count = torch.tensor(epsilon, dtype=torch.float64)
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.variance = torch.ones(dimension, dtype=torch.float64)
        self.training = True

    def update(self, observations: torch.Tensor) -> None:
        if not self.training:
            return
        values = (
            observations.detach()
            .to(dtype=torch.float64, device="cpu")
            .reshape(-1, self.mean.numel())
        )
        batch_count = torch.tensor(float(values.shape[0]), dtype=torch.float64)
        batch_mean = values.mean(dim=0)
        batch_variance = values.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.variance = (
            self.variance * self.count
            + batch_variance * batch_count
            + delta.square() * self.count * batch_count / total
        ) / total
        self.mean = self.mean + delta * batch_count / total
        self.count = total

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(observations)
        std = torch.sqrt(self.variance.to(observations) + 1e-8)
        return (observations - mean) / std

    def state_dict(self) -> dict[str, torch.Tensor | bool]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "training": self.training,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | bool]) -> None:
        self.count = torch.as_tensor(state["count"], dtype=torch.float64)
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float64)
        self.variance = torch.as_tensor(state["variance"], dtype=torch.float64)
        self.training = bool(state["training"])


__all__ = ["RunningObservationNormalizer"]
