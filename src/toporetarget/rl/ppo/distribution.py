"""The paper-named Softplus Gaussian action distribution."""

from __future__ import annotations

import torch
from torch.distributions import Normal


class SoftplusGaussian:
    """Tanh-bounded residual mean with state-independent Softplus std.

    The paper provides only the distribution name.  This exact parameterization
    is registered as an engineering assumption and is isolated here.
    """

    def __init__(self, mean: torch.Tensor, raw_std: torch.Tensor) -> None:
        self.mean = mean
        self.std = torch.nn.functional.softplus(raw_std).expand_as(mean) + 1e-5
        self.normal = Normal(mean, self.std)

    def sample(self) -> torch.Tensor:
        return self.normal.rsample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.normal.log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.normal.entropy().sum(dim=-1)


__all__ = ["SoftplusGaussian"]
