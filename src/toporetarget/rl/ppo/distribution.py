"""The paper-named Softplus Gaussian action distribution."""

from __future__ import annotations

import torch
from torch.distributions import Normal
from torch.nn import functional as F


class SoftplusGaussian:
    """Tanh-squashed Gaussian with state-independent Softplus std.

    The paper provides only the distribution name.  This exact parameterization
    is registered as an engineering assumption and is isolated here.
    """

    def __init__(self, location: torch.Tensor, raw_std: torch.Tensor) -> None:
        self.location = location
        self.mean = torch.tanh(location)
        self.std = F.softplus(raw_std).expand_as(location) + 1e-5
        self.normal = Normal(location, self.std)

    @staticmethod
    def _log_abs_det_jacobian(pre_tanh: torch.Tensor) -> torch.Tensor:
        # Stable equivalent of log(1 - tanh(x)^2), used by SAC-style policies.
        return 2.0 * (
            torch.log(torch.tensor(2.0, device=pre_tanh.device))
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.normal.rsample())

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(action.dtype).eps
        bounded = action.clamp(min=-1.0 + epsilon, max=1.0 - epsilon)
        pre_tanh = torch.atanh(bounded)
        corrected = self.normal.log_prob(pre_tanh) - self._log_abs_det_jacobian(pre_tanh)
        return corrected.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # The squashed distribution has no analytic entropy.  This is an
        # unbiased one-sample reparameterized estimate with the same Jacobian.
        pre_tanh = self.normal.rsample()
        corrected = self.normal.log_prob(pre_tanh) - self._log_abs_det_jacobian(pre_tanh)
        return -corrected.sum(dim=-1)


__all__ = ["SoftplusGaussian"]
