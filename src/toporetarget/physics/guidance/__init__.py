"""Explicit, bounded object-reference guidance for assisted simulations.

This package is an engineering extension.  It is deliberately independent of
the PPO observation and reward contracts, and its default mode is ``none``.
"""

from .contract import ObjectGuidanceContractV1
from .reference_wrench import GuidanceWrench, ReferenceWrenchGuidance

__all__ = ["GuidanceWrench", "ObjectGuidanceContractV1", "ReferenceWrenchGuidance"]
