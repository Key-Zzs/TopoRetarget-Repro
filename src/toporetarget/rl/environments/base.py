"""Backend-neutral contracts for the Stage-16 MDP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    vectorized: bool
    free_object_dynamics: bool
    contact: bool
    renderer: bool
    deterministic_reset: bool
    supported_randomizations: tuple[str, ...]


class SimulationBackend(ABC):
    """A physics backend, intentionally separate from policy/training code."""

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def reset(self, **kwargs: Any) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> dict[str, np.ndarray]: ...


class VectorizedReferenceTrackingEnv(ABC):
    """Gym-like batch interface with explicit terminal reason information."""

    @property
    @abstractmethod
    def num_envs(self) -> int: ...

    @property
    @abstractmethod
    def observation_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray: ...

    @abstractmethod
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]: ...


class PhysicsRandomizationBackend(ABC):
    """Explicit runtime randomization interface; unsupported fields must fail visibly."""

    @abstractmethod
    def apply_randomization(self, config: Any) -> None: ...


class RendererBackend(ABC):
    """Optional backend renderer interface used by review tooling."""

    @abstractmethod
    def render_rgb(self, *, width: int = 640, height: int = 480) -> np.ndarray: ...


__all__ = [
    "BackendCapabilities",
    "PhysicsRandomizationBackend",
    "RendererBackend",
    "SimulationBackend",
    "VectorizedReferenceTrackingEnv",
]
