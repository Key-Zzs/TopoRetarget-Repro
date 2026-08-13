"""Leakage-resistant Stage 16-D demonstration dataset construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicsCorrectionTrajectoryV1:
    trajectory_id: str
    clip: str
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    semantic_progress: np.ndarray
    contact_topology: np.ndarray
    done: np.ndarray
    success: bool
    source_hash: str
    corrected_hash: str

    def validate(self) -> None:
        frames = self.actions.shape[0]
        if self.observations.shape != (frames, 764):
            raise ValueError("demonstration observations must be [T,764]")
        if self.next_observations.shape != (frames, 764):
            raise ValueError("demonstration next observations must be [T,764]")
        if self.actions.shape != (frames, 26):
            raise ValueError("demonstration actions must be [T,26]")
        if self.rewards.shape != (frames,) or self.done.shape != (frames,):
            raise ValueError("demonstration scalar traces must align with actions")
        if self.semantic_progress.shape[0] != frames or self.contact_topology.shape[0] != frames:
            raise ValueError("demonstration semantic traces must align with actions")
        arrays = (
            self.observations,
            self.actions,
            self.next_observations,
            self.rewards,
            self.semantic_progress,
            self.contact_topology,
        )
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("demonstration contains non-finite values")
        if np.max(np.abs(self.actions)) > 1.0:
            raise ValueError("demonstration actions exceed [-1,1]")


def split_demonstrations_by_trajectory(
    trajectories: list[PhysicsCorrectionTrajectoryV1],
    *,
    validation_fraction: float = 0.20,
    seed: int = 20260806,
) -> tuple[list[PhysicsCorrectionTrajectoryV1], list[PhysicsCorrectionTrajectoryV1]]:
    if len(trajectories) < 2 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("trajectory split needs at least two trajectories and a valid fraction")
    for trajectory in trajectories:
        trajectory.validate()
    ordered = sorted(
        trajectories,
        key=lambda row: hashlib.sha256(f"{seed}:{row.trajectory_id}".encode()).hexdigest(),
    )
    validation_count = max(1, int(round(len(ordered) * validation_fraction)))
    validation = ordered[:validation_count]
    train = ordered[validation_count:]
    if not train:
        raise ValueError("trajectory split produced an empty training set")
    if {row.trajectory_id for row in train} & {row.trajectory_id for row in validation}:
        raise RuntimeError("trajectory-level train/validation leakage")
    return train, validation


def stack_actor_samples(trajectories: list[PhysicsCorrectionTrajectoryV1]) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("actor sample stack cannot be empty")
    for trajectory in trajectories:
        trajectory.validate()
    return {
        "observations": np.concatenate([row.observations for row in trajectories], axis=0),
        "actions": np.concatenate([row.actions for row in trajectories], axis=0),
        "trajectory_ids": [row.trajectory_id for row in trajectories],
        "future_state_in_observation": False,
        "hidden_simulator_state_in_observation": False,
    }


__all__ = [
    "PhysicsCorrectionTrajectoryV1",
    "split_demonstrations_by_trajectory",
    "stack_actor_samples",
]
