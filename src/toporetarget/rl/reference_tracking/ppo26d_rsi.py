"""Reference-state initialization helpers for Stage 16-D.5 PPO-26D."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Stage16DPPO26DRSIV1:
    identifier: str = "Stage16DPPO26DRSIV1"
    sampling: str = "uniform_valid_reference_indices"
    reset_wrist_from_reference: bool = True
    reset_fingers_from_reference: bool = True
    reset_object_from_reference: bool = True
    reset_state_writes_allowed: bool = True
    rollout_object_state_writes: int = 0
    rollout_wrist_root_state_writes: int = 0
    progression: str = "min(k_plus_1, final)"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_uniform_reference_indices(
    rng: np.random.Generator, *, count: int, frame_count: int
) -> np.ndarray:
    if count < 1 or frame_count < 2:
        raise ValueError("RSI requires positive count and at least two reference frames")
    return rng.integers(0, frame_count, size=count, endpoint=False, dtype=np.int64)


def rsi_histogram(indices: np.ndarray, *, frame_count: int) -> dict[str, object]:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("RSI histogram needs a non-empty index vector")
    if np.any(values < 0) or np.any(values >= frame_count):
        raise ValueError("RSI index outside reference range")
    counts = np.bincount(values, minlength=frame_count)
    thirds = np.array_split(np.arange(frame_count), 4)
    return {
        "frame_count": frame_count,
        "sample_count": int(values.size),
        "counts": counts.tolist(),
        "phase_counts": {
            name: int(counts[frames].sum())
            for name, frames in zip(
                ("approach", "first_contact", "persistent_contact", "terminal"),
                thirds,
                strict=True,
            )
        },
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
    }


__all__ = ["Stage16DPPO26DRSIV1", "rsi_histogram", "sample_uniform_reference_indices"]
