"""Shared Stage 16-D geometry/terminal optimization window extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationWindowV2:
    start: int
    end: int
    reasons: tuple[str, ...]


def extract_geometry_optimization_windows(
    masks: Mapping[str, np.ndarray], *, margin: int = 10, trajectory_length: int = 321
) -> tuple[OptimizationWindowV2, ...]:
    if margin < 0 or trajectory_length < 1 or not masks:
        raise ValueError("invalid geometry window extraction contract")
    tagged: list[tuple[int, int, str]] = []
    for reason, values in masks.items():
        mask = np.asarray(values, dtype=bool)
        if mask.shape != (trajectory_length,):
            raise ValueError(f"window mask has wrong shape: {reason}")
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        transitions = np.diff(padded)
        starts = [int(value) for value in np.flatnonzero(transitions == 1)]
        ends = [int(value) for value in np.flatnonzero(transitions == -1)]
        for start, end in zip(starts, ends, strict=True):
            tagged.append((max(0, start - margin), min(trajectory_length, end + margin), reason))
    if not tagged:
        return ()
    tagged.sort()
    merged: list[OptimizationWindowV2] = []
    start, end, reason = tagged[0]
    reasons = {reason}
    for next_start, next_end, next_reason in tagged[1:]:
        if next_start <= end:
            end = max(end, next_end)
            reasons.add(next_reason)
        else:
            merged.append(OptimizationWindowV2(start, end, tuple(sorted(reasons))))
            start, end, reasons = next_start, next_end, {next_reason}
    merged.append(OptimizationWindowV2(start, end, tuple(sorted(reasons))))
    return tuple(merged)


__all__ = ["OptimizationWindowV2", "extract_geometry_optimization_windows"]
