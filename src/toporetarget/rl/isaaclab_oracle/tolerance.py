"""Freeze state-replication tolerances from a measured no-clone baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

HARD_CAPS_V1: dict[str, float] = {
    "wrist_position_m": 1.0e-3,
    "object_position_m": 1.0e-3,
    "quaternion_geodesic_rad": 2.0e-3,
    "joint_position_rad": 1.0e-3,
    "linear_velocity_si": 1.0e-2,
    "angular_velocity_si": 1.0e-2,
    "reward": 1.0e-3,
}

FIXED_FLOORS_V1: dict[str, float] = {
    "wrist_position_m": 1.0e-7,
    "object_position_m": 1.0e-7,
    "quaternion_geodesic_rad": 1.0e-7,
    "joint_position_rad": 1.0e-7,
    "linear_velocity_si": 1.0e-6,
    "angular_velocity_si": 1.0e-6,
    "reward": 1.0e-7,
}


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for no samples")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_noise_floor(samples: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "p50": percentile(values, 50.0),
            "p95": percentile(values, 95.0),
            "p99": percentile(values, 99.0),
            "max": max(float(value) for value in values),
            "trials": float(len(values)),
        }
        for name, values in samples.items()
    }


def freeze_tolerances(samples: Mapping[str, Sequence[float]]) -> dict[str, object]:
    """Compute max(fixed floor, 10× p99) and fail if a hard cap is exceeded."""

    summary = summarize_noise_floor(samples)
    unknown = set(summary).difference(HARD_CAPS_V1)
    if unknown:
        raise ValueError(f"unknown tolerance metrics: {sorted(unknown)}")
    metrics = {}
    baseline_exceeds_hard_cap = False
    for name, row in summary.items():
        hard_cap = HARD_CAPS_V1[name]
        tolerance = max(FIXED_FLOORS_V1[name], 10.0 * row["p99"])
        exceeds = row["max"] > hard_cap or tolerance > hard_cap
        baseline_exceeds_hard_cap = baseline_exceeds_hard_cap or exceeds
        metrics[name] = {
            **row,
            "fixed_floor": FIXED_FLOORS_V1[name],
            "frozen_tolerance": tolerance,
            "hard_cap": hard_cap,
            "passes_hard_cap": not exceeds,
        }
    return {
        "version": "replication_noise_floor_v1",
        "formula": "max(fixed_floor, 10 * baseline_p99)",
        "metrics": metrics,
        "status": (
            "PHYSX_REPLICATION_BASELINE_NONDETERMINISM"
            if baseline_exceeds_hard_cap
            else "REPLICATION_TOLERANCES_FROZEN"
        ),
    }


__all__ = ["FIXED_FLOORS_V1", "HARD_CAPS_V1", "freeze_tolerances", "summarize_noise_floor"]
