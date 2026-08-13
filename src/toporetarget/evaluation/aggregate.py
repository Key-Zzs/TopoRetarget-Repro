"""Rollout aggregation and serializable metric-timeline helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("aggregate values must be non-empty finite scalars")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "p95_over_rollouts": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def aggregate_rollouts(rows: list[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate per-episode V2 metrics without conflating legacy success."""

    if not rows:
        raise ValueError("cannot aggregate zero rollout rows")
    metrics = ("E_r_mean_deg", "E_t_mean_cm", "E_j_mean_cm", "E_ft_mean_cm")
    result: dict[str, object] = {"rollout_count": len(rows)}
    for metric in metrics:
        values: list[float] = []
        for row in rows:
            value = row[metric]
            if not isinstance(value, (float, int)):
                raise ValueError(f"{metric} must be numeric")
            values.append(float(value))
        result[metric] = _summary(values)
    for name in ("kinematic_success", "physics_success", "qualified_success"):
        passed = sum(bool(row[name]) for row in rows)
        result[name] = {"pass_count": passed, "total": len(rows), "rate": passed / len(rows)}
    return result


def timeline_rows(
    metrics: Mapping[str, np.ndarray], *, contact: np.ndarray, terminal_stability: np.ndarray
) -> list[dict[str, object]]:
    """Make an optional replay-compatible sidecar; all inputs must share [T]."""

    required = ("e_r_deg", "e_t_cm", "e_j_cm", "e_ft_cm")
    length = len(np.asarray(metrics["e_r_deg"]))
    if any(len(np.asarray(metrics[name])) != length for name in required):
        raise ValueError("metric time series have inconsistent lengths")
    contact_values = np.asarray(contact, dtype=bool)
    stable_values = np.asarray(terminal_stability, dtype=bool)
    if contact_values.shape != (length,) or stable_values.shape != (length,):
        raise ValueError("contact and terminal stability must be [T]")
    return [
        {
            "frame": frame,
            **{name: float(np.asarray(metrics[name])[frame]) for name in required},
            "contact": bool(contact_values[frame]),
            "terminal_stability": bool(stable_values[frame]),
        }
        for frame in range(length)
    ]
