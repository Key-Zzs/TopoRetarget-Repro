"""Dataset-neutral metrics with explicit dynamic/static applicability."""

# ruff: noqa: E501

from __future__ import annotations

from typing import Any

import numpy as np


def applicability(dynamic: bool, metric_id: str) -> str:
    if not dynamic and metric_id in {
        "q_velocity",
        "q_acceleration",
        "q_jerk",
        "base_translation_velocity",
        "base_translation_acceleration",
        "base_rotation_velocity",
        "max_interframe_q_step",
        "max_base_step",
        "temporal_lag_diagnostic",
    }:
        return "NOT_APPLICABLE"
    return "APPLICABLE"


def _finite_difference(values: np.ndarray, timestamps: np.ndarray, order: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    for _ in range(order):
        if result.shape[0] < 2:
            return np.empty((0,) + result.shape[1:], dtype=np.float64)
        result = np.diff(result, axis=0) / np.diff(timestamps).reshape(
            (-1,) + (1,) * (result.ndim - 1)
        )
        timestamps = timestamps[1:]
    return result


def trajectory_metrics(
    *,
    dynamic: bool,
    qpos: Any | None = None,
    base_translation: Any | None = None,
    timestamps: Any | None = None,
    signed_distance: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if signed_distance is not None:
        values = np.asarray(signed_distance, dtype=np.float64)
        penetration = np.maximum(-values, 0.0)
        per_frame = penetration.max(axis=-1) if penetration.ndim > 1 else penetration
        result.update(
            {
                "raw_max_penetration": float(np.max(per_frame) * 1000.0),
                "raw_penetration_rate_2mm": float(np.mean(per_frame > 0.002)),
                "min_signed_distance": float(np.min(values) * 1000.0),
            }
        )
    if not dynamic:
        for key in (
            "q_velocity",
            "q_acceleration",
            "q_jerk",
            "base_translation_velocity",
            "base_translation_acceleration",
            "base_rotation_velocity",
            "max_interframe_q_step",
            "max_base_step",
            "temporal_lag_diagnostic",
        ):
            result[key] = "NOT_APPLICABLE"
        return result
    if qpos is None or timestamps is None:
        result["temporal_status"] = "N/A"
        return result
    q = np.asarray(qpos, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape[0] != t.size or t.size < 2 or not np.all(np.isfinite(q)):
        result["temporal_status"] = "N/A"
        return result
    velocity = _finite_difference(q, t, 1)
    acceleration = _finite_difference(q, t, 2)
    jerk = _finite_difference(q, t, 3)
    result.update(
        {
            "q_velocity": float(np.max(np.linalg.norm(velocity, axis=1))) if velocity.size else 0.0,
            "q_acceleration": float(np.max(np.linalg.norm(acceleration, axis=1)))
            if acceleration.size
            else 0.0,
            "q_jerk": float(np.max(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0,
            "max_interframe_q_step": float(np.max(np.linalg.norm(np.diff(q, axis=0), axis=1))),
        }
    )
    if base_translation is not None:
        base = np.asarray(base_translation, dtype=np.float64)
        if base.ndim == 2 and base.shape == (t.size, 3):
            bv = _finite_difference(base, t, 1)
            ba = _finite_difference(base, t, 2)
            result["base_translation_velocity"] = (
                float(np.max(np.linalg.norm(bv, axis=1))) if bv.size else 0.0
            )
            result["base_translation_acceleration"] = (
                float(np.max(np.linalg.norm(ba, axis=1))) if ba.size else 0.0
            )
    return result


__all__ = ["applicability", "trajectory_metrics"]
