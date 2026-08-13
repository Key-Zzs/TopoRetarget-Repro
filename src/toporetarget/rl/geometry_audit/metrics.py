"""Formal per-frame worst penetration aggregation and source comparison."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .contracts import (
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
    GeometryQueryContractV1,
    RuntimeCollisionProxyPenetrationV1,
)


def _windows(mask: np.ndarray) -> list[dict[str, int]]:
    flat = np.asarray(mask, dtype=bool)
    if flat.ndim != 1:
        raise ValueError("penetration window mask must be one-dimensional")
    result: list[dict[str, int]] = []
    start: int | None = None
    for index, active in enumerate(np.append(flat, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            result.append({"start_frame": start, "end_frame": index - 1, "length": index - start})
            start = None
    return result


def aggregate_penetration(
    frame_worst_penetration_m: np.ndarray,
    frame_worst_pair_index: np.ndarray,
    pair_ids: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(frame_worst_penetration_m, dtype=np.float64)
    pair_index = np.asarray(frame_worst_pair_index, dtype=np.int64)
    if values.ndim != 2 or pair_index.shape != values.shape:
        raise ValueError("formal penetration inputs must be [frames,replicas]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("formal penetration values must be finite and nonnegative")
    if not pair_ids:
        raise ValueError("formal penetration requires at least one pair")
    worst_flat = int(np.argmax(values))
    worst_frame, worst_replica = np.unravel_index(worst_flat, values.shape)
    worst_pair_index = int(pair_index[worst_frame, worst_replica])
    if not 0 <= worst_pair_index < len(pair_ids):
        raise ValueError("formal penetration pair index is invalid")
    active = values[values > 0.0]
    replicas = []
    for replica in range(values.shape[1]):
        replicas.append(
            {
                "replica": replica,
                "over_3mm_windows": _windows(values[:, replica] > 0.003),
                "over_10mm_windows": _windows(values[:, replica] > 0.010),
            }
        )
    return {
        "schema_version": "RuntimeCollisionProxyPenetrationAggregateV1",
        "frame_count": int(values.shape[0]),
        "replica_count": int(values.shape[1]),
        "pair_count": len(pair_ids),
        "max_penetration_m": float(values[worst_frame, worst_replica]),
        "p95_penetration_m": float(np.quantile(active, 0.95)) if active.size else 0.0,
        "active_p95_penetration_m": float(np.quantile(active, 0.95)) if active.size else 0.0,
        "all_frame_p95_penetration_m": float(np.quantile(values.reshape(-1), 0.95)),
        "active_mean_penetration_m": float(active.mean()) if active.size else 0.0,
        "penetrating_frame_replica_count": int(active.size),
        "over_3mm_frame_replica_count": int(np.count_nonzero(values > 0.003)),
        "over_10mm_frame_replica_count": int(np.count_nonzero(values > 0.010)),
        "worst_frame": int(worst_frame),
        "worst_replica": int(worst_replica),
        "worst_pair": pair_ids[worst_pair_index],
        "contiguous_windows": replicas,
    }


def qualify_source_corrected(
    source: dict[str, Any],
    corrected: dict[str, Any],
    *,
    metric_contract: RuntimeCollisionProxyPenetrationV1 = GEOMETRY_METRIC_CONTRACT,
    query_contract: GeometryQueryContractV1 = GEOMETRY_QUERY_CONTRACT,
) -> dict[str, Any]:
    relative = {}
    for metric in metric_contract.relative_metrics:
        source_value = float(source[metric])
        corrected_value = float(corrected[metric])
        limit = source_value * metric_contract.relative_degradation_factor
        limit += query_contract.metric_epsilon_m
        relative[metric] = {
            "source_m": source_value,
            "corrected_m": corrected_value,
            "limit_m": limit,
            "pass": corrected_value <= limit,
        }
    absolute = {
        "strict_max_below_10mm": (
            float(corrected["max_penetration_m"]) < metric_contract.strict_catastrophic_max_m
        ),
        "p95_at_most_3mm": (float(corrected["p95_penetration_m"]) <= metric_contract.maximum_p95_m),
    }
    passed = all(absolute.values()) and all(row["pass"] for row in relative.values())
    return {
        "schema_version": "Stage16DSourceCorrectedPenetrationQualificationV1",
        "absolute_gates": absolute,
        "relative_gates": relative,
        "formal_pass": passed,
        "status": (
            "STAGE16D_RUNTIME_COLLISION_PROXY_PENETRATION_VALIDATED"
            if passed
            else "STAGE16D_RUNTIME_COLLISION_PROXY_PENETRATION_BLOCKED"
        ),
    }


__all__ = ["aggregate_penetration", "qualify_source_corrected"]
