"""Frozen distributional metrics for Stage 16-C.5A-R4.

The functions in this module are simulator independent.  Thresholds are
derived from deterministic half-sample comparisons of the natural PhysX
population before a candidate population is inspected.  This prevents a
failed candidate from influencing the qualification metric or gate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_SCALE_FLOOR = 1.0e-6
_MMD_BANDWIDTH = 1.0


def _samples(values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("distribution samples must have shape [replicas>=2, features>=1]")
    if not np.isfinite(array).all():
        raise ValueError("distribution samples must be finite")
    return array


def _linear_percentile(values: np.ndarray, probability: float, *, axis: int = 0) -> np.ndarray:
    return np.asarray(np.quantile(values, probability, axis=axis, method="linear"))


def _baseline_scale(baseline: np.ndarray) -> np.ndarray:
    mean = np.mean(baseline, axis=0)
    std = np.std(baseline, axis=0)
    return np.maximum.reduce((std, np.abs(mean) * 1.0e-6, np.full_like(std, _SCALE_FLOOR)))


def _rbf_mmd_squared(first: np.ndarray, second: np.ndarray) -> float:
    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        delta = left[:, None, :] - right[None, :, :]
        squared = np.sum(delta * delta, axis=-1) / max(1, left.shape[1])
        return np.exp(-0.5 * squared / (_MMD_BANDWIDTH**2))

    xx = kernel(first, first)
    yy = kernel(second, second)
    xy = kernel(first, second)
    # The biased estimate is deliberately nonnegative and remains defined for
    # the small (10/20 replica) frozen qualification populations.
    return max(0.0, float(np.mean(xx) + np.mean(yy) - 2.0 * np.mean(xy)))


def _wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float:
    quantiles = (np.arange(max(len(first), len(second)), dtype=np.float64) + 0.5) / max(
        len(first), len(second)
    )
    first_q = np.quantile(first, quantiles, method="linear")
    second_q = np.quantile(second, quantiles, method="linear")
    return float(np.mean(np.abs(first_q - second_q)))


@dataclass(frozen=True)
class DistributionDistancesV1:
    """Scale-normalized distances required by the R4 gate."""

    mean_difference: float
    variance_difference: float
    p95_difference: float
    wasserstein_distance: float
    mmd: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_difference": self.mean_difference,
            "variance_difference": self.variance_difference,
            "p95_difference": self.p95_difference,
            "wasserstein_distance": self.wasserstein_distance,
            "mmd": self.mmd,
        }


def distribution_distances(
    baseline: np.ndarray | Sequence[Sequence[float]],
    candidate: np.ndarray | Sequence[Sequence[float]],
    *,
    scale_reference: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> DistributionDistancesV1:
    """Compare two replica populations without any fitted candidate parameter."""

    first = _samples(baseline)
    second = _samples(candidate)
    if first.shape[1] != second.shape[1]:
        raise ValueError("distribution feature dimensions must match")
    reference = first if scale_reference is None else _samples(scale_reference)
    if reference.shape[1] != first.shape[1]:
        raise ValueError("scale reference feature dimension must match")
    scale = _baseline_scale(reference)
    first_scaled = first / scale
    second_scaled = second / scale
    mean_difference = float(
        np.mean(np.abs(np.mean(first_scaled, axis=0) - np.mean(second_scaled, axis=0)))
    )
    # Variance is expressed in squared normalized units.
    variance_difference = float(
        np.mean(np.abs(np.var(first_scaled, axis=0) - np.var(second_scaled, axis=0)))
    )
    p95_difference = float(
        np.mean(
            np.abs(_linear_percentile(first_scaled, 0.95) - _linear_percentile(second_scaled, 0.95))
        )
    )
    wasserstein = float(
        np.mean(
            [
                _wasserstein_1d(first_scaled[:, index], second_scaled[:, index])
                for index in range(first.shape[1])
            ]
        )
    )
    return DistributionDistancesV1(
        mean_difference=mean_difference,
        variance_difference=variance_difference,
        p95_difference=p95_difference,
        wasserstein_distance=wasserstein,
        mmd=math.sqrt(_rbf_mmd_squared(first_scaled, second_scaled)),
    )


def termination_distribution_divergence(
    first: Sequence[str | int], second: Sequence[str | int]
) -> float:
    """Total-variation divergence between two termination populations."""

    if len(first) < 2 or len(second) < 2:
        raise ValueError("termination divergence needs at least two samples per population")
    categories = sorted(set(first).union(second), key=str)
    first_counts = Counter(first)
    second_counts = Counter(second)
    return 0.5 * sum(
        abs(first_counts[key] / len(first) - second_counts[key] / len(second)) for key in categories
    )


def wilson_confidence_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson binomial interval; R4 freezes the supported confidence at 95%."""

    if confidence != 0.95:
        raise ValueError("DistributionalReplicationContractV1 freezes confidence at 0.95")
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid success count")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def confidence_interval_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Zero for overlapping intervals, otherwise their nearest-endpoint gap."""

    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0.0


@dataclass(frozen=True)
class DistributionThresholdsV1:
    """Natural-variance envelope frozen before candidate qualification."""

    metric_limits: Mapping[str, float]
    termination_divergence_limit: float
    success_interval_distance_limit: float
    multiplier: float = 2.0
    split_count: int = 20

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_limits": dict(self.metric_limits),
            "termination_divergence_limit": self.termination_divergence_limit,
            "success_interval_distance_limit": self.success_interval_distance_limit,
            "multiplier": self.multiplier,
            "split_count": self.split_count,
            "derivation": "2x_p95_deterministic_half_split_natural_envelope",
        }


def _deterministic_splits(
    sample_count: int, split_count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if sample_count < 4:
        raise ValueError("natural envelope needs at least four replicas")
    half = sample_count // 2
    base = np.arange(sample_count)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(split_count):
        rng = np.random.default_rng(0x16C5A4 + index)
        permuted = rng.permutation(base)
        splits.append((permuted[:half], permuted[half : 2 * half]))
    return splits


def freeze_natural_envelope(
    fields: Mapping[str, np.ndarray | Sequence[Sequence[float]]],
    terminations: Sequence[str | int],
    successes: Sequence[bool],
    *,
    multiplier: float = 2.0,
    split_count: int = 20,
) -> DistributionThresholdsV1:
    """Freeze all R4 thresholds from a natural population only."""

    if multiplier != 2.0:
        raise ValueError("R4 freezes the natural envelope multiplier at 2")
    arrays = {name: _samples(value) for name, value in fields.items()}
    if not arrays:
        raise ValueError("natural envelope requires at least one field")
    counts = {value.shape[0] for value in arrays.values()}
    if len(counts) != 1:
        raise ValueError("all natural fields must use the same replica count")
    sample_count = counts.pop()
    if len(terminations) != sample_count or len(successes) != sample_count:
        raise ValueError("task distribution must align with natural replicas")
    splits = _deterministic_splits(sample_count, split_count)
    per_metric: dict[str, list[float]] = {
        "mean_difference": [],
        "variance_difference": [],
        "p95_difference": [],
        "wasserstein_distance": [],
        "mmd": [],
    }
    termination_values: list[float] = []
    success_values: list[float] = []
    success_array = np.asarray(successes, dtype=np.bool_)
    for left, right in splits:
        field_distances = [
            distribution_distances(value[left], value[right], scale_reference=value)
            for value in arrays.values()
        ]
        for metric in per_metric:
            per_metric[metric].append(max(getattr(row, metric) for row in field_distances))
        termination_values.append(
            termination_distribution_divergence(
                [terminations[index] for index in left],
                [terminations[index] for index in right],
            )
        )
        left_interval = wilson_confidence_interval(int(success_array[left].sum()), len(left))
        right_interval = wilson_confidence_interval(int(success_array[right].sum()), len(right))
        success_values.append(confidence_interval_distance(left_interval, right_interval))
    return DistributionThresholdsV1(
        metric_limits={
            name: multiplier * float(_linear_percentile(np.asarray(values), 0.95))
            for name, values in per_metric.items()
        },
        termination_divergence_limit=multiplier
        * float(_linear_percentile(np.asarray(termination_values), 0.95)),
        success_interval_distance_limit=multiplier
        * float(_linear_percentile(np.asarray(success_values), 0.95)),
        multiplier=multiplier,
        split_count=split_count,
    )


def summarize_samples(values: np.ndarray | Sequence[Sequence[float]]) -> dict[str, object]:
    """JSON-friendly mean/std/variance/p95 summary of a replica field."""

    array = _samples(values)
    return {
        "replicas": int(array.shape[0]),
        "features": int(array.shape[1]),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "variance": np.var(array, axis=0).tolist(),
        "p95": _linear_percentile(array, 0.95).tolist(),
    }


__all__ = [
    "DistributionDistancesV1",
    "DistributionThresholdsV1",
    "confidence_interval_distance",
    "distribution_distances",
    "freeze_natural_envelope",
    "summarize_samples",
    "termination_distribution_divergence",
    "wilson_confidence_interval",
]
