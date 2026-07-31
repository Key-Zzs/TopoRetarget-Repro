"""Exact signed-distance spatial gradients and ambiguity routing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import SignedDistanceQueryResult

ANALYTIC_CLOSEST_FEATURE_GRADIENT_ID = "analytic_closest_feature_v1"
SPATIAL_FD_FALLBACK_ID = "spatial_central_fd_v1"
AMBIGUITY_POLICY_ID = "SignedDistanceGradientAmbiguityPolicy.v1"


@dataclass(frozen=True)
class SignedDistanceGradientAmbiguityPolicy:
    surface_epsilon_m: float = 1e-7
    barycentric_epsilon: float = 1e-6
    closest_distance_tie_tolerance_m: float = 1e-9
    spatial_fd_step_m: float = 1e-5
    sign_safety_margin_m: float = 1e-9
    profile_id: str = AMBIGUITY_POLICY_ID


@dataclass
class SignedDistanceGradientResult:
    spatial_gradient_object: np.ndarray
    spatial_gradient_scene: np.ndarray
    gradient_source: np.ndarray
    analytic_mask: np.ndarray
    spatial_fd_mask: np.ndarray
    invalid_mask: np.ndarray
    ambiguity_flags: np.ndarray
    gradient_norm: np.ndarray
    fd_probe_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "gradient_source": self.gradient_source.tolist(),
            "analytic_mask": self.analytic_mask.tolist(),
            "spatial_fd_mask": self.spatial_fd_mask.tolist(),
            "invalid_mask": self.invalid_mask.tolist(),
            "ambiguity_flags": self.ambiguity_flags.tolist(),
            "gradient_norm": self.gradient_norm.tolist(),
            "fd_probe_count": int(self.fd_probe_count),
        }


def analytic_spatial_gradient(
    points: np.ndarray,
    result: SignedDistanceQueryResult,
    *,
    policy: SignedDistanceGradientAmbiguityPolicy | None = None,
    mesh_is_watertight: bool = True,
) -> SignedDistanceGradientResult:
    """Evaluate ``sign * (x - closest) / unsigned_distance`` where smooth.

    No face-normal shortcut is used.  Surface normals remain a diagnostic last
    resort and are never silently accepted by qualification.
    """

    selected = policy or SignedDistanceGradientAmbiguityPolicy()
    x = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    closest = np.asarray(result.closest_points, dtype=np.float64).reshape(-1, 3)
    unsigned = np.asarray(result.unsigned_distance, dtype=np.float64).reshape(-1)
    phi = np.asarray(result.signed_distance, dtype=np.float64).reshape(-1)
    bary = np.asarray(result.closest_barycentric, dtype=np.float64).reshape(-1, 3)
    sign = np.where(phi >= 0.0, 1.0, -1.0)
    direction = x - closest
    denominator = np.maximum(unsigned, np.finfo(np.float64).tiny)
    gradient = sign[:, None] * direction / denominator[:, None]
    flags = np.full(len(x), "", dtype=object)
    near_surface = unsigned <= selected.surface_epsilon_m
    near_feature = np.min(bary, axis=1) <= selected.barycentric_epsilon
    nonfinite = (~np.isfinite(gradient).all(axis=1)) | (~np.isfinite(phi))
    sign_invalid = ~np.asarray(result.sign_valid, dtype=bool).reshape(-1)
    reported_nonsmooth = (
        np.zeros(len(x), dtype=bool)
        if result.non_smooth is None
        else np.asarray(result.non_smooth, dtype=bool).reshape(-1)
    )
    reported_invalid = (
        np.zeros(len(x), dtype=bool)
        if result.gradient_valid is None
        else ~np.asarray(result.gradient_valid, dtype=bool).reshape(-1)
    )
    ambiguous = (
        near_surface
        | near_feature
        | reported_nonsmooth
        | reported_invalid
        | sign_invalid
        | nonfinite
        | (not mesh_is_watertight)
    )
    for mask, label in (
        (near_surface, "NEAR_SURFACE"),
        (near_feature, "EDGE_OR_VERTEX_FEATURE"),
        (reported_nonsmooth, "NONSMOOTH_QUERY"),
        (reported_invalid, "BACKEND_GRADIENT_INVALID"),
        (sign_invalid, "SIGN_UNRELIABLE"),
        (nonfinite, "NONFINITE"),
        (np.full(len(x), not mesh_is_watertight, dtype=bool), "OPEN_OR_NONWATERTIGHT_MESH"),
    ):
        flags[mask] = np.where(flags[mask] == "", label, flags[mask] + ";" + label)
    norm = np.linalg.norm(gradient, axis=1)
    norm_invalid = (~np.isfinite(norm)) | (np.abs(norm - 1.0) > 1e-5)
    ambiguous |= norm_invalid
    flags[norm_invalid] = np.where(
        flags[norm_invalid] == "", "GRADIENT_NORM", flags[norm_invalid] + ";GRADIENT_NORM"
    )
    return SignedDistanceGradientResult(
        spatial_gradient_object=gradient.copy(),
        spatial_gradient_scene=gradient.copy(),
        gradient_source=np.where(
            ambiguous, "SPATIAL_FD_REQUIRED", ANALYTIC_CLOSEST_FEATURE_GRADIENT_ID
        ),
        analytic_mask=~ambiguous,
        spatial_fd_mask=ambiguous.copy(),
        invalid_mask=nonfinite,
        ambiguity_flags=flags.astype(str),
        gradient_norm=norm,
    )


def ambiguity_reason_counts(value: SignedDistanceGradientResult) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in value.ambiguity_flags[value.spatial_fd_mask]:
        for reason in str(item).split(";"):
            if reason:
                counter[reason] += 1
    return dict(sorted(counter.items()))


__all__ = [
    "AMBIGUITY_POLICY_ID",
    "ANALYTIC_CLOSEST_FEATURE_GRADIENT_ID",
    "SPATIAL_FD_FALLBACK_ID",
    "SignedDistanceGradientAmbiguityPolicy",
    "SignedDistanceGradientResult",
    "ambiguity_reason_counts",
    "analytic_spatial_gradient",
]
