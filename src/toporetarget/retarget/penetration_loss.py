"""Generic paper-external dense signed-distance penetration objective terms.

The term deliberately knows nothing about a particular hand, sequence, or
degree-of-freedom count. A caller supplies the signed-distance query result,
the immutable collision-surface geometry IDs, and the point Jacobian with
respect to its state vector.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.geometry.signed_distance.base import SignedDistanceQueryResult

PENETRATION_LOSS_PROFILE_ID = "dense_squared_hinge_deadzone1mm_v2"
OBJECTIVE_TERM_REGISTRY: dict[str, type[DenseSDFPenetrationLoss]] = {}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PenetrationLossProfile:
    profile_id: str
    version: str
    loss_type: str
    signed_distance_convention: str
    d_ref_m: float
    penetration_tolerance_m: float
    normalization_depth_m: float
    reduction: str
    query_surface: str
    clearance_m: float
    gradient_backend: str
    inner_sdf_backend: str
    validation_sdf_backend: str
    paper_method: bool
    paper_external_extension: bool
    deprecated_for_zero_tolerance_comparison: bool
    profile_hash: str
    source_path: Path | None = None

    def validate(self) -> PenetrationLossProfile:
        if self.loss_type != "squared_hinge":
            raise ValueError("only the squared one-sided hinge is supported")
        if self.signed_distance_convention != "positive_outside":
            raise ValueError("penetration loss requires the positive-outside convention")
        if not np.isfinite(self.d_ref_m) or self.d_ref_m <= 0:
            raise ValueError("d_ref_m must be finite and positive")
        if not np.isfinite(self.penetration_tolerance_m) or self.penetration_tolerance_m < 0:
            raise ValueError("penetration_tolerance_m must be finite and non-negative")
        if not np.isfinite(self.normalization_depth_m) or self.normalization_depth_m <= 0:
            raise ValueError("normalization_depth_m must be finite and positive")
        if self.reduction != "mean_per_geometry_then_mean":
            raise ValueError("unsupported penetration-loss reduction")
        if self.query_surface != "full_robot_collision_surface":
            raise ValueError("penetration loss must use the full robot collision surface")
        if self.clearance_m != 0:
            raise ValueError("positive clearance is forbidden for penetration loss")
        if self.gradient_backend != "signed_distance_normal_times_point_jacobian":
            raise ValueError("unsupported penetration-loss gradient backend")
        if self.inner_sdf_backend not in {"solver_fast_backend", "convex_hull_exact_solver_only"}:
            raise ValueError("unsupported penetration-loss inner SDF backend")
        if self.validation_sdf_backend != "reference_winding_v1":
            raise ValueError("penetration-loss validation must use reference_winding_v1")
        if self.profile_id.endswith("_v2") and self.penetration_tolerance_m != 0.001:
            raise ValueError("the v2 dead-zone profile requires a 1 mm loss tolerance")
        return self

    @classmethod
    def load(
        cls,
        profile_id: str = PENETRATION_LOSS_PROFILE_ID,
        *,
        config_root: str | Path | None = None,
    ) -> PenetrationLossProfile:
        root = Path(config_root) if config_root is not None else Path(__file__).resolve().parents[3]
        path = root / "configs" / "retarget" / "penetration_losses" / f"{profile_id}.yaml"
        raw = path.read_bytes()
        values = yaml.safe_load(raw) or {}
        if not isinstance(values, dict):
            raise ValueError(f"penetration-loss profile must be a mapping: {path}")
        normalization_depth = values.get("normalization_depth_m")
        if normalization_depth is None:
            normalization_depth = values["d_ref_m"]
        result = cls(
            profile_id=str(values.get("profile_id", profile_id)),
            version=str(values.get("version", "1.0.0")),
            loss_type=str(values["loss_type"]),
            signed_distance_convention=str(values["signed_distance_convention"]),
            d_ref_m=float(values["d_ref_m"]),
            penetration_tolerance_m=float(values.get("penetration_tolerance_m", 0.0)),
            normalization_depth_m=float(normalization_depth),
            reduction=str(values["reduction"]),
            query_surface=str(values["query_surface"]),
            clearance_m=float(values.get("clearance_m", 0.0)),
            gradient_backend=str(values["gradient_backend"]),
            inner_sdf_backend=str(values.get("inner_sdf_backend", "convex_hull_exact_solver_only")),
            validation_sdf_backend=str(
                values.get("validation_sdf_backend", "reference_winding_v1")
            ),
            paper_method=bool(values.get("paper_method", False)),
            paper_external_extension=bool(values.get("paper_external_extension", True)),
            deprecated_for_zero_tolerance_comparison=bool(
                values.get("deprecated_for_zero_tolerance_comparison", False)
            ),
            profile_hash=_digest(raw),
            source_path=path,
        )
        return result.validate()

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "loss_type": self.loss_type,
            "signed_distance_convention": self.signed_distance_convention,
            "d_ref_m": self.d_ref_m,
            "penetration_tolerance_m": self.penetration_tolerance_m,
            "normalization_depth_m": self.normalization_depth_m,
            "reduction": self.reduction,
            "query_surface": self.query_surface,
            "clearance_m": self.clearance_m,
            "gradient_backend": self.gradient_backend,
            "inner_sdf_backend": self.inner_sdf_backend,
            "validation_sdf_backend": self.validation_sdf_backend,
            "paper_method": self.paper_method,
            "paper_external_extension": self.paper_external_extension,
            "deprecated_for_zero_tolerance_comparison": (
                self.deprecated_for_zero_tolerance_comparison
            ),
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class PenetrationLossEvaluation:
    value: float
    gradient: np.ndarray
    normalized_depth: np.ndarray
    signed_distance: np.ndarray
    geometry_ids: np.ndarray
    negative_sample_count: int
    fallback_count: int
    gradient_backend: str
    penetration_tolerance_m: float
    normalization_depth_m: float
    sdf_backend: str = "unknown"

    @property
    def negative_sample_fraction(self) -> float:
        return float(self.negative_sample_count / max(len(self.signed_distance), 1))

    def as_dict(self) -> dict[str, Any]:
        active = self.normalized_depth > 0.0
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "negative_sample_count": self.negative_sample_count,
            "negative_sample_fraction": self.negative_sample_fraction,
            "loss_active_sample_count": int(np.count_nonzero(active)),
            "over_tolerance_sample_count": int(np.count_nonzero(active)),
            "penetration_tolerance_m": self.penetration_tolerance_m,
            "normalization_depth_m": self.normalization_depth_m,
            "max_penetration_m": float(max(0.0, -float(np.min(self.signed_distance)))),
            "fallback_count": self.fallback_count,
            "gradient_backend": self.gradient_backend,
            "sdf_backend": self.sdf_backend,
            "geometry_sample_counts": {
                str(identifier): int(np.count_nonzero(self.geometry_ids == identifier))
                for identifier in np.unique(self.geometry_ids)
            },
        }


class DenseSDFPenetrationLoss:
    """Geometry-balanced squared one-sided signed-distance hinge."""

    term_id = "dense_sdf_penetration"

    def __init__(self, profile: PenetrationLossProfile, lambda_sdf: float) -> None:
        profile.validate()
        if not np.isfinite(lambda_sdf) or lambda_sdf < 0:
            raise ValueError("lambda_sdf must be finite and non-negative")
        self.profile = profile
        self.lambda_sdf = float(lambda_sdf)

    @property
    def profile_hash(self) -> str:
        payload = {"profile": self.profile.as_dict(), "lambda_sdf": self.lambda_sdf}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _weights(geometry_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(geometry_ids).astype(str).reshape(-1)
        if len(ids) == 0:
            raise ValueError("penetration loss requires at least one collision sample")
        groups = np.unique(ids)
        weights = np.zeros(len(ids), dtype=np.float64)
        for group in groups:
            members = np.flatnonzero(ids == group)
            weights[members] = 1.0 / (len(groups) * len(members))
        return weights

    def value_only(self, signed_distance: np.ndarray, geometry_ids: np.ndarray) -> float:
        phi = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
        weights = self._weights(geometry_ids)
        if len(phi) != len(weights) or not np.all(np.isfinite(phi)):
            raise ValueError("signed-distance values must be finite and match geometry IDs")
        depth = np.maximum(-phi, 0.0)
        excess = np.maximum(depth - self.profile.penetration_tolerance_m, 0.0)
        return float(np.sum(weights * np.square(excess / self.profile.normalization_depth_m)))

    def evaluate(
        self,
        signed_distance: np.ndarray,
        geometry_ids: np.ndarray,
        *,
        surface_normals: np.ndarray | None = None,
        point_jacobian: np.ndarray | None = None,
        gradient_valid: np.ndarray | None = None,
        non_smooth: np.ndarray | None = None,
        fallback_gradient: Callable[[], np.ndarray] | None = None,
        sdf_backend_id: str | None = None,
    ) -> PenetrationLossEvaluation:
        phi = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
        ids = np.asarray(geometry_ids).astype(str).reshape(-1)
        weights = self._weights(ids)
        if len(phi) != len(ids) or not np.all(np.isfinite(phi)):
            raise ValueError("signed-distance values must be finite and match geometry IDs")
        depth = np.maximum(-phi, 0.0)
        excess = np.maximum(depth - self.profile.penetration_tolerance_m, 0.0)
        value = float(np.sum(weights * np.square(excess / self.profile.normalization_depth_m)))
        backend = self.profile.gradient_backend
        if point_jacobian is None or surface_normals is None:
            gradient = np.zeros(0, dtype=np.float64)
            valid = np.ones(len(phi), dtype=bool)
            backend = "scalar_only_full_surface_diagnostic"
        else:
            normals = np.asarray(surface_normals, dtype=np.float64).reshape(len(phi), 3)
            jac = np.asarray(point_jacobian, dtype=np.float64)
            if jac.ndim != 3 or jac.shape[:2] != (len(phi), 3):
                raise ValueError("point_jacobian must have shape [sample,3,state]")
            if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(jac)):
                raise ValueError("SDF normals and point Jacobian must be finite")
            sample_grad = (
                weights[:, None]
                * (-2.0 * excess / (self.profile.normalization_depth_m**2))[:, None]
                * np.einsum("ni,nid->nd", normals, jac, optimize=True)
            )
            gradient = np.sum(sample_grad, axis=0)
            valid = np.ones(len(phi), dtype=bool)
            if gradient_valid is not None:
                valid &= np.asarray(gradient_valid, dtype=bool).reshape(-1)
            if non_smooth is not None:
                valid &= ~np.asarray(non_smooth, dtype=bool).reshape(-1)
        fallback_count = int(np.count_nonzero(~valid)) if len(valid) else 0
        if fallback_count and fallback_gradient is not None:
            replacement = np.asarray(fallback_gradient(), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(replacement)):
                raise ValueError("finite-difference penetration gradient is non-finite")
            gradient = replacement
            backend = f"{backend}+finite_difference_fallback"
        if not np.all(np.isfinite(gradient)):
            raise ValueError("penetration loss gradient is non-finite")
        return PenetrationLossEvaluation(
            value=value,
            gradient=gradient,
            normalized_depth=excess / self.profile.normalization_depth_m,
            signed_distance=phi.copy(),
            geometry_ids=ids.copy(),
            negative_sample_count=int(np.count_nonzero(phi < 0.0)),
            fallback_count=fallback_count,
            gradient_backend=backend,
            penetration_tolerance_m=self.profile.penetration_tolerance_m,
            normalization_depth_m=self.profile.normalization_depth_m,
            sdf_backend=str(sdf_backend_id or "unknown"),
        )

    def evaluate_query(
        self,
        query: SignedDistanceQueryResult,
        geometry_ids: np.ndarray,
        *,
        point_jacobian: np.ndarray | None = None,
        fallback_gradient: Callable[[], np.ndarray] | None = None,
    ) -> PenetrationLossEvaluation:
        gradient_valid = query.gradient_valid
        if gradient_valid is None:
            gradient_valid = query.sign_valid & query.valid
        return self.evaluate(
            query.signed_distance,
            geometry_ids,
            surface_normals=query.surface_normals,
            point_jacobian=point_jacobian,
            gradient_valid=gradient_valid,
            non_smooth=query.non_smooth,
            fallback_gradient=fallback_gradient,
            sdf_backend_id=query.backend_id,
        )


def register_objective_term(name: str, term_type: type[DenseSDFPenetrationLoss]) -> None:
    if not name or name in OBJECTIVE_TERM_REGISTRY:
        raise ValueError(f"objective term is already registered: {name!r}")
    OBJECTIVE_TERM_REGISTRY[name] = term_type


def build_objective_term(
    name: str,
    *,
    profile: PenetrationLossProfile,
    lambda_sdf: float,
) -> DenseSDFPenetrationLoss:
    try:
        term_type = OBJECTIVE_TERM_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown objective term: {name}") from exc
    return term_type(profile, lambda_sdf)


register_objective_term(DenseSDFPenetrationLoss.term_id, DenseSDFPenetrationLoss)


__all__ = [
    "DenseSDFPenetrationLoss",
    "OBJECTIVE_TERM_REGISTRY",
    "PENETRATION_LOSS_PROFILE_ID",
    "PenetrationLossEvaluation",
    "PenetrationLossProfile",
    "build_objective_term",
    "register_objective_term",
]
