"""Frozen query and aggregation contracts for the Stage 16-D geometry gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryQueryContractV1:
    """Numerical policy frozen before formal source/corrected queries."""

    schema_version: str = "Stage16DGeometryQueryBackendContractV1"
    backend: str = "python-fcl"
    backend_version: str = "0.7.0.11"
    algorithm: str = "FCL libccd GJK signed distance plus EPA/contact MTD"
    numerical_tolerance_m: float = 1.0e-8
    # Derived during G2 from the worst sphere overlap signed-distance/MTD
    # disagreement (1.35e-7 m), rounded upward before any trajectory query.
    metric_epsilon_m: float = 5.0e-7
    max_iterations: int | None = None
    max_iterations_control: str = (
        "not exposed by python-fcl 0.7.0.11 DistanceRequest or CollisionRequest"
    )
    sign_convention: str = "positive separated, zero touching, negative penetrating"
    convergence_policy: str = "finite result plus analytic qualification; no silent fallback"
    failure_policy: str = "fail closed on exception, non-finite result, or missing contact MTD"
    license: str = "BSD-3-Clause"
    installation_method: str = "fixed PyPI wheel in toporetarget-isaaclab environment"

    def __post_init__(self) -> None:
        if self.backend_version != "0.7.0.11":
            raise ValueError("Stage16D formal backend version is frozen at python-fcl 0.7.0.11")
        if not 0.0 < self.numerical_tolerance_m <= self.metric_epsilon_m <= 1.0e-5:
            raise ValueError("invalid Stage16D geometry numerical tolerances")
        if self.max_iterations is not None:
            raise ValueError("Stage16D must not claim an iteration control absent from the binding")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeCollisionProxyPenetrationV1:
    """Formal metric contract over actual authored runtime convex proxies."""

    schema_version: str = "RuntimeCollisionProxyPenetrationV1"
    geometry_authority: str = "C.1 authored runtime collision proxies"
    pair_scope: str = "all collision-bearing hand pieces versus active object pieces"
    frame_reduction: str = "maximum penetration depth across valid pairs"
    p95_population: str = "positive contact-active per-frame worst values only"
    all_frame_p95_role: str = "required diagnostic only; never the formal p95 gate"
    strict_catastrophic_max_m: float = 0.010
    maximum_p95_m: float = 0.003
    relative_degradation_factor: float = 1.10
    relative_metrics: tuple[str, ...] = ("max_penetration_m", "p95_penetration_m")
    visual_geometry_role: str = "unsigned diagnostic only"
    stage12_sdf_role: str = "historical diagnostic only; not directly comparable"
    excludes: tuple[str, ...] = (
        "visual-only geometry",
        "reference ghost",
        "inactive object",
        "self collision",
        "ground",
        "support",
    )

    def __post_init__(self) -> None:
        if self.strict_catastrophic_max_m != 0.010 or self.maximum_p95_m != 0.003:
            raise ValueError("Stage16D penetration limits are frozen at 10mm/3mm")
        if self.relative_degradation_factor != 1.10:
            raise ValueError("Stage16D relative penetration factor is frozen at 1.10")

    def as_dict(self, *, query_contract: GeometryQueryContractV1) -> dict[str, Any]:
        return {
            **asdict(self),
            "relative_comparison": ("corrected_metric <= source_metric * 1.10 + metric_epsilon_m"),
            "metric_epsilon_m": query_contract.metric_epsilon_m,
            "existing_contract_audit": {
                "source": "configs/rl/stage16/stage16d_trajectory_gate.yaml",
                "existing_limits_preserved": True,
                "previous_p95_population_defined": False,
                "previous_relative_zero_handling_defined": False,
                "resolution": "freeze this contract before formal trajectory results",
            },
        }


def geometry_contract_sha256(payload: dict[str, Any]) -> str:
    """Hash a geometry contract with stable JSON serialization."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RuntimeCollisionProxyPenetrationV2:
    """Evidence-gated normalization revision; absolute safety limits stay frozen."""

    parent_v1_sha256: str
    calibration_sha256: str
    geometry_epsilon_m: float
    dynamic_contact_floor_max_m: float
    dynamic_contact_floor_active_p95_m: float
    schema_version: str = "RuntimeCollisionProxyPenetrationV2"
    geometry_authority: str = "C.1 authored runtime collision proxies"
    strict_catastrophic_max_m: float = 0.010
    maximum_p95_m: float = 0.003
    relative_degradation_factor: float = 1.10

    def __post_init__(self) -> None:
        hashes = (self.parent_v1_sha256, self.calibration_sha256)
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ValueError("V2 requires lowercase SHA-256 parent and calibration hashes")
        values = (
            self.geometry_epsilon_m,
            self.dynamic_contact_floor_max_m,
            self.dynamic_contact_floor_active_p95_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("V2 geometry values must be finite and nonnegative")
        if self.geometry_epsilon_m > self.dynamic_contact_floor_max_m:
            raise ValueError("V2 max dynamic floor cannot be below geometry epsilon")
        if self.geometry_epsilon_m > self.dynamic_contact_floor_active_p95_m:
            raise ValueError("V2 p95 dynamic floor cannot be below geometry epsilon")
        if self.strict_catastrophic_max_m != 0.010 or self.maximum_p95_m != 0.003:
            raise ValueError("V2 cannot change the frozen 10mm/3mm absolute gates")
        if self.relative_degradation_factor != 1.10:
            raise ValueError("V2 cannot change the source degradation factor")
        if self.dynamic_contact_floor_max_m >= self.strict_catastrophic_max_m:
            raise ValueError("V2 max dynamic floor must remain below the absolute max gate")
        if self.dynamic_contact_floor_active_p95_m > self.maximum_p95_m:
            raise ValueError("V2 p95 dynamic floor must remain within the absolute p95 gate")

    def relative_limit(self, metric: str, source_value_m: float) -> float:
        if metric == "max_penetration_m":
            floor = self.dynamic_contact_floor_max_m
        elif metric in {"p95_penetration_m", "active_p95_penetration_m"}:
            floor = self.dynamic_contact_floor_active_p95_m
        else:
            raise ValueError(f"unsupported V2 relative metric: {metric}")
        source_limit = source_value_m * self.relative_degradation_factor + self.geometry_epsilon_m
        return max(source_limit, floor)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "relative_comparison": (
                "corrected_metric <= max(source_metric * 1.10 + geometry_epsilon_m, "
                "dynamic_contact_floor_metric)"
            ),
            "absolute_gates_unchanged": True,
            "clip_specific_thresholds": False,
            "formal_authority": "exact python-fcl RuntimeCollisionProxy query",
        }


GEOMETRY_QUERY_CONTRACT = GeometryQueryContractV1()
GEOMETRY_METRIC_CONTRACT = RuntimeCollisionProxyPenetrationV1()


__all__ = [
    "GEOMETRY_METRIC_CONTRACT",
    "GEOMETRY_QUERY_CONTRACT",
    "GeometryQueryContractV1",
    "RuntimeCollisionProxyPenetrationV1",
    "RuntimeCollisionProxyPenetrationV2",
    "geometry_contract_sha256",
]
