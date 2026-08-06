"""Evidence-gated empirical stable dynamic-contact reference and V1/V2 decision."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contracts import (
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
    RuntimeCollisionProxyPenetrationV2,
    geometry_contract_sha256,
)


def stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BootstrapUCBContractV1:
    """Frozen upper bound over a per-replica 95th-percentile statistic."""

    resamples: int = 10_000
    seed: int = 20260806
    one_sided_confidence: float = 0.95
    within_replica_statistic_quantile: float = 0.95
    schema_version: str = "StableDynamicContactBootstrapUCBContractV1"

    def __post_init__(self) -> None:
        if self.resamples != 10_000 or self.seed != 20260806:
            raise ValueError("dynamic-reference bootstrap count and seed are frozen")
        if self.one_sided_confidence != 0.95 or self.within_replica_statistic_quantile != 0.95:
            raise ValueError("dynamic-reference bootstrap quantiles are frozen at 95%")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


BOOTSTRAP_UCB_CONTRACT = BootstrapUCBContractV1()


def bootstrap_upper_confidence_quantile(
    values: Sequence[float],
    *,
    contract: BootstrapUCBContractV1 = BOOTSTRAP_UCB_CONTRACT,
) -> float:
    """Return the one-sided bootstrap UCB for the sample 95th percentile."""

    sample = np.asarray(values, dtype=np.float64)
    if sample.shape != (20,) or not np.all(np.isfinite(sample)) or np.any(sample < 0.0):
        raise ValueError("stable dynamic reference requires 20 finite nonnegative replicas")
    generator = np.random.default_rng(contract.seed)
    indices = generator.integers(0, sample.size, size=(contract.resamples, sample.size))
    resampled = sample[indices]
    statistics = np.quantile(
        resampled,
        contract.within_replica_statistic_quantile,
        axis=1,
        method="linear",
    )
    return float(np.quantile(statistics, contract.one_sided_confidence, method="higher"))


@dataclass(frozen=True)
class SelectedStableCalibrationV1:
    object_id: str
    family_id: str
    candidate_id: str
    qualification_sha256: str
    replica_max_penetration_m: tuple[float, ...]
    replica_active_p95_penetration_m: tuple[float, ...]
    v1_max_limit_m: float
    v1_active_p95_limit_m: float
    stable_gate_passed: bool = True
    corrected_trajectory_used: bool = False
    schema_version: str = "SelectedStableCalibrationV1"

    def __post_init__(self) -> None:
        if (
            len(self.replica_max_penetration_m) != 20
            or len(self.replica_active_p95_penetration_m) != 20
        ):
            raise ValueError("selected stable calibration requires 20 replicas")
        if self.corrected_trajectory_used:
            raise ValueError("dynamic reference cannot use corrected trajectories")
        if len(self.qualification_sha256) != 64:
            raise ValueError("selected calibration requires a qualification SHA-256")

    def v1_passed(self) -> bool:
        return (
            self.stable_gate_passed
            and all(value <= self.v1_max_limit_m for value in self.replica_max_penetration_m)
            and all(
                value <= self.v1_active_p95_limit_m
                for value in self.replica_active_p95_penetration_m
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "v1_passed_20_of_20": self.v1_passed()}


def freeze_empirical_dynamic_contact_reference(
    calibrations: Sequence[SelectedStableCalibrationV1],
    *,
    required_object_family_pairs: Sequence[tuple[str, str]],
    bootstrap: BootstrapUCBContractV1 = BOOTSTRAP_UCB_CONTRACT,
) -> dict[str, Any]:
    """Freeze the shared worst-case reference across objects and topology families."""

    if not calibrations:
        raise ValueError("at least one selected stable calibration is required")
    selected_pairs = {(row.object_id, row.family_id) for row in calibrations}
    required_pairs = set(required_object_family_pairs)
    if selected_pairs != required_pairs:
        missing = sorted(required_pairs - selected_pairs)
        extra = sorted(selected_pairs - required_pairs)
        raise RuntimeError(f"STAGE16D_DYNAMIC_REFERENCE_COVERAGE_FAILURE:{missing=}:{extra=}")
    if any(not row.stable_gate_passed for row in calibrations):
        raise RuntimeError("STAGE16D_STABLE_FREE_OBJECT_GRASP_CALIBRATION_BLOCKED")
    rows: list[dict[str, Any]] = []
    for calibration in calibrations:
        max_ucb = bootstrap_upper_confidence_quantile(
            calibration.replica_max_penetration_m, contract=bootstrap
        )
        p95_ucb = bootstrap_upper_confidence_quantile(
            calibration.replica_active_p95_penetration_m, contract=bootstrap
        )
        rows.append(
            {
                "object_id": calibration.object_id,
                "family_id": calibration.family_id,
                "candidate_id": calibration.candidate_id,
                "qualification_sha256": calibration.qualification_sha256,
                "ucb95_max_penetration_m": max_ucb,
                "ucb95_active_p95_penetration_m": p95_ucb,
            }
        )
    reference_max = max(float(row["ucb95_max_penetration_m"]) for row in rows)
    reference_p95 = max(float(row["ucb95_active_p95_penetration_m"]) for row in rows)
    within_absolute = (
        reference_max < GEOMETRY_METRIC_CONTRACT.strict_catastrophic_max_m
        and reference_p95 <= GEOMETRY_METRIC_CONTRACT.maximum_p95_m
    )
    payload = {
        "schema_version": "EmpiricalStableDynamicContactReferenceV1",
        "terminology": (
            "empirical engineering reference under the frozen PhysX contract; "
            "not a physical truth or mathematical lower bound"
        ),
        "bootstrap_contract": bootstrap.as_dict(),
        "selected_calibrations": rows,
        "aggregation": "maximum UCB95 across both objects and all required topology families",
        "dynamic_reference_max_m": reference_max,
        "dynamic_reference_active_p95_m": reference_p95,
        "within_absolute_gate": within_absolute,
        "shared_across_clips": True,
        "corrected_trajectory_used": False,
        "optimizer_result_observed_before_freeze": False,
    }
    return {**payload, "reference_sha256": stable_payload_sha256(payload)}


def decide_geometry_v1_v2(
    calibrations: Sequence[SelectedStableCalibrationV1],
    *,
    required_object_family_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Retain V1, create evidence-authorized V2, or fail closed."""

    required_pairs = set(required_object_family_pairs)
    rows_by_pair = {(row.object_id, row.family_id): row for row in calibrations}
    if set(rows_by_pair) != required_pairs or any(
        not row.stable_gate_passed for row in calibrations
    ):
        return {
            "schema_version": "Stage16DGeometryV1V2DecisionV1",
            "status": "STAGE16D_STABLE_GRASP_CALIBRATION_BLOCKED",
            "v1_attainable": False,
            "v2_created": False,
            "optimizer_authorized": False,
            "ppo_authorized": False,
            "reason": "no complete stable 20-replica calibration coverage",
        }
    if all(row.v1_passed() for row in calibrations):
        return {
            "schema_version": "Stage16DGeometryV1V2DecisionV1",
            "status": "STAGE16D_GEOMETRY_V1_ATTAINABLE",
            "v1_attainable": True,
            "v2_created": False,
            "selected_metric": "RuntimeCollisionProxyPenetrationV1",
            "optimizer_authorized": True,
            "ppo_authorized": False,
            "absolute_gate_unchanged": True,
        }
    reference = freeze_empirical_dynamic_contact_reference(
        calibrations,
        required_object_family_pairs=required_object_family_pairs,
    )
    if not reference["within_absolute_gate"]:
        return {
            "schema_version": "Stage16DGeometryV1V2DecisionV1",
            "status": "STAGE16D_DYNAMIC_CONTACT_REFERENCE_EXCEEDS_ABSOLUTE_GATE",
            "v1_attainable": False,
            "v2_created": False,
            "optimizer_authorized": False,
            "ppo_authorized": False,
            "absolute_gate_unchanged": True,
            "empirical_reference": reference,
        }
    parent = GEOMETRY_METRIC_CONTRACT.as_dict(query_contract=GEOMETRY_QUERY_CONTRACT)
    v2 = RuntimeCollisionProxyPenetrationV2(
        parent_v1_sha256=geometry_contract_sha256(parent),
        calibration_sha256=str(reference["reference_sha256"]),
        geometry_epsilon_m=GEOMETRY_QUERY_CONTRACT.metric_epsilon_m,
        dynamic_contact_floor_max_m=float(reference["dynamic_reference_max_m"]),
        dynamic_contact_floor_active_p95_m=float(reference["dynamic_reference_active_p95_m"]),
    )
    return {
        "schema_version": "Stage16DGeometryV1V2DecisionV1",
        "status": "STAGE16D_GEOMETRY_V2_VALIDATED",
        "v1_attainable": False,
        "v1_classification": "STAGE16D_GEOMETRY_V1_BELOW_EMPIRICAL_DYNAMIC_REFERENCE",
        "v2_created": True,
        "selected_metric": "RuntimeCollisionProxyPenetrationV2",
        "optimizer_authorized": True,
        "ppo_authorized": False,
        "absolute_gate_unchanged": True,
        "empirical_reference": reference,
        "v2_contract": v2.as_dict(),
    }


__all__ = [
    "BOOTSTRAP_UCB_CONTRACT",
    "BootstrapUCBContractV1",
    "SelectedStableCalibrationV1",
    "bootstrap_upper_confidence_quantile",
    "decide_geometry_v1_v2",
    "freeze_empirical_dynamic_contact_reference",
    "stable_payload_sha256",
]
