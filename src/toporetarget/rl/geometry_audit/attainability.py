"""Fail-closed attainability decision for the Stage 16-D relative geometry gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import (
    GeometryQueryContractV1,
    RuntimeCollisionProxyPenetrationV1,
    RuntimeCollisionProxyPenetrationV2,
    geometry_contract_sha256,
)


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_calibration_provenance(value: str) -> None:
    lowered = value.lower()
    forbidden = ("corrected", "optimizer", "candidate", "ppo", "policy")
    if any(token in lowered for token in forbidden):
        raise ValueError("dynamic floor calibration cannot use corrected or learned-policy data")


@dataclass(frozen=True)
class StableDynamicContactCalibrationV1:
    experiment_id: str
    provenance: str
    replicas: int
    control_steps: int
    free_object: bool
    formal_state_writes: int
    required_contact_present_rate: float
    replica_p99_max_penetration_m: float
    pooled_active_p95_penetration_m: float
    max_penetration_m: float
    bounded_normal_load: bool

    def __post_init__(self) -> None:
        _validate_calibration_provenance(self.provenance)
        if self.replicas != 20 or self.control_steps < 1:
            raise ValueError("stable dynamic calibration requires 20 complete replicas")
        if not self.free_object or self.formal_state_writes != 0:
            raise ValueError(
                "stable dynamic calibration requires a free object and no rollout writes"
            )
        values = (
            self.required_contact_present_rate,
            self.replica_p99_max_penetration_m,
            self.pooled_active_p95_penetration_m,
            self.max_penetration_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("stable contact metrics must be finite and nonnegative")
        if not 0.0 <= self.required_contact_present_rate <= 1.0:
            raise ValueError("contact-present rate must be in [0,1]")
        if self.required_contact_present_rate < 0.95 or not self.bounded_normal_load:
            raise ValueError("dynamic floor requires stable bounded contact")


@dataclass(frozen=True)
class ContactPreservingFeasibilityV1:
    clip: str
    free_object: bool
    formal_state_writes: int
    required_topology_pass: bool
    semantic_intent_pass: bool
    v1_max_limit_m: float
    v1_active_p95_limit_m: float
    best_contact_preserving_max_m: float
    best_contact_preserving_active_p95_m: float
    v1_pass_with_required_contact: bool
    lower_penetration_only_by_contact_or_task_degeneracy: bool

    def __post_init__(self) -> None:
        if self.clip not in {"hocap_170105", "hocap_170650"}:
            raise ValueError("unknown Stage16D clip")
        if not self.free_object or self.formal_state_writes != 0:
            raise ValueError("local feasibility requires a free object and no rollout writes")
        values = (
            self.v1_max_limit_m,
            self.v1_active_p95_limit_m,
            self.best_contact_preserving_max_m,
            self.best_contact_preserving_active_p95_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("local feasibility metrics must be finite and nonnegative")


def decide_penetration_gate_contract(
    *,
    query_contract: GeometryQueryContractV1,
    v1_contract: RuntimeCollisionProxyPenetrationV1,
    numerical_query_p99_m: float,
    no_contact_max_penetration_m: float,
    stable_contact: StableDynamicContactCalibrationV1,
    local_feasibility: tuple[ContactPreservingFeasibilityV1, ...],
) -> dict[str, Any]:
    """Keep V1 when demonstrated attainable, otherwise authorize evidence-based V2."""

    if len(local_feasibility) != 2 or {row.clip for row in local_feasibility} != {
        "hocap_170105",
        "hocap_170650",
    }:
        raise ValueError("attainability requires both frozen clips exactly once")
    if numerical_query_p99_m > query_contract.metric_epsilon_m:
        raise RuntimeError("STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED:NUMERICAL_FLOOR")
    if no_contact_max_penetration_m > query_contract.metric_epsilon_m:
        raise RuntimeError("STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED:NO_CONTACT_FLOOR")

    v1_attainable = all(
        row.required_topology_pass
        and row.semantic_intent_pass
        and row.v1_pass_with_required_contact
        for row in local_feasibility
    )
    calibration = asdict(stable_contact)
    calibration_sha256 = _stable_hash(calibration)
    v1_payload = v1_contract.as_dict(query_contract=query_contract)
    if v1_attainable:
        return {
            "schema_version": "RuntimePenetrationGateAttainabilityAuditV1",
            "status": "STAGE16D_GEOMETRY_V1_ATTAINABLE",
            "classification": "RUNTIME_COLLISION_PROXY_PENETRATION_V1_ATTAINABLE",
            "metric_version": "RuntimeCollisionProxyPenetrationV1",
            "v2_created": False,
            "numerical_query_p99_m": numerical_query_p99_m,
            "no_contact_max_penetration_m": no_contact_max_penetration_m,
            "stable_contact": calibration,
            "local_feasibility": [asdict(row) for row in local_feasibility],
            "metric_contract": v1_payload,
            "metric_contract_sha256": geometry_contract_sha256(v1_payload),
        }

    below_floor = all(
        row.lower_penetration_only_by_contact_or_task_degeneracy
        and row.required_topology_pass
        and row.semantic_intent_pass
        and (
            row.v1_max_limit_m < stable_contact.replica_p99_max_penetration_m
            or row.v1_active_p95_limit_m < stable_contact.pooled_active_p95_penetration_m
        )
        for row in local_feasibility
        if not row.v1_pass_with_required_contact
    )
    if not below_floor:
        raise RuntimeError("STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED:ATTAINABILITY_INCONCLUSIVE")

    v2 = RuntimeCollisionProxyPenetrationV2(
        parent_v1_sha256=geometry_contract_sha256(v1_payload),
        calibration_sha256=calibration_sha256,
        geometry_epsilon_m=query_contract.metric_epsilon_m,
        dynamic_contact_floor_max_m=stable_contact.replica_p99_max_penetration_m,
        dynamic_contact_floor_active_p95_m=stable_contact.pooled_active_p95_penetration_m,
    )
    return {
        "schema_version": "RuntimePenetrationGateAttainabilityAuditV1",
        "status": "STAGE16D_GEOMETRY_V2_VALIDATED",
        "classification": "SOURCE_RELATIVE_GATE_BELOW_DYNAMIC_CONTACT_FLOOR",
        "metric_version": v2.schema_version,
        "v2_created": True,
        "numerical_query_p99_m": numerical_query_p99_m,
        "no_contact_max_penetration_m": no_contact_max_penetration_m,
        "stable_contact": calibration,
        "local_feasibility": [asdict(row) for row in local_feasibility],
        "metric_contract": v2.as_dict(),
        "metric_contract_sha256": geometry_contract_sha256(v2.as_dict()),
    }


__all__ = [
    "ContactPreservingFeasibilityV1",
    "StableDynamicContactCalibrationV1",
    "decide_penetration_gate_contract",
]
