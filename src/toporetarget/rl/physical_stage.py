"""Frozen P0 contracts for the Stage 16 full-gravity preparation work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .reference_tracking.contact_reward_mode import ContactRewardMode

EARTH_NOMINAL_GRAVITY = (0.0, 0.0, -9.81)
PHYSICAL_BOOTSTRAP_SCHEMA = "Stage16PhysicalBootstrapContractV1"
P3_ENTRY_GATE_SCHEMA = "Stage16P3EntryGateV1"


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_MUST_BE_A_MAPPING")
    return value


def _bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name}_MUST_BE_A_BOOL")
    return value


@dataclass(frozen=True)
class Stage16PhysicalBootstrapContractV1:
    """The stable P0 bridge from the causal zero-g parent to P1/P2."""

    identifier: str = PHYSICAL_BOOTSTRAP_SCHEMA
    target_gravity_world_mps2: tuple[float, float, float] = EARTH_NOMINAL_GRAVITY
    target_gravity_status: str = "EARTH_NOMINAL_ENGINEERING_TARGET"
    support_semantics_required: bool = True
    external_guidance: bool = False
    rollout_object_state_write: bool = False
    rollout_wrist_root_write: bool = False
    rsi_target_version: str = "contact_ready_v2"
    contact_reward_candidates: tuple[str, ...] = (
        ContactRewardMode.AGGREGATE_V3.value,
        ContactRewardMode.STRICT_PER_FINGER_V4.value,
    )

    def __post_init__(self) -> None:
        if self.identifier != PHYSICAL_BOOTSTRAP_SCHEMA:
            raise ValueError("PHYSICAL_BOOTSTRAP_SCHEMA_DRIFT")
        if self.target_gravity_world_mps2 != EARTH_NOMINAL_GRAVITY:
            raise ValueError("PHYSICAL_BOOTSTRAP_TARGET_GRAVITY_INVALID")
        if self.target_gravity_status != "EARTH_NOMINAL_ENGINEERING_TARGET":
            raise ValueError("PHYSICAL_BOOTSTRAP_TARGET_GRAVITY_STATUS_INVALID")
        if not self.support_semantics_required:
            raise ValueError("PHYSICAL_BOOTSTRAP_SUPPORT_SEMANTICS_REQUIRED")
        if (
            self.external_guidance
            or self.rollout_object_state_write
            or self.rollout_wrist_root_write
        ):
            raise ValueError("PHYSICAL_BOOTSTRAP_CAUSALITY_BOUNDARY_VIOLATED")
        if self.rsi_target_version != "contact_ready_v2":
            raise ValueError("PHYSICAL_BOOTSTRAP_RSI_TARGET_INVALID")
        expected = tuple(item.value for item in ContactRewardMode)
        if self.contact_reward_candidates != expected:
            raise ValueError("PHYSICAL_BOOTSTRAP_CONTACT_CANDIDATES_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Stage16P1RSIAcceptanceContractV1:
    """Result-independent engineering thresholds frozen before P1 diagnostics."""

    max_pre_contact_displacement_m: float
    max_pre_contact_downward_displacement_m: float
    max_object_linear_speed_mps: float
    max_object_angular_speed_radps: float
    min_contact_persistence_control_steps: int
    replicas_per_state: int
    control_steps: int

    def __post_init__(self) -> None:
        if not 0.0 < self.max_pre_contact_displacement_m <= 0.05:
            raise ValueError("P1_RSI_MAX_PRE_CONTACT_DISPLACEMENT_INVALID")
        if not 0.0 < self.max_pre_contact_downward_displacement_m <= 0.05:
            raise ValueError("P1_RSI_MAX_DOWNWARD_DISPLACEMENT_INVALID")
        if self.max_object_linear_speed_mps <= 0.0 or self.max_object_angular_speed_radps <= 0.0:
            raise ValueError("P1_RSI_OBJECT_SPEED_THRESHOLD_INVALID")
        if self.min_contact_persistence_control_steps < 1:
            raise ValueError("P1_RSI_CONTACT_PERSISTENCE_INVALID")
        if not 1 <= self.replicas_per_state <= 4 or not 10 <= self.control_steps <= 20:
            raise ValueError("P1_RSI_DIAGNOSTIC_BUDGET_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_physical_bootstrap_contract(path: Path) -> Stage16PhysicalBootstrapContractV1:
    """Load and fail closed on a malformed P0 contract."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    document = _mapping(payload, name="PHYSICAL_BOOTSTRAP_CONFIG")
    if document.get("schema_version") != PHYSICAL_BOOTSTRAP_SCHEMA:
        raise ValueError("PHYSICAL_BOOTSTRAP_SCHEMA_INVALID")
    parent = _mapping(document.get("parent"), name="PHYSICAL_BOOTSTRAP_PARENT")
    if parent.get("milestone") != "stage16d_causal_zero_gravity":
        raise ValueError("PHYSICAL_BOOTSTRAP_PARENT_MILESTONE_INVALID")
    physical = _mapping(document.get("physical_target"), name="PHYSICAL_BOOTSTRAP_TARGET")
    gravity = tuple(physical.get("target_gravity_world_mps2", ()))
    if len(gravity) != 3 or any(not isinstance(value, (int, float)) for value in gravity):
        raise ValueError("PHYSICAL_BOOTSTRAP_TARGET_GRAVITY_SHAPE_INVALID")
    rsi = _mapping(document.get("rsi"), name="PHYSICAL_BOOTSTRAP_RSI")
    candidates = tuple(rsi.get("contact_reward_candidates", ()))
    if not all(isinstance(item, str) for item in candidates):
        raise ValueError("PHYSICAL_BOOTSTRAP_CONTACT_CANDIDATES_INVALID")
    return Stage16PhysicalBootstrapContractV1(
        target_gravity_world_mps2=(
            float(gravity[0]),
            float(gravity[1]),
            float(gravity[2]),
        ),
        target_gravity_status=str(physical.get("target_gravity_status")),
        support_semantics_required=_bool(
            physical.get("support_semantics_required"), name="PHYSICAL_BOOTSTRAP_SUPPORT"
        ),
        external_guidance=_bool(
            physical.get("external_guidance"), name="PHYSICAL_BOOTSTRAP_GUIDANCE"
        ),
        rollout_object_state_write=_bool(
            physical.get("rollout_object_state_write"), name="PHYSICAL_BOOTSTRAP_OBJECT_WRITE"
        ),
        rollout_wrist_root_write=_bool(
            physical.get("rollout_wrist_root_write"), name="PHYSICAL_BOOTSTRAP_WRIST_WRITE"
        ),
        rsi_target_version=str(rsi.get("target_version")),
        contact_reward_candidates=candidates,
    )


def load_p1_rsi_acceptance_contract(path: Path) -> Stage16P1RSIAcceptanceContractV1:
    """Read the immutable P1 diagnostic thresholds from the P3 entry gate."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    document = _mapping(payload, name="P3_ENTRY_GATE_CONFIG")
    if document.get("schema_version") != P3_ENTRY_GATE_SCHEMA:
        raise ValueError("P3_ENTRY_GATE_SCHEMA_INVALID")
    thresholds = _mapping(document.get("engineering_thresholds"), name="P3_ENGINEERING_THRESHOLDS")
    if thresholds.get("label") != "ENGINEERING_THRESHOLD":
        raise ValueError("P3_ENGINEERING_THRESHOLDS_NOT_DECLARED")
    return Stage16P1RSIAcceptanceContractV1(
        max_pre_contact_displacement_m=float(thresholds["max_pre_contact_displacement_m"]),
        max_pre_contact_downward_displacement_m=float(
            thresholds["max_pre_contact_downward_displacement_m"]
        ),
        max_object_linear_speed_mps=float(thresholds["max_object_linear_speed_mps"]),
        max_object_angular_speed_radps=float(thresholds["max_object_angular_speed_radps"]),
        min_contact_persistence_control_steps=int(
            thresholds["min_contact_persistence_control_steps"]
        ),
        replicas_per_state=4,
        control_steps=20,
    )


def load_p3_entry_gate(path: Path) -> dict[str, object]:
    """Return a validated copy of the machine-readable P3 gate."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    document = dict(_mapping(payload, name="P3_ENTRY_GATE_CONFIG"))
    if document.get("schema_version") != P3_ENTRY_GATE_SCHEMA:
        raise ValueError("P3_ENTRY_GATE_SCHEMA_INVALID")
    gates = _mapping(document.get("gates"), name="P3_ENTRY_GATES")
    expected = {
        "G0_provenance",
        "G1_rsi_v2",
        "G2_support",
        "G3_geometry",
        "G4_controller_actuator",
        "G5_causality",
    }
    if set(gates) != expected:
        raise ValueError("P3_ENTRY_GATES_INCOMPLETE")
    support = _mapping(gates["G2_support"], name="P3_ENTRY_SUPPORT_GATE")
    allowed = support.get("allowed_classifications")
    if not isinstance(allowed, list) or "CONTACT_READY_ONLY_VALIDATED" not in allowed:
        raise ValueError("P3_ENTRY_CONTACT_READY_CONSTRAINT_MISSING")
    decision = _mapping(document.get("decision_contract"), name="P3_ENTRY_DECISION_CONTRACT")
    banks = decision.get("allowed_initial_reset_banks")
    if banks != ["CONTACT_READY_SAFE", "PERSISTENT_SAFE", "MANIPULATION_SAFE"]:
        raise ValueError("P3_ENTRY_INITIAL_BANKS_DRIFT")
    return document


__all__ = [
    "EARTH_NOMINAL_GRAVITY",
    "PHYSICAL_BOOTSTRAP_SCHEMA",
    "P3_ENTRY_GATE_SCHEMA",
    "Stage16P1RSIAcceptanceContractV1",
    "Stage16PhysicalBootstrapContractV1",
    "load_p1_rsi_acceptance_contract",
    "load_p3_entry_gate",
    "load_physical_bootstrap_contract",
]
