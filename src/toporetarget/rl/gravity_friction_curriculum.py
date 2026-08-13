"""Frozen Stage 16 P3/P4 gravity-friction curriculum contracts.

This module is intentionally Isaac-free so the schedule, checkpoint state, and
promotion decisions can be checked before a simulator process is started.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .physical_stage import EARTH_NOMINAL_GRAVITY, load_physical_bootstrap_contract

CURRICULUM_SCHEMA = "Stage16GravityFrictionCurriculumV1"
P3_ENTRY_GATE_V2_SCHEMA = "Stage16P3EntryGateV2"
P4_QUALIFICATION_SCHEMA = "ContactReadyFullGravityQualificationV1"
CURRICULUM_STAGES = ("C0", "C1", "C2", "C3", "C4")
INITIAL_SAFE_BANKS = (
    "CONTACT_READY_SAFE",
    "PERSISTENT_SAFE",
    "MANIPULATION_SAFE",
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_MUST_BE_A_MAPPING")
    return value


def _finite_scale(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name}_MUST_BE_NUMERIC")
    result = float(value)
    if result < 0.0 or result > 2.0:
        raise ValueError(f"{name}_OUT_OF_RANGE")
    return result


@dataclass(frozen=True)
class FrictionMaterialRoleV1:
    """One frozen material role whose coefficients scale together."""

    name: str
    static_friction: float
    dynamic_friction: float
    restitution: float
    source: str

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise ValueError("CURRICULUM_MATERIAL_ROLE_IDENTITY_INVALID")
        if self.static_friction <= 0.0 or self.dynamic_friction <= 0.0:
            raise ValueError("CURRICULUM_NOMINAL_FRICTION_INVALID")
        if self.restitution != 0.0:
            raise ValueError("CURRICULUM_RESTITUTION_MUST_REMAIN_FROZEN_ZERO")

    def scaled(self, friction_scale: float) -> dict[str, float]:
        return {
            "static_friction": self.static_friction * friction_scale,
            "dynamic_friction": self.dynamic_friction * friction_scale,
            "restitution": self.restitution,
        }


@dataclass(frozen=True)
class CurriculumStageV1:
    identifier: str
    gravity_scale: float
    friction_scale: float

    def __post_init__(self) -> None:
        if self.identifier not in CURRICULUM_STAGES:
            raise ValueError("CURRICULUM_STAGE_UNKNOWN")
        _finite_scale(self.gravity_scale, name="CURRICULUM_GRAVITY_SCALE")
        _finite_scale(self.friction_scale, name="CURRICULUM_FRICTION_SCALE")


@dataclass(frozen=True)
class Stage16GravityFrictionCurriculumV1:
    """Machine-readable P3/P4 physics contract frozen before training."""

    target_gravity_world_mps2: tuple[float, float, float]
    material_roles: tuple[FrictionMaterialRoleV1, ...]
    stages: tuple[CurriculumStageV1, ...]
    scientific_label: str = "ENGINEERING_CURRICULUM_V1"
    identifier: str = CURRICULUM_SCHEMA

    def __post_init__(self) -> None:
        if self.identifier != CURRICULUM_SCHEMA:
            raise ValueError("CURRICULUM_SCHEMA_DRIFT")
        if self.scientific_label != "ENGINEERING_CURRICULUM_V1":
            raise ValueError("CURRICULUM_SCIENTIFIC_LABEL_DRIFT")
        if self.target_gravity_world_mps2 != EARTH_NOMINAL_GRAVITY:
            raise ValueError("CURRICULUM_TARGET_GRAVITY_INVALID")
        if tuple(stage.identifier for stage in self.stages) != CURRICULUM_STAGES:
            raise ValueError("CURRICULUM_STAGE_ORDER_DRIFT")
        expected = ((0.0, 2.0), (0.25, 1.75), (0.5, 1.5), (0.75, 1.25), (1.0, 1.0))
        observed = tuple((stage.gravity_scale, stage.friction_scale) for stage in self.stages)
        if observed != expected:
            raise ValueError("CURRICULUM_SCHEDULE_DRIFT")
        names = tuple(role.name for role in self.material_roles)
        if names != ("global_default_rigid_body", "hocap_bound_object_material"):
            raise ValueError("CURRICULUM_MATERIAL_ROLE_SET_DRIFT")

    def stage(self, identifier: str) -> CurriculumStageV1:
        for stage in self.stages:
            if stage.identifier == identifier:
                return stage
        raise ValueError("CURRICULUM_STAGE_UNKNOWN")

    def physics(self, identifier: str) -> dict[str, object]:
        stage = self.stage(identifier)
        return {
            "curriculum_stage": stage.identifier,
            "gravity_scale": stage.gravity_scale,
            "friction_scale": stage.friction_scale,
            "gravity_world_mps2": [
                stage.gravity_scale * value for value in self.target_gravity_world_mps2
            ],
            "material_roles": {
                role.name: {**role.scaled(stage.friction_scale), "source": role.source}
                for role in self.material_roles
            },
        }

    def checkpoint_state(
        self,
        *,
        stage: str,
        allowed_reset_banks: tuple[str, ...],
        selected_contact_mode: str | None,
    ) -> dict[str, object]:
        if tuple(allowed_reset_banks) != INITIAL_SAFE_BANKS:
            raise ValueError("CURRICULUM_RESET_BANKS_DRIFT")
        if selected_contact_mode not in {None, "aggregate_v3", "strict_per_finger_v4"}:
            raise ValueError("CURRICULUM_CONTACT_MODE_INVALID")
        return {
            "schema_version": "Stage16GravityFrictionCurriculumCheckpointStateV1",
            "curriculum_contract": self.identifier,
            "selected_contact_mode": selected_contact_mode,
            "allowed_reset_banks": list(allowed_reset_banks),
            **self.physics(stage),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "scientific_label": self.scientific_label,
            "target_gravity_world_mps2": list(self.target_gravity_world_mps2),
            "material_roles": [asdict(role) for role in self.material_roles],
            "stages": [asdict(stage) for stage in self.stages],
        }


def load_gravity_friction_curriculum(path: Path) -> Stage16GravityFrictionCurriculumV1:
    """Load the frozen P3/P4 curriculum and reject silent schedule changes."""

    resolved = path.resolve()
    document = _mapping(yaml.safe_load(resolved.read_text(encoding="utf-8")), name="CURRICULUM")
    if document.get("schema_version") != CURRICULUM_SCHEMA:
        raise ValueError("CURRICULUM_SCHEMA_INVALID")
    parent_rel = document.get("parent_physical_bootstrap")
    if not isinstance(parent_rel, str):
        raise ValueError("CURRICULUM_PARENT_BOOTSTRAP_MISSING")
    parent = (resolved.parents[3] / parent_rel).resolve()
    bootstrap = load_physical_bootstrap_contract(parent)
    gravity = tuple(document.get("target_gravity_world_mps2", ()))
    if gravity != bootstrap.target_gravity_world_mps2:
        raise ValueError("CURRICULUM_PARENT_GRAVITY_DRIFT")
    nominal = _mapping(document.get("nominal_friction"), name="CURRICULUM_NOMINAL_FRICTION")
    raw_roles = _mapping(nominal.get("material_roles"), name="CURRICULUM_MATERIAL_ROLES")
    roles_list: list[FrictionMaterialRoleV1] = []
    for name, value in raw_roles.items():
        role = _mapping(value, name=f"CURRICULUM_ROLE_{name}")
        roles_list.append(
            FrictionMaterialRoleV1(
                name=name,
                static_friction=float(role["static_friction"]),
                dynamic_friction=float(role["dynamic_friction"]),
                restitution=float(role["restitution"]),
                source=str(role["source"]),
            )
        )
    roles = tuple(roles_list)
    raw_schedule = _mapping(document.get("schedule"), name="CURRICULUM_SCHEDULE")
    stages = tuple(
        CurriculumStageV1(
            identifier=identifier,
            gravity_scale=_finite_scale(
                _mapping(raw_schedule.get(identifier), name=f"CURRICULUM_{identifier}").get(
                    "gravity_scale"
                ),
                name=f"CURRICULUM_{identifier}_GRAVITY",
            ),
            friction_scale=_finite_scale(
                _mapping(raw_schedule.get(identifier), name=f"CURRICULUM_{identifier}").get(
                    "friction_scale"
                ),
                name=f"CURRICULUM_{identifier}_FRICTION",
            ),
        )
        for identifier in CURRICULUM_STAGES
    )
    if set(raw_schedule) != set(CURRICULUM_STAGES):
        raise ValueError("CURRICULUM_STAGE_SET_DRIFT")
    reset = _mapping(document.get("reset_domain"), name="CURRICULUM_RESET_DOMAIN")
    if tuple(reset.get("allowed_initial_banks", ())) != INITIAL_SAFE_BANKS:
        raise ValueError("CURRICULUM_ALLOWED_RESET_BANKS_DRIFT")
    causal_flags = (
        "frame_zero_full_gravity_authorized",
        "invented_support_allowed",
        "external_guidance",
    )
    if any(bool(reset.get(key)) for key in causal_flags):
        raise ValueError("CURRICULUM_CAUSAL_BOUNDARY_DRIFT")
    p4 = _mapping(document.get("p4_formal"), name="CURRICULUM_P4_FORMAL")
    if (
        p4.get("schema_version") != P4_QUALIFICATION_SCHEMA
        or int(p4.get("episodes_per_clip", -1)) != 20
        or float(p4.get("milestone_srqualified_min", -1.0)) != 0.8
        or p4.get("support") != "none"
        or bool(p4.get("external_guidance"))
        or bool(p4.get("frame_zero_full_gravity"))
    ):
        raise ValueError("CURRICULUM_P4_CONTRACT_DRIFT")
    return Stage16GravityFrictionCurriculumV1(
        target_gravity_world_mps2=tuple(float(value) for value in gravity),
        material_roles=roles,
        stages=stages,
        scientific_label=str(document.get("scientific_label")),
    )


def load_p3_entry_gate_v2(path: Path) -> dict[str, object]:
    """Read the versioned P3 entry contract without erasing historical V1."""

    document = dict(_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="P3_ENTRY_V2"))
    if document.get("schema_version") != P3_ENTRY_GATE_V2_SCHEMA:
        raise ValueError("P3_ENTRY_V2_SCHEMA_INVALID")
    gates = _mapping(document.get("gates"), name="P3_ENTRY_V2_GATES")
    expected_gates = {
        "G0_provenance",
        "G1_rsi_v2",
        "G2_support",
        "G4_controller_actuator",
        "G5_causality",
    }
    if set(gates) != expected_gates:
        raise ValueError("P3_ENTRY_V2_GATE_SET_INVALID")
    decision = _mapping(document.get("decision_contract"), name="P3_ENTRY_V2_DECISION")
    if (
        decision.get("status_on_pass") != "P3_READY_WITH_CONSTRAINTS"
        or tuple(decision.get("allowed_initial_reset_banks", ())) != INITIAL_SAFE_BANKS
        or bool(decision.get("frame_zero_full_gravity_authorized"))
        or bool(decision.get("invented_support_allowed"))
        or bool(decision.get("external_guidance"))
    ):
        raise ValueError("P3_ENTRY_V2_DECISION_DRIFT")
    promotion = _mapping(document.get("promotion_gate"), name="P3_ENTRY_V2_PROMOTION")
    if promotion.get("identifier") != "G3" or promotion.get("placement") != "between_C2_and_C3":
        raise ValueError("P3_ENTRY_V2_PROMOTION_PLACEMENT_INVALID")
    return document


__all__ = [
    "CURRICULUM_SCHEMA",
    "CURRICULUM_STAGES",
    "INITIAL_SAFE_BANKS",
    "P3_ENTRY_GATE_V2_SCHEMA",
    "P4_QUALIFICATION_SCHEMA",
    "CurriculumStageV1",
    "FrictionMaterialRoleV1",
    "Stage16GravityFrictionCurriculumV1",
    "load_gravity_friction_curriculum",
    "load_p3_entry_gate_v2",
]
