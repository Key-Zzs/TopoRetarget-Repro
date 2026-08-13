"""Versioned, bounded PhysX contracts for Stage 16-C.5A-R2.

This module deliberately has no Isaac imports.  It defines the serializable
contract before an Isaac process is created, validates the small authorized
dimension set, and applies it only to an already-constructed Isaac Lab config.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

DeviceKind = Literal["gpu", "cpu"]

_MAX_SOLVER_ITERATIONS = 255
_BASELINE_IDENTIFIER = "physx_factor8_baseline_v1"


@dataclass(frozen=True)
class Stage16PhysxContractV1:
    """All authorized mutable PhysX dimensions, shared by both clips and all envs."""

    identifier: str
    device_kind: DeviceKind
    device: str
    parent_contract_hash: str | None
    reason_for_change: str
    solver_type: int
    enhanced_determinism: bool
    solve_articulation_contact_last: bool
    scene_min_position_iterations: int
    scene_max_position_iterations: int
    scene_min_velocity_iterations: int
    scene_max_velocity_iterations: int
    actor_position_iterations: int
    actor_velocity_iterations: int
    gpu_max_rigid_contact_count: int
    gpu_max_rigid_patch_count: int
    scene_construction: str
    status: str = "AVAILABLE"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.device_kind not in {"gpu", "cpu"}:
            raise ValueError(f"unknown PhysX device kind: {self.device_kind}")
        if self.device_kind == "gpu" and not self.device.startswith("cuda"):
            raise ValueError("GPU PhysX contract requires a CUDA device")
        if self.device_kind == "cpu" and self.device != "cpu":
            raise ValueError("CPU PhysX contract requires device='cpu'")
        if self.solver_type not in {0, 1}:
            raise ValueError("PhysX solver type must be PGS=0 or TGS=1")
        for name, value in (
            ("scene_min_position_iterations", self.scene_min_position_iterations),
            ("scene_max_position_iterations", self.scene_max_position_iterations),
            ("scene_min_velocity_iterations", self.scene_min_velocity_iterations),
            ("scene_max_velocity_iterations", self.scene_max_velocity_iterations),
            ("actor_position_iterations", self.actor_position_iterations),
            ("actor_velocity_iterations", self.actor_velocity_iterations),
        ):
            if not isinstance(value, int) or not 0 <= value <= _MAX_SOLVER_ITERATIONS:
                raise ValueError(f"invalid PhysX iteration count {name}={value!r}")
        if not 1 <= self.actor_position_iterations <= _MAX_SOLVER_ITERATIONS:
            raise ValueError("actor position iterations must be in [1, 255]")
        if (
            not self.scene_min_position_iterations
            <= self.actor_position_iterations
            <= (self.scene_max_position_iterations)
        ):
            raise ValueError("actor position iterations would be clamped by the scene")
        if (
            not self.scene_min_velocity_iterations
            <= self.actor_velocity_iterations
            <= (self.scene_max_velocity_iterations)
        ):
            raise ValueError("actor velocity iterations would be clamped by the scene")
        if self.gpu_max_rigid_contact_count <= 0 or self.gpu_max_rigid_patch_count <= 0:
            raise ValueError("GPU contact capacities must be positive")
        if self.scene_construction != "usd_clone_no_fabric_v1":
            raise ValueError("Stage 16 R2 freezes USD clone/no-Fabric scene construction")
        if self.status not in {"AVAILABLE", "UNAVAILABLE"}:
            raise ValueError(f"invalid contract availability state: {self.status}")

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {**self.canonical_payload(), "config_sha256": self.config_hash}


def baseline_contract() -> Stage16PhysxContractV1:
    """Return the immutable historical G0 configuration as a named contract."""

    contract = Stage16PhysxContractV1(
        identifier=_BASELINE_IDENTIFIER,
        device_kind="gpu",
        device="cuda:0",
        parent_contract_hash=None,
        reason_for_change="historical C3/C4 contract retained as G0 control",
        solver_type=1,
        enhanced_determinism=False,
        solve_articulation_contact_last=False,
        scene_min_position_iterations=4,
        scene_max_position_iterations=8,
        scene_min_velocity_iterations=1,
        scene_max_velocity_iterations=2,
        actor_position_iterations=8,
        actor_velocity_iterations=2,
        gpu_max_rigid_contact_count=2**22,
        gpu_max_rigid_patch_count=2**20,
        scene_construction="usd_clone_no_fabric_v1",
    )
    contract.validate()
    return contract


def candidate_matrix() -> dict[str, Stage16PhysxContractV1]:
    """Freeze the entire authorized R2 matrix before any qualification run."""

    g0 = baseline_contract()
    parent = g0.config_hash

    def derived(
        identifier: str,
        reason: str,
        *,
        enhanced: bool = True,
        position: int = 8,
        velocity: int = 2,
        contact_last: bool = False,
        device_kind: DeviceKind = "gpu",
    ) -> Stage16PhysxContractV1:
        contract = Stage16PhysxContractV1(
            identifier=identifier,
            device_kind=device_kind,
            device="cuda:0" if device_kind == "gpu" else "cpu",
            parent_contract_hash=parent,
            reason_for_change=reason,
            solver_type=g0.solver_type,
            enhanced_determinism=enhanced,
            solve_articulation_contact_last=contact_last,
            scene_min_position_iterations=g0.scene_min_position_iterations,
            scene_max_position_iterations=max(g0.scene_max_position_iterations, position),
            scene_min_velocity_iterations=g0.scene_min_velocity_iterations,
            scene_max_velocity_iterations=max(g0.scene_max_velocity_iterations, velocity),
            actor_position_iterations=position,
            actor_velocity_iterations=velocity,
            gpu_max_rigid_contact_count=g0.gpu_max_rigid_contact_count,
            gpu_max_rigid_patch_count=g0.gpu_max_rigid_patch_count,
            scene_construction=g0.scene_construction,
        )
        contract.validate()
        return contract

    return {
        "G0": g0,
        "G1": derived(
            "physx_factor8_determinism_candidate_g1",
            "enable documented PhysX enhanced determinism only",
        ),
        "G2": derived(
            "physx_factor8_determinism_candidate_g2",
            "enhanced determinism plus two-times actor position iterations",
            position=16,
        ),
        "G3": derived(
            "physx_factor8_determinism_candidate_g3",
            "enhanced determinism plus two-times actor position and velocity iterations",
            position=16,
            velocity=4,
        ),
        "G4": derived(
            "physx_factor8_determinism_candidate_g4",
            "enhanced determinism plus four-times position and two-times velocity iterations",
            position=32,
            velocity=4,
        ),
        "G5": derived(
            "physx_factor8_determinism_candidate_g5",
            "G2 plus documented articulation-contact-last solver ordering",
            position=16,
            contact_last=True,
        ),
        "C0": derived(
            "physx_factor8_cpu_diagnostic_c0",
            "CPU-only diagnostic equivalent to G0; never authorizes GPU Oracle",
            enhanced=False,
            device_kind="cpu",
        ),
    }


def contract_from_mapping(payload: Mapping[str, object]) -> Stage16PhysxContractV1:
    fields = {key: value for key, value in payload.items() if key != "config_sha256"}
    contract = Stage16PhysxContractV1(**fields)  # type: ignore[arg-type]
    contract.validate()
    expected_hash = payload.get("config_sha256")
    if expected_hash is not None and expected_hash != contract.config_hash:
        raise ValueError("PhysX contract config SHA256 mismatch")
    return contract


def load_contract(matrix_path: Path, candidate_id: str) -> Stage16PhysxContractV1:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("matrix_frozen") is not True:
        raise ValueError("Stage16 C5A R2 candidate matrix is not frozen")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ValueError(f"unknown frozen Stage16 C5A R2 candidate: {candidate_id}")
    candidate = candidates[candidate_id]
    if not isinstance(candidate, dict):
        raise ValueError(f"malformed frozen candidate: {candidate_id}")
    return contract_from_mapping(candidate)


def apply_physx_contract(cfg: Any, contract: Stage16PhysxContractV1) -> None:
    """Apply an already validated contract before scene construction only."""

    contract.validate()
    if contract.status != "AVAILABLE":
        raise ValueError(f"cannot apply unavailable PhysX contract: {contract.identifier}")
    cfg.sim.device = contract.device
    cfg.sim.physx.solver_type = contract.solver_type
    cfg.sim.physx.enable_enhanced_determinism = contract.enhanced_determinism
    cfg.sim.physx.solve_articulation_contact_last = contract.solve_articulation_contact_last
    cfg.sim.physx.min_position_iteration_count = contract.scene_min_position_iterations
    cfg.sim.physx.max_position_iteration_count = contract.scene_max_position_iterations
    cfg.sim.physx.min_velocity_iteration_count = contract.scene_min_velocity_iterations
    cfg.sim.physx.max_velocity_iteration_count = contract.scene_max_velocity_iterations
    cfg.sim.physx.gpu_max_rigid_contact_count = contract.gpu_max_rigid_contact_count
    cfg.sim.physx.gpu_max_rigid_patch_count = contract.gpu_max_rigid_patch_count
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = (
        contract.actor_position_iterations
    )
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = (
        contract.actor_velocity_iterations
    )
    for object_cfg in (cfg.object_170105, cfg.object_170650):
        object_cfg.spawn.rigid_props.solver_position_iteration_count = (
            contract.actor_position_iterations
        )
        object_cfg.spawn.rigid_props.solver_velocity_iteration_count = (
            contract.actor_velocity_iterations
        )
    cfg.scene.clone_in_fabric = False
    cfg.scene.env_spacing = 0.75


def expected_runtime_config(contract: Stage16PhysxContractV1) -> dict[str, object]:
    """Stable requested values against which the live USD inspection is checked."""

    return {
        "device": contract.device,
        "enhanced_determinism": contract.enhanced_determinism,
        "solve_articulation_contact_last": contract.solve_articulation_contact_last,
        "solver_type": contract.solver_type,
        "scene_min_position_iterations": contract.scene_min_position_iterations,
        "scene_max_position_iterations": contract.scene_max_position_iterations,
        "scene_min_velocity_iterations": contract.scene_min_velocity_iterations,
        "scene_max_velocity_iterations": contract.scene_max_velocity_iterations,
        "actor_position_iterations": contract.actor_position_iterations,
        "actor_velocity_iterations": contract.actor_velocity_iterations,
        "gpu_max_rigid_contact_count": contract.gpu_max_rigid_contact_count,
        "gpu_max_rigid_patch_count": contract.gpu_max_rigid_patch_count,
        "scene_construction": contract.scene_construction,
    }


__all__ = [
    "Stage16PhysxContractV1",
    "apply_physx_contract",
    "baseline_contract",
    "candidate_matrix",
    "contract_from_mapping",
    "expected_runtime_config",
    "load_contract",
]
