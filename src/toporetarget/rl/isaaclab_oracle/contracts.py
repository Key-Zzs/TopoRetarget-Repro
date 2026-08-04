"""Versioned, explicit contracts for the bounded C.5A replication stage."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import torch

STATE_VERSION = "Stage16C5CandidateStateV1"


class CandidateStateValidationError(ValueError):
    """Raised when a candidate state is incompatible or incomplete."""


@dataclass(frozen=True)
class CandidateStateFieldV1:
    """A single state field and its causal classification."""

    name: str
    classification: str
    restore_method: str
    required: bool = True


REQUIRED_STATE_FIELDS: tuple[CandidateStateFieldV1, ...] = (
    CandidateStateFieldV1("robot_joint_pos", "simulation", "write_joint_state_to_sim"),
    CandidateStateFieldV1("robot_joint_vel", "simulation", "write_joint_state_to_sim"),
    CandidateStateFieldV1("robot_root_state", "simulation", "write_root_state_to_sim"),
    CandidateStateFieldV1("object_170105_root_state", "simulation", "write_root_state_to_sim"),
    CandidateStateFieldV1("object_170650_root_state", "simulation", "write_root_state_to_sim"),
    CandidateStateFieldV1("clip_index", "task", "copy_"),
    CandidateStateFieldV1("reference_index", "task", "copy_"),
    CandidateStateFieldV1("target_reference_index", "task", "copy_"),
    CandidateStateFieldV1("actions", "action_history", "copy_"),
    CandidateStateFieldV1("previous_actions", "action_history", "copy_"),
    CandidateStateFieldV1("second_previous_actions", "action_history", "copy_"),
    CandidateStateFieldV1("joint_target_isaac", "controller", "copy_and_set_target"),
    CandidateStateFieldV1("explicit_wrist_joint_target", "controller", "copy_and_set_target"),
    CandidateStateFieldV1(
        "explicit_wrist_joint_velocity_target", "controller", "copy_and_set_target"
    ),
    CandidateStateFieldV1("previous_explicit_wrist_joint_target", "controller", "copy_"),
    CandidateStateFieldV1("wrist_target_position", "controller", "copy_"),
    CandidateStateFieldV1("wrist_target_quaternion", "controller", "copy_"),
    CandidateStateFieldV1("wrist_target_twist", "controller", "copy_"),
    CandidateStateFieldV1("success", "task", "copy_"),
    CandidateStateFieldV1("reason_codes", "task", "copy_"),
    CandidateStateFieldV1("source_env_origins", "replication", "origin_rebase"),
)


@dataclass
class Stage16C5CandidateStateV1:
    """GPU tensor snapshot used by the C.5A candidate pool.

    ``tensors`` contains only per-environment tensors whose leading dimension
    equals ``env_count``.  Scalars are immutable run metadata or explicitly
    documented software state.  PhysX solver/contact caches are deliberately
    absent because Isaac Lab exposes no supported API to restore them.
    """

    config_hashes: dict[str, str]
    tensors: dict[str, torch.Tensor]
    scalars: dict[str, Any]
    env_count: int
    version: str = STATE_VERSION
    inaccessible_physx_state: tuple[str, ...] = (
        "solver_warm_start",
        "contact_manifold_cache",
        "friction_patch_state",
        "internal_constraint_cache",
    )

    def field_manifest(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        fields = {field.name: field for field in REQUIRED_STATE_FIELDS}
        for name in sorted(self.tensors):
            value = self.tensors[name]
            definition = fields.get(name)
            rows.append(
                {
                    "field": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "classification": (
                        definition.classification if definition is not None else "auxiliary"
                    ),
                    "restore_method": (
                        definition.restore_method if definition is not None else "copy_"
                    ),
                    "required": definition.required if definition is not None else False,
                }
            )
        return rows

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "env_count": self.env_count,
            "config_hashes": dict(self.config_hashes),
            "fields": self.field_manifest(),
            "scalars": self.scalars,
            "inaccessible_physx_state": list(self.inaccessible_physx_state),
        }


@dataclass
class Stage16C5WriteAuditV1:
    """Separate reset/setup/formal-write accounting with compact call stacks."""

    reset_writes: int = 0
    candidate_setup_writes: int = 0
    formal_execution_rollout_writes: int = 0
    events: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        *,
        category: str,
        operation: str,
        env_ids: torch.Tensor | list[int],
        tensor_names: list[str],
    ) -> None:
        if category not in {"reset", "candidate_setup", "formal_execution_rollout"}:
            raise ValueError(f"unknown write category: {category}")
        count = int(torch.as_tensor(env_ids).numel())
        if category == "reset":
            self.reset_writes += count
        elif category == "candidate_setup":
            self.candidate_setup_writes += count
        else:
            self.formal_execution_rollout_writes += count
        stack = traceback.extract_stack(limit=8)
        self.events.append(
            {
                "category": category,
                "operation": operation,
                "env_ids": torch.as_tensor(env_ids).detach().cpu().tolist(),
                "tensors": tensor_names,
                "stack": [f"{row.filename}:{row.lineno}:{row.name}" for row in stack[:-1]],
            }
        )

    def clear_candidate_setup(self) -> None:
        self.candidate_setup_writes = 0
        self.events = [event for event in self.events if event["category"] != "candidate_setup"]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "Stage16C5WriteAuditV1",
            "reset_writes": self.reset_writes,
            "candidate_setup_writes": self.candidate_setup_writes,
            "formal_execution_rollout_writes": self.formal_execution_rollout_writes,
            "events": self.events,
        }


__all__ = [
    "CandidateStateFieldV1",
    "CandidateStateValidationError",
    "REQUIRED_STATE_FIELDS",
    "STATE_VERSION",
    "Stage16C5CandidateStateV1",
    "Stage16C5WriteAuditV1",
]
