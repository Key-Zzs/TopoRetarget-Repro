"""Versioned Stage 16-D semantic, contact, and qualification contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

PHYSICS_CONSISTENT_RETARGETING_PROTOCOL = "physics_consistent_retargeting_v1"
TASK_CLASSES = (
    "grasp_only",
    "grasp_and_hold",
    "transport",
    "in_hand_translation",
    "in_hand_rotation",
    "release",
    "mixed_or_ambiguous",
    "generic_contact_preserving_motion",
)
CONTACT_GROUPS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ContactWindowV1:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("contact window must be ordered and nonnegative")


@dataclass(frozen=True)
class TaskSemanticContractV1:
    clip: str
    task_class: str
    classification_confidence: float
    classification_status: str
    source_motion_class: str
    source_frame_count: int
    retimed_frame_count: int
    initial_object_pose_wxyz: tuple[float, ...]
    initial_wrist_object_transform: tuple[tuple[float, ...], ...]
    final_wrist_object_transform: tuple[tuple[float, ...], ...]
    contact_onset_window: ContactWindowV1
    persistent_contact_window: ContactWindowV1
    contact_end_window: ContactWindowV1
    final_hold_window: ContactWindowV1
    observed_contact_bodies: tuple[str, ...]
    observed_contact_groups: tuple[str, ...]
    source_contact_control_steps: int
    source_contact_duration_s: float
    source_object_translation_m: float
    source_object_rotation_deg: float
    source_object_relative_palm_translation_m: float
    source_object_relative_palm_rotation_deg: float
    source_final_linear_speed_mps: float
    source_final_angular_speed_radps: float
    limitations: tuple[str, ...] = ()
    protocol: str = PHYSICS_CONSISTENT_RETARGETING_PROTOCOL

    def __post_init__(self) -> None:
        if self.task_class not in TASK_CLASSES or self.source_motion_class not in TASK_CLASSES:
            raise ValueError("unknown task class")
        if not 0.0 <= self.classification_confidence <= 1.0:
            raise ValueError("classification confidence must be in [0,1]")
        if self.source_frame_count < 2 or self.retimed_frame_count < self.source_frame_count:
            raise ValueError("semantic contract requires a complete retimed trajectory")
        if len(self.initial_object_pose_wxyz) != 7:
            raise ValueError("initial object pose must be xyz+wxyz")
        if self.source_contact_control_steps < 0:
            raise ValueError("contact duration cannot be negative")
        for name in (
            "source_contact_duration_s",
            "source_object_translation_m",
            "source_object_rotation_deg",
            "source_object_relative_palm_translation_m",
            "source_object_relative_palm_rotation_deg",
            "source_final_linear_speed_mps",
            "source_final_angular_speed_radps",
        ):
            _finite(float(getattr(self, name)), name)

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "TaskSemanticContractV1", **asdict(self)}


@dataclass(frozen=True)
class PersistentContactTopologyV1:
    clip: str
    required_body_groups: tuple[str, ...]
    optional_body_groups: tuple[str, ...]
    forbidden_unrelated_contacts: tuple[str, ...]
    minimum_persistence_control_steps: int
    source_onset_window: ContactWindowV1
    final_hold_window: ContactWindowV1
    contact_graph_edges: tuple[tuple[str, str], ...]
    group_weights: dict[str, float]
    source_group_step_counts: dict[str, int]
    transient_filter_control_steps: int
    raw_point_precision: str
    protocol: str = PHYSICS_CONSISTENT_RETARGETING_PROTOCOL

    def __post_init__(self) -> None:
        known = set(CONTACT_GROUPS)
        if not set(self.required_body_groups).issubset(known):
            raise ValueError("required contact group is unknown")
        if not set(self.optional_body_groups).issubset(known):
            raise ValueError("optional contact group is unknown")
        if set(self.required_body_groups) & set(self.optional_body_groups):
            raise ValueError("required and optional contact groups must be disjoint")
        if self.minimum_persistence_control_steps < 1:
            raise ValueError("contact persistence must be positive")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.group_weights.values()):
            raise ValueError("contact group weights must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "PersistentContactTopologyV1", **asdict(self)}


@dataclass(frozen=True)
class PhysicsConsistentTaskGateV1:
    clip: str
    object_bbox_diagonal_m: float
    minimum_contact_recall: float
    minimum_semantic_progress: float
    minimum_object_motion_m: float
    minimum_object_rotation_deg: float
    terminal_window_control_steps: int
    workspace_radius_m: float
    catastrophic_penetration_m: float = 0.010
    p95_penetration_m: float = 0.003
    action_limit: float = 1.0
    seed_success_rate: float = 0.80
    ppo_success_rate: float = 0.90
    protocol: str = PHYSICS_CONSISTENT_RETARGETING_PROTOCOL
    hard_gates: tuple[str, ...] = field(
        default=(
            "finite",
            "no_formal_wrist_or_object_state_writes",
            "no_hidden_force",
            "no_hidden_attachment",
            "action_bounds",
            "wrist_safety",
            "finger_joint_limits",
            "workspace",
            "catastrophic_penetration",
            "terminal_semantics",
            "contact_topology",
            "contact_driven_object_motion",
        )
    )

    def __post_init__(self) -> None:
        for name in (
            "object_bbox_diagonal_m",
            "minimum_contact_recall",
            "minimum_semantic_progress",
            "minimum_object_motion_m",
            "minimum_object_rotation_deg",
            "workspace_radius_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.minimum_contact_recall <= 1.0:
            raise ValueError("contact recall gate must be in [0,1]")
        if not 0.0 <= self.minimum_semantic_progress <= 1.0:
            raise ValueError("semantic progress gate must be in [0,1]")
        if self.terminal_window_control_steps < 1:
            raise ValueError("terminal window must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "PhysicsConsistentTaskGateV1", **asdict(self)}


def derive_task_gate(
    semantic: TaskSemanticContractV1,
    *,
    object_bbox_diagonal_m: float,
) -> PhysicsConsistentTaskGateV1:
    """Freeze task-scaled gates before any optimization result is observed."""

    if not math.isfinite(object_bbox_diagonal_m) or object_bbox_diagonal_m <= 0.0:
        raise ValueError("object bbox diagonal must be finite and positive")
    translation_scale = semantic.source_object_translation_m
    rotation_scale = semantic.source_object_rotation_deg
    return PhysicsConsistentTaskGateV1(
        clip=semantic.clip,
        object_bbox_diagonal_m=object_bbox_diagonal_m,
        minimum_contact_recall=0.50 if semantic.classification_confidence < 0.60 else 0.70,
        minimum_semantic_progress=0.30 if semantic.classification_confidence < 0.60 else 0.60,
        minimum_object_motion_m=max(
            0.0025,
            min(0.25 * object_bbox_diagonal_m, 0.10 * translation_scale),
        ),
        minimum_object_rotation_deg=(
            max(1.0, 0.10 * rotation_scale)
            if semantic.source_motion_class in {"in_hand_rotation", "mixed_or_ambiguous"}
            else 0.0
        ),
        terminal_window_control_steps=max(3, min(20, semantic.retimed_frame_count // 16)),
        workspace_radius_m=max(0.30, 4.0 * object_bbox_diagonal_m),
    )


__all__ = [
    "CONTACT_GROUPS",
    "PHYSICS_CONSISTENT_RETARGETING_PROTOCOL",
    "TASK_CLASSES",
    "ContactWindowV1",
    "PersistentContactTopologyV1",
    "PhysicsConsistentTaskGateV1",
    "TaskSemanticContractV1",
    "derive_task_gate",
]
