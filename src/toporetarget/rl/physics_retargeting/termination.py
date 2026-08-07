"""Stage 16-D task gates and termination, independent of strict object tracking."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import PhysicsConsistentTaskGateV1

STAGE16D_TERMINATION_REASONS = (
    "NONE",
    "SUCCESS_TASK_SEMANTICS",
    "FAILURE_NUMERICAL",
    "FAILURE_CATASTROPHIC_PENETRATION",
    "FAILURE_OBJECT_WORKSPACE",
    "FAILURE_WRIST_SAFETY",
    "FAILURE_JOINT_LIMIT",
    "FAILURE_ACTION_INVALID",
    "FAILURE_OBJECT_EXPLOSION",
    "TIMEOUT",
    "FAILURE_INTER_FINGER_PENETRATION",
)


def physics_consistent_termination(
    metrics: dict[str, torch.Tensor],
    gate: PhysicsConsistentTaskGateV1,
    *,
    final_step: torch.Tensor,
) -> dict[str, Any]:
    required = {
        "finite",
        "penetration_m",
        "inter_finger_penetration_m",
        "workspace_distance_m",
        "wrist_safe",
        "joint_limits_safe",
        "action_valid",
        "object_speed_mps",
        "semantic_progress",
        "contact_recall",
        "contact_causality",
        "terminal_stable",
        "object_motion_m",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Stage16D termination misses metrics: {sorted(missing)}")
    finite = metrics["finite"].bool()
    numerical = ~finite
    penetration = metrics["penetration_m"] > gate.catastrophic_penetration_m
    inter_finger = metrics["inter_finger_penetration_m"] > gate.maximum_inter_finger_penetration_m
    workspace = metrics["workspace_distance_m"] > gate.workspace_radius_m
    wrist = ~metrics["wrist_safe"].bool()
    joints = ~metrics["joint_limits_safe"].bool()
    action = ~metrics["action_valid"].bool()
    explosion = metrics["object_speed_mps"] > 5.0
    hard_failure = (
        numerical | penetration | inter_finger | workspace | wrist | joints | action | explosion
    )
    semantic_success = (
        (metrics["semantic_progress"] >= gate.minimum_semantic_progress)
        & (metrics["contact_recall"] >= gate.minimum_contact_recall)
        & metrics["contact_causality"].bool()
        & metrics["terminal_stable"].bool()
        & (metrics["object_motion_m"] >= gate.minimum_object_motion_m)
        & ~hard_failure
        & final_step.bool()
    )
    timed_out = final_step.bool() & ~semantic_success & ~hard_failure
    reason = torch.zeros_like(final_step, dtype=torch.long)
    ordered = (
        (timed_out, 9),
        (semantic_success, 1),
        (inter_finger, 10),
        (explosion, 8),
        (action, 7),
        (joints, 6),
        (wrist, 5),
        (workspace, 4),
        (penetration, 3),
        (numerical, 2),
    )
    for mask, code in ordered:
        reason = torch.where(mask, code, reason)
    return {
        "terminated": hard_failure | semantic_success,
        "timed_out": timed_out,
        "success": semantic_success,
        "primary_reason_code": reason,
        "all_triggered": {
            "numerical": numerical,
            "catastrophic_penetration": penetration,
            "inter_finger_penetration": inter_finger,
            "workspace": workspace,
            "wrist_safety": wrist,
            "joint_limits": joints,
            "action_invalid": action,
            "object_explosion": explosion,
            "timeout": timed_out,
            "success": semantic_success,
        },
    }


__all__ = ["STAGE16D_TERMINATION_REASONS", "physics_consistent_termination"]
