"""Task-aware, consecutive-window terminal stability predicates."""

from __future__ import annotations

import torch

from .contracts import PhysicsConsistentTaskGateV1

TERMINAL_CONTACT_MODES = ("required", "forbidden", "optional")


def derive_terminal_contact_mode(task_class: str, source_motion_class: str) -> str:
    """Map shared semantic classes to one terminal contact requirement."""

    if task_class == "release" or source_motion_class == "release":
        return "forbidden"
    contact_held = {
        "grasp_only",
        "grasp_and_hold",
        "transport",
        "in_hand_translation",
        "in_hand_rotation",
    }
    if task_class in contact_held or source_motion_class in contact_held:
        return "required"
    return "optional"


def terminal_kinematic_step_pass(
    linear_speed_mps: torch.Tensor,
    angular_speed_radps: torch.Tensor,
    contact_present: torch.Tensor,
    gate: PhysicsConsistentTaskGateV1,
) -> torch.Tensor:
    """Use stricter thresholds whenever an unsupported object is free."""

    linear_limit = torch.where(
        contact_present.bool(),
        torch.full_like(linear_speed_mps, gate.terminal_linear_speed_mps),
        torch.full_like(linear_speed_mps, gate.terminal_free_object_linear_speed_mps),
    )
    angular_limit = torch.where(
        contact_present.bool(),
        torch.full_like(angular_speed_radps, gate.terminal_angular_speed_radps),
        torch.full_like(angular_speed_radps, gate.terminal_free_object_angular_speed_radps),
    )
    return (linear_speed_mps <= linear_limit) & (angular_speed_radps <= angular_limit)


def terminal_contact_window_pass(
    contact_steps: torch.Tensor,
    observed_steps: torch.Tensor,
    gate: PhysicsConsistentTaskGateV1,
) -> torch.Tensor:
    """Evaluate required/forbidden/optional contact over the complete window."""

    complete = observed_steps >= gate.terminal_window_control_steps
    if gate.terminal_contact_mode == "required":
        required = int(
            gate.terminal_required_contact_fraction * gate.terminal_window_control_steps + 0.999999
        )
        return complete & (contact_steps >= required)
    if gate.terminal_contact_mode == "forbidden":
        return complete & (contact_steps == 0)
    return complete


__all__ = [
    "TERMINAL_CONTACT_MODES",
    "derive_terminal_contact_mode",
    "terminal_contact_window_pass",
    "terminal_kinematic_step_pass",
]
