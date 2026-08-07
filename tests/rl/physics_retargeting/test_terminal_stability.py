from __future__ import annotations

import torch

from toporetarget.rl.physics_retargeting.contracts import PhysicsConsistentTaskGateV1
from toporetarget.rl.physics_retargeting.terminal_stability import (
    derive_terminal_contact_mode,
    terminal_contact_window_pass,
    terminal_kinematic_step_pass,
)


def _gate(**overrides) -> PhysicsConsistentTaskGateV1:
    values = {
        "clip": "synthetic",
        "object_bbox_diagonal_m": 0.1,
        "minimum_contact_recall": 0.5,
        "minimum_semantic_progress": 0.3,
        "minimum_object_motion_m": 0.01,
        "minimum_object_rotation_deg": 0.0,
        "terminal_window_control_steps": 20,
        "workspace_radius_m": 0.5,
        "terminal_contact_mode": "required",
    }
    values.update(overrides)
    return PhysicsConsistentTaskGateV1(**values)


def test_in_hand_motion_requires_terminal_contact() -> None:
    assert (
        derive_terminal_contact_mode("generic_contact_preserving_motion", "in_hand_rotation")
        == "required"
    )
    assert derive_terminal_contact_mode("release", "mixed_or_ambiguous") == "forbidden"


def test_terminal_contact_gate_requires_complete_consecutive_window() -> None:
    gate = _gate()
    assert not terminal_contact_window_pass(torch.tensor([16]), torch.tensor([19]), gate).item()
    assert terminal_contact_window_pass(torch.tensor([16]), torch.tensor([20]), gate).item()
    assert not terminal_contact_window_pass(torch.tensor([15]), torch.tensor([20]), gate).item()


def test_free_fast_object_fails_even_when_old_threshold_would_pass() -> None:
    gate = _gate(terminal_contact_mode="optional")
    passed = terminal_kinematic_step_pass(
        torch.tensor([0.0386, 0.0]),
        torch.tensor([0.824, 0.0]),
        torch.tensor([False, False]),
        gate,
    )
    assert passed.tolist() == [False, True]


def test_contacted_object_uses_bounded_contact_thresholds() -> None:
    gate = _gate()
    passed = terminal_kinematic_step_pass(
        torch.tensor([0.04, 0.06]),
        torch.tensor([0.4, 0.4]),
        torch.tensor([True, True]),
        gate,
    )
    assert passed.tolist() == [True, False]
