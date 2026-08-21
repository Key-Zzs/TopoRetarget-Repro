# ruff: noqa: E501

from __future__ import annotations

import numpy as np

from toporetarget.rl.dynamic_physical_qualification import (
    DynamicTerminalGate,
    dynamic_qualification,
    dynamic_twist_metrics,
    phase_labels_from_reference_index,
)


def _gate() -> DynamicTerminalGate:
    return DynamicTerminalGate(
        terminal_window_control_steps=2,
        terminal_linear_speed_mps=0.05,
        terminal_angular_speed_radps=0.5,
        terminal_free_object_linear_speed_mps=0.01,
        terminal_free_object_angular_speed_radps=0.25,
    )


def test_phase_labels_match_existing_reference_index_phase_code() -> None:
    phases = phase_labels_from_reference_index(
        np.asarray([0, 45, 46, 91, 92, 137, 138, 183, 184, 229, 230, 275, 276, 320])
    )
    assert phases.tolist() == [
        "PRE_CONTACT",
        "PRE_CONTACT",
        "APPROACH",
        "APPROACH",
        "CONTACT",
        "CONTACT",
        "GRASP",
        "GRASP",
        "LIFT",
        "LIFT",
        "MANIPULATION",
        "MANIPULATION",
        "TERMINAL",
        "TERMINAL",
    ]


def test_dynamic_twist_uses_world_frame_reference_alignment() -> None:
    reference = np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0, 0.4]] * 3)
    actual = reference.copy()
    actual[-1, 0] += 0.02
    actual[-1, 5] += 0.10
    result = dynamic_twist_metrics(
        actual_twist_world=actual,
        reference_twist_world=reference,
        hand_object_contact=np.asarray([True, True, True]),
        valid=np.asarray([False, True, True]),
        gate=_gate(),
    )
    assert result["frame_convention"] == "world_frame__delta_equals_actual_minus_reference"
    assert result["reference_twist_dynamic_pass"] is True
    assert result["absolute_world_terminal_velocity_used"] is False
    assert result["legacy_terminal_window_equivalent"]["Delta_v_terminal_max_mps"] == 0.02


def test_inherited_legacy_thresholds_are_applied_to_delta_not_absolute_velocity() -> None:
    reference = np.asarray([[3.0, 0.0, 0.0, 0.0, 0.0, 2.0]] * 3)
    actual = reference.copy()
    result = dynamic_twist_metrics(
        actual_twist_world=actual,
        reference_twist_world=reference,
        hand_object_contact=np.asarray([True, True, True]),
        valid=np.asarray([False, True, True]),
        gate=_gate(),
    )
    assert result["reference_twist_dynamic_pass"] is True
    assert result["legacy_terminal_window_equivalent"]["linear_limit_mps"] == [0.05, 0.05]
    assert result["legacy_terminal_window_equivalent"]["angular_limit_radps"] == [0.5, 0.5]


def test_dynamic_and_legacy_results_are_separate_receipts() -> None:
    interaction = {"interaction_dynamic_pass": True}
    twist = {"reference_twist_dynamic_pass": True}
    result = dynamic_qualification(
        legacy_kinematic_success=True,
        interaction=interaction,
        twist=twist,
        geometry_safe=True,
        action_bounds_safe=True,
        causal_execution_safe=True,
    )
    assert result["SR_dynamic"] is True
    assert result["ABSOLUTE_WORLD_TERMINAL_ZERO_SPEED_REQUIRED"] == "NO"
    assert result["SR_HOLD_IMPLEMENTED"] == "NO"
