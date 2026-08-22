"""Focused unit coverage for the object-agnostic profile objective."""

from __future__ import annotations

import torch

from toporetarget.rl.reference_tracking.source_profile_tracking import (
    Stage16SourceProfileTrackingV1,
    pose_derived_coupling_ratios,
    source_profile_tracking_terms,
)


def _inputs() -> dict[str, torch.Tensor | float]:
    return {
        "source_contact_activity": torch.zeros((1, 5)),
        "source_geometry_object_normalized": torch.zeros((1, 3)),
        "source_geometry_valid": torch.zeros(1, dtype=torch.bool),
        "source_linear_coupling_normalized": torch.zeros(1),
        "source_angular_coupling_normalized": torch.zeros(1),
        "object_characteristic_length_m": torch.ones(1),
        "robot_tip_positions_world": torch.zeros((1, 5, 3)),
        "robot_tip_pair_force_world": torch.zeros((1, 5, 3)),
        "object_pose_wxyz": torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        "robot_linear_coupling_normalized": torch.zeros(1),
        "robot_angular_coupling_normalized": torch.zeros(1),
        "contact_force_scale_n": 1.0,
    }


def test_identical_profile_is_near_zero_and_reward_is_bounded() -> None:
    terms = source_profile_tracking_terms(**_inputs())
    assert float(terms["l_profile"].item()) == 0.0
    assert float(terms["r_profile"].item()) == 1.0


def test_contact_geometry_and_coupling_residuals_are_separate() -> None:
    contact = _inputs()
    contact["source_contact_activity"] = torch.ones((1, 5))
    contact_terms = source_profile_tracking_terms(**contact)
    assert float(contact_terms["l_profile_contact"].item()) > 0.0
    assert float(contact_terms["l_profile_geometry"].item()) == 0.0

    geometry = _inputs()
    geometry["source_geometry_valid"] = torch.ones(1, dtype=torch.bool)
    geometry["source_geometry_object_normalized"] = torch.ones((1, 3))
    geometry_terms = source_profile_tracking_terms(**geometry)
    assert float(geometry_terms["l_profile_geometry"].item()) > 0.0
    assert float(geometry_terms["l_profile_contact"].item()) == 0.0

    coupling = _inputs()
    coupling["source_linear_coupling_normalized"] = torch.ones(1)
    coupling["source_angular_coupling_normalized"] = torch.ones(1)
    coupling_terms = source_profile_tracking_terms(**coupling)
    assert float(coupling_terms["l_profile_linear_coupling"].item()) > 0.0
    assert float(coupling_terms["l_profile_angular_coupling"].item()) > 0.0


def test_object_scale_normalization_is_invariant() -> None:
    small = _inputs()
    small["source_geometry_valid"] = torch.ones(1, dtype=torch.bool)
    small["source_geometry_object_normalized"] = torch.ones((1, 3))
    small["object_characteristic_length_m"] = torch.full((1,), 0.1)
    small["robot_tip_positions_world"] = torch.full((1, 5, 3), 0.1)
    large = _inputs()
    large["source_geometry_valid"] = torch.ones(1, dtype=torch.bool)
    large["source_geometry_object_normalized"] = torch.ones((1, 3))
    large["object_characteristic_length_m"] = torch.full((1,), 0.2)
    large["robot_tip_positions_world"] = torch.full((1, 5, 3), 0.2)
    assert float(source_profile_tracking_terms(**small)["l_profile_geometry"].item()) == 0.0
    assert float(source_profile_tracking_terms(**large)["l_profile_geometry"].item()) == 0.0


def test_pose_derived_coupling_is_finite_without_physx_omega() -> None:
    previous_wrist = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    current_wrist = previous_wrist.clone()
    previous_object = torch.tensor([[0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    current_object = previous_object.clone()
    linear, angular = pose_derived_coupling_ratios(
        previous_wrist_pose_wxyz=previous_wrist,
        current_wrist_pose_wxyz=current_wrist,
        previous_object_pose_wxyz=previous_object,
        current_object_pose_wxyz=current_object,
        dt_s=0.05,
    )
    assert torch.isfinite(linear).all()
    assert torch.isfinite(angular).all()
    assert float(linear.item()) == 0.0
    assert float(angular.item()) == 0.0


def test_contract_forbids_profile_tuning_and_grasp_gate() -> None:
    assert Stage16SourceProfileTrackingV1().profile_reward_weight == 1.0
    try:
        Stage16SourceProfileTrackingV1(fixed_pre_lift_grasp_gate_added=True)
    except ValueError as error:
        assert "FORBIDDEN" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("fixed pre-LIFT grasp gate was accepted")
