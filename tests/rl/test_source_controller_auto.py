from __future__ import annotations

import math

import pytest
import torch

from toporetarget.rl.environments.isaaclab_backend.explicit_virtual_wrist import (
    continuous_angle_branch,
    explicit_3p3r_rotation_matrix,
    nearest_equivalent_angle,
    se3_target_to_explicit_3p3r,
)
from toporetarget.rl.environments.isaaclab_backend.tensor_math import quaternion_exp_wxyz
from toporetarget.rl.environments.isaaclab_backend.virtual_wrist_asset import (
    explicit_virtual_wrist_recipe,
)
from toporetarget.rl.source_controller import (
    SourceControllerDecision,
    SourceControllerMode,
    SourceControllerSafetyContractV1,
    select_source_controller_route,
    selected_mode_for_clip,
)


def _receipt(clip: str, passed: bool) -> dict[str, object]:
    names = (
        "reference_tracking_pass",
        "contact_execution_pass",
        "reference_progression_pass",
        "finite_safe",
        "controller_authority_pass",
        "joint_limits_safe",
        "actuator_limits_safe",
        "action_bounds_safe",
        "collision_safety_pass",
    )
    return {"clip_id": clip, **{name: passed for name in names}}


def test_continuous_angle_branch_crosses_principal_boundary_without_jump() -> None:
    principal = torch.deg2rad(
        torch.tensor([[178.0], [179.0], [-179.0], [-178.0]], dtype=torch.float64)
    )
    continuous = continuous_angle_branch(principal)
    assert torch.rad2deg(continuous[:, 0]).tolist() == pytest.approx([178.0, 179.0, 181.0, 182.0])
    assert float(torch.max(torch.abs(torch.diff(continuous[:, 0])))) < math.radians(3.0)


def test_223_872_degrees_is_an_equivalent_branch_not_a_physical_limit_failure() -> None:
    principal = torch.tensor([math.radians(-136.128)], dtype=torch.float64)
    previous = torch.tensor([math.radians(220.0)], dtype=torch.float64)
    continuous = nearest_equivalent_angle(principal, previous)
    assert math.degrees(float(continuous[0])) == pytest.approx(223.872, abs=1.0e-9)
    q_principal = quaternion_exp_wxyz(torch.tensor([[0.0, 0.0, principal[0]]], dtype=torch.float64))
    joints = se3_target_to_explicit_3p3r(
        torch.zeros((1, 3), dtype=torch.float64),
        q_principal,
        previous_joint_position=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, previous[0]]], dtype=torch.float64
        ),
    )
    assert math.degrees(float(joints[0, 5])) == pytest.approx(223.872, abs=1.0e-9)
    principal_joints = joints.clone()
    principal_joints[0, 5] = principal[0]
    assert torch.allclose(
        explicit_3p3r_rotation_matrix(joints),
        explicit_3p3r_rotation_matrix(principal_joints),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_source_controller_auto_uses_zero_residual_then_corrected_l0_fallback() -> None:
    zero = [_receipt("a", True), _receipt("b", False)]
    corrected = [_receipt("b", True)]
    assert (
        select_source_controller_route(zero, corrected)
        is SourceControllerDecision.AUTO_ZERO_RESIDUAL_THEN_L0_FALLBACK
    )
    assert selected_mode_for_clip(zero[0]) is SourceControllerMode.ZERO_RESIDUAL
    assert selected_mode_for_clip(zero[1]) is SourceControllerMode.CORRECTED_L0


def test_missing_l0_fallback_evidence_is_fail_closed_inconclusive() -> None:
    assert (
        select_source_controller_route([_receipt("a", False)], [])
        is SourceControllerDecision.L0_AUTHORITY_INCONCLUSIVE
    )


def test_real_limits_remain_active_under_auto_contract() -> None:
    safety = SourceControllerSafetyContractV1()
    assert safety.real_finger_joint_limits is True
    assert safety.actuator_effort_limits is True
    assert safety.actuator_velocity_limits is True
    assert safety.action_bounds is True
    assert safety.singularity_detection is True
    assert safety.collision_safety is True


def test_continuous_virtual_wrist_recipe_retains_real_finger_and_translation_limits() -> None:
    recipe = explicit_virtual_wrist_recipe(
        "high_authority_bounded", continuous_virtual_wrist_angles=True
    )
    assert recipe["continuous_virtual_wrist_angles"] is True
    assert recipe["virtual_wrist_rotation_limits_enforced"] is False
    assert recipe["rotation_limits_deg"] is None
    assert recipe["finger_joint_position_limits_enforced"] is True
    assert recipe["virtual_wrist_translation_limits_enforced"] is True
    assert recipe["translation_limits_m"] == [-0.4, 0.4]
