from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from toporetarget.rl.environments.isaaclab_backend.explicit_virtual_wrist import (
    continuous_angle_branch,
    explicit_3p3r_rotation_matrix,
    nearest_equivalent_angle,
    se3_target_to_explicit_3p3r,
)
from toporetarget.rl.environments.isaaclab_backend.tensor_math import quaternion_exp_wxyz
from toporetarget.rl.environments.isaaclab_backend.termination_terms import (
    source_controller_admission_dones_v2,
)
from toporetarget.rl.environments.isaaclab_backend.virtual_wrist_asset import (
    explicit_virtual_wrist_recipe,
)
from toporetarget.rl.ppo.networks import ActorCritic
from toporetarget.rl.source_controller import (
    EXECUTABILITY_V2_REQUIRED_TRUE,
    SourceControllerDecision,
    SourceControllerExecutability,
    SourceControllerFidelity,
    SourceControllerMode,
    SourceControllerRouteV2,
    SourceControllerSafetyContractV1,
    make_zero_output_residual_actor_,
    select_source_controller_route,
    select_source_controller_route_v2,
    selected_mode_for_clip,
    source_controller_executability_v2,
    source_controller_fidelity_v2,
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


def _v2_receipt(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        **{name: True for name in EXECUTABILITY_V2_REQUIRED_TRUE},
        "wrist_position_tracking_pass": True,
        "wrist_rotation_tracking_pass": True,
        "finger_tracking_pass": True,
        "link_tracking_pass": True,
        "source_contact_recall_pass": True,
        "object_tracking_pass": True,
        "interaction_progression_pass": True,
        "command_clamp_pass": True,
        "actuator_saturation_pass": True,
        "reference_completion_pass": True,
        "normalized_wrist_tracking_error": 0.1,
        "normalized_finger_tracking_error": 0.1,
        "command_clamp_fraction": 0.0,
        "actuator_saturation_fraction": 0.0,
        "source_contact_recall": 1.0,
        "object_tracking_score": 1.0,
    }
    result.update(overrides)
    return result


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


@pytest.mark.parametrize(
    "diagnostic",
    (
        "source_contact_recall_pass",
        "object_tracking_pass",
        "interaction_progression_pass",
        "reference_completion_pass",
    ),
)
def test_task_fidelity_failure_is_degraded_but_executable(diagnostic: str) -> None:
    receipt = _v2_receipt(**{diagnostic: False})
    assert (
        source_controller_executability_v2(receipt)
        is SourceControllerExecutability.PASS
    )
    assert source_controller_fidelity_v2(receipt) is SourceControllerFidelity.DEGRADED
    assert select_source_controller_route_v2(receipt) is SourceControllerRouteV2.ZERO_RESIDUAL


@pytest.mark.parametrize(
    "hard_gate",
    (
        "real_finger_joint_limits_safe",
        "virtual_wrist_translation_limits_safe",
        "actuator_effort_limits_safe",
        "actuator_velocity_limits_safe",
        "action_bounds_safe",
        "singularity_safety_pass",
        "catastrophic_collision_safe",
        "nonfinite_dynamics_absent",
        "controller_divergence_absent",
    ),
)
def test_true_execution_or_physical_limit_remains_hard(hard_gate: str) -> None:
    receipt = _v2_receipt(**{hard_gate: False})
    assert (
        source_controller_executability_v2(receipt)
        is SourceControllerExecutability.FAIL
    )


def test_auto_v2_uses_l0_only_as_executable_fallback_or_source_side_improvement() -> None:
    non_executable_zero = _v2_receipt(real_finger_joint_limits_safe=False)
    l0 = _v2_receipt()
    assert (
        select_source_controller_route_v2(non_executable_zero, l0)
        is SourceControllerRouteV2.CORRECTED_L0
    )
    assert (
        select_source_controller_route_v2(non_executable_zero, None)
        is SourceControllerRouteV2.SOURCE_CONTROLLER_HARD_FAILURE
    )
    worse_l0 = _v2_receipt(
        normalized_wrist_tracking_error=0.4,
        normalized_finger_tracking_error=0.4,
    )
    assert (
        select_source_controller_route_v2(_v2_receipt(), worse_l0)
        is SourceControllerRouteV2.ZERO_RESIDUAL
    )


def test_object_fidelity_termination_does_not_stop_v2_admission() -> None:
    terminated, success = source_controller_admission_dones_v2(
        {"primary_reason_code": torch.tensor([2, 3, 4, 1, 5, 6])},
        reference_index=torch.tensor([4, 4, 4, 4, 4, 4]),
        final_reference_index=4,
    )
    assert terminated.tolist() == [False, False, False, True, True, True]
    assert success.tolist() == [True, True, True, False, False, False]


def test_zero_output_network_matches_deterministic_zero_residual_after_reload() -> None:
    torch.manual_seed(7)
    model = ActorCritic(observation_dim=11, action_dim=26)
    make_zero_output_residual_actor_(model)
    observations = torch.randn(5, 11)
    expected = torch.zeros(5, 26)
    assert torch.equal(model.mean(observations), expected)
    restored = ActorCritic(observation_dim=11, action_dim=26)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.mean(observations), expected)


def test_unbounded_profile_is_diagnostic_only_not_production_authority() -> None:
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "configs/contracts/source_controller_auto_v2.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    assert contract["fallback"]["bounded_targets_required"] is True
    assert contract["fallback"]["wrapped_principal_angle_gate"] == "forbidden"
    assert contract["diagnostic_only_profiles"] == [
        "l0_unbounded_joint_targets_then_physical_grouped_rse_v1"
    ]
