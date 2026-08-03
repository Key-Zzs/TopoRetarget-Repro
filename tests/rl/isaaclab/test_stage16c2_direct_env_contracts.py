"""CPU-only contracts for the optional Stage 16-C.2 Isaac backend."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from toporetarget.rl.environments.isaaclab_backend.action_adapter import Stage16ActionAdapter
from toporetarget.rl.environments.isaaclab_backend.finite_virtual_wrist_actuator import (
    VIRTUAL_6D_JOINT_ORDER,
    FiniteVirtual6DWristActuator,
)
from toporetarget.rl.environments.isaaclab_backend.inverse_wrench_controller import (
    BatchedEffectiveWrenchMapIdentifier,
    DampedSVDInverseWrenchController,
    EffectiveWrenchMap,
    IdentifiedInverseWrenchProfileV1,
)
from toporetarget.rl.environments.isaaclab_backend.recovery_state_machine import (
    RecoveryStage,
    Stage16C3R2C5RecoveryStateMachine,
)
from toporetarget.rl.environments.isaaclab_backend.reference_bank import WorldWristReferenceBank
from toporetarget.rl.environments.isaaclab_backend.reward_terms import world_wrist_reward_terms
from toporetarget.rl.environments.isaaclab_backend.scene_frame import (
    global_to_scene,
    scene_to_global,
)
from toporetarget.rl.environments.isaaclab_backend.semantic_checks import (
    derive_fully_kinematic_link_targets,
)
from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
    quaternion_exp_wxyz,
    quaternion_geodesic,
    quaternion_log_wxyz,
)
from toporetarget.rl.environments.isaaclab_backend.termination_terms import stage16_termination
from toporetarget.rl.environments.isaaclab_backend.wrist_controller import (
    ArticulatedHandCompositeInertiaEstimator,
    IsaacComputedWrenchWristControllerV2,
    IsaacComputedWrenchWristProfileV2,
    IsaacEffectiveDynamicsWristControllerV3,
    IsaacEffectiveDynamicsWristProfileV3,
    PhysicsSubstepWristTargetInterpolator,
    WristEffectiveDynamicsIdentifier,
)

JOINTS = tuple(f"joint_{index}" for index in range(20))


def adapter() -> Stage16ActionAdapter:
    return Stage16ActionAdapter(
        canonical_joint_names=JOINTS,
        isaac_joint_names=tuple(reversed(JOINTS)),
        joint_lower=torch.full((20,), -1.0),
        joint_upper=torch.full((20,), 1.0),
    )


def identity(batch: int = 1) -> torch.Tensor:
    value = torch.zeros((batch, 4))
    value[:, 0] = 1.0
    return value


def test_scene_origin_invariance() -> None:
    points = torch.tensor([[[0.1, -0.2, 0.3]], [[0.1, -0.2, 0.3]]])
    origins = torch.tensor([[0.0, 0.0, 0.0], [1.5, -2.0, 4.0]])
    global_points = scene_to_global(points, origins[:, None, :])
    assert torch.allclose(global_to_scene(global_points, origins[:, None, :]), points)
    assert torch.allclose(
        global_to_scene(global_points, origins[:, None, :])[0],
        global_to_scene(global_points, origins[:, None, :])[1],
    )


def test_action_mapping_round_trip_and_bounds() -> None:
    value = torch.arange(20, dtype=torch.float32).reshape(1, 20)
    mapping = adapter()
    assert torch.equal(mapping.isaac_to_canonical(mapping.canonical_to_isaac(value)), value)
    action = torch.ones((1, 26))
    target = mapping.finger_target_canonical(torch.zeros((1, 20)), action)
    assert torch.allclose(target, torch.full((1, 20), 0.2))


def test_action_rejects_wrong_shape_or_nonfinite() -> None:
    mapping = adapter()
    with pytest.raises(ValueError, match="shape"):
        mapping.validate_action(torch.zeros((1, 25)))
    action = torch.zeros((1, 26))
    action[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        mapping.validate_action(action)


def test_quaternion_identity_and_shortest_arc() -> None:
    rotation = torch.tensor([[0.0, 0.0, math.pi / 2]])
    quaternion = quaternion_exp_wxyz(rotation)
    assert torch.allclose(quaternion_log_wxyz(identity()), torch.zeros((1, 3)))
    assert torch.allclose(quaternion_log_wxyz(quaternion), rotation, atol=1.0e-6)
    assert torch.allclose(quaternion_geodesic(quaternion, -quaternion), torch.zeros(1))


def test_reward_is_monotonic_from_zero_error() -> None:
    zeros3 = torch.zeros((1, 3))
    zeros6 = torch.zeros((1, 6, 3))
    zeros16 = torch.zeros((1, 16, 3))
    zeros20 = torch.zeros((1, 20))
    kwargs = {
        "object_axis_points": zeros6,
        "object_axis_points_ref": zeros6,
        "tracked_links": zeros16,
        "tracked_links_ref": zeros16,
        "finger_q": zeros20,
        "finger_q_ref": zeros20,
        "joint_lower": torch.full((20,), -1.0),
        "joint_upper": torch.full((20,), 1.0),
        "wrist_position": zeros3,
        "wrist_quaternion_wxyz": identity(),
        "wrist_position_ref": zeros3,
        "wrist_quaternion_ref_wxyz": identity(),
        "action": zeros20.new_zeros((1, 26)),
        "previous_action": zeros20.new_zeros((1, 26)),
        "second_previous_action": zeros20.new_zeros((1, 26)),
    }
    reward = world_wrist_reward_terms(**kwargs)["total"]
    kwargs["object_axis_points"] = torch.full((1, 6, 3), 0.1)
    perturbed = world_wrist_reward_terms(**kwargs)["total"]
    assert torch.all(perturbed < reward)


def test_termination_failure_precedes_success_at_final_frame() -> None:
    kwargs = {
        "object_position": torch.tensor([[0.051, 0.0, 0.0]]),
        "object_quaternion_wxyz": identity(),
        "object_axis_points": torch.zeros((1, 6, 3)),
        "object_position_ref": torch.zeros((1, 3)),
        "object_quaternion_ref_wxyz": identity(),
        "object_axis_points_ref": torch.zeros((1, 6, 3)),
        "wrist_position": torch.zeros((1, 3)),
        "wrist_quaternion_wxyz": identity(),
        "wrist_position_ref": torch.zeros((1, 3)),
        "wrist_quaternion_ref_wxyz": identity(),
        "reference_index": torch.tensor([40]),
        "final_reference_index": 40,
    }
    result = stage16_termination(**kwargs)
    assert bool(result["terminated"].item())
    assert not bool(result["success"].item())
    kwargs["object_position"] = torch.zeros((1, 3))
    result = stage16_termination(**kwargs)
    assert not bool(result["terminated"].item())
    assert bool(result["success"].item())


def test_reference_bank_preserves_two_clips_if_local_artifacts_exist() -> None:
    root = Path(".local/stage16_reference_tracking_ppo/world_wrist_references")
    paths = {
        "hocap_170105": root / "hocap_170105.world_wrist.stage16.npz",
        "hocap_170650": root / "hocap_170650.world_wrist.stage16.npz",
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("local immutable Stage 16 reference artifacts are unavailable")
    bank = WorldWristReferenceBank(paths, device="cpu")
    assert bank.frame_count == 41
    assert bank.q_finger_ref.shape == (2, 41, 20)
    assert bank.tracked_link_positions_world_ref.shape == (2, 41, 16, 3)


def test_fully_kinematic_targets_are_derived_without_rewriting_local_references() -> None:
    root = Path(".local/stage16_reference_tracking_ppo/world_wrist_references")
    paths = {
        "hocap_170105": root / "hocap_170105.world_wrist.stage16.npz",
        "hocap_170650": root / "hocap_170650.world_wrist.stage16.npz",
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("local immutable Stage 16 reference artifacts are unavailable")
    result = derive_fully_kinematic_link_targets(paths, repo_root=Path("."))
    assert set(result) == set(paths)
    assert result["hocap_170105"].positions_world.shape == (41, 16, 3)
    assert result["hocap_170105"].manifest["stored_link_field_preserved"] is True
    assert result["hocap_170105"].manifest["stored_link_interpolation_residual_max_m"] > 0.0


def test_base_import_does_not_start_or_import_isaac() -> None:
    assert "isaaclab.app" not in sys.modules


def test_substep_wrist_targets_preserve_keys_and_shortest_arc() -> None:
    interpolator = PhysicsSubstepWristTargetInterpolator(decimation=6, control_dt_s=0.05)
    start_position = torch.zeros((1, 3))
    end_position = torch.tensor([[0.06, -0.02, 0.01]])
    start_quaternion = identity()
    end_quaternion = quaternion_exp_wxyz(torch.tensor([[0.0, 0.0, math.pi / 2.0]]))
    start_twist = torch.zeros((1, 6))
    end_twist = torch.tensor([[1.2, -0.4, 0.2, 0.0, 0.0, math.pi / 0.1]])
    first = interpolator.sample(
        position_k=start_position,
        quaternion_k_wxyz=start_quaternion,
        twist_k_world=start_twist,
        position_k1=end_position,
        quaternion_k1_wxyz=end_quaternion,
        twist_k1_world=end_twist,
        substep=0,
    )
    last = interpolator.sample(
        position_k=start_position,
        quaternion_k_wxyz=start_quaternion,
        twist_k_world=start_twist,
        position_k1=end_position,
        quaternion_k1_wxyz=end_quaternion,
        twist_k1_world=end_twist,
        substep=6,
    )
    assert torch.allclose(first.position_world, start_position)
    assert torch.allclose(last.position_world, end_position)
    assert torch.allclose(
        quaternion_geodesic(first.quaternion_wxyz, start_quaternion), torch.zeros(1)
    )
    assert torch.allclose(quaternion_geodesic(last.quaternion_wxyz, end_quaternion), torch.zeros(1))
    assert torch.isfinite(first.acceleration_world).all()
    assert torch.isfinite(last.acceleration_world).all()


def test_composite_inertia_includes_parallel_axis_terms_for_all_links() -> None:
    masses = torch.tensor([[2.0, 1.0]])
    inertias = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    quaternions = identity(2).reshape(1, 2, 4)
    centers = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = ArticulatedHandCompositeInertiaEstimator.estimate(
        masses_kg=masses,
        inertia_link_kgm2=inertias,
        link_quaternion_world_wxyz=quaternions,
        center_of_mass_world=centers,
        root_origin_world=torch.zeros((1, 3)),
    )
    assert torch.allclose(result.mass_kg, torch.tensor([3.0]))
    assert torch.allclose(result.center_of_mass_world, torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0]]))
    assert torch.allclose(
        result.inertia_world_kgm2, torch.diag(torch.tensor([3.0, 4.0, 5.0]))[None]
    )
    assert torch.all(result.eigenvalues_kgm2 > 0.0)
    assert torch.allclose(result.spatial_inertia_world[:, :3, :3], 3.0 * torch.eye(3)[None])


def test_effective_dynamics_identifier_recovers_cross_axis_response() -> None:
    wrench = torch.eye(6)
    response = torch.eye(6) * 0.02
    response[0, 3] = 0.005
    delta_twist = wrench @ response.transpose(0, 1)
    estimate = WristEffectiveDynamicsIdentifier.estimate(
        applied_wrench=wrench, delta_twist=delta_twist, dt_s=0.01
    )
    assert torch.allclose(estimate.response_matrix_s_per_kg, response)
    assert estimate.inverse_spatial_inertia[0, 3] == pytest.approx(0.5)
    assert torch.allclose(estimate.residual_rms, torch.zeros(6), atol=1.0e-7)


def test_batched_wrench_map_central_difference_recovers_cross_axis_response() -> None:
    response = torch.eye(6)
    response[2, 4] = 0.25
    amplitudes = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    positive = (response * amplitudes).transpose(0, 1)
    negative = -positive
    result = BatchedEffectiveWrenchMapIdentifier.central_difference(
        positive_acceleration=positive,
        negative_acceleration=negative,
        amplitudes=amplitudes,
    )
    diagnostics = BatchedEffectiveWrenchMapIdentifier.diagnostics(result)
    assert torch.allclose(result, response)
    assert diagnostics["condition_number"] == pytest.approx(1.2831955)
    assert diagnostics["cross_axis_coupling_ratio"][2] == pytest.approx(0.25)


def test_damped_svd_inverse_wrench_is_bounded_and_condition_gated() -> None:
    response = torch.eye(6).reshape(1, 1, 6, 6).repeat(2, 2, 1, 1)
    response[1, 1, -1, -1] = 1.0e-5
    effective_map = EffectiveWrenchMap(
        clip_ids=("hocap_170105", "hocap_170650"),
        frame_indices=torch.tensor([0, 40]),
        response_acceleration_per_wrench_world=response,
        zero_wrench_acceleration_world=torch.zeros((2, 2, 6)),
        source_path="test",
    )
    controller = DampedSVDInverseWrenchController(
        effective_map=effective_map,
        regularization=0.1,
        profile=IdentifiedInverseWrenchProfileV1(
            force_limit_n=50.0,
            torque_limit_nm=6.0,
            condition_number_max=4000.0,
        ),
    )
    result = controller.compute(
        clip_index=torch.tensor([0, 1]),
        reference_index=torch.tensor([0, 40]),
        target_position_world=torch.full((2, 3), 10.0),
        target_quaternion_wxyz=identity(2),
        target_twist_world=torch.zeros((2, 6)),
        target_acceleration_world=torch.zeros((2, 6)),
        current_position_world=torch.zeros((2, 3)),
        current_quaternion_wxyz=identity(2),
        current_linear_velocity_world=torch.zeros((2, 3)),
        current_angular_velocity_world=torch.zeros((2, 3)),
    )
    assert torch.linalg.vector_norm(result["force_world"][0]).item() <= 50.0
    assert torch.linalg.vector_norm(result["torque_world"][0]).item() <= 6.0
    assert bool(result["map_condition_gate_pass"][0])
    assert not bool(result["map_condition_gate_pass"][1])
    assert torch.equal(result["force_world"][1], torch.zeros(3))
    assert torch.equal(result["torque_world"][1], torch.zeros(3))


def test_finite_virtual_six_dof_wrist_is_bounded_and_uses_quaternion_error() -> None:
    actuator = FiniteVirtual6DWristActuator.from_profile_identifier("nominal")
    actuator.reset(position_world=torch.zeros((1, 3)), quaternion_wxyz=identity())
    result = actuator.compute(
        target_position_world=torch.tensor([[10.0, -10.0, 10.0]]),
        target_quaternion_wxyz=quaternion_exp_wxyz(torch.tensor([[0.0, 0.0, math.pi]])),
        target_twist_world=torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 10.0]]),
        current_position_world=torch.zeros((1, 3)),
        current_quaternion_wxyz=identity(),
        current_linear_velocity_world=torch.zeros((1, 3)),
        current_angular_velocity_world=torch.zeros((1, 3)),
        dt_s=1.0 / 120.0,
    )
    profile = actuator.profile
    assert torch.isfinite(result["force_world"]).all()
    assert torch.isfinite(result["torque_world"]).all()
    assert torch.linalg.vector_norm(result["force_world"], dim=-1).item() <= profile.force_limit_n
    assert (
        torch.linalg.vector_norm(result["torque_world"], dim=-1).item() <= profile.torque_limit_nm
    )
    assert not bool(result["position_deflection_limited"][0])
    assert not bool(result["rotation_deflection_limited"][0])
    assert result["virtual_joint_order"] == VIRTUAL_6D_JOINT_ORDER
    assert result["rotation_coordinate"] == "shortest_rotation_log_rad_bounded_below_pi"
    assert torch.linalg.vector_norm(result["virtual_position_world"], dim=-1).item() <= (
        profile.translation_velocity_limit_mps / 120.0 + 1.0e-8
    )


def test_computed_wrench_is_finite_and_respects_shared_authority_limits() -> None:
    controller = IsaacComputedWrenchWristControllerV2(
        IsaacComputedWrenchWristProfileV2(force_limit_n=25.0, torque_limit_nm=1.5)
    )
    result = controller.compute(
        mass_kg=torch.tensor([2.0]),
        inertia_world_kgm2=torch.eye(3)[None],
        target_position_world=torch.tensor([[1.0, -1.0, 2.0]]),
        target_quaternion_wxyz=quaternion_exp_wxyz(torch.tensor([[0.0, 0.0, math.pi]])),
        target_twist_world=torch.zeros((1, 6)),
        target_acceleration_world=torch.zeros((1, 6)),
        current_position_world=torch.zeros((1, 3)),
        current_quaternion_wxyz=identity(),
        current_linear_velocity_world=torch.zeros((1, 3)),
        current_angular_velocity_world=torch.zeros((1, 3)),
    )
    assert torch.isfinite(result["force_world"]).all()
    assert torch.isfinite(result["torque_world"]).all()
    assert torch.linalg.vector_norm(result["force_world"], dim=-1).item() <= 25.0
    assert torch.linalg.vector_norm(result["torque_world"], dim=-1).item() <= 1.5


def test_effective_dynamics_wrench_is_finite_and_uses_one_shared_body_matrix() -> None:
    profile = IsaacEffectiveDynamicsWristProfileV3(force_limit_n=50.0, torque_limit_nm=4.0)
    controller = IsaacEffectiveDynamicsWristControllerV3(profile)
    result = controller.compute(
        target_position_world=torch.tensor([[0.5, -0.5, 1.0]]),
        target_quaternion_wxyz=quaternion_exp_wxyz(torch.tensor([[0.0, 0.0, math.pi / 2.0]])),
        target_twist_world=torch.zeros((1, 6)),
        target_acceleration_world=torch.zeros((1, 6)),
        current_position_world=torch.zeros((1, 3)),
        current_quaternion_wxyz=identity(),
        current_linear_velocity_world=torch.zeros((1, 3)),
        current_angular_velocity_world=torch.zeros((1, 3)),
    )
    assert torch.isfinite(result["force_world"]).all()
    assert torch.isfinite(result["torque_world"]).all()
    assert torch.linalg.vector_norm(result["force_world"], dim=-1).item() <= 50.0
    assert torch.linalg.vector_norm(result["torque_world"], dim=-1).item() <= 4.0
    assert result["effective_spatial_inertia_world"].shape == (1, 6, 6)


def test_stage16c3_d6_recipe_has_six_finite_axes_and_quaternion_orientation() -> None:
    from toporetarget.rl.environments.isaaclab_backend.d6_wrist_asset import (
        D6_WRIST_AXES,
        D6_WRIST_PROFILES,
        d6_wrist_recipe,
    )

    recipe = d6_wrist_recipe()
    assert D6_WRIST_AXES == ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
    assert recipe["implementation"] == "finite_d6_wrist_actuator_v1"
    assert recipe["anchor"] == {"kind": "static", "path": "WristAnchor"}
    assert recipe["joint"]["type"] == "PhysicsJoint"
    assert recipe["joint"]["constraint_model"] == "D6_via_six_LimitAPI_and_DriveAPI_axes"
    assert recipe["joint"]["axes"] == list(D6_WRIST_AXES)
    assert recipe["joint"]["target_orientation_representation"] == "quaternion_then_rotation_log"
    assert recipe["joint"]["target_orientation_residual_representation"] == "not_euler"
    assert [profile.identifier for profile in D6_WRIST_PROFILES] == [
        "conservative",
        "nominal",
        "high_authority_bounded",
    ]
    assert all(profile.translation_effort_limit_n > 0.0 for profile in D6_WRIST_PROFILES)
    assert all(profile.rotation_effort_limit_nm > 0.0 for profile in D6_WRIST_PROFILES)


def test_recovery_state_machine_blocks_c4_and_c5_after_wrist_architecture_failure() -> None:
    state = Stage16C3R2C5RecoveryStateMachine()
    state.transition(RecoveryStage.FREE_ROOT_FINAL_ATTEMPT, reason="Path A map condition gate")
    state.transition(RecoveryStage.WRIST_ARCHITECTURE_SWITCH, reason="Path A exhausted")
    state.record_wrist_architecture_switch()
    state.transition(RecoveryStage.D6_WRAPPER_IMPORT, reason="D6 tensor discovery")
    state.transition(RecoveryStage.WRIST_QUALIFICATION, reason="virtual fallback")
    for _ in range(3):
        state.record_wrist_profile_run()
    state.block_c3(reason="C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED")
    assert state.as_dict()["stage"] == RecoveryStage.CLOSEOUT.value
    assert state.as_dict()["budgets"]["free_root_qualification_runs"] == 0
    with pytest.raises(RuntimeError, match="C3R2_C5_RECOVERY_CLOSED"):
        state.transition(RecoveryStage.C4_BENCHMARK, reason="must not run")


def test_recovery_state_machine_enforces_all_declared_recovery_budgets() -> None:
    state = Stage16C3R2C5RecoveryStateMachine()
    for _ in range(3):
        state.record_failure_class_repair("contact_api")
    with pytest.raises(RuntimeError, match="FAILURE_CLASS_REPAIR_BUDGET_EXHAUSTED"):
        state.record_failure_class_repair("contact_api")
    for _ in range(5):
        state.record_phase_rerun("contact_readout")
    with pytest.raises(RuntimeError, match="PHASE_RERUN_BUDGET_EXHAUSTED"):
        state.record_phase_rerun("contact_readout")
    state.record_free_root_controller_implementation()
    with pytest.raises(RuntimeError, match="FREE_ROOT_CONTROLLER_IMPLEMENTATION_BUDGET_EXHAUSTED"):
        state.record_free_root_controller_implementation()
    for _ in range(2):
        state.record_free_root_run()
    with pytest.raises(RuntimeError, match="FREE_ROOT_QUALIFICATION_BUDGET_EXHAUSTED"):
        state.record_free_root_run()
    state.record_wrist_architecture_switch()
    with pytest.raises(RuntimeError, match="WRIST_ARCHITECTURE_SWITCH_BUDGET_EXHAUSTED"):
        state.record_wrist_architecture_switch()
    for _ in range(3):
        state.record_wrist_profile_run()
    with pytest.raises(RuntimeError, match="WRIST_PROFILE_BUDGET_EXHAUSTED"):
        state.record_wrist_profile_run()
    for _ in range(3):
        state.record_contact_api_strategy()
    with pytest.raises(RuntimeError, match="CONTACT_API_STRATEGY_BUDGET_EXHAUSTED"):
        state.record_contact_api_strategy()
    state.record_cem_upgrade()
    with pytest.raises(RuntimeError, match="CEM_UPGRADE_BUDGET_EXHAUSTED"):
        state.record_cem_upgrade()
    state.record_replication_switch()
    with pytest.raises(RuntimeError, match="REPLICATION_SWITCH_BUDGET_EXHAUSTED"):
        state.record_replication_switch()
    for _ in range(36):
        state.transition(RecoveryStage.CONTACT_API_ISOLATION, reason="bounded test")
    with pytest.raises(RuntimeError, match="MAJOR_TRANSITION_BUDGET_EXHAUSTED"):
        state.transition(RecoveryStage.CONTACT_API_ISOLATION, reason="must fail closed")
    budgets = state.as_dict()["budgets"]
    assert budgets["failure_class_repairs"] == {"contact_api": 3}
    assert budgets["phase_reruns"] == {"contact_readout": 5}
    assert budgets["major_transitions"] == 36


def test_recovery_state_machine_requires_validated_upstream_gates_for_c4_and_c5() -> None:
    state = Stage16C3R2C5RecoveryStateMachine()
    with pytest.raises(RuntimeError, match="C3_VALIDATION_REQUIRED_FOR_C4"):
        state.transition(RecoveryStage.C4_BENCHMARK, reason="must fail closed")
    state.validate_c3()
    state.transition(RecoveryStage.C4_BENCHMARK, reason="C3 passed in this synthetic contract test")
    with pytest.raises(RuntimeError, match="C4_VALIDATION_REQUIRED_FOR_C5"):
        state.transition(RecoveryStage.C5_STATE_REPLICATION, reason="must fail closed")
    state.validate_c4()
    state.transition(RecoveryStage.C5_STATE_REPLICATION, reason="C4 passed in this synthetic test")
