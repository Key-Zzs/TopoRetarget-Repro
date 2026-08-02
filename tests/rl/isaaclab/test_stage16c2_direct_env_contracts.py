"""CPU-only contracts for the optional Stage 16-C.2 Isaac backend."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from toporetarget.rl.environments.isaaclab_backend.action_adapter import Stage16ActionAdapter
from toporetarget.rl.environments.isaaclab_backend.reference_bank import WorldWristReferenceBank
from toporetarget.rl.environments.isaaclab_backend.reward_terms import world_wrist_reward_terms
from toporetarget.rl.environments.isaaclab_backend.scene_frame import (
    global_to_scene,
    scene_to_global,
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
        substep=5,
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
