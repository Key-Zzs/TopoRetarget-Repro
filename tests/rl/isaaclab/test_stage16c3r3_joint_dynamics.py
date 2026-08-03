"""CPU-only contracts for Stage 16-C.3R3 joint-space dynamics code."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
    FullArticulationComputedTorqueProfileV1,
    FullArticulationComputedTorqueWristControllerV1,
    inferred_generalized_bias,
    mass_matrix_diagnostics,
)
from toporetarget.rl.environments.isaaclab_backend.explicit_wrist_reference import (
    ExplicitWristJointReferenceV2,
)
from toporetarget.rl.environments.isaaclab_backend.tensor_math import quaternion_exp_wxyz
from toporetarget.rl.environments.isaaclab_backend.tvlqr_wrist import (
    BoundedMPCWristControllerV1,
    BoundedMPCWristProfileV1,
    BoundedTVLQRWristControllerV1,
    BoundedTVLQRWristProfileV1,
)


def _bank() -> SimpleNamespace:
    quaternion = quaternion_exp_wxyz(
        torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, math.radians(20.0), 0.0],
                    [0.0, math.radians(40.0), 0.0],
                ]
            ]
        )
    )
    return SimpleNamespace(
        clip_ids=("test",),
        manifest=SimpleNamespace(control_hz=20.0),
        wrist_pose_translation_world_ref=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.1, -0.1, 0.05], [0.2, -0.2, 0.10]]]
        ),
        wrist_pose_quaternion_world_ref_wxyz=quaternion,
        q_finger_ref=torch.zeros((1, 3, 20)),
    )


def test_joint_reference_preserves_keys_and_has_analytic_substep_derivatives() -> None:
    reference = ExplicitWristJointReferenceV2.from_reference_bank(_bank())
    start = reference.sample(torch.tensor([0]), torch.tensor([0]), substep=0, decimation=6)
    end = reference.sample(torch.tensor([0]), torch.tensor([0]), substep=5, decimation=6)
    assert torch.allclose(start.q_wrist, reference.q_wrist_ref[:, 0])
    assert torch.allclose(end.q_wrist, reference.q_wrist_ref[:, 1])
    assert torch.isfinite(start.qd_wrist).all()
    assert torch.isfinite(end.qdd_finger).all()
    assert reference.validation()["keyframes_preserved"] is True


def test_full_articulation_controller_retains_wrist_finger_coupling() -> None:
    mass = torch.eye(26).reshape(1, 26, 26)
    mass[:, 0, 6] = 0.25
    mass[:, 6, 0] = 0.25
    profile = FullArticulationComputedTorqueProfileV1(
        identifier="test", kp=(1.0,) * 6, zeta=1.0, effort_limit=(10.0,) * 6
    )
    result = FullArticulationComputedTorqueWristControllerV1(profile, device="cpu").compute(
        mass_matrix=mass,
        generalized_bias=torch.zeros((1, 26)),
        wrist_joint_ids=list(range(6)),
        finger_joint_ids=list(range(6, 26)),
        q_wrist=torch.zeros((1, 6)),
        qd_wrist=torch.zeros((1, 6)),
        q_wrist_ref=torch.zeros((1, 6)),
        qd_wrist_ref=torch.zeros((1, 6)),
        qdd_wrist_ref=torch.zeros((1, 6)),
        qdd_finger_ref=torch.tensor([[2.0] + [0.0] * 19]),
    )
    assert result.coupling[0, 0].item() == 0.5
    assert result.effort_applied[0, 0].item() == 0.5


def test_bias_estimate_and_mass_diagnostics_are_finite_and_complete() -> None:
    mass = torch.eye(26).reshape(1, 26, 26)
    acceleration = torch.full((1, 26), 2.0)
    effort = torch.full((1, 26), 3.0)
    bias = inferred_generalized_bias(
        mass_matrix=mass, applied_effort=effort, joint_acceleration=acceleration
    )
    assert torch.allclose(bias, torch.ones((1, 26)))
    diagnostics = mass_matrix_diagnostics(
        mass, wrist_joint_ids=list(range(6)), finger_joint_ids=list(range(6, 26))
    )
    assert diagnostics["blocks"]["M_wf"] == [6, 20]
    assert diagnostics["symmetric_max_abs"] == 0.0


def test_bounded_tvlqr_uses_live_mass_and_strict_effort_box() -> None:
    controller = BoundedTVLQRWristControllerV1(
        BoundedTVLQRWristProfileV1(effort_limit=0.1), device="cpu"
    )
    result = controller.compute(
        mass_wrist=torch.eye(6).reshape(1, 6, 6),
        feedforward=torch.zeros((1, 6)),
        q_wrist=torch.ones((1, 6)),
        qd_wrist=torch.ones((1, 6)),
        q_wrist_ref=torch.zeros((1, 6)),
        qd_wrist_ref=torch.zeros((1, 6)),
        dt_s=1.0 / 120.0,
    )
    assert result["a"].shape == (1, 12, 12)
    assert result["b"].shape == (1, 12, 6)
    assert bool(result["saturation"].all())
    assert torch.all(result["applied"].abs() <= 0.1)


def test_bounded_mpc_respects_the_same_effort_box() -> None:
    controller = BoundedMPCWristControllerV1(
        BoundedMPCWristProfileV1(effort_limit=0.1, projected_gradient_iterations=1), device="cpu"
    )
    result = controller.compute(
        dynamics_a=torch.eye(12).reshape(1, 12, 12),
        dynamics_b=torch.cat((torch.zeros((6, 6)), torch.eye(6))).reshape(1, 12, 6),
        feedforward=torch.zeros((1, 6)),
        q_wrist=torch.ones((1, 6)),
        qd_wrist=torch.ones((1, 6)),
        q_wrist_ref=torch.zeros((1, 6)),
        qd_wrist_ref=torch.zeros((1, 6)),
        model_source="test",
    )
    assert isinstance(result["applied"], torch.Tensor)
    assert torch.all(result["applied"].abs() <= 0.1)
    assert bool(result["saturation"].any())
