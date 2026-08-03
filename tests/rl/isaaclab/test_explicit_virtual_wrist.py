from __future__ import annotations

import math

import pytest
import torch

from toporetarget.rl.environments.isaaclab_backend.explicit_virtual_wrist import (
    EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER,
    explicit_3p3r_rotation_matrix,
    se3_target_to_explicit_3p3r,
    serial_xyz_singularity_margin_deg,
)
from toporetarget.rl.environments.isaaclab_backend.reference_bank import (
    quaternion_to_matrix_wxyz,
)
from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
    quaternion_exp_wxyz,
    quaternion_multiply_wxyz,
)
from toporetarget.rl.environments.isaaclab_backend.virtual_wrist_asset import (
    explicit_virtual_wrist_recipe,
)


def _serial_xyz_quaternion(xyz: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(xyz[..., 0])
    qx = quaternion_exp_wxyz(torch.stack((xyz[..., 0], zero, zero), dim=-1))
    qy = quaternion_exp_wxyz(torch.stack((zero, xyz[..., 1], zero), dim=-1))
    qz = quaternion_exp_wxyz(torch.stack((zero, zero, xyz[..., 2]), dim=-1))
    return quaternion_multiply_wxyz(quaternion_multiply_wxyz(qx, qy), qz)


def test_explicit_virtual_wrist_recipe_is_six_dof_and_not_a_real_arm() -> None:
    recipe = explicit_virtual_wrist_recipe("nominal")
    assert recipe["implementation"] == "finite_virtual_6d_wrist_actuator_v1"
    assert recipe["articulation_model"] == "explicit_serial_3p3r"
    assert tuple(recipe["joint_order"]) == EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER
    assert list(recipe["joint_types"].values()).count("PrismaticJoint") == 3
    assert list(recipe["joint_types"].values()).count("RevoluteJoint") == 3
    assert recipe["labels"] == [
        "ENGINEERING_WRIST_ACTUATION",
        "ABSTRACT_6DOF_WRIST_ACTUATOR",
        "NOT_A_REAL_ARM_MODEL",
        "NOT_PAPER_MINIMAL_CONTROLLER",
    ]
    assert "no_real_arm" in recipe["rollout_prohibitions"]
    assert recipe["profile"]["translation_effort_limit_n"] == 50.0
    assert recipe["profile"]["rotation_effort_limit_nm"] == 5.0


def test_high_authority_profile_remains_finite_and_explicitly_abstract() -> None:
    recipe = explicit_virtual_wrist_recipe("high_authority_bounded")
    profile = recipe["profile"]
    assert profile == {
        "identifier": "high_authority_bounded",
        "translation_stiffness_npm": 10000.0,
        "translation_damping_ns_per_m": 500.0,
        "translation_effort_limit_n": 500.0,
        "translation_velocity_limit_mps": 2.0,
        "rotation_stiffness_nm_per_rad": 3000.0,
        "rotation_damping_nm_s_per_rad": 45.0,
        "rotation_effort_limit_nm": 500.0,
        "rotation_velocity_limit_radps": 6.0,
    }
    assert all(math.isfinite(value) for value in profile.values() if isinstance(value, float))
    assert recipe["engineering_model"] == "abstract_6dof_wrist_not_real_arm"
    assert "no_real_arm" in recipe["rollout_prohibitions"]


@pytest.mark.parametrize(
    "xyz_deg",
    [
        (0.0, 0.0, 0.0),
        (75.0, 10.0, 120.0),
        (118.0, 23.0, 168.0),
        (-45.0, -30.0, 60.0),
    ],
)
def test_se3_target_conversion_reconstructs_serial_xyz_rotation(
    xyz_deg: tuple[float, float, float],
) -> None:
    xyz = torch.deg2rad(torch.tensor([xyz_deg], dtype=torch.float64))
    quaternion = _serial_xyz_quaternion(xyz)
    position = torch.tensor([[0.20, -0.30, 0.32]], dtype=torch.float64)
    joints = se3_target_to_explicit_3p3r(position, quaternion)
    assert torch.allclose(joints[..., :3], position, atol=1.0e-12, rtol=0.0)
    expected = quaternion_to_matrix_wxyz(quaternion)
    actual = explicit_3p3r_rotation_matrix(joints)
    assert torch.allclose(actual, expected, atol=1.0e-10, rtol=0.0)


def test_se3_target_conversion_unwraps_to_previous_target() -> None:
    xyz = torch.tensor([[0.0, 0.0, math.radians(-179.0)]], dtype=torch.float64)
    quaternion = _serial_xyz_quaternion(xyz)
    previous = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.radians(179.0)]], dtype=torch.float64)
    joints = se3_target_to_explicit_3p3r(
        torch.zeros((1, 3), dtype=torch.float64),
        quaternion,
        previous_joint_position=previous,
    )
    assert math.degrees(float(joints[0, 5])) == pytest.approx(181.0, abs=1.0e-9)


def test_singularity_margin_reports_distance_from_pitch_ninety() -> None:
    joints = torch.zeros((2, 6), dtype=torch.float64)
    joints[:, 4] = torch.deg2rad(torch.tensor([23.0, -80.0], dtype=torch.float64))
    margin = serial_xyz_singularity_margin_deg(joints)
    assert margin.tolist() == pytest.approx([67.0, 10.0])
