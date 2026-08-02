"""Batched finite-wrench Cartesian impedance controller for Isaac PhysX."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference_bank import quaternion_to_matrix_wxyz
from .tensor_math import relative_rotation_log_local


@dataclass(frozen=True)
class IsaacWristImpedanceProfileV1:
    translation_stiffness_npm: float = 250.0
    translation_damping_ratio: float = 1.0
    rotation_stiffness_nmprad: float = 2.0
    rotation_damping_ratio: float = 0.5
    force_limit_n: float = 25.0
    torque_limit_nm: float = 1.5
    feedforward_twist_gain: float = 1.0


def clip_vector_norm(value: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    saturated = norm.squeeze(-1) > limit
    return value * torch.clamp(limit / norm.clamp_min(1.0e-12), max=1.0), saturated


class IsaacCartesianWristImpedanceController:
    """Compute world-frame wrenches, then let PhysX integrate the floating root."""

    def __init__(
        self,
        *,
        mass_kg: torch.Tensor,
        inertia_kgm2: torch.Tensor,
        profile: IsaacWristImpedanceProfileV1 = IsaacWristImpedanceProfileV1(),
    ) -> None:
        if bool(torch.any(mass_kg <= 0.0)) or bool(torch.any(inertia_kgm2 <= 0.0)):
            raise ValueError("actual wrist mass and inertia must be positive")
        self.profile = profile
        self.translation_damping = (
            2.0
            * torch.sqrt(profile.translation_stiffness_npm * mass_kg)
            * profile.translation_damping_ratio
        )
        self.rotation_damping = (
            2.0
            * torch.sqrt(profile.rotation_stiffness_nmprad * inertia_kgm2.mean(dim=-1))
            * profile.rotation_damping_ratio
        )

    def compute(
        self,
        *,
        target_position: torch.Tensor,
        target_quaternion_wxyz: torch.Tensor,
        target_twist_world: torch.Tensor,
        current_position: torch.Tensor,
        current_quaternion_wxyz: torch.Tensor,
        current_linear_velocity_world: torch.Tensor,
        current_angular_velocity_world: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        position_error = target_position - current_position
        rotation_error_local = relative_rotation_log_local(
            current_quaternion_wxyz, target_quaternion_wxyz
        )
        rotation_matrix = quaternion_to_matrix_wxyz(current_quaternion_wxyz)
        rotation_error_world = torch.matmul(
            rotation_matrix, rotation_error_local.unsqueeze(-1)
        ).squeeze(-1)
        linear_velocity_error = target_twist_world[:, :3] - current_linear_velocity_world
        angular_velocity_error = target_twist_world[:, 3:] - current_angular_velocity_world
        raw_force = (
            self.profile.translation_stiffness_npm * position_error
            + self.translation_damping[:, None]
            * self.profile.feedforward_twist_gain
            * linear_velocity_error
        )
        raw_torque = (
            self.profile.rotation_stiffness_nmprad * rotation_error_world
            + self.rotation_damping[:, None]
            * self.profile.feedforward_twist_gain
            * angular_velocity_error
        )
        force, force_saturated = clip_vector_norm(raw_force, self.profile.force_limit_n)
        torque, torque_saturated = clip_vector_norm(raw_torque, self.profile.torque_limit_nm)
        return {
            "force_world": force,
            "torque_world": torque,
            "force_saturated": force_saturated,
            "torque_saturated": torque_saturated,
            "position_error_world": position_error,
            "rotation_error_local": rotation_error_local,
        }


__all__ = [
    "IsaacCartesianWristImpedanceController",
    "IsaacWristImpedanceProfileV1",
    "clip_vector_norm",
]
