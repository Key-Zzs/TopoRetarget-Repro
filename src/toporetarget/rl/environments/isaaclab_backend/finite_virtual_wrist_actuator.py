"""Finite virtual six-DoF wrist actuator used only after the D6 tensor gate.

This is an engineering wrist, not an arm model.  It consumes a world-frame
target pose represented by translation plus quaternion, computes the shortest
rotation-vector error (never Euler residuals), and applies a bounded physical
wrench at ``r_wrist``.  It has no Isaac dependency and writes no state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch

from .d6_wrist_asset import D6_WRIST_PROFILES, FiniteD6WristProfile
from .reference_bank import quaternion_to_matrix_wxyz
from .tensor_math import (
    quaternion_exp_wxyz,
    quaternion_multiply_wxyz,
    relative_rotation_log_local,
)
from .wrist_controller import clip_vector_norm

VIRTUAL_6D_JOINT_ORDER = (
    "virtual_prismatic_x",
    "virtual_prismatic_y",
    "virtual_prismatic_z",
    "virtual_revolute_x",
    "virtual_revolute_y",
    "virtual_revolute_z",
)


class FiniteVirtualWristResult(TypedDict):
    """Tensor wrench result plus immutable actuator-contract metadata."""

    force_world: torch.Tensor
    torque_world: torch.Tensor
    force_saturated: torch.Tensor
    torque_saturated: torch.Tensor
    position_deflection_limited: torch.Tensor
    rotation_deflection_limited: torch.Tensor
    virtual_position_world: torch.Tensor
    virtual_quaternion_wxyz: torch.Tensor
    virtual_joint_order: tuple[str, str, str, str, str, str]
    rotation_coordinate: str


@dataclass(frozen=True)
class FiniteVirtual6DWristActuatorProfile:
    """One shared bounded virtual-wrist profile in SI units."""

    identifier: str
    translation_stiffness_npm: float
    translation_damping_ns_per_m: float
    force_limit_n: float
    translation_velocity_limit_mps: float
    rotation_stiffness_nm_per_rad: float
    rotation_damping_nm_s_per_rad: float
    torque_limit_nm: float
    rotation_velocity_limit_radps: float
    translation_deflection_limit_m: float = 0.30
    rotation_deflection_limit_rad: float = math.radians(100.0)

    @classmethod
    def from_d6_profile_identifier(cls, identifier: str) -> FiniteVirtual6DWristActuatorProfile:
        source: FiniteD6WristProfile | None = next(
            (profile for profile in D6_WRIST_PROFILES if profile.identifier == identifier),
            None,
        )
        if source is None:
            valid = ", ".join(profile.identifier for profile in D6_WRIST_PROFILES)
            raise ValueError(
                f"unknown finite virtual wrist profile {identifier!r}; expected {valid}"
            )
        return cls(
            identifier=source.identifier,
            translation_stiffness_npm=source.translation_stiffness_npm,
            translation_damping_ns_per_m=source.translation_damping_ns_per_m,
            force_limit_n=source.translation_effort_limit_n,
            translation_velocity_limit_mps=source.translation_velocity_limit_mps,
            rotation_stiffness_nm_per_rad=source.rotation_stiffness_nm_per_rad,
            rotation_damping_nm_s_per_rad=source.rotation_damping_nm_s_per_rad,
            torque_limit_nm=source.rotation_effort_limit_nm,
            rotation_velocity_limit_radps=source.rotation_velocity_limit_radps,
        )


class FiniteVirtual6DWristActuator:
    """Batched virtual 3P+3R wrist with bounded physical output and no writes.

    The virtual state is a world-frame prismatic triplet plus an SO(3)
    quaternion.  Its rotational coordinate is the shortest rotation-log
    vector, restricted below pi by the 100-degree joint limit; it therefore
    does not use Euler residuals or cross a representation singularity.
    """

    def __init__(self, profile: FiniteVirtual6DWristActuatorProfile) -> None:
        self.profile = profile
        self._virtual_position_world: torch.Tensor | None = None
        self._virtual_quaternion_wxyz: torch.Tensor | None = None

    @classmethod
    def from_profile_identifier(cls, identifier: str) -> FiniteVirtual6DWristActuator:
        return cls(FiniteVirtual6DWristActuatorProfile.from_d6_profile_identifier(identifier))

    def reset(
        self,
        *,
        position_world: torch.Tensor,
        quaternion_wxyz: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        num_envs: int | None = None,
    ) -> None:
        """Reset virtual joints at episode setup; never writes a PhysX state."""

        if env_ids is None:
            self._virtual_position_world = position_world.clone()
            self._virtual_quaternion_wxyz = quaternion_wxyz.clone()
            return
        if self._virtual_position_world is None or self._virtual_quaternion_wxyz is None:
            if num_envs is None:
                raise ValueError("num_envs is required for an indexed first virtual reset")
            self._virtual_position_world = torch.zeros(
                (num_envs, 3), dtype=position_world.dtype, device=position_world.device
            )
            self._virtual_quaternion_wxyz = torch.zeros(
                (num_envs, 4), dtype=quaternion_wxyz.dtype, device=quaternion_wxyz.device
            )
            self._virtual_quaternion_wxyz[:, 0] = 1.0
        self._virtual_position_world[env_ids] = position_world
        self._virtual_quaternion_wxyz[env_ids] = quaternion_wxyz

    def _advance_virtual_target(
        self,
        *,
        target_position_world: torch.Tensor,
        target_quaternion_wxyz: torch.Tensor,
        dt_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if dt_s <= 0.0:
            raise ValueError("virtual wrist dt_s must be positive")
        if (
            self._virtual_position_world is None
            or self._virtual_quaternion_wxyz is None
            or self._virtual_position_world.shape != target_position_world.shape
        ):
            self.reset(position_world=target_position_world, quaternion_wxyz=target_quaternion_wxyz)
        assert self._virtual_position_world is not None
        assert self._virtual_quaternion_wxyz is not None
        previous_position = self._virtual_position_world
        previous_quaternion = self._virtual_quaternion_wxyz
        position_delta, _ = clip_vector_norm(
            target_position_world - previous_position,
            self.profile.translation_velocity_limit_mps * dt_s,
        )
        rotation_delta_local, _ = clip_vector_norm(
            relative_rotation_log_local(previous_quaternion, target_quaternion_wxyz),
            self.profile.rotation_velocity_limit_radps * dt_s,
        )
        next_position = previous_position + position_delta
        next_quaternion = quaternion_multiply_wxyz(
            previous_quaternion, quaternion_exp_wxyz(rotation_delta_local)
        )
        rotation_delta_world = torch.matmul(
            quaternion_to_matrix_wxyz(previous_quaternion), rotation_delta_local.unsqueeze(-1)
        ).squeeze(-1)
        self._virtual_position_world = next_position
        self._virtual_quaternion_wxyz = next_quaternion
        return (
            next_position,
            next_quaternion,
            position_delta / dt_s,
            rotation_delta_world / dt_s,
        )

    def compute(
        self,
        *,
        target_position_world: torch.Tensor,
        target_quaternion_wxyz: torch.Tensor,
        target_twist_world: torch.Tensor,
        current_position_world: torch.Tensor,
        current_quaternion_wxyz: torch.Tensor,
        current_linear_velocity_world: torch.Tensor,
        current_angular_velocity_world: torch.Tensor,
        dt_s: float,
    ) -> FiniteVirtualWristResult:
        """Return finite world-frame wrench for the physical wrist link."""

        (
            virtual_position,
            virtual_quaternion,
            virtual_linear_velocity,
            virtual_angular_velocity,
        ) = self._advance_virtual_target(
            target_position_world=target_position_world,
            target_quaternion_wxyz=target_quaternion_wxyz,
            dt_s=dt_s,
        )
        position_error, position_deflection_limited = clip_vector_norm(
            virtual_position - current_position_world,
            self.profile.translation_deflection_limit_m,
        )
        rotation_error_local, rotation_deflection_limited = clip_vector_norm(
            relative_rotation_log_local(current_quaternion_wxyz, virtual_quaternion),
            self.profile.rotation_deflection_limit_rad,
        )
        rotation_error_world = torch.matmul(
            quaternion_to_matrix_wxyz(current_quaternion_wxyz), rotation_error_local.unsqueeze(-1)
        ).squeeze(-1)
        raw_force = (
            self.profile.translation_stiffness_npm * position_error
            + self.profile.translation_damping_ns_per_m
            * (virtual_linear_velocity - current_linear_velocity_world)
        )
        raw_torque = (
            self.profile.rotation_stiffness_nm_per_rad * rotation_error_world
            + self.profile.rotation_damping_nm_s_per_rad
            * (virtual_angular_velocity - current_angular_velocity_world)
        )
        force_world, force_saturated = clip_vector_norm(raw_force, self.profile.force_limit_n)
        torque_world, torque_saturated = clip_vector_norm(raw_torque, self.profile.torque_limit_nm)
        return {
            "force_world": force_world,
            "torque_world": torque_world,
            "force_saturated": force_saturated,
            "torque_saturated": torque_saturated,
            "position_deflection_limited": position_deflection_limited,
            "rotation_deflection_limited": rotation_deflection_limited,
            "virtual_position_world": virtual_position,
            "virtual_quaternion_wxyz": virtual_quaternion,
            "virtual_joint_order": VIRTUAL_6D_JOINT_ORDER,
            "rotation_coordinate": "shortest_rotation_log_rad_bounded_below_pi",
        }
