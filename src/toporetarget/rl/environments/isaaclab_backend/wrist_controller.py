"""Batched finite-wrench wrist control and articulated-inertia utilities.

This module intentionally has no Isaac Lab import.  The DirectRLEnv adapter
supplies live CUDA tensors while the interpolation, inertia, effective-response
and finite-wrench contracts remain CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference_bank import quaternion_to_matrix_wxyz
from .tensor_math import (
    quaternion_slerp_shortest_wxyz,
    relative_rotation_log_local,
)


@dataclass(frozen=True)
class IsaacWristImpedanceProfileV1:
    translation_stiffness_npm: float = 250.0
    translation_damping_ratio: float = 1.0
    rotation_stiffness_nmprad: float = 2.0
    rotation_damping_ratio: float = 0.5
    force_limit_n: float = 25.0
    torque_limit_nm: float = 1.5
    feedforward_twist_gain: float = 1.0


@dataclass(frozen=True)
class IsaacComputedWrenchWristProfileV2:
    """Shared finite computed-wrench profile expressed in SI units."""

    translation_position_gain_s2: float = 225.0
    translation_damping_ratio: float = 1.0
    rotation_position_gain_s2: float = 100.0
    rotation_damping_ratio: float = 1.0
    force_limit_n: float = 100.0
    torque_limit_nm: float = 6.0


@dataclass(frozen=True)
class IsaacEffectiveDynamicsWristProfileV3:
    """Finite shared effective-dynamics profile identified under F2 finger drives.

    The matrix maps a body-frame desired root spatial acceleration
    ``[linear, angular]`` to the body-frame wrench at ``r_wrist``.  It is the
    mean of baseline-subtracted +/− six-axis PhysX probes across the two frozen
    clips, transformed into the wrist body frame before averaging.  It is a
    single shared model, not a clip selection or a per-episode lookup.
    """

    translation_position_gain_s2: float = 100.0
    translation_damping_ratio: float = 1.0
    rotation_position_gain_s2: float = 36.0
    rotation_damping_ratio: float = 1.0
    force_limit_n: float = 50.0
    torque_limit_nm: float = 4.0
    effective_spatial_inertia_body: tuple[tuple[float, ...], ...] = (
        (9.4982626660, 0.3744082609, -0.9678138316, 0.0321487276, -1.1389509824, 0.1747912952),
        (-0.1466771389, 1.5700400694, 0.2885948304, 0.1131413393, 0.0363228500, 0.0055860196),
        (-1.3110787704, 0.3865998100, 5.0137627571, 0.0010454760, 0.1837709657, -0.0197304592),
        (-0.0323279132, 0.0891488073, -0.0006840359, 0.0098987564, 0.0040780384, -0.0015275030),
        (-1.1230356297, -0.0296889646, 0.1249947697, -0.0051830574, 0.1689674053, -0.0252322092),
        (0.1798068640, 0.0545032588, -0.0344558717, 0.0042687673, -0.0279033179, 0.0098710218),
    )


@dataclass(frozen=True)
class WristTargetSample:
    """One exact physics-substep reference target in world coordinates."""

    position_world: torch.Tensor
    quaternion_wxyz: torch.Tensor
    twist_world: torch.Tensor
    acceleration_world: torch.Tensor
    alpha: torch.Tensor


@dataclass(frozen=True)
class ArticulatedHandCompositeInertia:
    """Composite mass properties about the wrist/root origin in world axes."""

    mass_kg: torch.Tensor
    center_of_mass_world: torch.Tensor
    inertia_world_kgm2: torch.Tensor
    parallel_axis_world_kgm2: torch.Tensor
    spatial_inertia_world: torch.Tensor
    eigenvalues_kgm2: torch.Tensor
    condition: torch.Tensor


@dataclass(frozen=True)
class WristEffectiveDynamicsEstimate:
    """Least-squares local response to short applied 6-D wrench probes."""

    response_matrix_s_per_kg: torch.Tensor
    inverse_spatial_inertia: torch.Tensor
    condition: torch.Tensor
    residual_rms: torch.Tensor


def clip_vector_norm(value: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    saturated = norm.squeeze(-1) > limit
    return value * torch.clamp(limit / norm.clamp_min(1.0e-12), max=1.0), saturated


class PhysicsSubstepWristTargetInterpolator:
    """Preserve 20 Hz keys while supplying a continuous 120 Hz wrist target.

    Physics-step calls use the six start boundaries ``0/6`` through ``5/6``;
    boundary ``6/6`` is the post-step endpoint and exactly key ``k+1``.  This
    preserves the 20 Hz timing without commanding the endpoint one physics step
    early.  Translation uses cubic Hermite interpolation and orientation uses
    shortest-arc SLERP; no future key beyond ``k+1`` is read.
    """

    def __init__(self, *, decimation: int, control_dt_s: float) -> None:
        if decimation < 2:
            raise ValueError("substep interpolation requires decimation >= 2")
        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        self.decimation = decimation
        self.control_dt_s = control_dt_s

    def alpha(
        self, substep: int, *, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if not 0 <= substep <= self.decimation:
            raise ValueError(f"boundary {substep} outside [0, {self.decimation}]")
        return torch.full((batch,), substep / float(self.decimation), device=device, dtype=dtype)

    def sample(
        self,
        *,
        position_k: torch.Tensor,
        quaternion_k_wxyz: torch.Tensor,
        twist_k_world: torch.Tensor,
        position_k1: torch.Tensor,
        quaternion_k1_wxyz: torch.Tensor,
        twist_k1_world: torch.Tensor,
        substep: int,
    ) -> WristTargetSample:
        if position_k.shape[-1] != 3 or twist_k_world.shape[-1] != 6:
            raise ValueError("wrist target tensors must end in position=3 and twist=6")
        alpha = self.alpha(
            substep,
            batch=position_k.shape[0],
            device=position_k.device,
            dtype=position_k.dtype,
        )
        s = alpha[:, None]
        dt = self.control_dt_s
        p0, p1 = position_k, position_k1
        v0, v1 = twist_k_world[:, :3], twist_k1_world[:, :3]
        h00 = 2.0 * s.pow(3) - 3.0 * s.pow(2) + 1.0
        h10 = s.pow(3) - 2.0 * s.pow(2) + s
        h01 = -2.0 * s.pow(3) + 3.0 * s.pow(2)
        h11 = s.pow(3) - s.pow(2)
        position = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1
        velocity = (
            (6.0 * s.pow(2) - 6.0 * s) / dt * p0
            + (3.0 * s.pow(2) - 4.0 * s + 1.0) * v0
            + (-6.0 * s.pow(2) + 6.0 * s) / dt * p1
            + (3.0 * s.pow(2) - 2.0 * s) * v1
        )
        acceleration = (
            (12.0 * s - 6.0) / dt**2 * p0
            + (6.0 * s - 4.0) / dt * v0
            + (-12.0 * s + 6.0) / dt**2 * p1
            + (6.0 * s - 2.0) / dt * v1
        )
        angular_velocity = (1.0 - s) * twist_k_world[:, 3:] + s * twist_k1_world[:, 3:]
        angular_acceleration = (twist_k1_world[:, 3:] - twist_k_world[:, 3:]) / dt
        return WristTargetSample(
            position_world=position,
            quaternion_wxyz=quaternion_slerp_shortest_wxyz(
                quaternion_k_wxyz, quaternion_k1_wxyz, alpha
            ),
            twist_world=torch.cat((velocity, angular_velocity), dim=-1),
            acceleration_world=torch.cat((acceleration, angular_acceleration), dim=-1),
            alpha=alpha,
        )


class ArticulatedHandCompositeInertiaEstimator:
    """Compose every hand-link inertia about the live wrist/root origin."""

    @staticmethod
    def estimate(
        *,
        masses_kg: torch.Tensor,
        inertia_link_kgm2: torch.Tensor,
        link_quaternion_world_wxyz: torch.Tensor,
        center_of_mass_world: torch.Tensor,
        root_origin_world: torch.Tensor,
    ) -> ArticulatedHandCompositeInertia:
        if masses_kg.ndim != 2 or inertia_link_kgm2.shape[-2:] != (3, 3):
            raise ValueError("expected masses [N,B] and inertia [N,B,3,3]")
        if bool(torch.any(masses_kg <= 0.0)):
            raise ValueError("all articulated links require positive mass")
        rotations = quaternion_to_matrix_wxyz(link_quaternion_world_wxyz)
        inertia_world_links = rotations @ inertia_link_kgm2 @ rotations.transpose(-1, -2)
        mass = masses_kg.sum(dim=1)
        combined_com = (masses_kg[..., None] * center_of_mass_world).sum(dim=1) / mass[:, None]
        offset = center_of_mass_world - root_origin_world[:, None, :]
        squared_distance = torch.sum(offset.square(), dim=-1)
        eye = torch.eye(3, dtype=offset.dtype, device=offset.device).expand(
            *offset.shape[:-1], 3, 3
        )
        parallel_axis_links = masses_kg[..., None, None] * (
            squared_distance[..., None, None] * eye - offset[..., :, None] * offset[..., None, :]
        )
        parallel_axis = parallel_axis_links.sum(dim=1)
        inertia_world = inertia_world_links.sum(dim=1) + parallel_axis
        eigenvalues = torch.linalg.eigvalsh(inertia_world)
        condition = eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1.0e-12)
        spatial = torch.zeros(
            (masses_kg.shape[0], 6, 6), dtype=masses_kg.dtype, device=masses_kg.device
        )
        spatial[:, :3, :3] = mass[:, None, None] * torch.eye(
            3, dtype=masses_kg.dtype, device=masses_kg.device
        )
        spatial[:, 3:, 3:] = inertia_world
        return ArticulatedHandCompositeInertia(
            mass_kg=mass,
            center_of_mass_world=combined_com,
            inertia_world_kgm2=inertia_world,
            parallel_axis_world_kgm2=parallel_axis,
            spatial_inertia_world=spatial,
            eigenvalues_kgm2=eigenvalues,
            condition=condition,
        )


class WristEffectiveDynamicsIdentifier:
    """Estimate local effective inverse spatial inertia from basis probes."""

    @staticmethod
    def estimate(
        *, applied_wrench: torch.Tensor, delta_twist: torch.Tensor, dt_s: float
    ) -> WristEffectiveDynamicsEstimate:
        if applied_wrench.ndim != 2 or applied_wrench.shape[-1] != 6:
            raise ValueError("applied_wrench must have shape [samples, 6]")
        if delta_twist.shape != applied_wrench.shape or dt_s <= 0.0:
            raise ValueError("delta_twist must match wrench and dt_s must be positive")
        # delta_twist = wrench @ response.T.  ``lstsq`` permits overdetermined
        # repeated probes without assuming a perfectly diagonal response.
        response = torch.linalg.lstsq(applied_wrench, delta_twist).solution.transpose(0, 1)
        predicted = applied_wrench @ response.transpose(0, 1)
        residual_rms = torch.sqrt(torch.mean((predicted - delta_twist).square(), dim=0))
        inverse = response / dt_s
        singular_values = torch.linalg.svdvals(inverse)
        condition = singular_values.max() / singular_values.min().clamp_min(1.0e-12)
        return WristEffectiveDynamicsEstimate(
            response_matrix_s_per_kg=response,
            inverse_spatial_inertia=inverse,
            condition=condition,
            residual_rms=residual_rms,
        )


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


class IsaacComputedWrenchWristControllerV2:
    """Finite SE(3) computed-wrench controller with explicit world-frame inputs."""

    def __init__(
        self, profile: IsaacComputedWrenchWristProfileV2 = IsaacComputedWrenchWristProfileV2()
    ) -> None:
        if profile.force_limit_n <= 0.0 or profile.torque_limit_nm <= 0.0:
            raise ValueError("finite force and torque limits must be positive")
        self.profile = profile

    def compute(
        self,
        *,
        mass_kg: torch.Tensor,
        inertia_world_kgm2: torch.Tensor,
        target_position_world: torch.Tensor,
        target_quaternion_wxyz: torch.Tensor,
        target_twist_world: torch.Tensor,
        target_acceleration_world: torch.Tensor,
        current_position_world: torch.Tensor,
        current_quaternion_wxyz: torch.Tensor,
        current_linear_velocity_world: torch.Tensor,
        current_angular_velocity_world: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if inertia_world_kgm2.shape[-2:] != (3, 3) or target_acceleration_world.shape[-1] != 6:
            raise ValueError(
                "computed-wrench inputs require inertia [N,3,3] and acceleration [N,6]"
            )
        position_error = target_position_world - current_position_world
        rotation_error_local = relative_rotation_log_local(
            current_quaternion_wxyz, target_quaternion_wxyz
        )
        current_rotation = quaternion_to_matrix_wxyz(current_quaternion_wxyz)
        rotation_error_world = (current_rotation @ rotation_error_local.unsqueeze(-1)).squeeze(-1)
        linear_velocity_error = target_twist_world[:, :3] - current_linear_velocity_world
        angular_velocity_error = target_twist_world[:, 3:] - current_angular_velocity_world
        linear_kd = (
            2.0
            * self.profile.translation_damping_ratio
            * (self.profile.translation_position_gain_s2**0.5)
        )
        angular_kd = (
            2.0
            * self.profile.rotation_damping_ratio
            * (self.profile.rotation_position_gain_s2**0.5)
        )
        commanded_linear_acceleration = (
            target_acceleration_world[:, :3]
            + self.profile.translation_position_gain_s2 * position_error
            + linear_kd * linear_velocity_error
        )
        commanded_angular_acceleration = (
            target_acceleration_world[:, 3:]
            + self.profile.rotation_position_gain_s2 * rotation_error_world
            + angular_kd * angular_velocity_error
        )
        raw_force = mass_kg[:, None] * commanded_linear_acceleration
        angular_momentum = (
            inertia_world_kgm2 @ current_angular_velocity_world.unsqueeze(-1)
        ).squeeze(-1)
        gyroscopic_torque = torch.linalg.cross(current_angular_velocity_world, angular_momentum)
        raw_torque = (inertia_world_kgm2 @ commanded_angular_acceleration.unsqueeze(-1)).squeeze(
            -1
        ) + gyroscopic_torque
        force, force_saturated = clip_vector_norm(raw_force, self.profile.force_limit_n)
        torque, torque_saturated = clip_vector_norm(raw_torque, self.profile.torque_limit_nm)
        return {
            "force_world": force,
            "torque_world": torque,
            "force_saturated": force_saturated,
            "torque_saturated": torque_saturated,
            "position_error_world": position_error,
            "rotation_error_local": rotation_error_local,
            "rotation_error_world": rotation_error_world,
            "commanded_linear_acceleration_world": commanded_linear_acceleration,
            "commanded_angular_acceleration_world": commanded_angular_acceleration,
            "gyroscopic_torque_world": gyroscopic_torque,
        }


class IsaacEffectiveDynamicsWristControllerV3:
    """Computed wrench using the measured, coupled F2 effective dynamics."""

    def __init__(
        self, profile: IsaacEffectiveDynamicsWristProfileV3 = IsaacEffectiveDynamicsWristProfileV3()
    ) -> None:
        if profile.force_limit_n <= 0.0 or profile.torque_limit_nm <= 0.0:
            raise ValueError("finite force and torque limits must be positive")
        matrix = torch.tensor(profile.effective_spatial_inertia_body, dtype=torch.float32)
        if matrix.shape != (6, 6) or not bool(torch.isfinite(matrix).all()):
            raise ValueError("effective spatial inertia must be a finite 6x6 matrix")
        if float(torch.linalg.svdvals(matrix).min().item()) <= 1.0e-6:
            raise ValueError("effective spatial inertia must be nonsingular")
        self.profile = profile
        self._effective_spatial_inertia_body = matrix

    def compute(
        self,
        *,
        target_position_world: torch.Tensor,
        target_quaternion_wxyz: torch.Tensor,
        target_twist_world: torch.Tensor,
        target_acceleration_world: torch.Tensor,
        current_position_world: torch.Tensor,
        current_quaternion_wxyz: torch.Tensor,
        current_linear_velocity_world: torch.Tensor,
        current_angular_velocity_world: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        position_error = target_position_world - current_position_world
        rotation_error_local = relative_rotation_log_local(
            current_quaternion_wxyz, target_quaternion_wxyz
        )
        rotation = quaternion_to_matrix_wxyz(current_quaternion_wxyz)
        rotation_error_world = (rotation @ rotation_error_local.unsqueeze(-1)).squeeze(-1)
        linear_velocity_error = target_twist_world[:, :3] - current_linear_velocity_world
        angular_velocity_error = target_twist_world[:, 3:] - current_angular_velocity_world
        linear_kd = (
            2.0
            * self.profile.translation_damping_ratio
            * (self.profile.translation_position_gain_s2**0.5)
        )
        angular_kd = (
            2.0
            * self.profile.rotation_damping_ratio
            * (self.profile.rotation_position_gain_s2**0.5)
        )
        desired_acceleration_world = torch.cat(
            (
                target_acceleration_world[:, :3]
                + self.profile.translation_position_gain_s2 * position_error
                + linear_kd * linear_velocity_error,
                target_acceleration_world[:, 3:]
                + self.profile.rotation_position_gain_s2 * rotation_error_world
                + angular_kd * angular_velocity_error,
            ),
            dim=-1,
        )
        body_to_world = torch.zeros(
            (rotation.shape[0], 6, 6), dtype=rotation.dtype, device=rotation.device
        )
        body_to_world[:, :3, :3] = rotation
        body_to_world[:, 3:, 3:] = rotation
        effective_body = self._effective_spatial_inertia_body.to(
            dtype=rotation.dtype, device=rotation.device
        )
        effective_world = body_to_world @ effective_body @ body_to_world.transpose(-1, -2)
        raw_wrench = (effective_world @ desired_acceleration_world.unsqueeze(-1)).squeeze(-1)
        force, force_saturated = clip_vector_norm(raw_wrench[:, :3], self.profile.force_limit_n)
        torque, torque_saturated = clip_vector_norm(raw_wrench[:, 3:], self.profile.torque_limit_nm)
        return {
            "force_world": force,
            "torque_world": torque,
            "force_saturated": force_saturated,
            "torque_saturated": torque_saturated,
            "position_error_world": position_error,
            "rotation_error_local": rotation_error_local,
            "rotation_error_world": rotation_error_world,
            "desired_spatial_acceleration_world": desired_acceleration_world,
            "effective_spatial_inertia_world": effective_world,
        }


__all__ = [
    "IsaacCartesianWristImpedanceController",
    "IsaacComputedWrenchWristControllerV2",
    "IsaacComputedWrenchWristProfileV2",
    "IsaacEffectiveDynamicsWristControllerV3",
    "IsaacEffectiveDynamicsWristProfileV3",
    "IsaacWristImpedanceProfileV1",
    "ArticulatedHandCompositeInertia",
    "ArticulatedHandCompositeInertiaEstimator",
    "PhysicsSubstepWristTargetInterpolator",
    "WristEffectiveDynamicsEstimate",
    "WristEffectiveDynamicsIdentifier",
    "WristTargetSample",
    "clip_vector_norm",
]
