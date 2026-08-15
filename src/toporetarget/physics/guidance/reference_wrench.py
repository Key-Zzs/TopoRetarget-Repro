"""Pure, deterministic reference-restoring wrench computation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Protocol

import torch

from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
    quaternion_conjugate_wxyz,
    quaternion_log_wxyz,
    quaternion_multiply_wxyz,
)

from .contract import ObjectGuidanceContractV1


@dataclass(frozen=True)
class GuidanceWrench:
    """Wrench and auditable per-step diagnostics, all in the world frame."""

    force_world: torch.Tensor
    torque_world: torch.Tensor
    position_error_world: torch.Tensor
    orientation_error_world: torch.Tensor
    linear_velocity_error_world: torch.Tensor
    angular_velocity_error_world: torch.Tensor
    force_clipped: torch.Tensor
    torque_clipped: torch.Tensor
    guidance_active: torch.Tensor
    force_limit_n: torch.Tensor
    torque_limit_nm: torch.Tensor


class _InstantaneousWrenchComposer(Protocol):
    def set_forces_and_torques(
        self,
        *,
        forces: torch.Tensor,
        torques: torch.Tensor,
        is_global: bool,
    ) -> None: ...


class _GuidedRigidObject(Protocol):
    instantaneous_wrench_composer: _InstantaneousWrenchComposer


def _zero_small(vector: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold == 0.0:
        return vector
    return torch.where(
        torch.linalg.vector_norm(vector, dim=-1, keepdim=True) <= threshold,
        torch.zeros_like(vector),
        vector,
    )


def _clamp_vector_norm(
    vector: torch.Tensor, limit: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(vector, dim=-1)
    clipped = norm > limit
    scale = torch.minimum(torch.ones_like(norm), limit / norm.clamp_min(1.0e-12))
    return vector * scale.unsqueeze(-1), clipped


class ReferenceWrenchGuidance:
    """Compute ``F=m a`` and ``tau=I_world alpha`` without mutating state."""

    def __init__(self, contract: ObjectGuidanceContractV1) -> None:
        self.contract = contract

    def compute(
        self,
        *,
        reference_position_world: torch.Tensor,
        reference_quaternion_wxyz: torch.Tensor,
        reference_twist_world: torch.Tensor,
        object_position_world: torch.Tensor,
        object_quaternion_wxyz: torch.Tensor,
        object_twist_world: torch.Tensor,
        mass_kg: torch.Tensor,
        inertia_world_kgm2: torch.Tensor,
    ) -> GuidanceWrench:
        """Return a bounded world-frame wrench for each batch item.

        Every vector argument is ``(N, 3)`` (twists ``(N, 6)``); mass is
        ``(N,)`` or ``(N, 1)`` and inertia is ``(N, 3, 3)``.  This function is
        intentionally side-effect-free so it is testable outside IsaacLab.
        """

        batch = object_position_world.shape[0]
        expected = {
            "reference_position_world": reference_position_world,
            "object_position_world": object_position_world,
            "reference_twist_world": reference_twist_world,
            "object_twist_world": object_twist_world,
        }
        if any(value.shape[0] != batch for value in expected.values()):
            raise ValueError("OBJECT_GUIDANCE_BATCH_MISMATCH")
        if reference_position_world.shape != (batch, 3) or object_position_world.shape != (
            batch,
            3,
        ):
            raise ValueError("OBJECT_GUIDANCE_POSITION_SHAPE_INVALID")
        if reference_twist_world.shape != (batch, 6) or object_twist_world.shape != (batch, 6):
            raise ValueError("OBJECT_GUIDANCE_TWIST_SHAPE_INVALID")
        if inertia_world_kgm2.shape != (batch, 3, 3):
            raise ValueError("OBJECT_GUIDANCE_INERTIA_SHAPE_INVALID")
        mass = mass_kg.reshape(batch)
        if not torch.isfinite(mass).all() or bool((mass <= 0.0).any()):
            raise ValueError("OBJECT_GUIDANCE_MASS_INVALID")
        inputs = tuple(expected.values()) + (
            reference_quaternion_wxyz,
            object_quaternion_wxyz,
            inertia_world_kgm2,
        )
        if not all(torch.isfinite(value).all() for value in inputs):
            raise ValueError("OBJECT_GUIDANCE_NONFINITE_INPUT")

        position_error = reference_position_world - object_position_world
        orientation_error = quaternion_log_wxyz(
            quaternion_multiply_wxyz(
                reference_quaternion_wxyz,
                quaternion_conjugate_wxyz(object_quaternion_wxyz),
            )
        )
        linear_velocity_error = reference_twist_world[:, :3] - object_twist_world[:, :3]
        angular_velocity_error = reference_twist_world[:, 3:] - object_twist_world[:, 3:]
        # The default mode is observationally inert.  Keep useful zero-wrench
        # telemetry, but do not run the controller (notably its eigensolve) on
        # a path that is contractually forbidden from influencing physics.
        if self.contract.mode == "none":
            zero = torch.zeros_like(position_error)
            return GuidanceWrench(
                force_world=zero,
                torque_world=zero,
                position_error_world=position_error,
                orientation_error_world=orientation_error,
                linear_velocity_error_world=linear_velocity_error,
                angular_velocity_error_world=angular_velocity_error,
                force_clipped=torch.zeros(batch, dtype=torch.bool, device=mass.device),
                torque_clipped=torch.zeros(batch, dtype=torch.bool, device=mass.device),
                guidance_active=torch.zeros(batch, dtype=torch.bool, device=mass.device),
                force_limit_n=mass * self.contract.translation_acceleration_cap_mps2,
                # This is only a diagnostic bound in disabled mode.  No torque
                # is submitted to PhysX, so an inexpensive finite upper bound
                # is preferable to an eigensolve on every substep.
                torque_limit_nm=inertia_world_kgm2.abs().sum(dim=(-2, -1))
                * self.contract.rotation_acceleration_cap_radps2,
            )
        position_control_error = _zero_small(position_error, self.contract.position_deadband_m)
        orientation_control_error = _zero_small(
            orientation_error, self.contract.rotation_deadband_rad
        )
        linear_velocity_control_error = _zero_small(
            linear_velocity_error, self.contract.linear_velocity_deadband_mps
        )
        angular_velocity_control_error = _zero_small(
            angular_velocity_error, self.contract.angular_velocity_deadband_radps
        )
        omega_p = 2.0 * pi * self.contract.translation_natural_frequency_hz
        omega_r = 2.0 * pi * self.contract.rotation_natural_frequency_hz
        acceleration_raw = omega_p**2 * position_control_error + (
            2.0 * self.contract.translation_damping_ratio * omega_p * linear_velocity_control_error
        )
        angular_acceleration_raw = omega_r**2 * orientation_control_error + (
            2.0 * self.contract.rotation_damping_ratio * omega_r * angular_velocity_control_error
        )
        acceleration, force_clipped = _clamp_vector_norm(
            acceleration_raw,
            torch.full(
                (batch,),
                self.contract.translation_acceleration_cap_mps2,
                device=mass.device,
                dtype=mass.dtype,
            ),
        )
        angular_acceleration, torque_clipped = _clamp_vector_norm(
            angular_acceleration_raw,
            torch.full(
                (batch,),
                self.contract.rotation_acceleration_cap_radps2,
                device=mass.device,
                dtype=mass.dtype,
            ),
        )
        force_limit = mass * self.contract.translation_acceleration_cap_mps2
        principal_inertia_bound = (
            torch.linalg.eigvalsh(inertia_world_kgm2).amax(dim=-1).clamp_min(0.0)
        )
        torque_limit = principal_inertia_bound * self.contract.rotation_acceleration_cap_radps2
        force = mass.unsqueeze(-1) * acceleration
        torque = torch.bmm(inertia_world_kgm2, angular_acceleration.unsqueeze(-1)).squeeze(-1)
        # Numerical roundoff cannot be permitted to violate the public caps.
        force, force_roundoff_clipped = _clamp_vector_norm(force, force_limit)
        torque, torque_roundoff_clipped = _clamp_vector_norm(torque, torque_limit)
        force_clipped |= force_roundoff_clipped
        torque_clipped |= torque_roundoff_clipped
        if not torch.isfinite(force).all() or not torch.isfinite(torque).all():
            raise RuntimeError("OBJECT_GUIDANCE_NONFINITE_WRENCH")
        return GuidanceWrench(
            force_world=force,
            torque_world=torque,
            position_error_world=position_error,
            orientation_error_world=orientation_error,
            linear_velocity_error_world=linear_velocity_error,
            angular_velocity_error_world=angular_velocity_error,
            force_clipped=force_clipped,
            torque_clipped=torque_clipped,
            guidance_active=(torch.linalg.vector_norm(force, dim=-1) > 0.0)
            | (torch.linalg.vector_norm(torque, dim=-1) > 0.0),
            force_limit_n=force_limit,
            torque_limit_nm=torque_limit,
        )

    @staticmethod
    def apply(
        *,
        wrench: GuidanceWrench,
        active_first: torch.Tensor,
        first_object: _GuidedRigidObject,
        second_object: _GuidedRigidObject,
    ) -> None:
        """Apply the wrench through IsaacLab's instantaneous PhysX interface.

        The runtime object type is deliberately structural here: keeping this
        pure-Python module free of Isaac imports preserves unit-testability.
        Both inactive and active objects receive an explicit per-step value so
        no stale wrench can persist across a clip switch or reset.
        """

        if (
            active_first.dtype is not torch.bool
            or active_first.shape != wrench.force_world.shape[:1]
        ):
            raise ValueError("OBJECT_GUIDANCE_ACTIVE_OBJECT_SELECTOR_INVALID")
        zeros = torch.zeros_like(wrench.force_world)
        first_force = torch.where(active_first[:, None], wrench.force_world, zeros)
        second_force = torch.where(active_first[:, None], zeros, wrench.force_world)
        first_torque = torch.where(active_first[:, None], wrench.torque_world, zeros)
        second_torque = torch.where(active_first[:, None], zeros, wrench.torque_world)
        first_object.instantaneous_wrench_composer.set_forces_and_torques(
            forces=first_force.unsqueeze(1),
            torques=first_torque.unsqueeze(1),
            is_global=True,
        )
        second_object.instantaneous_wrench_composer.set_forces_and_torques(
            forces=second_force.unsqueeze(1),
            torques=second_torque.unsqueeze(1),
            is_global=True,
        )
