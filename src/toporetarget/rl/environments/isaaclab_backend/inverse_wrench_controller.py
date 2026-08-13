"""Bounded identified free-root inverse-wrench control for Stage 16-C.3R2.

This module intentionally has no Isaac imports.  It holds the one permitted
Path-A architecture: a finite, reference-indexed local wrench map, global
damped-SVD regularization candidates, and bounded world-frame wrench output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .reference_bank import quaternion_to_matrix_wxyz
from .tensor_math import relative_rotation_log_local
from .wrist_controller import clip_vector_norm


@dataclass(frozen=True)
class IdentifiedInverseWrenchProfileV1:
    """Predeclared global Path-A profile; no clip-specific gains are allowed."""

    identifier: str = "identified_inverse_wrench_v1"
    translation_position_gain_s2: float = 100.0
    translation_damping_ratio: float = 1.0
    rotation_position_gain_s2: float = 36.0
    rotation_damping_ratio: float = 1.0
    force_limit_n: float = 50.0
    torque_limit_nm: float = 6.0
    condition_number_max: float = 4000.0
    singular_value_relative_cutoff: float = 1.0e-3


@dataclass(frozen=True)
class EffectiveWrenchMap:
    """Read-only response map produced by a bounded PhysX probe."""

    clip_ids: tuple[str, ...]
    frame_indices: torch.Tensor
    response_acceleration_per_wrench_world: torch.Tensor
    zero_wrench_acceleration_world: torch.Tensor
    source_path: str

    @classmethod
    def from_json(cls, path: str | Path, *, device: torch.device) -> EffectiveWrenchMap:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("status") != "C3_EFFECTIVE_WRENCH_MAP_IDENTIFIED":
            raise ValueError(f"invalid Path-A wrench-map status: {payload.get('status')}")
        map_data = payload.get("control_map")
        if not isinstance(map_data, dict):
            raise ValueError("Path-A wrench map has no control_map")
        clip_ids = tuple(str(value) for value in map_data.get("clip_ids", ()))
        frames = torch.as_tensor(map_data.get("frame_indices"), dtype=torch.long, device=device)
        response = torch.as_tensor(
            map_data.get("response_acceleration_per_wrench_world"),
            dtype=torch.float32,
            device=device,
        )
        drift = torch.as_tensor(
            map_data.get("zero_wrench_acceleration_world"),
            dtype=torch.float32,
            device=device,
        )
        if clip_ids != ("hocap_170105", "hocap_170650") or frames.ndim != 1 or frames.numel() < 1:
            raise ValueError("Path-A map must cover both frozen clips and at least one frame")
        expected_map_shape = (len(clip_ids), frames.numel(), 6, 6)
        expected_drift_shape = (len(clip_ids), frames.numel(), 6)
        if response.shape != expected_map_shape or drift.shape != expected_drift_shape:
            raise ValueError(
                "Path-A map shapes must be "
                f"response={expected_map_shape}, drift={expected_drift_shape}"
            )
        if not bool(torch.isfinite(response).all()) or not bool(torch.isfinite(drift).all()):
            raise ValueError("Path-A map contains non-finite values")
        if not bool(torch.all(frames[1:] > frames[:-1])):
            raise ValueError("Path-A map frame indices must be strictly increasing")
        return cls(
            clip_ids=clip_ids,
            frame_indices=frames,
            response_acceleration_per_wrench_world=response,
            zero_wrench_acceleration_world=drift,
            source_path=str(source),
        )

    def gather(
        self, *, clip_index: torch.Tensor, reference_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clip_index.ndim != 1 or reference_index.shape != clip_index.shape:
            raise ValueError("Path-A clip and reference index tensors must have shape [num_envs]")
        if bool(torch.any(clip_index < 0)) or bool(torch.any(clip_index >= len(self.clip_ids))):
            raise ValueError("Path-A clip index is outside the frozen map")
        nearest = torch.argmin(
            torch.abs(reference_index[:, None] - self.frame_indices[None, :]), dim=1
        )
        return (
            self.response_acceleration_per_wrench_world[clip_index.long(), nearest],
            self.zero_wrench_acceleration_world[clip_index.long(), nearest],
            self.frame_indices[nearest],
        )


class BatchedEffectiveWrenchMapIdentifier:
    """Central-difference map estimation and diagnostics for batched probes."""

    @staticmethod
    def central_difference(
        *,
        positive_acceleration: torch.Tensor,
        negative_acceleration: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``B`` where ``a = B w + a0`` from signed basis samples."""

        if positive_acceleration.shape != (6, 6) or negative_acceleration.shape != (6, 6):
            raise ValueError(
                "signed accelerations must have shape [axis, spatial_component] = [6,6]"
            )
        if amplitudes.shape != (6,) or bool(torch.any(amplitudes <= 0.0)):
            raise ValueError("six positive signed-basis amplitudes are required")
        response_by_axis = (positive_acceleration - negative_acceleration) / (
            2.0 * amplitudes[:, None]
        )
        result = response_by_axis.transpose(0, 1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("identified effective wrench map is non-finite")
        return result

    @staticmethod
    def diagnostics(response: torch.Tensor) -> dict[str, torch.Tensor]:
        if response.shape[-2:] != (6, 6):
            raise ValueError("effective wrench response must end in [6,6]")
        singular_values = torch.linalg.svdvals(response)
        condition = singular_values[..., 0] / singular_values[..., -1].clamp_min(1.0e-12)
        diagonal = torch.diagonal(response, dim1=-2, dim2=-1).abs().clamp_min(1.0e-12)
        off_diagonal = response.abs().sum(dim=-1) - torch.diagonal(response.abs(), dim1=-2, dim2=-1)
        return {
            "singular_values": singular_values,
            "condition_number": condition,
            "cross_axis_coupling_ratio": off_diagonal / diagonal,
        }


class DampedSVDInverseWrenchController:
    """Recompute a bounded DLS wrench at every physics substep."""

    def __init__(
        self,
        *,
        effective_map: EffectiveWrenchMap,
        regularization: float,
        profile: IdentifiedInverseWrenchProfileV1 = IdentifiedInverseWrenchProfileV1(),
    ) -> None:
        if regularization <= 0.0:
            raise ValueError("Path-A regularization must be positive")
        if profile.force_limit_n <= 0.0 or profile.torque_limit_nm <= 0.0:
            raise ValueError("Path-A wrench limits must be finite and positive")
        if profile.condition_number_max <= 1.0:
            raise ValueError("Path-A condition gate must exceed one")
        if not 0.0 < profile.singular_value_relative_cutoff < 1.0:
            raise ValueError("Path-A singular-value cutoff must be in (0,1)")
        self.effective_map = effective_map
        self.regularization = regularization
        self.profile = profile

    def compute(
        self,
        *,
        clip_index: torch.Tensor,
        reference_index: torch.Tensor,
        target_position_world: torch.Tensor,
        target_quaternion_wxyz: torch.Tensor,
        target_twist_world: torch.Tensor,
        target_acceleration_world: torch.Tensor,
        current_position_world: torch.Tensor,
        current_quaternion_wxyz: torch.Tensor,
        current_linear_velocity_world: torch.Tensor,
        current_angular_velocity_world: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        response, drift, selected_frame = self.effective_map.gather(
            clip_index=clip_index, reference_index=reference_index
        )
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
        desired_acceleration = torch.cat(
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
        u, singular_values, vh = torch.linalg.svd(response)
        condition = singular_values[:, 0] / singular_values[:, -1].clamp_min(1.0e-12)
        cutoff = singular_values[:, :1] * self.profile.singular_value_relative_cutoff
        retained = singular_values >= cutoff
        damped_inverse = torch.where(
            retained,
            singular_values / (singular_values.square() + self.regularization**2),
            torch.zeros_like(singular_values),
        )
        rhs = desired_acceleration - drift
        raw_wrench = (
            vh.transpose(-1, -2)
            @ (damped_inverse * (u.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
        ).squeeze(-1)
        condition_gate_pass = condition <= self.profile.condition_number_max
        raw_wrench = torch.where(
            condition_gate_pass[:, None], raw_wrench, torch.zeros_like(raw_wrench)
        )
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
            "desired_spatial_acceleration_world": desired_acceleration,
            "identified_zero_wrench_acceleration_world": drift,
            "map_singular_values": singular_values,
            "map_condition_number": condition,
            "map_condition_gate_pass": condition_gate_pass,
            "map_retained_singular_values": retained.sum(dim=-1),
            "map_selected_reference_frame": selected_frame,
            "regularization": torch.full_like(condition, self.regularization),
        }


__all__ = [
    "BatchedEffectiveWrenchMapIdentifier",
    "DampedSVDInverseWrenchController",
    "EffectiveWrenchMap",
    "IdentifiedInverseWrenchProfileV1",
]
