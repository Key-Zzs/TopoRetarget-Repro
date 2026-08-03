"""Immutable-key joint references for the explicit serial 3P+3R wrist.

This module deliberately stays free of Isaac imports.  It converts the frozen
SE(3) references once, then evaluates a cubic-Hermite segment analytically at
every physics substep.  The original 20 Hz keys are never filtered or retimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .explicit_virtual_wrist import (
    EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER,
    explicit_3p3r_rotation_matrix,
    se3_target_to_explicit_3p3r,
    serial_xyz_singularity_margin_deg,
)
from .reference_bank import quaternion_to_matrix_wxyz


@dataclass(frozen=True)
class ExplicitWristJointReferenceSample:
    """One batched, continuous 26-DoF desired joint state."""

    q_wrist: torch.Tensor
    qd_wrist: torch.Tensor
    qdd_wrist: torch.Tensor
    q_finger: torch.Tensor
    qd_finger: torch.Tensor
    qdd_finger: torch.Tensor


def _key_derivative(values: torch.Tensor, dt_s: float) -> torch.Tensor:
    """Second-order interior / one-sided endpoint derivatives at frozen keys."""

    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("joint keys must have shape [clip, frame>=2, dof]")
    derivative = torch.empty_like(values)
    derivative[:, 0] = (values[:, 1] - values[:, 0]) / dt_s
    derivative[:, -1] = (values[:, -1] - values[:, -2]) / dt_s
    derivative[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * dt_s)
    return derivative


def _cubic_hermite(
    q0: torch.Tensor,
    q1: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    alpha: torch.Tensor,
    dt_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic cubic-Hermite q/qd/qdd without moving the endpoints."""

    u = alpha[..., None]
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    q = h00 * q0 + h10 * (dt_s * v0) + h01 * q1 + h11 * (dt_s * v1)
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -6.0 * u2 + 6.0 * u
    dh11 = 3.0 * u2 - 2.0 * u
    qd = (dh00 * q0 + dh10 * (dt_s * v0) + dh01 * q1 + dh11 * (dt_s * v1)) / dt_s
    d2h00 = 12.0 * u - 6.0
    d2h10 = 6.0 * u - 4.0
    d2h01 = -12.0 * u + 6.0
    d2h11 = 6.0 * u - 2.0
    qdd = (d2h00 * q0 + d2h10 * (dt_s * v0) + d2h01 * q1 + d2h11 * (dt_s * v1)) / (dt_s * dt_s)
    return q, qd, qdd


@dataclass(frozen=True)
class ExplicitWristJointReferenceV2:
    """Joint-space contract derived only from the frozen WorldWrist bank."""

    clip_ids: tuple[str, ...]
    dt_s: float
    q_wrist_ref: torch.Tensor
    qd_wrist_ref: torch.Tensor
    q_finger_ref: torch.Tensor
    qd_finger_ref: torch.Tensor
    wrist_rotation_ref: torch.Tensor
    interpolation: str = "cubic_hermite_physics_boundary_analytic_qdd_v2"

    @classmethod
    def from_reference_bank(cls, bank: Any) -> ExplicitWristJointReferenceV2:
        positions = bank.wrist_pose_translation_world_ref
        quaternions = bank.wrist_pose_quaternion_world_ref_wxyz
        joints: list[torch.Tensor] = []
        previous: torch.Tensor | None = None
        for frame in range(positions.shape[1]):
            current = se3_target_to_explicit_3p3r(
                positions[:, frame], quaternions[:, frame], previous_joint_position=previous
            )
            joints.append(current)
            previous = current
        q_wrist = torch.stack(joints, dim=1)
        dt_s = 1.0 / float(bank.manifest.control_hz)
        return cls(
            clip_ids=tuple(bank.clip_ids),
            dt_s=dt_s,
            q_wrist_ref=q_wrist,
            qd_wrist_ref=_key_derivative(q_wrist, dt_s),
            q_finger_ref=bank.q_finger_ref.clone(),
            qd_finger_ref=_key_derivative(bank.q_finger_ref, dt_s),
            wrist_rotation_ref=quaternion_to_matrix_wxyz(quaternions),
        )

    @property
    def frame_count(self) -> int:
        return int(self.q_wrist_ref.shape[1])

    def sample(
        self, clip_index: torch.Tensor, key_index: torch.Tensor, *, substep: int, decimation: int
    ) -> ExplicitWristJointReferenceSample:
        if not 0 <= substep <= decimation:
            raise ValueError("physics boundary outside [0, decimation]")
        index0 = key_index.clamp(0, self.frame_count - 2)
        index1 = index0 + 1
        alpha = torch.full_like(index0, float(substep) / float(decimation), dtype=torch.float32)

        def gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return values[clip_index, index]

        qw, qdw, qddw = _cubic_hermite(
            gather(self.q_wrist_ref, index0),
            gather(self.q_wrist_ref, index1),
            gather(self.qd_wrist_ref, index0),
            gather(self.qd_wrist_ref, index1),
            alpha,
            self.dt_s,
        )
        qf, qdf, qddf = _cubic_hermite(
            gather(self.q_finger_ref, index0),
            gather(self.q_finger_ref, index1),
            gather(self.qd_finger_ref, index0),
            gather(self.qd_finger_ref, index1),
            alpha,
            self.dt_s,
        )
        return ExplicitWristJointReferenceSample(qw, qdw, qddw, qf, qdf, qddf)

    def validation(self, *, wrist_limits: torch.Tensor | None = None) -> dict[str, Any]:
        rotation = explicit_3p3r_rotation_matrix(self.q_wrist_ref)
        fk_error = torch.linalg.matrix_norm(rotation - self.wrist_rotation_ref, dim=(-2, -1)).amax()
        margin = serial_xyz_singularity_margin_deg(self.q_wrist_ref)
        in_limits = True
        if wrist_limits is not None:
            in_limits = bool(
                (
                    (self.q_wrist_ref >= wrist_limits[None, None, :, 0])
                    & (self.q_wrist_ref <= wrist_limits[None, None, :, 1])
                ).all()
            )
        return {
            "identifier": "ExplicitWristJointReferenceV2",
            "joint_order": list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER),
            "interpolation": self.interpolation,
            "frame_count": self.frame_count,
            "keyframes_preserved": True,
            "finite": bool(
                torch.isfinite(self.q_wrist_ref).all() and torch.isfinite(self.q_finger_ref).all()
            ),
            "no_unwrap_jump": bool(
                (self.q_wrist_ref[:, 1:, 3:] - self.q_wrist_ref[:, :-1, 3:]).abs().amax() < torch.pi
            ),
            "minimum_singularity_margin_deg": float(margin.amin().detach().cpu()),
            "joint_limits_pass": in_limits,
            "fk_round_trip_matrix_residual": float(fk_error.detach().cpu()),
        }


__all__ = ["ExplicitWristJointReferenceSample", "ExplicitWristJointReferenceV2"]
