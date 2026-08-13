"""Full-articulation dynamics utilities for the explicit 3P+3R wrist.

The only simulator-specific assumption is the verified PhysX tensor call
``get_generalized_mass_matrices``.  Bias is inferred from the preceding
zero-wrist-effort substep when the runtime does not expose a bias tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def generalized_mass_matrix(robot: Any) -> torch.Tensor:
    matrix = robot.root_physx_view.get_generalized_mass_matrices()
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise RuntimeError(f"MASS_MATRIX_INVALID: shape={tuple(matrix.shape)}")
    if matrix.shape[-1] != robot.num_joints:
        raise RuntimeError(
            f"DYNAMICS_API_ORDER_MISMATCH: matrix_dof={matrix.shape[-1]} joints={robot.num_joints}"
        )
    if not bool(torch.isfinite(matrix).all()):
        raise RuntimeError("MASS_MATRIX_INVALID: non-finite")
    return matrix


def generalized_bias_compensation(robot: Any) -> torch.Tensor:
    """Return the live PhysX h(q, qdot) compensation tensor.

    Isaac Sim 5.1 exposes compensation forces directly on the articulation
    tensor view.  Both terms are forces required to counteract the named
    dynamics, so their sum has the sign used by ``tau = M qdd + h``.
    """

    view = robot.root_physx_view
    coriolis = view.get_coriolis_and_centrifugal_compensation_forces()
    gravity = view.get_gravity_compensation_forces()
    expected = (robot.num_instances, robot.num_joints)
    if not isinstance(coriolis, torch.Tensor) or not isinstance(gravity, torch.Tensor):
        raise RuntimeError("BIAS_FORCE_INVALID: PhysX did not return torch tensors")
    if tuple(coriolis.shape) != expected or tuple(gravity.shape) != expected:
        raise RuntimeError(
            "BIAS_FORCE_INVALID: "
            f"coriolis={tuple(coriolis.shape)} gravity={tuple(gravity.shape)} "
            f"expected={expected}"
        )
    result = coriolis + gravity
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("BIAS_FORCE_INVALID: non-finite live compensation")
    return result


def inferred_generalized_bias(
    *, mass_matrix: torch.Tensor, applied_effort: torch.Tensor, joint_acceleration: torch.Tensor
) -> torch.Tensor:
    """Estimate h from the previous fully observed PhysX substep: tau - M qdd."""

    if applied_effort.shape != joint_acceleration.shape:
        raise ValueError("effort and acceleration must share [env, dof] shape")
    result = applied_effort - torch.bmm(mass_matrix, joint_acceleration.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("BIAS_FORCE_INVALID")
    return result


@dataclass(frozen=True)
class FullArticulationComputedTorqueProfileV1:
    identifier: str
    kp: tuple[float, float, float, float, float, float]
    zeta: float
    effort_limit: tuple[float, float, float, float, float, float]

    @property
    def kd(self) -> tuple[float, float, float, float, float, float]:
        return tuple(2.0 * self.zeta * value**0.5 for value in self.kp)


@dataclass(frozen=True)
class ComputedTorqueResult:
    effort_command: torch.Tensor
    effort_applied: torch.Tensor
    feedforward: torch.Tensor
    coupling: torch.Tensor
    bias: torch.Tensor
    feedback: torch.Tensor
    saturation: torch.Tensor


class FullArticulationComputedTorqueWristControllerV1:
    """Batched six-joint controller retaining the complete wrist-finger block."""

    def __init__(
        self, profile: FullArticulationComputedTorqueProfileV1, *, device: torch.device | str
    ):
        self.profile = profile
        self.kp = torch.tensor(profile.kp, dtype=torch.float32, device=device)
        self.kd = torch.tensor(profile.kd, dtype=torch.float32, device=device)
        self.effort_limit = torch.tensor(profile.effort_limit, dtype=torch.float32, device=device)

    def compute(
        self,
        *,
        mass_matrix: torch.Tensor,
        generalized_bias: torch.Tensor,
        wrist_joint_ids: list[int],
        finger_joint_ids: list[int],
        q_wrist: torch.Tensor,
        qd_wrist: torch.Tensor,
        q_wrist_ref: torch.Tensor,
        qd_wrist_ref: torch.Tensor,
        qdd_wrist_ref: torch.Tensor,
        qdd_finger_ref: torch.Tensor,
    ) -> ComputedTorqueResult:
        if len(wrist_joint_ids) != 6 or len(finger_joint_ids) != 20:
            raise ValueError("computed torque requires six wrist and twenty finger joint ids")
        wrist_ids = torch.tensor(wrist_joint_ids, device=mass_matrix.device)
        finger_ids = torch.tensor(finger_joint_ids, device=mass_matrix.device)
        mww = mass_matrix.index_select(1, wrist_ids).index_select(2, wrist_ids)
        mwf = mass_matrix.index_select(1, wrist_ids).index_select(2, finger_ids)
        e_q = q_wrist_ref - q_wrist
        e_qd = qd_wrist_ref - qd_wrist
        wrist_ff = torch.bmm(mww, qdd_wrist_ref.unsqueeze(-1)).squeeze(-1)
        coupling = torch.bmm(mwf, qdd_finger_ref.unsqueeze(-1)).squeeze(-1)
        bias = generalized_bias.index_select(1, wrist_ids)
        feedback_acceleration = self.kp * e_q + self.kd * e_qd
        feedback = torch.bmm(mww, feedback_acceleration.unsqueeze(-1)).squeeze(-1)
        raw = wrist_ff + coupling + bias + feedback
        applied = torch.clamp(raw, min=-self.effort_limit, max=self.effort_limit)
        return ComputedTorqueResult(
            effort_command=raw,
            effort_applied=applied,
            feedforward=wrist_ff,
            coupling=coupling,
            bias=bias,
            feedback=feedback,
            saturation=raw.abs() > self.effort_limit,
        )


def mass_matrix_diagnostics(
    mass_matrix: torch.Tensor, *, wrist_joint_ids: list[int], finger_joint_ids: list[int]
) -> dict[str, Any]:
    symmetry = (mass_matrix - mass_matrix.mT).abs().amax()
    eigenvalues = torch.linalg.eigvalsh(0.5 * (mass_matrix + mass_matrix.mT))
    condition = torch.linalg.cond(mass_matrix)
    return {
        "shape": list(mass_matrix.shape),
        "device": str(mass_matrix.device),
        "dtype": str(mass_matrix.dtype),
        "finite": bool(torch.isfinite(mass_matrix).all()),
        "symmetric_max_abs": float(symmetry.detach().cpu()),
        "eigenvalue_min": float(eigenvalues.amin().detach().cpu()),
        "eigenvalue_max": float(eigenvalues.amax().detach().cpu()),
        "condition_number_max": float(condition.amax().detach().cpu()),
        "wrist_joint_ids": wrist_joint_ids,
        "finger_joint_ids": finger_joint_ids,
        "blocks": {"M_ww": [6, 6], "M_wf": [6, 20], "M_fw": [20, 6], "M_ff": [20, 20]},
    }


FULL_ARTICULATION_COMPUTED_TORQUE_PROFILES = (
    FullArticulationComputedTorqueProfileV1(
        identifier="CT-low",
        kp=(400.0, 400.0, 400.0, 144.0, 144.0, 144.0),
        zeta=0.7,
        effort_limit=(500.0, 500.0, 500.0, 500.0, 500.0, 500.0),
    ),
    FullArticulationComputedTorqueProfileV1(
        identifier="CT-nominal",
        kp=(1000.0, 1000.0, 1000.0, 400.0, 400.0, 400.0),
        zeta=1.0,
        effort_limit=(500.0, 500.0, 500.0, 500.0, 500.0, 500.0),
    ),
)


def computed_torque_profile(identifier: str) -> FullArticulationComputedTorqueProfileV1:
    for profile in FULL_ARTICULATION_COMPUTED_TORQUE_PROFILES:
        if profile.identifier == identifier:
            return profile
    valid = ", ".join(profile.identifier for profile in FULL_ARTICULATION_COMPUTED_TORQUE_PROFILES)
    raise ValueError(f"unknown computed-torque profile {identifier!r}; expected {valid}")


__all__ = [
    "ComputedTorqueResult",
    "FULL_ARTICULATION_COMPUTED_TORQUE_PROFILES",
    "FullArticulationComputedTorqueProfileV1",
    "FullArticulationComputedTorqueWristControllerV1",
    "computed_torque_profile",
    "generalized_bias_compensation",
    "generalized_mass_matrix",
    "inferred_generalized_bias",
    "mass_matrix_diagnostics",
]
