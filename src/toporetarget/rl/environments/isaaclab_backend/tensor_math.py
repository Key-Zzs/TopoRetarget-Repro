"""Small, batched SO(3) helpers using Isaac's documented ``wxyz`` convention."""

from __future__ import annotations

import torch

from .reference_bank import quaternion_to_matrix_wxyz


def normalize_quaternion_wxyz(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-12)


def quaternion_conjugate_wxyz(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_exp_wxyz(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(angle > 1.0e-8, torch.sin(half) / angle, 0.5 - angle.square() / 48.0)
    return normalize_quaternion_wxyz(torch.cat((torch.cos(half), rotation_vector * scale), dim=-1))


def quaternion_log_wxyz(value: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion_wxyz(value)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    vector = q[..., 1:]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, q[..., :1].clamp_min(1.0e-12))
    scale = torch.where(norm > 1.0e-8, angle / norm, 2.0 + norm.square() / 3.0)
    return vector * scale


def relative_rotation_log_local(
    current_wxyz: torch.Tensor, target_wxyz: torch.Tensor
) -> torch.Tensor:
    return quaternion_log_wxyz(
        quaternion_multiply_wxyz(quaternion_conjugate_wxyz(current_wxyz), target_wxyz)
    )


def quaternion_geodesic(current_wxyz: torch.Tensor, target_wxyz: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(
        normalize_quaternion_wxyz(current_wxyz) * normalize_quaternion_wxyz(target_wxyz), dim=-1
    )
    # Exact identity (including the equivalent ``q``/``-q`` representation)
    # must remain zero; clamping below one introduced a measurable fake error.
    return 2.0 * torch.acos(dot.abs().clamp(min=0.0, max=1.0))


def apply_local_residual(
    reference_position: torch.Tensor,
    reference_quaternion_wxyz: torch.Tensor,
    translation_local: torch.Tensor,
    rotation_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = quaternion_to_matrix_wxyz(reference_quaternion_wxyz)
    position = reference_position + torch.matmul(rotation, translation_local.unsqueeze(-1)).squeeze(
        -1
    )
    quaternion = quaternion_multiply_wxyz(
        reference_quaternion_wxyz, quaternion_exp_wxyz(rotation_local)
    )
    return position, normalize_quaternion_wxyz(quaternion)


__all__ = [
    "apply_local_residual",
    "normalize_quaternion_wxyz",
    "quaternion_conjugate_wxyz",
    "quaternion_exp_wxyz",
    "quaternion_geodesic",
    "quaternion_log_wxyz",
    "quaternion_multiply_wxyz",
    "relative_rotation_log_local",
]
