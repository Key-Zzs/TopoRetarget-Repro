"""Base-pose seed alignment and numerical observability audits."""

from __future__ import annotations

from typing import Any

import numpy as np

from toporetarget.geometry.se3 import invert_transform, pose_rotation_error, pose_translation_error

from .bones import BoneDirectionProfile
from .frames import BoneDirectionFrameProfile
from .objectives import BoneDirectionResidual


def base_seed_from_hand_frames(
    source_hand_frame_scene: np.ndarray, robot_hand_frame_base: np.ndarray
) -> np.ndarray:
    """Return ``T^S_B = T^S_Hs (T^B_Hr)^-1`` without entering Eq. (1)."""

    return np.matmul(source_hand_frame_scene, invert_transform(robot_hand_frame_base))


def apply_base_pose_to_points(points_base: np.ndarray, base_pose_scene: np.ndarray) -> np.ndarray:
    rotation = base_pose_scene[..., :3, :3]
    translation = base_pose_scene[..., :3, 3]
    return np.einsum("...ij,...nj->...ni", rotation, points_base) + translation[..., None, :]


def alignment_errors(
    source_hand_frame_scene: np.ndarray,
    robot_hand_frame_base: np.ndarray,
    base_pose_scene: np.ndarray,
) -> dict[str, Any]:
    aligned = np.matmul(base_pose_scene, robot_hand_frame_base)
    translation = pose_translation_error(aligned, source_hand_frame_scene)
    rotation = pose_rotation_error(aligned, source_hand_frame_scene)
    return {
        "max_translation_m": float(np.max(translation)),
        "mean_translation_m": float(np.mean(translation)),
        "max_rotation_rad": float(np.max(rotation)),
        "mean_rotation_rad": float(np.mean(rotation)),
        "per_frame_translation_m": translation.tolist(),
        "per_frame_rotation_rad": rotation.tolist(),
    }


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    axis = vector / angle
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _perturb_base(base: np.ndarray, coordinate: int, delta: float) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    if coordinate < 3:
        result[:3, 3][coordinate] += delta
    else:
        result[:3, :3] = result[:3, :3] @ _rodrigues(np.eye(3)[coordinate - 3] * delta)
    return result


def observability_report(
    source_feature: Any,
    robot_model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    qpos: np.ndarray,
    *,
    side: str,
    finite_difference_epsilon: float = 1e-6,
) -> dict[str, Any]:
    import torch

    residual_model = BoneDirectionResidual(
        source_feature, frame_profile, bone_profile, robot_model, side
    )
    q = torch.as_tensor(qpos, dtype=torch.float64).detach().clone().requires_grad_(True)
    jacobian_q = (
        torch.autograd.functional.jacobian(
            lambda item: residual_model.residual_tensor(item).reshape(-1), q, create_graph=False
        )
        .detach()
        .cpu()
        .numpy()
    )
    singular = np.linalg.svd(jacobian_q, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian_q))
    base_identity = np.eye(4)
    base_jacobian = np.empty((jacobian_q.shape[0], 6), dtype=np.float64)
    for coordinate in range(6):
        plus = _perturb_base(base_identity, coordinate, finite_difference_epsilon)
        minus = _perturb_base(base_identity, coordinate, -finite_difference_epsilon)
        plus_value = (
            residual_model.residual_tensor(q.detach(), base_pose=plus)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        minus_value = (
            residual_model.residual_tensor(q.detach(), base_pose=minus)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        base_jacobian[:, coordinate] = (plus_value - minus_value) / (
            2.0 * finite_difference_epsilon
        )
    base_translation = base_jacobian[:, :3]
    base_rotation = base_jacobian[:, 3:]
    return {
        "frame_profile_id": frame_profile.profile_id,
        "qpos": np.asarray(qpos).tolist(),
        "qpos_jacobian_shape": list(jacobian_q.shape),
        "qpos_column_norms": np.linalg.norm(jacobian_q, axis=0).tolist(),
        "qpos_rank": rank,
        "qpos_null_space_dimension": int(jacobian_q.shape[1] - rank),
        "qpos_singular_values": singular.tolist(),
        "base_translation_jacobian_norm": float(np.linalg.norm(base_translation)),
        "base_rotation_jacobian_norm": float(np.linalg.norm(base_rotation)),
        "base_translation_column_norms": np.linalg.norm(base_translation, axis=0).tolist(),
        "base_rotation_column_norms": np.linalg.norm(base_rotation, axis=0).tolist(),
        "base_jacobian_shape": list(base_jacobian.shape),
        "finite_difference_epsilon": finite_difference_epsilon,
        "expected_local_base_translation_zero": frame_profile.strategy
        == "canonical_keypoint_wrist",
        "expected_local_base_rotation_zero": frame_profile.strategy == "canonical_keypoint_wrist",
    }


def finite_difference_jacobian_check(
    residual_model: Any, qpos: np.ndarray, *, epsilon: float = 1e-6
) -> dict[str, Any]:
    """Compare the Torch Jacobian with a central finite-difference Jacobian."""

    import torch

    if epsilon <= 0.0:
        raise ValueError("finite-difference epsilon must be positive")
    q = torch.as_tensor(qpos, dtype=torch.float64).detach().clone().requires_grad_(True)

    def residual(item: Any) -> Any:
        return residual_model.residual_tensor(item).reshape(-1)

    analytic = (
        torch.autograd.functional.jacobian(residual, q, create_graph=False).detach().cpu().numpy()
    )
    numeric = np.empty_like(analytic)
    q_numpy = np.asarray(qpos, dtype=np.float64)
    for column in range(q_numpy.size):
        plus = q_numpy.copy()
        minus = q_numpy.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        plus_value = (
            residual_model.residual_tensor(torch.as_tensor(plus, dtype=torch.float64))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        minus_value = (
            residual_model.residual_tensor(torch.as_tensor(minus, dtype=torch.float64))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        numeric[:, column] = (plus_value - minus_value) / (2.0 * epsilon)
    difference = analytic - numeric
    scale = max(float(np.linalg.norm(numeric)), 1e-15)
    return {
        "epsilon": float(epsilon),
        "jacobian_shape": list(analytic.shape),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_frobenius_error": float(np.linalg.norm(difference) / scale),
    }


__all__ = [
    "alignment_errors",
    "apply_base_pose_to_points",
    "base_seed_from_hand_frames",
    "finite_difference_jacobian_check",
    "observability_report",
]
