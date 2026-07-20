"""Stage 7 Eq. (1) and Eq. (2) residuals and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bones import BoneDirectionProfile, BoneFeatures, extract_bone_features
from .frames import BoneDirectionFrameProfile


@dataclass(frozen=True)
class BoneDirectionResidual:
    """Exact Eq. (1) residual: adjacent direction difference, then source subtraction."""

    source_features: Any
    frame_profile: BoneDirectionFrameProfile
    bone_profile: BoneDirectionProfile
    robot_model: Any
    side: str

    def robot_features(self, qpos: Any, *, base_pose: Any | None = None) -> BoneFeatures:
        if base_pose is None:
            keypoints = self.robot_model.keypoints_base(qpos, layout=self.bone_profile.layout_name)
        else:
            keypoints = self.robot_model.keypoints_scene(
                qpos, base_pose, layout=self.bone_profile.layout_name
            )
        return extract_bone_features(
            keypoints, self.frame_profile, self.bone_profile, side=self.side, strict=True
        )

    def residual_tensor(self, qpos: Any, *, base_pose: Any | None = None) -> Any:
        import torch

        robot = self.robot_features(qpos, base_pose=base_pose)
        source = self.source_features
        if not isinstance(source, torch.Tensor):
            source = torch.as_tensor(
                source, dtype=robot.adjacent_features.dtype, device=robot.adjacent_features.device
            )
        return robot.adjacent_features - source

    def __call__(self, qpos: Any, *, base_pose: Any | None = None) -> Any:
        return self.residual_tensor(qpos, base_pose=base_pose)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def per_finger_losses(residual: Any, pair_fingers: tuple[str, ...]) -> dict[str, float]:
    values = _as_numpy(residual)
    losses: dict[str, float] = {}
    for index, finger in enumerate(pair_fingers):
        losses[finger] = losses.get(finger, 0.0) + float(np.sum(values[index] ** 2))
    return losses


def direction_angle_diagnostics(source_directions: Any, robot_directions: Any) -> dict[str, Any]:
    source = _as_numpy(source_directions)
    robot = _as_numpy(robot_directions)
    cosine = np.sum(source * robot, axis=-1) / np.maximum(
        np.linalg.norm(source, axis=-1) * np.linalg.norm(robot, axis=-1), 1e-15
    )
    return {
        "mean_rad": float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "max_rad": float(np.max(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "per_bone_rad": np.arccos(np.clip(cosine, -1.0, 1.0)).tolist(),
    }


def equation_1_report(
    source: BoneFeatures, robot: BoneFeatures, *, initial: Any | None = None
) -> dict[str, Any]:
    residual = robot.adjacent_features - source.adjacent_features
    values = _as_numpy(residual)
    pair_losses = np.sum(values**2, axis=-1)
    payload: dict[str, Any] = {
        "sum_loss": float(np.sum(values**2)),
        "mean_diagnostic": float(np.mean(values**2)),
        "per_pair_loss": pair_losses.tolist(),
        "per_finger_loss": per_finger_losses(residual, robot.pair_fingers),
        "per_axis_residual": values.tolist(),
        "direction_angles": direction_angle_diagnostics(
            source.unit_directions, robot.unit_directions
        ),
        "pair_names": list(robot.pair_names),
    }
    if initial is not None:
        initial_values = _as_numpy(initial)
        payload["initial_sum_loss"] = float(np.sum(initial_values**2))
        payload["final_sum_loss"] = payload["sum_loss"]
    return payload


@dataclass(frozen=True)
class BoneDirectionObjective:
    residual_model: BoneDirectionResidual
    lambda_warm: float
    lambda_smooth: float
    previous_qpos: Any | None = None

    def residual_tensor(self, qpos: Any) -> Any:
        import torch

        bone = self.residual_model.residual_tensor(qpos)
        pieces = [(torch.sqrt(qpos.new_tensor(self.lambda_warm)) * bone).reshape(-1)]
        if self.previous_qpos is not None:
            previous = self.previous_qpos
            if not isinstance(previous, torch.Tensor):
                previous = qpos.new_tensor(previous)
            else:
                previous = previous.to(dtype=qpos.dtype, device=qpos.device)
            pieces.append(torch.sqrt(qpos.new_tensor(self.lambda_smooth)) * (qpos - previous))
        return torch.cat(pieces)

    def paper_objective(self, qpos: Any) -> dict[str, Any]:
        import torch

        bone = self.residual_model.residual_tensor(qpos)
        ebone = self.lambda_warm * torch.sum(bone * bone)
        temporal = qpos.new_zeros(())
        if self.previous_qpos is not None:
            previous = self.previous_qpos
            if not isinstance(previous, torch.Tensor):
                previous = qpos.new_tensor(previous)
            temporal = self.lambda_smooth * torch.sum((qpos - previous.to(qpos)) ** 2)
        return {
            "ebone": ebone,
            "temporal": temporal,
            "total": ebone + temporal,
            "bone_residual": bone,
            "library_cost_half": 0.5 * (ebone + temporal),
        }


__all__ = [
    "BoneDirectionObjective",
    "BoneDirectionResidual",
    "direction_angle_diagnostics",
    "equation_1_report",
    "per_finger_losses",
]
