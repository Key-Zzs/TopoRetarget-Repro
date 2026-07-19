"""Torch FK and an independent NumPy reference FK."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import JointSpec, UrdfModel


def _np_motion(joint: JointSpec, q: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if joint.joint_type in {"revolute", "continuous"}:
        axis = joint.axis
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        result[:3, :3] = np.eye(3) + np.sin(q) * skew + (1.0 - np.cos(q)) * (skew @ skew)
    elif joint.joint_type == "prismatic":
        result[:3, 3] = joint.axis * q
    return result


def forward_kinematics_numpy(model: UrdfModel, qpos: Any) -> dict[str, np.ndarray]:
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape[-1] != len(model.actuated_joints):
        raise ValueError(f"qpos must end in {len(model.actuated_joints)}, got {q.shape}")
    if q.ndim != 1:
        flat = q.reshape(-1, q.shape[-1])
        values = [forward_kinematics_numpy(model, item) for item in flat]
        return {
            name: np.stack([item[name] for item in values]).reshape(*q.shape[:-1], 4, 4)
            for name in model.link_names
        }
    by_child = model.parent_joint_by_child
    transforms: dict[str, np.ndarray] = {model.root_link: np.eye(4, dtype=np.float64)}
    for link in model.link_names:
        if link == model.root_link:
            continue
        joint = by_child[link]
        motion = 0.0 if joint.dof_index is None else float(q[joint.dof_index])
        transforms[link] = transforms[joint.parent] @ joint.origin @ _np_motion(joint, motion)
    return transforms


def joint_origins_numpy(model: UrdfModel, qpos: Any) -> dict[str, np.ndarray]:
    transforms = forward_kinematics_numpy(model, qpos)
    result: dict[str, np.ndarray] = {}
    for joint in model.joints:
        result[joint.name] = transforms[joint.parent] @ joint.origin
    return result


def _torch_pose(pose: np.ndarray, *, batch_shape: tuple[int, ...], dtype: Any, device: Any) -> Any:
    import torch

    value = torch.as_tensor(pose, dtype=dtype, device=device)
    return value.expand(*batch_shape, 4, 4) if batch_shape else value


def _torch_motion(joint: JointSpec, q: Any, *, dtype: Any, device: Any) -> Any:
    import torch

    batch_shape = tuple(q.shape)
    result = torch.eye(4, dtype=dtype, device=device).expand(*batch_shape, 4, 4).clone()
    axis = torch.as_tensor(joint.axis, dtype=dtype, device=device)
    if joint.joint_type in {"revolute", "continuous"}:
        skew = torch.zeros((3, 3), dtype=dtype, device=device)
        skew[0, 1], skew[0, 2] = -axis[2], axis[1]
        skew[1, 0], skew[1, 2] = axis[2], -axis[0]
        skew[2, 0], skew[2, 1] = -axis[1], axis[0]
        eye = torch.eye(3, dtype=dtype, device=device).expand(*batch_shape, 3, 3)
        outer = torch.einsum("i,j->ij", axis, axis).expand(*batch_shape, 3, 3)
        result[..., :3, :3] = (
            eye * torch.cos(q)[..., None, None]
            + (1.0 - torch.cos(q))[..., None, None] * outer
            + torch.sin(q)[..., None, None] * skew
        )
    elif joint.joint_type == "prismatic":
        result[..., :3, 3] = q[..., None] * axis
    return result


def forward_kinematics_torch(model: UrdfModel, qpos: Any) -> dict[str, Any]:
    import torch

    if isinstance(qpos, torch.Tensor):
        q = qpos
    else:
        q = torch.as_tensor(qpos)
    if not q.is_floating_point():
        q = q.to(dtype=torch.get_default_dtype())
    if q.ndim < 1 or q.shape[-1] != len(model.actuated_joints):
        raise ValueError(f"qpos must end in {len(model.actuated_joints)}, got {tuple(q.shape)}")
    batch_shape = tuple(q.shape[:-1])
    by_child = model.parent_joint_by_child
    transforms: dict[str, Any] = {
        model.root_link: torch.eye(4, dtype=q.dtype, device=q.device)
        .expand(*batch_shape, 4, 4)
        .clone()
    }
    for link in model.link_names:
        if link == model.root_link:
            continue
        joint = by_child[link]
        parent = transforms[joint.parent]
        origin = _torch_pose(joint.origin, batch_shape=batch_shape, dtype=q.dtype, device=q.device)
        motion_q = (
            torch.zeros(batch_shape, dtype=q.dtype, device=q.device)
            if joint.dof_index is None
            else q[..., joint.dof_index]
        )
        transforms[link] = (
            parent @ origin @ _torch_motion(joint, motion_q, dtype=q.dtype, device=q.device)
        )
    return transforms


def joint_origins_torch(model: UrdfModel, qpos: Any) -> dict[str, Any]:
    transforms = forward_kinematics_torch(model, qpos)
    return {
        joint.name: transforms[joint.parent]
        @ _torch_pose(
            joint.origin,
            batch_shape=tuple(transforms[joint.parent].shape[:-2]),
            dtype=transforms[joint.parent].dtype,
            device=transforms[joint.parent].device,
        )
        for joint in model.joints
    }


__all__ = [
    "forward_kinematics_numpy",
    "forward_kinematics_torch",
    "joint_origins_numpy",
    "joint_origins_torch",
]
