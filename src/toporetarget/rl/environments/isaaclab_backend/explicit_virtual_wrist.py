"""SE(3)-to-joint conversion for the explicit virtual 3P+3R wrist.

The policy action remains a translation residual plus a rotation-vector
residual.  Only after the complete quaternion target has been formed do we
convert it to the serial X-Y-Z revolute coordinates required by PhysX.
"""

from __future__ import annotations

import math

import torch

from .tensor_math import quaternion_to_matrix_wxyz

EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER = (
    "virtual_prismatic_x",
    "virtual_prismatic_y",
    "virtual_prismatic_z",
    "virtual_revolute_x",
    "virtual_revolute_y",
    "virtual_revolute_z",
)
EXPLICIT_VIRTUAL_WRIST_TRANSLATION_JOINTS = EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[:3]
EXPLICIT_VIRTUAL_WRIST_ROTATION_JOINTS = EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[3:]


def _nearest_equivalent_angle(angle: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Choose the 2-pi-equivalent joint angle nearest the previous target."""

    delta = torch.remainder(angle - previous + math.pi, 2.0 * math.pi) - math.pi
    return previous + delta


def quaternion_to_serial_xyz_wxyz(
    quaternion_wxyz: torch.Tensor,
    *,
    previous_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a quaternion to coordinates satisfying ``R = Rx Ry Rz``.

    This is an actuator-side inverse-kinematics conversion, not the policy
    residual representation.  The caller must separately enforce a pitch
    singularity margin.
    """

    rotation = quaternion_to_matrix_wxyz(quaternion_wxyz)
    sin_y = rotation[..., 0, 2].clamp(-1.0, 1.0)
    y = torch.asin(sin_y)
    cos_y = torch.cos(y)
    if bool((cos_y.abs() <= 1.0e-6).any()):
        raise RuntimeError("C3_EXPLICIT_WRIST_XYZ_SINGULARITY")
    x = torch.atan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    z = torch.atan2(-rotation[..., 0, 1], rotation[..., 0, 0])
    xyz = torch.stack((x, y, z), dim=-1)
    if previous_xyz is not None:
        xyz = _nearest_equivalent_angle(xyz, previous_xyz)
    return xyz


def se3_target_to_explicit_3p3r(
    position_scene: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    *,
    previous_joint_position: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a complete scene-local SE(3) target to the six serial joints."""

    if position_scene.shape[-1] != 3 or quaternion_wxyz.shape[-1] != 4:
        raise ValueError("explicit wrist target must contain position[3] and quaternion[4]")
    previous_rotation = (
        None if previous_joint_position is None else previous_joint_position[..., 3:]
    )
    rotation_xyz = quaternion_to_serial_xyz_wxyz(quaternion_wxyz, previous_xyz=previous_rotation)
    return torch.cat((position_scene, rotation_xyz), dim=-1)


def serial_xyz_singularity_margin_deg(joint_position: torch.Tensor) -> torch.Tensor:
    """Return distance from the X-Y-Z pitch singularity at +/-90 degrees."""

    pitch_deg = torch.rad2deg(joint_position[..., 4].abs())
    return 90.0 - pitch_deg


def explicit_3p3r_rotation_matrix(joint_position: torch.Tensor) -> torch.Tensor:
    """Forward rotation for tests and asset-contract diagnostics."""

    x, y, z = joint_position[..., 3:].unbind(dim=-1)
    sx, cx = torch.sin(x), torch.cos(x)
    sy, cy = torch.sin(y), torch.cos(y)
    sz, cz = torch.sin(z), torch.cos(z)
    row0 = torch.stack((cy * cz, -cy * sz, sy), dim=-1)
    row1 = torch.stack((cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy), dim=-1)
    row2 = torch.stack((sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


__all__ = [
    "EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER",
    "EXPLICIT_VIRTUAL_WRIST_ROTATION_JOINTS",
    "EXPLICIT_VIRTUAL_WRIST_TRANSLATION_JOINTS",
    "explicit_3p3r_rotation_matrix",
    "quaternion_to_serial_xyz_wxyz",
    "se3_target_to_explicit_3p3r",
    "serial_xyz_singularity_margin_deg",
]
