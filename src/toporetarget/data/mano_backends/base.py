"""MANO backend protocol with no torch/smplx import at package import time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ManoBackendError(RuntimeError):
    """Raised for missing optional MANO dependencies or model resources."""


@dataclass
class ManoRenderResult:
    vertices_scene: np.ndarray
    faces: np.ndarray
    wrist_pose_scene: np.ndarray
    joints_scene: np.ndarray | None = None
    keypoint_layout: str | None = None
    model_profile: str = "unknown"


class ManoBackend(Protocol):
    def render(
        self,
        *,
        params: dict[str, np.ndarray],
        v_template: np.ndarray,
        side: str,
        frame_count: int,
    ) -> ManoRenderResult:
        """Reconstruct one selected hand track without resampling."""


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion for GRAB's radians axis-angle values."""

    value = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(value, axis=1, keepdims=True)
    axis = value / np.where(theta > 1e-12, theta, 1.0)
    x, y, z = axis.T
    zeros = np.zeros_like(x)
    skew = np.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), axis=-1).reshape(-1, 3, 3)
    identity = np.eye(3)[None, ...]
    sin_theta = np.sin(theta)[:, None]
    cos_theta = np.cos(theta)[:, None]
    result = (
        cos_theta * identity
        + (1.0 - cos_theta) * np.einsum("bi,bj->bij", axis, axis)
        + sin_theta * skew
    )
    zero_mask = theta[:, 0] <= 1e-12
    result[zero_mask] = identity
    return result


__all__ = ["ManoBackend", "ManoBackendError", "ManoRenderResult", "axis_angle_to_matrix"]
