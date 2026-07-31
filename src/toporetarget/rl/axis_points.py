"""Object-axis feature construction used by the paper's MDP."""

from __future__ import annotations

import numpy as np

OBJECT_AXIS_PROFILE_ID = "object_axis_points_v1"


def object_axis_points_from_poses(poses: np.ndarray, *, axis_length_m: float = 0.05) -> np.ndarray:
    """Return six base-frame endpoints (+/- x, y, z) for each SE(3) pose.

    The paper states that it uses six attached axis points but does not give
    their local spatial offsets.  A fixed 5 cm length is a frozen engineering
    assumption, not an inferred paper value.
    """

    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape [T,4,4], got {values.shape}")
    if not np.isfinite(values).all() or axis_length_m <= 0.0:
        raise ValueError("poses must be finite and axis_length_m must be positive")
    local = axis_length_m * np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return np.einsum("tij,mj->tmi", values[:, :3, :3], local) + values[:, None, :3, 3]


__all__ = ["OBJECT_AXIS_PROFILE_ID", "object_axis_points_from_poses"]
