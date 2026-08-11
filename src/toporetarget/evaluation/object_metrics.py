"""Object tracking metrics with explicit SO(3) and unit conventions."""

from __future__ import annotations

import numpy as np


def _as_pose(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"{name} must have shape [T, 7]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized WXYZ active quaternions to rotation matrices."""

    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError("quaternion must end in four WXYZ values")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("quaternion contains zero norm")
    w, x, y, z = (values / norm).T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def quaternion_geodesic_deg(actual_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    """Return raw SO(3) geodesic error in degrees for aligned trajectories."""

    actual = np.asarray(actual_wxyz, dtype=np.float64)
    reference = np.asarray(reference_wxyz, dtype=np.float64)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 4:
        raise ValueError("actual/reference quaternion trajectories must both have shape [T, 4]")
    actual_matrix = quaternion_to_matrix_wxyz(actual)
    reference_matrix = quaternion_to_matrix_wxyz(reference)
    trace = np.trace(np.swapaxes(reference_matrix, -1, -2) @ actual_matrix, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def object_metric_series(
    actual_pose_world: np.ndarray, reference_pose_world: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute V2 object time series in degree/cm units."""

    actual = _as_pose(actual_pose_world, name="actual_pose_world")
    reference = _as_pose(reference_pose_world, name="reference_pose_world")
    if actual.shape != reference.shape:
        raise ValueError("actual/reference object poses must have the same shape")
    return {
        "e_r_deg": quaternion_geodesic_deg(actual[:, 3:], reference[:, 3:]),
        "e_t_cm": np.linalg.norm(actual[:, :3] - reference[:, :3], axis=-1) * 100.0,
    }
