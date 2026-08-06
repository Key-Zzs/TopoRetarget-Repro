"""Pose utilities with the Stage 16 wxyz quaternion convention."""

from __future__ import annotations

import numpy as np


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite wxyz")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-15:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be finite [3,3]")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            q = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            q = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            q = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def pose_matrix(pose_xyz_wxyz: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite xyz+wxyz")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_matrix_wxyz(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def pose_from_matrix(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be finite [4,4]")
    return np.concatenate((value[:3, 3], quaternion_wxyz_from_matrix(value[:3, :3])))


def compose_poses(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return ``T_world_first @ T_first_second`` as xyz+wxyz."""

    return pose_from_matrix(pose_matrix(first) @ pose_matrix(second))


def transform_points(points: np.ndarray, pose_xyz_wxyz: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    return vertices @ quaternion_matrix_wxyz(pose[3:]).T + pose[:3]


__all__ = [
    "compose_poses",
    "pose_from_matrix",
    "pose_matrix",
    "quaternion_matrix_wxyz",
    "quaternion_wxyz_from_matrix",
    "transform_points",
]
