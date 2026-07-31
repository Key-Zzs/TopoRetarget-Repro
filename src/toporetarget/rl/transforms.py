"""Explicit world/base conversion helpers for reference export."""

from __future__ import annotations

import numpy as np


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape[-2:] != (4, 4):
        raise ValueError("transform must end in [4,4]")
    result = np.broadcast_to(np.eye(4), value.shape).copy()
    result[..., :3, :3] = np.swapaxes(value[..., :3, :3], -1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], value[..., :3, 3])
    return result


def relative_pose(world_base: np.ndarray, world_item: np.ndarray) -> np.ndarray:
    """T_base_item = inverse(T_world_base) @ T_world_item."""

    return inverse_transform(world_base) @ np.asarray(world_item, dtype=np.float64)


def points_in_base(world_base: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    inverse = inverse_transform(world_base)
    points = np.asarray(world_points, dtype=np.float64)
    return np.einsum("...ij,...pj->...pi", inverse[..., :3, :3], points) + inverse[..., None, :3, 3]


__all__ = ["inverse_transform", "points_in_base", "relative_pose"]
