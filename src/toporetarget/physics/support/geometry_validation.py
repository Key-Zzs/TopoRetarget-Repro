"""Geometry qualification for object/table and hand/table separation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .planar_inference import transform_mesh_trajectory
from .types import GeometryValidation, SupportPlaneConsistencyGateV1


def _series(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[2] != 3 or not np.isfinite(result).all():
        raise ValueError(f"SUPPORT_GEOMETRY_SERIES_INVALID:{name}")
    return result


def _surface_metrics(
    points_world: np.ndarray,
    *,
    normal: np.ndarray,
    plane_offset: float,
    extent: tuple[float, float] | None = None,
    table_center: np.ndarray | None = None,
) -> dict[str, object]:
    coordinates = np.einsum("tvi,i->tv", points_world, normal) - plane_offset
    minimum = np.min(coordinates, axis=1)
    penetration = np.maximum(0.0, -minimum)
    gap = np.maximum(0.0, minimum)
    metrics: dict[str, object] = {
        "frame_count": int(len(points_world)),
        "max_penetration_m": float(np.max(penetration)),
        "p95_penetration_m": float(np.percentile(penetration, 95.0)),
        "max_gap_m": float(np.max(gap)),
        "p95_gap_m": float(np.percentile(gap, 95.0)),
        "minimum_signed_distance_m": float(np.min(coordinates)),
        "per_frame_minimum_signed_distance_m": minimum.tolist(),
        "finite": bool(np.isfinite(coordinates).all()),
    }
    if extent is not None and table_center is not None:
        tangent_u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(tangent_u) < 1.0e-8:
            tangent_u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        tangent_u /= np.linalg.norm(tangent_u)
        tangent_v = np.cross(normal, tangent_u)
        relative = points_world - table_center[None, None, :]
        u = np.einsum("tvi,i->tv", relative, tangent_u)
        v = np.einsum("tvi,i->tv", relative, tangent_v)
        metrics["out_of_extent_frames"] = int(
            np.count_nonzero(
                (np.max(np.abs(u), axis=1) > extent[0] / 2.0)
                | (np.max(np.abs(v), axis=1) > extent[1] / 2.0)
            )
        )
    return metrics


def validate_object_table_geometry(
    *,
    visual_vertices_local: np.ndarray,
    collision_vertices_local: np.ndarray | None,
    object_translation_world: np.ndarray,
    object_quaternion_world_wxyz: np.ndarray,
    plane_normal: Sequence[float],
    plane_offset: float,
    table_extent: tuple[float, float],
    table_pose: Sequence[float],
    relevant_interval: tuple[int, int] | None = None,
    gate: SupportPlaneConsistencyGateV1 | None = None,
) -> dict[str, object]:
    """Measure visual and runtime collision geometry against the same plane."""

    active_gate = gate or SupportPlaneConsistencyGateV1()
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    visual_world = transform_mesh_trajectory(
        visual_vertices_local, object_translation_world, object_quaternion_world_wxyz
    )
    collision_world = transform_mesh_trajectory(
        collision_vertices_local if collision_vertices_local is not None else visual_vertices_local,
        object_translation_world,
        object_quaternion_world_wxyz,
    )
    start, stop = relevant_interval or (0, len(visual_world))
    if not 0 <= start < stop <= len(visual_world):
        raise ValueError("SUPPORT_GEOMETRY_INTERVAL_INVALID")
    visual = visual_world[start:stop]
    collision = collision_world[start:stop]
    pose = np.asarray(table_pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("SUPPORT_TABLE_POSE_INVALID")
    center = pose[:3]
    visual_metrics = _surface_metrics(
        visual,
        normal=normal,
        plane_offset=plane_offset,
        extent=table_extent,
        table_center=center,
    )
    collision_metrics = _surface_metrics(
        collision,
        normal=normal,
        plane_offset=plane_offset,
        extent=table_extent,
        table_center=center,
    )
    return {
        "status": "PASS"
        if float(visual_metrics["max_penetration_m"]) <= active_gate.max_object_table_penetration_m
        and float(collision_metrics["max_penetration_m"])
        <= active_gate.max_object_table_penetration_m
        and float(visual_metrics["max_gap_m"]) <= active_gate.max_object_table_gap_m
        else "FAIL",
        "interval": [start, stop],
        "plane_normal": normal.tolist(),
        "plane_offset": float(plane_offset),
        "visual": visual_metrics,
        "collision": collision_metrics,
        "visual_collision_delta_support_height_m": float(
            np.median(np.min(np.einsum("tvi,i->tv", visual, normal), axis=1))
            - np.median(np.min(np.einsum("tvi,i->tv", collision, normal), axis=1))
        ),
        "gate": active_gate.as_dict(),
    }


def validate_hand_table_geometry(
    *,
    hand_points_world: np.ndarray | None,
    plane_normal: Sequence[float],
    plane_offset: float,
    gate: SupportPlaneConsistencyGateV1 | None = None,
    source: str = "not_provided",
) -> dict[str, object]:
    """Validate hand geometry if available; never infer a full hand from links."""

    active_gate = gate or SupportPlaneConsistencyGateV1()
    if hand_points_world is None:
        return {
            "status": "DEFERRED",
            "source": source,
            "reason": "full_hand_geometry_not_provided",
            "gate": active_gate.as_dict(),
        }
    points = _series(hand_points_world, "hand_points_world")
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    metrics = _surface_metrics(points, normal=normal, plane_offset=plane_offset)
    return {
        "status": "PASS"
        if float(metrics["max_penetration_m"]) <= active_gate.max_hand_table_penetration_m
        else "FAIL",
        "source": source,
        "metrics": metrics,
        "gate": active_gate.as_dict(),
    }


def qualify_geometry(
    *,
    object_table: dict[str, object],
    hand_table: dict[str, object],
    visual_collision_consistent: bool,
) -> GeometryValidation:
    object_pass = object_table.get("status") == "PASS"
    hand_pass = hand_table.get("status") in {"PASS", "DEFERRED"}
    status = "PASS" if object_pass and hand_pass and visual_collision_consistent else "FAIL"
    return GeometryValidation(
        object_table=object_table,
        hand_table=hand_table,
        visual_collision_consistent=visual_collision_consistent,
        status=status,
    )


__all__ = [
    "qualify_geometry",
    "validate_hand_table_geometry",
    "validate_object_table_geometry",
]
