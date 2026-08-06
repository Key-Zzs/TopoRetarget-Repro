"""Independent collision-proxy and visual-mesh audit for Stage 16-D traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite wxyz")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, pose_xyz_wxyz: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or pose.shape != (7,):
        raise ValueError("point transform requires [N,3] points and [7] pose")
    return vertices @ quaternion_matrix_wxyz(pose[3:]).T + pose[:3]


def inverse_transform_points(points: np.ndarray, pose_xyz_wxyz: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    return (vertices - pose[:3]) @ quaternion_matrix_wxyz(pose[3:])


def convex_proxy_point_metrics(
    object_vertices_local: np.ndarray, query_points_local: np.ndarray
) -> tuple[float, float]:
    """Return vertex penetration lower bound and supporting-plane gap in metres."""

    hull = ConvexHull(np.asarray(object_vertices_local, dtype=np.float64))
    return _convex_proxy_metrics_from_equations(hull.equations, query_points_local)


def _convex_proxy_metrics_from_equations(
    equations: np.ndarray, query_points_local: np.ndarray
) -> tuple[float, float]:
    plane = np.asarray(query_points_local, dtype=np.float64) @ equations[:, :3].T
    plane += equations[:, 3]
    maximum = plane.max(axis=1)
    inside = maximum <= 1.0e-9
    penetration = 0.0
    if bool(inside.any()):
        penetration = float((-plane[inside]).min(axis=1).max())
    outside = maximum[~inside]
    gap = 0.0 if inside.any() else float(max(0.0, outside.min(initial=np.inf)))
    return penetration, gap


def audit_trace_geometry(
    *,
    object_vertices_local: np.ndarray,
    visual_vertices_local: np.ndarray,
    hand_vertices_by_body: Mapping[str, np.ndarray],
    body_names: Sequence[str],
    object_poses: np.ndarray,
    hand_body_poses: np.ndarray,
    source_penetration_m: np.ndarray,
) -> dict[str, Any]:
    """Audit all frames and replicas without using simulator contact state."""

    objects = np.asarray(object_poses, dtype=np.float64)
    hands = np.asarray(hand_body_poses, dtype=np.float64)
    if objects.ndim != 3 or objects.shape[2] != 7:
        raise ValueError("object poses must be [steps,replicas,7]")
    if hands.shape[:2] != objects.shape[:2] or hands.shape[2:] != (len(body_names), 7):
        raise ValueError("hand poses must be [steps,replicas,bodies,7]")
    if set(body_names) != set(hand_vertices_by_body):
        raise ValueError("body mesh inventory does not match trace")
    if not np.isfinite(objects).all() or not np.isfinite(hands).all():
        raise ValueError("geometry trace must be finite")
    steps, replicas = objects.shape[:2]
    penetration = np.zeros((steps, replicas), dtype=np.float64)
    gap = np.full((steps, replicas), np.inf, dtype=np.float64)
    object_vertices = np.asarray(object_vertices_local, dtype=np.float64)
    object_hull = ConvexHull(object_vertices)
    object_hull_equations = np.unique(np.round(object_hull.equations, decimals=12), axis=0)
    object_center_local = object_vertices[object_hull.vertices].mean(axis=0)
    object_radius = float(
        np.linalg.norm(object_vertices[object_hull.vertices] - object_center_local, axis=1).max()
    )
    body_broad_phase: dict[str, tuple[np.ndarray, float]] = {}
    body_proxy_queries: dict[str, np.ndarray] = {}
    for name, vertices in hand_vertices_by_body.items():
        local_vertices = np.asarray(vertices, dtype=np.float64)
        center = local_vertices.mean(axis=0)
        body_broad_phase[name] = (
            center,
            float(np.linalg.norm(local_vertices - center, axis=1).max()),
        )
        extrema = {int(np.argmin(local_vertices[:, axis])) for axis in range(3)} | {
            int(np.argmax(local_vertices[:, axis])) for axis in range(3)
        }
        body_proxy_queries[name] = local_vertices[sorted(extrema)]
    for step in range(steps):
        for replica in range(replicas):
            object_center_world = transform_points(
                object_center_local[None], objects[step, replica]
            )[0]
            near_points: list[np.ndarray] = []
            broad_gap = np.inf
            for body_index, body_name in enumerate(body_names):
                body_center_local, body_radius = body_broad_phase[body_name]
                body_center_world = transform_points(
                    body_center_local[None], hands[step, replica, body_index]
                )[0]
                sphere_gap = float(
                    np.linalg.norm(body_center_world - object_center_world)
                    - object_radius
                    - body_radius
                )
                broad_gap = min(broad_gap, sphere_gap)
                if sphere_gap > 0.010:
                    continue
                world = transform_points(
                    body_proxy_queries[body_name], hands[step, replica, body_index]
                )
                near_points.append(inverse_transform_points(world, objects[step, replica]))
            if near_points:
                penetration[step, replica], gap[step, replica] = (
                    _convex_proxy_metrics_from_equations(
                        object_hull_equations, np.concatenate(near_points, axis=0)
                    )
                )
            else:
                gap[step, replica] = max(0.0, broad_gap)
    contact = (penetration > 0.0) | (gap <= 0.002)
    longest_runs: list[int] = []
    for replica in range(replicas):
        longest = current = 0
        for present in contact[:, replica]:
            current = current + 1 if present else 0
            longest = max(longest, current)
        longest_runs.append(longest)
    worst_flat = np.argsort(penetration.reshape(-1))[-min(12, penetration.size) :]
    visual_frames = sorted(
        {0, steps - 1, *range(0, steps, 32), *(int(row // replicas) for row in worst_flat)}
    )
    visual_tree = cKDTree(np.asarray(visual_vertices_local, dtype=np.float64))
    visual_unsigned_gap: list[dict[str, float | int]] = []
    for step in visual_frames:
        replica = int(np.argmax(penetration[step]))
        minimum = np.inf
        for body_index, body_name in enumerate(body_names):
            world = transform_points(
                hand_vertices_by_body[body_name], hands[step, replica, body_index]
            )
            local = inverse_transform_points(world, objects[step, replica])
            distance, _ = visual_tree.query(local, k=1)
            minimum = min(minimum, float(np.min(distance)))
        visual_unsigned_gap.append(
            {"step": step, "replica": replica, "vertex_sampled_unsigned_gap_m": minimum}
        )
    corrected_max = float(penetration.max(initial=0.0))
    corrected_p95 = float(np.quantile(penetration, 0.95))
    source = np.maximum(np.asarray(source_penetration_m, dtype=np.float64).reshape(-1), 0.0)
    source_max = float(source.max(initial=0.0))
    source_p95 = float(np.quantile(source, 0.95))
    catastrophic_pass = corrected_max <= 0.010
    p95_pass = corrected_p95 <= 0.003
    return {
        "schema_version": "Stage16DIndependentGeometryAuditV1",
        "scope": {"steps": steps, "replicas": replicas, "body_count": len(body_names)},
        "per_step_replica": {
            "penetration_lower_bound_m": penetration.tolist(),
            "supporting_plane_gap_m": gap.tolist(),
        },
        "collision_proxy": {
            "method": "object_convex_hull_vs_transformed_hand_proxy_axis_extrema",
            "metric_role": "penetration_lower_bound_and_supporting_plane_gap",
            "max_penetration_lower_bound_m": corrected_max,
            "p95_penetration_lower_bound_m": corrected_p95,
            "minimum_contact_gap_m": float(gap.min(initial=np.inf)),
            "contact_frame_rate": float(contact.mean()),
            "longest_contact_run_steps": longest_runs,
            "passes_catastrophic_10mm": catastrophic_pass,
            "passes_p95_3mm": p95_pass,
            "formal_pass": catastrophic_pass and p95_pass,
        },
        "original_visual_mesh": {
            "watertight_sign_available": False,
            "metric_role": "vertex_sampled_unsigned_gap_diagnostic_only",
            "sampled_frames": visual_unsigned_gap,
            "limitation": "source visual OBJ is non-watertight; signed penetration is not inferred",
        },
        "source_stage12": {
            "max_penetration_m": source_max,
            "p95_penetration_m": source_p95,
            "definition": "Stage12 persisted reference SDF max_penetration",
        },
        "source_vs_corrected": {
            "directly_comparable": False,
            "reason": (
                "Stage12 reference SDF and Stage16D convex collision-proxy lower bound differ"
            ),
            "passes_no_more_than_10pct_degradation": False,
        },
        "formal_geometry_gate": "FAIL"
        if not (catastrophic_pass and p95_pass)
        else "BLOCKED_METRIC_COMPARABILITY_AND_VISUAL_SIGN",
    }


__all__ = [
    "audit_trace_geometry",
    "convex_proxy_point_metrics",
    "inverse_transform_points",
    "quaternion_matrix_wxyz",
    "transform_points",
]
