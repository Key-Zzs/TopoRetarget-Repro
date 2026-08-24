"""Gravity-aware planar support inference from meshes and pose trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import (
    FinitePlanarSupportProxy,
    StableIntervalResult,
    StablePreContactDetectionContractV1,
    SupportExtentContractV1,
    SupportInterval,
    SupportPatchType,
    SupportPlaneFit,
)


def _array(value: object, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"SUPPORT_ARRAY_INVALID:{name}")
    return result


def normalize_gravity(gravity: Sequence[float]) -> np.ndarray:
    vector = _array(gravity, ndim=1, name="gravity")
    if vector.shape != (3,):
        raise ValueError("SUPPORT_GRAVITY_SHAPE_INVALID")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1.0e-9:
        raise ValueError("SUPPORT_GRAVITY_MUST_BE_NONZERO")
    return vector / magnitude


def support_normal_from_gravity(gravity: Sequence[float]) -> np.ndarray:
    return -normalize_gravity(gravity)


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = _array(quaternion_wxyz, ndim=2, name="quaternion")
    if q.shape[1] != 4:
        raise ValueError("SUPPORT_QUATERNION_SHAPE_INVALID")
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm <= 1.0e-9):
        raise ValueError("SUPPORT_QUATERNION_ZERO")
    q = q / norm[:, None]
    w, x, y, z = q.T
    result = np.empty((len(q), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - z * w)
    result[:, 0, 2] = 2.0 * (x * z + y * w)
    result[:, 1, 0] = 2.0 * (x * y + z * w)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - x * w)
    result[:, 2, 0] = 2.0 * (x * z - y * w)
    result[:, 2, 1] = 2.0 * (y * z + x * w)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def rotation_step_angle(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = _array(quaternion_wxyz, ndim=2, name="quaternion")
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    return np.concatenate(([0.0], 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))))


def _stats(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "median": median,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "mad": mad,
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "range": float(np.max(values) - np.min(values)),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        if start is not None and (not enabled or index == len(mask) - 1):
            stop = index if not enabled else index + 1
            result.append((start, stop))
            start = None
    return result


def _persistent_motion(mask: np.ndarray, *, minimum_frames: int) -> np.ndarray:
    result = np.zeros_like(np.asarray(mask, dtype=bool))
    for start, stop in _runs(np.asarray(mask, dtype=bool)):
        if stop - start >= minimum_frames:
            result[start:stop] = True
    return result


def detect_stable_pre_contact_interval(
    *,
    timestamps: Sequence[float],
    object_translation_world: np.ndarray,
    object_quaternion_world_wxyz: np.ndarray,
    gravity: Sequence[float],
    contact_mask: np.ndarray | None = None,
    object_twist_world: np.ndarray | None = None,
    contract: StablePreContactDetectionContractV1 | None = None,
) -> StableIntervalResult:
    """Find stable pre-manipulation geometry; contact timing is diagnostic."""

    del gravity  # The interval is kinematic; gravity enters plane inference.
    thresholds = contract or StablePreContactDetectionContractV1()
    time = _array(timestamps, ndim=1, name="timestamps")
    translation = _array(object_translation_world, ndim=2, name="object_translation_world")
    quaternion = _array(object_quaternion_world_wxyz, ndim=2, name="object_quaternion_world_wxyz")
    if translation.shape != (len(time), 3) or quaternion.shape != (len(time), 4):
        raise ValueError("SUPPORT_TRAJECTORY_SHAPE_INVALID")
    if len(time) < thresholds.min_consecutive_frames:
        return StableIntervalResult(
            status="PLANAR_SUPPORT_INFERENCE_NOT_AUTHORIZED",
            interval=None,
            reason="trajectory_shorter_than_minimum_interval",
        )
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("SUPPORT_TIMESTAMPS_NOT_STRICTLY_INCREASING")
    translation_step = np.concatenate(([0.0], np.linalg.norm(np.diff(translation, axis=0), axis=1)))
    rotation_step = rotation_step_angle(quaternion)
    finite_linear = np.concatenate(
        ([0.0], np.linalg.norm(np.diff(translation, axis=0) / dt[:, None], axis=1))
    )
    finite_angular = rotation_step.copy()
    finite_angular[1:] /= dt
    if object_twist_world is not None:
        twist = _array(object_twist_world, ndim=2, name="object_twist_world")
        if twist.shape != (len(time), 6):
            raise ValueError("SUPPORT_TWIST_SHAPE_INVALID")
        linear_speed = np.linalg.norm(twist[:, :3], axis=1)
        angular_speed = np.linalg.norm(twist[:, 3:], axis=1)
    else:
        linear_speed = finite_linear
        angular_speed = finite_angular
    if contact_mask is None:
        contact = np.zeros(len(time), dtype=bool)
        contact_used = False
    else:
        contact = np.asarray(contact_mask, dtype=bool)
        if contact.shape != (len(time),):
            raise ValueError("SUPPORT_CONTACT_MASK_SHAPE_INVALID")
        contact_used = True
    kinematically_stable = (
        (linear_speed <= thresholds.max_linear_speed_mps)
        & (angular_speed <= thresholds.max_angular_speed_radps)
        & (translation_step <= thresholds.max_translation_step_m)
        & (rotation_step <= thresholds.max_rotation_step_rad)
    )
    eligible = kinematically_stable
    eligible_runs = [
        (start, stop)
        for start, stop in _runs(eligible)
        if stop - start >= thresholds.min_consecutive_frames
    ]
    manipulation = _persistent_motion(~kinematically_stable, minimum_frames=1)
    first_manipulation = (
        int(np.flatnonzero(manipulation)[0]) if np.any(manipulation) else len(contact)
    )
    candidates = tuple(
        SupportInterval(
            start,
            stop,
            (
                "stable_kinematic_pre_manipulation"
                if stop <= first_manipulation
                else "stable_kinematic_post_manipulation_diagnostic"
            ),
        )
        for start, stop in eligible_runs
    )
    authoritative = tuple(
        interval
        for interval in candidates
        if interval.reason == "stable_kinematic_pre_manipulation"
    )
    if not authoritative:
        return StableIntervalResult(
            status="PLANAR_SUPPORT_INFERENCE_NOT_AUTHORIZED",
            interval=None,
            candidate_intervals=candidates,
            linear_speed_mps=tuple(float(v) for v in linear_speed),
            angular_speed_radps=tuple(float(v) for v in angular_speed),
            translation_step_m=tuple(float(v) for v in translation_step),
            rotation_step_rad=tuple(float(v) for v in rotation_step),
            contact_mask_used=contact_used,
            reason=(
                "no_stable_interval_before_manipulation"
                if candidates
                else "no_stable_pre_contact_interval"
            ),
        )
    selected = authoritative[0]
    return StableIntervalResult(
        status="STABLE_PRE_CONTACT_INTERVAL_FOUND",
        interval=selected,
        candidate_intervals=candidates,
        linear_speed_mps=tuple(float(v) for v in linear_speed),
        angular_speed_radps=tuple(float(v) for v in angular_speed),
        translation_step_m=tuple(float(v) for v in translation_step),
        rotation_step_rad=tuple(float(v) for v in rotation_step),
        contact_mask_used=contact_used,
        reason=(
            "earliest stable pre-manipulation geometry selected; "
            "hand contact timing retained as separate provenance"
        ),
    )


def transform_mesh_trajectory(
    vertices_local: np.ndarray,
    translation_world: np.ndarray,
    quaternion_world_wxyz: np.ndarray,
) -> np.ndarray:
    vertices = _array(vertices_local, ndim=2, name="vertices_local")
    if vertices.shape[1] != 3:
        raise ValueError("SUPPORT_MESH_SHAPE_INVALID")
    rotations = quaternion_to_rotation_matrix(quaternion_world_wxyz)
    translation = _array(translation_world, ndim=2, name="translation_world")
    if translation.shape != (len(rotations), 3):
        raise ValueError("SUPPORT_POSE_SHAPE_INVALID")
    return np.einsum("tij,vj->tvi", rotations, vertices) + translation[:, None, :]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    tangent_u = np.cross(normal, reference)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    return tangent_u, tangent_v


def _patch_type(projected: np.ndarray) -> SupportPatchType:
    if len(projected) < 2:
        return SupportPatchType.UNSTABLE_SUPPORT_PATCH
    span = np.ptp(projected, axis=0)
    if max(span) <= 0.01:
        return SupportPatchType.POINT_SUPPORT
    if min(span) <= 0.01:
        return SupportPatchType.EDGE_SUPPORT
    return SupportPatchType.AREA_SUPPORT


def infer_planar_support(
    *,
    visual_vertices_local: np.ndarray,
    collision_vertices_local: np.ndarray | None,
    object_translation_world: np.ndarray,
    object_quaternion_world_wxyz: np.ndarray,
    gravity: Sequence[float],
    stable_interval: SupportInterval,
    extent_contract: SupportExtentContractV1 | None = None,
    detection_contract: StablePreContactDetectionContractV1 | None = None,
) -> tuple[SupportPlaneFit, FinitePlanarSupportProxy, dict[str, object]]:
    """Fit one finite support proxy using the stable interval and object meshes."""

    extent = extent_contract or SupportExtentContractV1()
    detection = detection_contract or StablePreContactDetectionContractV1()
    normal = support_normal_from_gravity(gravity)
    visual_world = transform_mesh_trajectory(
        visual_vertices_local, object_translation_world, object_quaternion_world_wxyz
    )
    collision_world = transform_mesh_trajectory(
        collision_vertices_local if collision_vertices_local is not None else visual_vertices_local,
        object_translation_world,
        object_quaternion_world_wxyz,
    )
    h_visual = np.min(np.einsum("tvi,i->tv", visual_world, normal), axis=1)
    h_collision = np.min(np.einsum("tvi,i->tv", collision_world, normal), axis=1)
    stable = slice(stable_interval.start_frame, stable_interval.end_frame_exclusive)
    visual_stats = _stats(h_visual[stable])
    collision_stats = _stats(h_collision[stable])
    if (
        visual_stats["mad"] > detection.max_height_mad_m
        or visual_stats["range"] > detection.max_height_range_m
    ):
        raise ValueError("SUPPORT_PLANE_TEMPORAL_CONSISTENCY_FAILED")
    plane_offset = float(np.median(h_visual[stable]))
    tangent_u, tangent_v = _plane_basis(normal)
    # Use stable plus the immediately following approach/contact frames for a
    # finite swept footprint, while never using post-lift frames.
    footprint_stop = min(len(visual_world), stable_interval.end_frame_exclusive + 4)
    footprint_world = visual_world[stable_interval.start_frame : footprint_stop].reshape(-1, 3)
    projected = np.column_stack((footprint_world @ tangent_u, footprint_world @ tangent_v))
    bounds = np.array(
        [
            projected[:, 0].min(),
            projected[:, 0].max(),
            projected[:, 1].min(),
            projected[:, 1].max(),
        ]
    )
    margin = extent.support_extent_margin_m
    table_extent = (
        float(bounds[1] - bounds[0] + 2.0 * margin),
        float(bounds[3] - bounds[2] + 2.0 * margin),
    )
    center_u = float((bounds[0] + bounds[1]) / 2.0)
    center_v = float((bounds[2] + bounds[3]) / 2.0)
    # ``table_pose`` is the audited top-surface pose. Runtime scene builders
    # subtract half the box thickness when deriving the rigid actor centre.
    center_world = normal * plane_offset + tangent_u * center_u + tangent_v * center_v
    patch_vertices = visual_world[stable]
    patch_coordinates = np.einsum("tvi,i->tv", patch_vertices, normal)
    patch_mask = patch_coordinates <= (
        np.min(patch_coordinates, axis=1, keepdims=True) + detection.support_patch_tolerance_m
    )
    patch_points = patch_vertices[patch_mask]
    patch_projected = np.column_stack((patch_points @ tangent_u, patch_points @ tangent_v))
    patch_area = float(np.prod(np.ptp(patch_projected, axis=0))) if len(patch_projected) else 0.0
    patch_type = _patch_type(patch_projected)
    rotation = np.column_stack((tangent_u, tangent_v, normal))
    table_pose = (*center_world.tolist(), *_rotation_matrix_to_quaternion(rotation).tolist())
    fit = SupportPlaneFit(
        plane_normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        plane_offset=plane_offset,
        h_visual=tuple(float(v) for v in h_visual),
        h_collision=tuple(float(v) for v in h_collision),
        h_visual_stats=visual_stats,
        h_collision_stats=collision_stats,
        delta_support_geometry=float(visual_stats["median"] - collision_stats["median"]),
        support_patch_type=patch_type,
        support_patch_vertex_count=int(len(patch_points)),
        support_patch_projected_area_m2=patch_area,
        patch_connected_components=None,
        stable_interval=stable_interval,
    )
    proxy = FinitePlanarSupportProxy(
        table_pose=tuple(float(v) for v in table_pose),
        table_extent=table_extent,
        table_thickness=extent.table_thickness_m,
        plane_normal=fit.plane_normal,
        plane_offset=plane_offset,
    )
    evidence = {
        "footprint_bounds_uv": bounds.tolist(),
        "tangent_u": tangent_u.tolist(),
        "tangent_v": tangent_v.tolist(),
        "footprint_frames": [stable_interval.start_frame, footprint_stop],
        "collision_geometry_source": "caller_supplied_collision_vertices"
        if collision_vertices_local is not None
        else "visual_mesh_fallback_not_runtime_qualified",
    }
    return fit, proxy, evidence


def audit_candidate_support_intervals(
    *,
    visual_vertices_local: np.ndarray,
    collision_vertices_local: np.ndarray | None,
    object_translation_world: np.ndarray,
    object_quaternion_world_wxyz: np.ndarray,
    gravity: Sequence[float],
    candidates: Sequence[SupportInterval],
    extent_contract: SupportExtentContractV1 | None = None,
    detection_contract: StablePreContactDetectionContractV1 | None = None,
) -> list[dict[str, object]]:
    """Audit every eligible interval without changing frame-zero authority.

    A later AREA-support placement is useful diagnostic evidence, but it cannot
    silently replace an initially POINT/EDGE-supported reset.
    """

    rows: list[dict[str, object]] = []
    for interval in candidates:
        try:
            fit, _proxy, _evidence = infer_planar_support(
                visual_vertices_local=visual_vertices_local,
                collision_vertices_local=collision_vertices_local,
                object_translation_world=object_translation_world,
                object_quaternion_world_wxyz=object_quaternion_world_wxyz,
                gravity=gravity,
                stable_interval=interval,
                extent_contract=extent_contract,
                detection_contract=detection_contract,
            )
        except ValueError as error:
            rows.append(
                {
                    "interval": interval.as_dict(),
                    "status": "INELIGIBLE",
                    "reason": str(error),
                }
            )
            continue
        rows.append(
            {
                "interval": interval.as_dict(),
                "status": "ELIGIBLE",
                "support_patch_type": fit.support_patch_type.value,
                "support_patch_projected_area_m2": fit.support_patch_projected_area_m2,
                "plane_offset": fit.plane_offset,
                "height_mad_m": fit.h_visual_stats["mad"],
                "height_range_m": fit.h_visual_stats["range"],
                "support_inference_authorized": (
                    interval.reason == "stable_kinematic_pre_manipulation"
                ),
                "frame_zero_observed": interval.start_frame == 0,
            }
        )
    return rows


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return an active wxyz quaternion for a proper rotation matrix."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(m)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * np.sqrt(
            max(
                1.0e-12,
                1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index],
            )
        )
        q = np.zeros(4)
        q[index + 1] = 0.25 * scale
        q[0] = (m[last_index, next_index] - m[next_index, last_index]) / scale
        q[next_index + 1] = (m[next_index, index] + m[index, next_index]) / scale
        q[last_index + 1] = (m[last_index, index] + m[index, last_index]) / scale
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


__all__ = [
    "audit_candidate_support_intervals",
    "detect_stable_pre_contact_interval",
    "infer_planar_support",
    "normalize_gravity",
    "quaternion_to_rotation_matrix",
    "rotation_step_angle",
    "support_normal_from_gravity",
    "transform_mesh_trajectory",
]
