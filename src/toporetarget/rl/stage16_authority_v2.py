"""Stage16 angular-velocity and raw-human grasp authority contracts.

These helpers are deliberately offline and outcome-independent.  They do not
load a policy, advance a simulator, change a reward, or reinterpret historical
trace bytes in place.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

from toporetarget.evaluation.source_contact_semantics import (
    REGION_ORDER,
    SEGMENT_ORDER,
    ManoSurfaceRegionMap,
    SourceContactThresholdContractV1,
    mesh_adjacency,
    persistent_mask,
)
from toporetarget.rl.reference_tracking.reference_kinematics import (
    derive_angular_velocity_world_wxyz,
    quaternion_to_matrix_wxyz,
    so3_log,
)

ACTUAL_ANGULAR_AUTHORITY_V2: Final = "Stage16ActualAngularVelocityAuthorityV2"
RAW_GRASP_PROFILE_V1: Final = "RawHumanGraspReadinessProfileV1"


def timing_layer_profile(
    *,
    raw_frame: int | None,
    retarget_frame: int,
    actual_frame: int,
    lift_frame: int,
    control_dt_s: float,
) -> dict[str, int | float | str | None]:
    """Report timing signs without manufacturing an unavailable raw authority."""

    if min(retarget_frame, actual_frame, lift_frame) < 0 or control_dt_s <= 0.0:
        raise ValueError("STAGE16_TIMING_PROFILE_INPUT_INVALID")
    if raw_frame is not None and raw_frame < 0:
        raise ValueError("STAGE16_TIMING_PROFILE_RAW_INPUT_INVALID")
    raw_margin = None if raw_frame is None else lift_frame - raw_frame
    raw_to_retarget = None if raw_frame is None else retarget_frame - raw_frame
    return {
        "raw_frame": raw_frame,
        "retarget_frame": retarget_frame,
        "actual_frame": actual_frame,
        "lift_frame": lift_frame,
        "raw_margin_frames": raw_margin,
        "raw_margin_s": None if raw_margin is None else raw_margin * control_dt_s,
        "retarget_margin_frames": lift_frame - retarget_frame,
        "retarget_margin_s": (lift_frame - retarget_frame) * control_dt_s,
        "actual_margin_frames": lift_frame - actual_frame,
        "actual_margin_s": (lift_frame - actual_frame) * control_dt_s,
        "raw_to_retarget_frames": (
            "NOT_IDENTIFIABLE" if raw_to_retarget is None else raw_to_retarget
        ),
        "raw_to_retarget_s": (
            "NOT_IDENTIFIABLE" if raw_to_retarget is None else raw_to_retarget * control_dt_s
        ),
        "retarget_to_actual_frames": actual_frame - retarget_frame,
        "retarget_to_actual_s": (actual_frame - retarget_frame) * control_dt_s,
        "margin_sign_convention": "positive_before_lift_negative_after_lift",
    }


@dataclass(frozen=True)
class Stage16ActualAngularVelocityAuthorityV2:
    """Comparable actual angular-velocity authority for Reference Kinematics V2."""

    identifier: str = ACTUAL_ANGULAR_AUTHORITY_V2
    source: str = "trace.object_pose actor-frame quaternion wxyz"
    frame: str = "WORLD"
    body: str = "single active rigid object"
    timestamp: str = "same post-physics control row as trace pose"
    estimator: str = "ReferenceKinematicsV2.SO3_log_centered_world_with_one_sided_endpoints"
    conversion: str = "none; derive world omega directly from world object orientation"
    legacy_trace_source: str = (
        "RigidObjectData.root_state_w[10:13] -> root_com_vel_w -> "
        "PhysX RigidBodyView.get_velocities()[3:6]"
    )
    legacy_trace_frame: str = "WORLD"
    legacy_trace_body: str = "active rigid-object COM; angular velocity is point invariant"
    legacy_trace_sampling: str = "instantaneous lazy-buffer read after the final physics substep"
    reference_sampling: str = "control-rate centered pose displacement over adjacent samples"
    historical_trace_rewritten: bool = False
    angular_threshold_tuned: bool = False

    def __post_init__(self) -> None:
        if self.identifier != ACTUAL_ANGULAR_AUTHORITY_V2:
            raise ValueError("ACTUAL_ANGULAR_AUTHORITY_IDENTIFIER_DRIFT")
        if self.frame != "WORLD" or self.legacy_trace_frame != "WORLD":
            raise ValueError("ACTUAL_ANGULAR_AUTHORITY_FRAME_DRIFT")
        if self.historical_trace_rewritten or self.angular_threshold_tuned:
            raise ValueError("ACTUAL_ANGULAR_AUTHORITY_MUTATION_FORBIDDEN")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector_distribution(values: np.ndarray) -> dict[str, float]:
    magnitude = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=-1)
    if magnitude.ndim != 1 or not magnitude.size or not np.isfinite(magnitude).all():
        raise ValueError("ANGULAR_SEMANTIC_DISTRIBUTION_INVALID")
    return {
        "mean": float(magnitude.mean()),
        "median": float(np.median(magnitude)),
        "p95": float(np.quantile(magnitude, 0.95)),
        "max": float(magnitude.max()),
    }


def angular_velocity_semantic_alignment(
    *,
    object_pose_wxyz: np.ndarray,
    trace_angular_velocity: np.ndarray,
    timestamps_s: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, object]:
    """Audit documented frame/time candidates without fit-based authority selection."""

    pose = np.asarray(object_pose_wxyz, dtype=np.float64)
    trace = np.asarray(trace_angular_velocity, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7 or trace.shape != (len(pose), 3):
        raise ValueError("ANGULAR_SEMANTIC_TRACE_SHAPE_INVALID")
    if timestamps.shape != (len(pose),) or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("ANGULAR_SEMANTIC_TIMESTAMP_INVALID")
    rows = np.ones(len(pose), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if rows.shape != (len(pose),) or not rows.any():
        raise ValueError("ANGULAR_SEMANTIC_VALID_MASK_INVALID")
    if not np.isfinite(pose).all() or not np.isfinite(trace).all():
        raise ValueError("ANGULAR_SEMANTIC_NONFINITE")

    authority = derive_angular_velocity_world_wxyz(pose[:, 3:], timestamps)
    rotation = quaternion_to_matrix_wxyz(pose[:, 3:])
    # These transforms are diagnostics only.  Source authority says the trace
    # is already world-frame, so neither candidate can be selected by fit.
    local_to_world_candidate = np.einsum("tij,tj->ti", rotation, trace)
    world_to_local_candidate = np.einsum("tji,tj->ti", rotation, trace)
    previous = np.vstack((trace[0], trace[:-1]))
    following = np.vstack((trace[1:], trace[-1]))
    dt = np.diff(timestamps)
    interval_pose = so3_log(rotation[1:] @ np.swapaxes(rotation[:-1], -1, -2)) / dt[:, None]
    trapezoid_trace = 0.5 * (trace[:-1] + trace[1:])
    interval_closure = trapezoid_trace - interval_pose
    pose_speed = np.linalg.norm(authority, axis=-1)
    trace_speed = np.linalg.norm(trace, axis=-1)
    static_pose_nonzero_trace = rows & (pose_speed <= 1.0e-3) & (trace_speed >= 0.05)

    def selected(error: np.ndarray) -> dict[str, float]:
        return _vector_distribution(np.asarray(error)[rows])

    return {
        "authority_omega_world": authority,
        "historical_trace_omega_world": trace,
        "semantic_corrected_trace_omega_world": trace.copy(),
        "documented_transform": "NONE_TRACE_ALREADY_WORLD",
        "offset_diagnostics": {
            "t": selected(trace - authority),
            "t_minus_1": selected(previous - authority),
            "t_plus_1": selected(following - authority),
            "selection_rule": "diagnostic_only_code_semantics_precedes_fit",
        },
        "frame_diagnostics": {
            "documented_world": selected(trace - authority),
            "hypothetical_local_to_world": selected(local_to_world_candidate - authority),
            "hypothetical_world_to_local": selected(world_to_local_candidate - authority),
            "selection_rule": "documented_world_only",
        },
        "kinematic_closure": {
            "trapezoidal_trace_vs_interval_pose": _vector_distribution(interval_closure),
            "pose_speed": _vector_distribution(authority[rows]),
            "trace_speed": _vector_distribution(trace[rows]),
            "static_pose_nonzero_trace_frame_count": int(static_pose_nonzero_trace.sum()),
            "static_pose_nonzero_trace_fraction": float(static_pose_nonzero_trace[rows].mean()),
            "diagnostic_thresholds": {
                "pose_speed_max_radps": 1.0e-3,
                "trace_speed_min_radps": 0.05,
                "authority_use": "diagnostic_only_not_gate_or_selection",
            },
        },
    }


@dataclass(frozen=True)
class RawHumanGraspReadinessProfileV1:
    """Outcome-independent profile; deliberately not a functional-grasp bool."""

    identifier: str = RAW_GRASP_PROFILE_V1
    source_geometry: str = "raw HOCap MANO surface to raw selected-object exact triangles"
    region_authority: str = "ManoSurfaceRegionMapV1 from MANO v1.2 LBS joint chains"
    nominal_contact_distance_m: float = 0.002
    component_distance_m: float = 0.005
    minimum_component_vertices: int = 3
    source_rate_hz: float = 30.0
    source_persistence_frames: int = 2
    runtime_rate_hz: float = 20.0
    runtime_persistence_frames: int = 2
    multi_region_minimum: int = 2
    opposition_normal_angle_deg: float = 90.0
    opposition_minimum_separation_m: float = 0.004
    binary_functional_authority_validated: bool = False
    force_closure_claimed: bool = False
    outcome_tuned: bool = False

    def __post_init__(self) -> None:
        source = SourceContactThresholdContractV1()
        if (
            self.nominal_contact_distance_m != source.nominal_min_distance_m
            or self.component_distance_m != source.component_distance_m
            or self.minimum_component_vertices != source.minimum_component_vertices
            or self.source_persistence_frames != source.native_persistence_frames
        ):
            raise ValueError("RAW_GRASP_SOURCE_THRESHOLD_DRIFT")
        duration_s = self.source_persistence_frames / self.source_rate_hz
        expected_runtime = math.ceil(duration_s * self.runtime_rate_hz - 1.0e-12)
        if self.runtime_persistence_frames != expected_runtime:
            raise ValueError("RAW_GRASP_PERSISTENCE_DURATION_DRIFT")
        if self.multi_region_minimum != 2:
            raise ValueError("RAW_GRASP_MULTI_REGION_DRIFT")
        if self.binary_functional_authority_validated or self.force_closure_claimed:
            raise ValueError("RAW_GRASP_UNSUPPORTED_AUTHORITY_CLAIM")
        if self.outcome_tuned:
            raise ValueError("RAW_GRASP_OUTCOME_TUNING_FORBIDDEN")

    @property
    def persistence_duration_s(self) -> float:
        return self.source_persistence_frames / self.source_rate_hz

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "persistence_duration_s": self.persistence_duration_s}


def _largest_components_per_region(
    distances_m: np.ndarray,
    region_map: ManoSurfaceRegionMap,
    faces: np.ndarray,
    *,
    threshold_m: float,
) -> np.ndarray:
    """Compute per-frame connected-component sizes at the frozen 5 mm radius."""

    distance = np.asarray(distances_m, dtype=np.float64)
    adjacency = mesh_adjacency(distance.shape[1], faces)
    result = np.zeros((len(distance), 6), dtype=np.int32)
    for region in range(6):
        member = region_map.region_id == region
        for frame in range(len(distance)):
            active = member & (distance[frame] <= threshold_m)
            visited = np.zeros(len(active), dtype=bool)
            largest = 0
            for start in np.flatnonzero(active):
                if visited[start]:
                    continue
                stack = [int(start)]
                visited[start] = True
                count = 0
                while stack:
                    current = stack.pop()
                    count += 1
                    for neighbor in adjacency[current]:
                        if active[neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(int(neighbor))
                largest = max(largest, count)
            result[frame, region] = largest
    return result


def opposing_contact_topology(
    *,
    closest_points_object: np.ndarray,
    contact_normals_object: np.ndarray,
    contact_region_ids: np.ndarray,
    minimum_separation_m: float,
    minimum_angle_deg: float,
) -> dict[str, object]:
    """Evaluate geometric opposition without claiming force closure."""

    points = np.asarray(closest_points_object, dtype=np.float64)
    normals = np.asarray(contact_normals_object, dtype=np.float64)
    regions = np.asarray(contact_region_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("RAW_GRASP_TOPOLOGY_POINT_SHAPE_INVALID")
    if regions.shape != (len(points),) or minimum_separation_m <= 0.0:
        raise ValueError("RAW_GRASP_TOPOLOGY_REGION_SHAPE_INVALID")
    if not 0.0 < minimum_angle_deg <= 180.0:
        raise ValueError("RAW_GRASP_TOPOLOGY_ANGLE_INVALID")
    norms = np.linalg.norm(normals, axis=-1)
    if len(points) < 2 or np.any(norms <= 1.0e-12):
        return {
            "opposing": False,
            "minimum_normal_dot": None,
            "maximum_normal_angle_deg": None,
            "maximum_qualifying_separation_m": None,
        }
    unit = normals / norms[:, None]
    dot = np.clip(unit @ unit.T, -1.0, 1.0)
    separation = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    distinct = regions[:, None] != regions[None, :]
    upper = np.triu(np.ones_like(distinct, dtype=bool), k=1)
    candidate = distinct & upper & (separation >= minimum_separation_m)
    if not candidate.any():
        return {
            "opposing": False,
            "minimum_normal_dot": None,
            "maximum_normal_angle_deg": None,
            "maximum_qualifying_separation_m": None,
        }
    minimum_dot = float(dot[candidate].min())
    maximum_angle = float(np.degrees(np.arccos(minimum_dot)))
    return {
        "opposing": bool(maximum_angle >= minimum_angle_deg),
        "minimum_normal_dot": minimum_dot,
        "maximum_normal_angle_deg": maximum_angle,
        "maximum_qualifying_separation_m": float(separation[candidate].max()),
    }


def raw_human_grasp_profile(
    *,
    distances_m: np.ndarray,
    region_map: ManoSurfaceRegionMap,
    mano_faces: np.ndarray,
    opposing_topology_raw: np.ndarray | None = None,
    contract: RawHumanGraspReadinessProfileV1 | None = None,
) -> dict[str, np.ndarray]:
    """Build all profile layers from the same frozen source-distance convention."""

    frozen = contract or RawHumanGraspReadinessProfileV1()
    distance = np.asarray(distances_m, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[1] != region_map.region_id.size:
        raise ValueError("RAW_GRASP_DISTANCE_SHAPE_INVALID")
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise ValueError("RAW_GRASP_DISTANCE_INVALID")
    component = _largest_components_per_region(
        distance,
        region_map,
        mano_faces,
        threshold_m=frozen.component_distance_m,
    )
    minimum = np.full((len(distance), 6), np.inf, dtype=np.float64)
    for region in range(6):
        member = region_map.region_id == region
        minimum[:, region] = distance[:, member].min(axis=1)
    robust_region = (minimum <= frozen.nominal_contact_distance_m) & (
        component >= frozen.minimum_component_vertices
    )
    any_surface_raw = distance.min(axis=1) <= frozen.nominal_contact_distance_m
    any_robust_raw = robust_region.any(axis=1)
    multi_region_raw = robust_region.sum(axis=1) >= frozen.multi_region_minimum
    thumb_non_thumb_raw = robust_region[:, 0] & robust_region[:, 1:5].any(axis=1)
    topology_raw = (
        np.zeros(len(distance), dtype=bool)
        if opposing_topology_raw is None
        else np.asarray(opposing_topology_raw, dtype=bool)
    )
    if topology_raw.shape != (len(distance),):
        raise ValueError("RAW_GRASP_OPPOSING_TOPOLOGY_SHAPE_INVALID")
    minimum_steps = frozen.runtime_persistence_frames

    segment_contact = np.zeros((len(distance), len(SEGMENT_ORDER)), dtype=bool)
    for segment in range(len(SEGMENT_ORDER)):
        member = region_map.segment_id == segment
        if np.any(member):
            segment_contact[:, segment] = (
                distance[:, member].min(axis=1) <= frozen.nominal_contact_distance_m
            )

    return {
        "minimum_surface_distance_m": distance.min(axis=1),
        "minimum_region_distance_m": minimum,
        "largest_component_vertices_at_5mm": component,
        "robust_region_contact_raw": robust_region,
        "robust_region_contact_persistent": np.stack(
            [persistent_mask(robust_region[:, index], minimum_steps) for index in range(6)],
            axis=-1,
        ),
        "any_hand_surface_contact_raw": any_surface_raw,
        "any_hand_surface_contact": persistent_mask(any_surface_raw, minimum_steps),
        "any_robust_region_contact_raw": any_robust_raw,
        "any_robust_region_contact": persistent_mask(any_robust_raw, minimum_steps),
        "multi_region_contact_raw": multi_region_raw,
        "multi_region_contact": persistent_mask(multi_region_raw, minimum_steps),
        "thumb_non_thumb_contact_raw": thumb_non_thumb_raw,
        "thumb_non_thumb_contact": persistent_mask(thumb_non_thumb_raw, minimum_steps),
        "opposing_topology_raw": topology_raw,
        "opposing_topology": persistent_mask(topology_raw, minimum_steps),
        "segment_contact": segment_contact,
        "region_order": np.asarray(REGION_ORDER[:6]),
        "segment_order": np.asarray(SEGMENT_ORDER),
    }


__all__ = [
    "ACTUAL_ANGULAR_AUTHORITY_V2",
    "RAW_GRASP_PROFILE_V1",
    "RawHumanGraspReadinessProfileV1",
    "Stage16ActualAngularVelocityAuthorityV2",
    "angular_velocity_semantic_alignment",
    "opposing_contact_topology",
    "raw_human_grasp_profile",
    "timing_layer_profile",
]
