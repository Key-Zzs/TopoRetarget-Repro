"""Object-canonical initialization for free-object stable-grasp calibration.

The functions in this module are simulator independent.  They consume only the
frozen runtime convex proxies plus live calibration-hand poses; source and
corrected trajectories are deliberately absent from the API.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .transforms import quaternion_matrix_wxyz

RESET_MAXIMUM_PAIR_PENETRATION_M = 5.0e-7


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-10:
        raise ValueError(f"{name} must be finite and nonzero")
    return value / norm


def _deterministic_axis_sign(axis: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(axis)))
    return axis if axis[pivot] >= 0.0 else -axis


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a canonical-sign quaternion."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        first = (index + 1) % 3
        second = (index + 2) % 3
        scale = (
            math.sqrt(
                1.0 + rotation[index, index] - rotation[first, first] - rotation[second, second]
            )
            * 2.0
        )
        values = np.zeros(4, dtype=np.float64)
        values[0] = (rotation[second, first] - rotation[first, second]) / scale
        values[index + 1] = 0.25 * scale
        values[first + 1] = (rotation[first, index] + rotation[index, first]) / scale
        values[second + 1] = (rotation[second, index] + rotation[index, second]) / scale
    values /= np.linalg.norm(values)
    return values if values[0] >= 0.0 else -values


@dataclass(frozen=True)
class ObjectCanonicalFrameV1:
    """Deterministic PCA/OBB description of one runtime object proxy."""

    centroid_m: tuple[float, float, float]
    pca_axes_columns: tuple[tuple[float, float, float], ...]
    principal_extents_m: tuple[float, float, float]
    projection_min_m: tuple[float, float, float]
    projection_max_m: tuple[float, float, float]
    schema_version: str = "ObjectCanonicalFrameV1"

    def __post_init__(self) -> None:
        axes: np.ndarray = np.asarray(self.pca_axes_columns, dtype=np.float64)
        extents: np.ndarray = np.asarray(self.principal_extents_m, dtype=np.float64)
        if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-7):
            raise ValueError("PCA axes must form an orthonormal 3x3 basis")
        if not math.isclose(float(np.linalg.det(axes)), 1.0, abs_tol=1.0e-7):
            raise ValueError("PCA axes must be right handed")
        if extents.shape != (3,) or np.any(extents <= 0.0) or not np.all(np.isfinite(extents)):
            raise ValueError("principal extents must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def object_canonical_frame(vertices_m: np.ndarray) -> ObjectCanonicalFrameV1:
    """Compute a deterministic object-local PCA basis and oriented bounds."""

    vertices = np.asarray(vertices_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 4:
        raise ValueError("object proxy vertices must have shape [N>=4,3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("object proxy vertices must be finite")
    centroid = vertices.mean(axis=0)
    covariance = np.cov(vertices - centroid, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    axes[:, 0] = _deterministic_axis_sign(axes[:, 0])
    axes[:, 1] = _deterministic_axis_sign(axes[:, 1])
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    axes[:, 2] = _unit(axes[:, 2], name="third PCA axis")
    axes[:, 1] = _unit(np.cross(axes[:, 2], axes[:, 0]), name="second PCA axis")
    projections = (vertices - centroid) @ axes
    lower = projections.min(axis=0)
    upper = projections.max(axis=0)
    extents = upper - lower
    return ObjectCanonicalFrameV1(
        centroid_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        pca_axes_columns=(
            (float(axes[0, 0]), float(axes[0, 1]), float(axes[0, 2])),
            (float(axes[1, 0]), float(axes[1, 1]), float(axes[1, 2])),
            (float(axes[2, 0]), float(axes[2, 1]), float(axes[2, 2])),
        ),
        principal_extents_m=(float(extents[0]), float(extents[1]), float(extents[2])),
        projection_min_m=(float(lower[0]), float(lower[1]), float(lower[2])),
        projection_max_m=(float(upper[0]), float(upper[1]), float(upper[2])),
    )


@dataclass(frozen=True)
class ObjectCanonicalGraspInitializationV1:
    """One object pose generated from a contact pair and canonical OBB."""

    object_pose_scene_xyz_wxyz: tuple[float, ...]
    object_pca_axis_for_opposition: int
    object_pca_axis_for_approach: int
    opposition_span_m: float
    object_opposition_extent_m: float
    placement_mode: str
    palm_approach_direction_scene: tuple[float, float, float]
    contact_support_plane_scene: tuple[float, float, float]
    precontact_clearance_m: float
    approach_offset_m: float
    corrected_trajectory_used: bool = False
    source_object_pose_used: bool = False
    rollout_state_writes: int = 0
    schema_version: str = "ObjectCanonicalGraspInitializationV1"

    def __post_init__(self) -> None:
        if len(self.object_pose_scene_xyz_wxyz) != 7:
            raise ValueError("object calibration pose must contain xyz+wxyz")
        axes = {self.object_pca_axis_for_opposition, self.object_pca_axis_for_approach}
        if not axes.issubset({0, 1, 2}) or len(axes) != 2:
            raise ValueError("opposition and approach PCA axes must be distinct 0/1/2 axes")
        if self.corrected_trajectory_used or self.source_object_pose_used:
            raise ValueError("object-canonical initialization cannot use trajectory object poses")
        if self.rollout_state_writes != 0:
            raise ValueError("calibration initialization is reset-only")
        if self.placement_mode not in {"inserted_between_contacts", "external_approach"}:
            raise ValueError("unknown object-canonical placement mode")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactBalancedContactRefinementV1:
    """Bounded exact-proxy refinement for a balanced precontact clearance."""

    initial_pose_scene_xyz_wxyz: tuple[float, ...]
    refined_pose_scene_xyz_wxyz: tuple[float, ...]
    selected_groups: tuple[str, ...]
    target_signed_separation_m: float
    maximum_translation_correction_m: float
    maximum_rotation_correction_rad: float
    translation_correction_m: tuple[float, float, float]
    rotation_correction_rad: tuple[float, float, float]
    iterations: int
    selected_signed_separation_before_m: tuple[float, ...]
    selected_signed_separation_after_m: tuple[float, ...]
    selected_direction_balance_norm: float
    maximum_pair_penetration_after_m: float
    safe_reset: bool
    converged: bool
    corrected_trajectory_used: bool = False
    source_object_pose_used: bool = False
    schema_version: str = "ExactBalancedContactRefinementV1"

    def __post_init__(self) -> None:
        if len(self.initial_pose_scene_xyz_wxyz) != 7 or len(self.refined_pose_scene_xyz_wxyz) != 7:
            raise ValueError("exact contact refinement poses must contain xyz+wxyz")
        if len(self.selected_groups) < 2:
            raise ValueError("exact contact refinement requires at least two groups")
        if self.corrected_trajectory_used or self.source_object_pose_used:
            raise ValueError("exact contact refinement cannot use trajectory object poses")
        if self.maximum_translation_correction_m <= 0.0:
            raise ValueError("exact contact correction bound must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rotation_vector_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def apply_bounded_pose_delta(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    change = np.asarray(delta, dtype=np.float64)
    result[:3] += change[:3]
    rotation = _rotation_vector_matrix(change[3:]) @ quaternion_matrix_wxyz(result[3:])
    result[3:] = _matrix_to_quaternion_wxyz(rotation)
    return result


def refine_balanced_contact_pose(
    *,
    initial_pose_scene_xyz_wxyz: np.ndarray,
    selected_groups: Sequence[str],
    group_slots: Mapping[str, Sequence[int]],
    query_pose: Callable[[np.ndarray], Sequence[Any]],
    target_signed_separation_m: float = 0.00025,
    maximum_pair_penetration_m: float = RESET_MAXIMUM_PAIR_PENETRATION_M,
    maximum_translation_correction_m: float = 0.010,
    maximum_rotation_correction_rad: float = math.radians(10.0),
    maximum_iterations: int = 12,
) -> ExactBalancedContactRefinementV1:
    """Balance selected exact convex contacts with a bounded 6-DoF solve.

    ``query_pose`` must return one qualified exact convex result per hand proxy.
    For each iteration, signed-distance linearizations form a minimum-norm
    least-squares correction.  This is deterministic, clip agnostic, and does
    not consume any source/corrected trajectory pose.
    """

    pose = np.asarray(initial_pose_scene_xyz_wxyz, dtype=np.float64).copy()
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("initial exact contact pose must be a finite xyz+wxyz vector")
    groups = tuple(dict.fromkeys(str(group) for group in selected_groups))
    if len(groups) < 2 or any(group not in group_slots for group in groups):
        raise ValueError("selected exact contact groups are unavailable")
    if not 0.0 < target_signed_separation_m <= 0.003:
        raise ValueError("exact reset clearance target must be in (0,3] mm")
    if not 0.0 < maximum_pair_penetration_m <= 0.003:
        raise ValueError("exact reset penetration cap must be in (0,3] mm")
    if not 0.0 < maximum_translation_correction_m <= 0.015:
        raise ValueError("exact contact correction bound must be in (0,15] mm")
    if not 0.0 < maximum_rotation_correction_rad <= math.radians(20.0):
        raise ValueError("exact contact rotation correction must be in (0,20] degrees")
    if not 1 <= maximum_iterations <= 32:
        raise ValueError("exact contact refinement iteration budget must be in [1,32]")

    # The reset pose is deliberately separated.  Contact and any resulting
    # proxy penetration must be produced by the frozen 321-step action
    # schedule, not injected as an initial overlap that can kick the free
    # object before the calibration rollout begins.
    desired_signed = float(target_signed_separation_m)
    initial = pose.copy()

    def snapshot(
        candidate_pose: np.ndarray,
    ) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray]:
        results = list(query_pose(candidate_pose))
        slot_count = max(slot for slots in group_slots.values() for slot in slots) + 1
        if len(results) < slot_count:
            raise ValueError("exact contact query did not cover all group slots")
        selected_results: list[Any] = []
        selected_signed: list[float] = []
        selected_directions: list[np.ndarray] = []
        for group in groups:
            slots = tuple(int(slot) for slot in group_slots[group])
            signed: np.ndarray = np.asarray(
                [float(results[slot].signed_separation_m) for slot in slots], dtype=np.float64
            )
            local_index = int(np.argmin(signed))
            result = results[slots[local_index]]
            direction = np.asarray(result.depenetration_direction_for_second, dtype=np.float64)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                raise RuntimeError("STAGE16D_EXACT_CONTACT_REFINEMENT_INVALID_DIRECTION")
            selected_results.append(result)
            selected_signed.append(float(result.signed_separation_m))
            selected_directions.append(direction)
        all_signed: np.ndarray = np.asarray(
            [float(result.signed_separation_m) for result in results], dtype=np.float64
        )
        return (
            selected_results,
            np.asarray(selected_signed, dtype=np.float64),
            np.stack(selected_directions),
            all_signed,
        )

    _, before_signed, before_directions, before_all_signed = snapshot(pose)
    best_pose = pose.copy()
    best_quality: tuple[bool, float, float, float] = (
        float(np.maximum(-before_all_signed, 0.0).max(initial=0.0)) > maximum_pair_penetration_m,
        float(np.maximum(-before_all_signed, 0.0).max(initial=0.0)),
        float(np.max(np.abs(before_signed - desired_signed))),
        float(np.linalg.norm(before_directions.sum(axis=0))),
    )
    completed_iterations = 0
    for iteration in range(maximum_iterations):
        _, signed, directions, all_signed = snapshot(pose)
        maximum_penetration = float(np.maximum(-all_signed, 0.0).max(initial=0.0))
        quality = (
            maximum_penetration > maximum_pair_penetration_m,
            maximum_penetration,
            float(np.max(np.abs(signed - desired_signed))),
            float(np.linalg.norm(directions.sum(axis=0))),
        )
        if quality < best_quality:
            best_quality = quality
            best_pose = pose.copy()
        completed_iterations = iteration + 1
        if (
            np.max(np.abs(signed - desired_signed)) <= 5.0e-5
            and maximum_penetration <= maximum_pair_penetration_m
        ):
            break

        rows = [direction for direction in directions]
        right = [desired_signed - value for value in signed]
        # Add active half-space constraints only for pairs exceeding the fixed
        # reset penetration cap.  Their qualified MTD directions move the
        # object toward the safe side without fabricating forces.
        results = list(query_pose(pose))
        for result in results:
            signed_value = float(result.signed_separation_m)
            if signed_value < -maximum_pair_penetration_m:
                rows.append(np.asarray(result.depenetration_direction_for_second, dtype=np.float64))
                right.append(desired_signed - signed_value)
        matrix = np.stack(rows)
        delta = np.linalg.lstsq(matrix, np.asarray(right), rcond=None)[0]
        step_norm = float(np.linalg.norm(delta))
        if step_norm > 0.002:
            delta *= 0.002 / step_norm
        candidate_translation = pose[:3] + delta
        total = candidate_translation - initial[:3]
        total_norm = float(np.linalg.norm(total))
        if total_norm > maximum_translation_correction_m:
            candidate_translation = (
                initial[:3] + total * maximum_translation_correction_m / total_norm
            )
        pose[:3] = candidate_translation

    pose = best_pose
    cumulative_rotation: np.ndarray = np.zeros(3, dtype=np.float64)

    def residual(candidate_pose: np.ndarray) -> np.ndarray:
        _, signed, directions, all_signed = snapshot(candidate_pose)
        selected_error = (signed - desired_signed) / 0.001
        direction_balance = 2.0 * directions.sum(axis=0)
        penetration_violation = np.minimum(all_signed + maximum_pair_penetration_m, 0.0) / 0.001
        return np.concatenate((selected_error, direction_balance, penetration_violation))

    # Translation alone equalizes the two signed clearances.  A bounded local
    # Gauss-Newton refinement then rotates the canonical OBB slightly so the
    # two prospective contact normals oppose one another.  Direction balance
    # is a deterministic ranking metric at positive clearance, not a physical
    # hard gate: before contact there is no force or torque to balance.
    for _orientation_iteration in range(maximum_iterations):
        base_residual = residual(pose)
        base_objective = float(np.dot(base_residual, base_residual))
        jacobian: np.ndarray = np.empty((len(base_residual), 6), dtype=np.float64)
        perturbations = np.asarray(
            [0.0001, 0.0001, 0.0001, math.radians(0.1), math.radians(0.1), math.radians(0.1)]
        )
        for axis, epsilon in enumerate(perturbations):
            delta = np.zeros(6, dtype=np.float64)
            delta[axis] = epsilon
            jacobian[:, axis] = (
                residual(apply_bounded_pose_delta(pose, delta)) - base_residual
            ) / epsilon
        system = jacobian.T @ jacobian + 1.0e-4 * np.eye(6)
        step = -np.linalg.solve(system, jacobian.T @ base_residual)
        translation_norm = float(np.linalg.norm(step[:3]))
        if translation_norm > 0.001:
            step[:3] *= 0.001 / translation_norm
        rotation_norm = float(np.linalg.norm(step[3:]))
        if rotation_norm > math.radians(1.0):
            step[3:] *= math.radians(1.0) / rotation_norm
        accepted = False
        for factor in (1.0, 0.5, 0.25):
            trial_step = factor * step
            total_translation = pose[:3] + trial_step[:3] - initial[:3]
            if np.linalg.norm(total_translation) > maximum_translation_correction_m:
                continue
            total_rotation = cumulative_rotation + trial_step[3:]
            if np.linalg.norm(total_rotation) > maximum_rotation_correction_rad:
                continue
            trial = apply_bounded_pose_delta(pose, trial_step)
            trial_residual = residual(trial)
            trial_objective = float(np.dot(trial_residual, trial_residual))
            if trial_objective < base_objective:
                pose = trial
                cumulative_rotation = total_rotation
                accepted = True
                break
        completed_iterations += 1
        if not accepted:
            break
        _, candidate_signed, candidate_directions, candidate_all_signed = snapshot(pose)
        candidate_penetration = float(np.maximum(-candidate_all_signed, 0.0).max(initial=0.0))
        candidate_quality = (
            candidate_penetration > maximum_pair_penetration_m,
            candidate_penetration,
            float(np.max(np.abs(candidate_signed - desired_signed))),
            float(np.linalg.norm(candidate_directions.sum(axis=0))),
        )
        if candidate_quality < best_quality:
            best_quality = candidate_quality
            best_pose = pose.copy()
        _, signed, directions, all_signed = snapshot(pose)
        if (
            np.max(np.abs(signed - desired_signed)) <= 1.5e-4
            and float(np.linalg.norm(directions.sum(axis=0))) <= 0.25
            and float(np.maximum(-all_signed, 0.0).max(initial=0.0)) <= maximum_pair_penetration_m
        ):
            break

    pose = best_pose
    _, after_signed, after_directions, all_after = snapshot(pose)
    maximum_after = float(np.maximum(-all_after, 0.0).max(initial=0.0))
    correction = pose[:3] - initial[:3]
    direction_balance = float(np.linalg.norm(after_directions.sum(axis=0)))
    converged = bool(
        np.max(np.abs(after_signed - desired_signed)) <= 1.5e-4
        and maximum_after <= maximum_pair_penetration_m
    )
    safe_reset = bool(maximum_after <= maximum_pair_penetration_m)
    return ExactBalancedContactRefinementV1(
        initial_pose_scene_xyz_wxyz=tuple(float(value) for value in initial),
        refined_pose_scene_xyz_wxyz=tuple(float(value) for value in pose),
        selected_groups=groups,
        target_signed_separation_m=desired_signed,
        maximum_translation_correction_m=float(maximum_translation_correction_m),
        maximum_rotation_correction_rad=float(maximum_rotation_correction_rad),
        translation_correction_m=(
            float(correction[0]),
            float(correction[1]),
            float(correction[2]),
        ),
        rotation_correction_rad=(
            float(cumulative_rotation[0]),
            float(cumulative_rotation[1]),
            float(cumulative_rotation[2]),
        ),
        iterations=completed_iterations,
        selected_signed_separation_before_m=tuple(float(value) for value in before_signed),
        selected_signed_separation_after_m=tuple(float(value) for value in after_signed),
        selected_direction_balance_norm=direction_balance,
        maximum_pair_penetration_after_m=maximum_after,
        safe_reset=safe_reset,
        converged=converged,
    )


def initialize_object_between_contacts(
    *,
    frame: ObjectCanonicalFrameV1,
    first_contact_center_scene: np.ndarray,
    second_contact_center_scene: np.ndarray,
    palm_center_scene: np.ndarray,
    palm_rotation_scene: np.ndarray,
    approach_offset_m: float,
    precontact_clearance_m: float = 0.003,
    first_contact_vertices_scene: np.ndarray | None = None,
    second_contact_vertices_scene: np.ndarray | None = None,
) -> ObjectCanonicalGraspInitializationV1:
    """Place an object OBB between two live calibration contact bodies.

    The object PCA axis whose extent best matches the contact span is aligned
    to the contact opposition direction.  A palm axis supplies the remaining
    orientation without consulting either source or corrected object poses.
    """

    first = np.asarray(first_contact_center_scene, dtype=np.float64)
    second = np.asarray(second_contact_center_scene, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("contact centers must be 3-vectors")
    palm_center = np.asarray(palm_center_scene, dtype=np.float64)
    if palm_center.shape != (3,) or not np.all(np.isfinite(palm_center)):
        raise ValueError("palm center must be a finite 3-vector")
    palm_rotation = np.asarray(palm_rotation_scene, dtype=np.float64)
    if palm_rotation.shape != (3, 3):
        raise ValueError("palm rotation must be 3x3")
    first_anchor = first
    second_anchor = second
    if first_contact_vertices_scene is not None or second_contact_vertices_scene is not None:
        if first_contact_vertices_scene is None or second_contact_vertices_scene is None:
            raise ValueError("both contact proxy vertex sets must be supplied together")
        first_vertices = np.asarray(first_contact_vertices_scene, dtype=np.float64)
        second_vertices = np.asarray(second_contact_vertices_scene, dtype=np.float64)
        if (
            first_vertices.ndim != 2
            or second_vertices.ndim != 2
            or first_vertices.shape[1] != 3
            or second_vertices.shape[1] != 3
            or first_vertices.shape[0] < 4
            or second_vertices.shape[0] < 4
            or not np.all(np.isfinite(first_vertices))
            or not np.all(np.isfinite(second_vertices))
        ):
            raise ValueError("contact proxy vertices must be finite [N>=4,3] arrays")
    else:
        first_vertices = None
        second_vertices = None

    # Fit a common fingertip support plane.  Its normal is palm-outward and
    # orthogonal to the line joining the two selected convex-proxy support
    # points, so both contacts have the same normal gap.  Re-selecting support
    # points a few times resolves the normal/support-point dependency without
    # any clip-specific branch or trajectory pose.
    opposition = _unit(second - first, name="contact opposition")
    for _ in range(4):
        midpoint = 0.5 * (first_anchor + second_anchor)
        outward_hint = midpoint - palm_center
        if np.linalg.norm(outward_hint) <= 1.0e-10:
            palm_axes = [palm_rotation[:, index] for index in range(3)]
            outward_hint = min(palm_axes, key=lambda axis: abs(float(np.dot(axis, opposition))))
        outward = _unit(
            outward_hint - np.dot(outward_hint, opposition) * opposition,
            name="palm outward axis",
        )
        if first_vertices is not None and second_vertices is not None:
            first_anchor = first_vertices[int(np.argmax(first_vertices @ outward))]
            second_anchor = second_vertices[int(np.argmax(second_vertices @ outward))]
            opposition = _unit(second_anchor - first_anchor, name="contact support opposition")
    support_point = 0.5 * (first_anchor + second_anchor)
    opposition_span = float(np.linalg.norm(second_anchor - first_anchor))

    extents: np.ndarray = np.asarray(frame.principal_extents_m, dtype=np.float64)
    selected_axis = int(np.argmin(np.abs(extents - opposition_span)))
    remaining = [index for index in range(3) if index != selected_axis]
    approach_axis = max(remaining, key=lambda index: extents[index])
    third_axis = next(index for index in remaining if index != approach_axis)
    local_axes: np.ndarray = np.asarray(frame.pca_axes_columns, dtype=np.float64)
    world_targets: np.ndarray = np.zeros((3, 3), dtype=np.float64)
    world_targets[:, selected_axis] = opposition
    world_targets[:, approach_axis] = outward
    world_targets[:, third_axis] = _unit(np.cross(opposition, outward), name="third grasp axis")
    if np.linalg.det(world_targets) < 0.0:
        world_targets[:, third_axis] *= -1.0
    rotation = world_targets @ local_axes.T
    lower: np.ndarray = np.asarray(frame.projection_min_m, dtype=np.float64)
    near_face_to_centroid_m = -float(lower[approach_axis])
    if not 0.0 <= precontact_clearance_m <= 0.005:
        raise ValueError("precontact clearance must be in [0,5] mm")
    if float(extents[selected_axis]) <= opposition_span:
        placement_mode = "inserted_between_contacts"
    else:
        placement_mode = "external_approach"
    # In both modes, keep the object approach-facing support plane outside the
    # hand.  Centering an elongated object on the fingertip plane would place
    # half of its approach extent through the palm even when its opposition
    # extent fits cleanly between the selected sides.
    desired_centroid = (
        support_point
        + (near_face_to_centroid_m + float(precontact_clearance_m) + float(approach_offset_m))
        * outward
    )
    centroid_local: np.ndarray = np.asarray(frame.centroid_m, dtype=np.float64)
    translation = desired_centroid - rotation @ centroid_local
    quaternion = _matrix_to_quaternion_wxyz(rotation)
    pose = np.concatenate((translation, quaternion))
    return ObjectCanonicalGraspInitializationV1(
        object_pose_scene_xyz_wxyz=tuple(float(value) for value in pose),
        object_pca_axis_for_opposition=selected_axis,
        object_pca_axis_for_approach=approach_axis,
        opposition_span_m=opposition_span,
        object_opposition_extent_m=float(extents[selected_axis]),
        placement_mode=placement_mode,
        palm_approach_direction_scene=(
            float(outward[0]),
            float(outward[1]),
            float(outward[2]),
        ),
        contact_support_plane_scene=(
            float(support_point[0]),
            float(support_point[1]),
            float(support_point[2]),
        ),
        precontact_clearance_m=float(precontact_clearance_m),
        approach_offset_m=float(approach_offset_m),
    )


__all__ = [
    "ExactBalancedContactRefinementV1",
    "ObjectCanonicalFrameV1",
    "ObjectCanonicalGraspInitializationV1",
    "apply_bounded_pose_delta",
    "initialize_object_between_contacts",
    "object_canonical_frame",
    "refine_balanced_contact_pose",
]
