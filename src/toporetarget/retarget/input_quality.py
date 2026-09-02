"""Outcome-independent retarget input validation and bounded gap repair.

This module intentionally stops before any robot model or retarget objective is
loaded.  It validates source tracking, applies one dataset-wide seconds-domain
gap policy, and freezes the wrist-frame authority used by downstream retarget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class RetargetInputQualityError(RuntimeError):
    """Raised when source tracking cannot satisfy the frozen input contract."""


@dataclass(frozen=True)
class RetargetInputQualityContractV1:
    """Global, clip-independent thresholds for retarget input preflight."""

    schema_version: str = "RetargetInputQualityV1"
    maximum_repair_gap_seconds: float = 0.100
    near_zero_bone_length_m: float = 1.0e-8
    duplicate_joint_distance_m: float = 1.0e-10
    wrist_axis_minimum_norm_m: float = 1.0e-8
    wrist_axis_minimum_sine: float = 1.0e-4
    orientation_discontinuity_rad: float = 2.0943951023931953
    translation_discontinuity_m: float = 0.250
    object_quaternion_norm_tolerance: float = 1.0e-3
    mano_pose_interpolation: str = "linear_pca_pose_space"
    mano_orientation_interpolation: str = "SO3_geodesic"
    object_orientation_interpolation: str = "SO3_geodesic"
    translation_interpolation: str = "linear"
    wrist_authority_priority: tuple[str, ...] = (
        "MANO_GLOBAL_WRIST_ORIENTATION",
        "MANO_LAYER_RECONSTRUCTED_WRIST_FRAME",
        "TEMPORALLY_PROPAGATED_VALID_WRIST_FRAME",
        "KEYPOINT_DERIVED_WRIST_FRAME_DIAGNOSTIC_ONLY",
        "RAW_TRACKING_QUALITY_FAILED",
    )

    def __post_init__(self) -> None:
        positive = (
            self.maximum_repair_gap_seconds,
            self.near_zero_bone_length_m,
            self.duplicate_joint_distance_m,
            self.wrist_axis_minimum_norm_m,
            self.wrist_axis_minimum_sine,
            self.orientation_discontinuity_rad,
            self.translation_discontinuity_m,
            self.object_quaternion_norm_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("RETARGET_INPUT_QUALITY_THRESHOLD_INVALID")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def invalid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open invalid runs for a one-dimensional validity mask."""

    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("RETARGET_INPUT_VALID_MASK_MUST_BE_1D")
    boundaries = np.diff(np.pad((~mask).astype(np.int8), (1, 1)))
    return [
        (int(start), int(end))
        for start, end in zip(
            np.flatnonzero(boundaries == 1),
            np.flatnonzero(boundaries == -1),
            strict=True,
        )
    ]


def _repairable_runs(
    valid: np.ndarray, timestamps: np.ndarray, maximum_gap_seconds: float
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    short: list[tuple[int, int]] = []
    long: list[tuple[int, int]] = []
    count = len(valid)
    for start, end in invalid_runs(valid):
        # Interpolation is two-sided. Boundary gaps are never extrapolated.
        duration = (
            float(timestamps[end] - timestamps[start - 1])
            if start > 0 and end < count
            else float("inf")
        )
        if start > 0 and end < count and duration <= maximum_gap_seconds + 1.0e-12:
            short.append((start, end))
        else:
            long.append((start, end))
    return short, long


def _linear_fill(values: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    for start, end in runs:
        alpha = np.arange(1, end - start + 1, dtype=np.float64) / (end - start + 1)
        output[start:end] = (1.0 - alpha)[(...,) + (None,) * (output.ndim - 1)] * output[
            start - 1
        ] + alpha[(...,) + (None,) * (output.ndim - 1)] * output[end]
    return output


def _rotation_fill(
    rotvecs: np.ndarray, timestamps: np.ndarray, runs: list[tuple[int, int]]
) -> np.ndarray:
    output = np.asarray(rotvecs, dtype=np.float64).copy()
    for start, end in runs:
        key_times = np.asarray([timestamps[start - 1], timestamps[end]], dtype=np.float64)
        key_rotations = Rotation.from_rotvec(output[[start - 1, end]])
        output[start:end] = Slerp(key_times, key_rotations)(timestamps[start:end]).as_rotvec()
    return output


def repair_mano_pose(
    pose_51: np.ndarray,
    timestamps: np.ndarray,
    contract: RetargetInputQualityContractV1 | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair bounded interior MANO gaps with SO(3) and linear interpolation."""

    frozen = contract or RetargetInputQualityContractV1()
    pose = np.asarray(pose_51, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1:] != (51,) or time.shape != (len(pose),):
        raise ValueError("RETARGET_INPUT_MANO_SHAPE_INVALID")
    valid = np.asarray(np.isfinite(pose).all(axis=1), dtype=np.bool_)
    short, long = _repairable_runs(valid, time, frozen.maximum_repair_gap_seconds)
    if long:
        raise RetargetInputQualityError(f"UNRECOVERABLE_TRACKING_GAP:MANO:{long}")
    repaired = pose.copy()
    repaired[:, :3] = _rotation_fill(repaired[:, :3], time, short)
    repaired[:, 3:48] = _linear_fill(repaired[:, 3:48], short)
    repaired[:, 48:51] = _linear_fill(repaired[:, 48:51], short)
    return repaired, {
        "valid_before": int(np.count_nonzero(valid)),
        "invalid_before": int(np.count_nonzero(~valid)),
        "short_invalid_gaps": [list(item) for item in short],
        "long_invalid_gaps": [list(item) for item in long],
        "repaired_frames": sorted(index for start, end in short for index in range(start, end)),
    }


def repair_object_pose_qxyzw(
    pose_7: np.ndarray,
    timestamps: np.ndarray,
    contract: RetargetInputQualityContractV1 | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair one or more HOCap ``[qx,qy,qz,qw,tx,ty,tz]`` object tracks."""

    frozen = contract or RetargetInputQualityContractV1()
    value = np.asarray(pose_7, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    squeeze = value.ndim == 2
    if squeeze:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[0] != len(time) or value.shape[2:] != (7,):
        raise ValueError("RETARGET_INPUT_OBJECT_POSE_SHAPE_INVALID")
    output = value.copy()
    reports: list[dict[str, Any]] = []
    for object_index in range(value.shape[1]):
        track = value[:, object_index]
        quaternion_norm = np.linalg.norm(track[:, :4], axis=1)
        valid = (
            np.isfinite(track).all(axis=1)
            & np.isfinite(quaternion_norm)
            & (np.abs(quaternion_norm - 1.0) <= frozen.object_quaternion_norm_tolerance)
        )
        short, long = _repairable_runs(valid, time, frozen.maximum_repair_gap_seconds)
        if long:
            raise RetargetInputQualityError(
                f"UNRECOVERABLE_TRACKING_GAP:OBJECT_{object_index}:{long}"
            )
        rotations = Rotation.from_quat(track[valid, :4])
        # Normalize only valid source quaternions; this preserves their physical rotation.
        output[valid, object_index, :4] = rotations.as_quat()
        for start, end in short:
            key_times = time[[start - 1, end]]
            keys = Rotation.from_quat(output[[start - 1, end], object_index, :4])
            output[start:end, object_index, :4] = Slerp(key_times, keys)(time[start:end]).as_quat()
        output[:, object_index, 4:7] = _linear_fill(track[:, 4:7], short)
        reports.append(
            {
                "object_index": object_index,
                "valid_before": int(np.count_nonzero(valid)),
                "invalid_before": int(np.count_nonzero(~valid)),
                "short_invalid_gaps": [list(item) for item in short],
                "long_invalid_gaps": [list(item) for item in long],
                "repaired_frames": sorted(
                    index for start, end in short for index in range(start, end)
                ),
            }
        )
    return (output[:, 0] if squeeze else output), {"objects": reports}


def keypoint_frame_diagnostics(
    keypoints: np.ndarray, contract: RetargetInputQualityContractV1 | None = None
) -> dict[str, np.ndarray]:
    """Measure canonical keypoint axes without granting them production authority."""

    frozen = contract or RetargetInputQualityContractV1()
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (21, 3):
        raise ValueError("RETARGET_INPUT_KEYPOINT_SHAPE_INVALID")
    longitudinal = points[:, 9] - points[:, 0]
    lateral = points[:, 5] - points[:, 17]
    longitudinal_norm = np.linalg.norm(longitudinal, axis=1)
    lateral_norm = np.linalg.norm(lateral, axis=1)
    cross_norm = np.linalg.norm(np.cross(longitudinal, lateral), axis=1)
    denominator = np.maximum(longitudinal_norm * lateral_norm, np.finfo(np.float64).tiny)
    sine = cross_norm / denominator
    valid = (
        np.isfinite(points).all(axis=(1, 2))
        & (longitudinal_norm > frozen.wrist_axis_minimum_norm_m)
        & (lateral_norm > frozen.wrist_axis_minimum_norm_m)
        & (sine > frozen.wrist_axis_minimum_sine)
    )
    return {
        "longitudinal_norm_m": longitudinal_norm,
        "lateral_norm_m": lateral_norm,
        "axis_sine": sine,
        "valid": valid,
    }


def select_mano_primary_wrist_frames(
    mano_global_orient_aa: np.ndarray,
    mano_translation: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    contract: RetargetInputQualityContractV1 | None = None,
    reconstructed_wrist_pose: np.ndarray | None = None,
    keypoint_wrist_pose: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve wrist frames according to the frozen MANO-first priority."""

    frozen = contract or RetargetInputQualityContractV1()
    orient = np.asarray(mano_global_orient_aa, dtype=np.float64)
    translation = np.asarray(mano_translation, dtype=np.float64)
    if orient.ndim != 2 or orient.shape[1:] != (3,) or translation.shape != orient.shape:
        raise ValueError("RETARGET_INPUT_MANO_WRIST_SHAPE_INVALID")
    frames = np.broadcast_to(np.eye(4, dtype=np.float64), (len(orient), 4, 4)).copy()
    authority = np.full(len(orient), "RAW_TRACKING_QUALITY_FAILED", dtype="U64")
    direct = np.isfinite(orient).all(axis=1) & np.isfinite(translation).all(axis=1)
    if np.any(direct):
        frames[direct, :3, :3] = Rotation.from_rotvec(orient[direct]).as_matrix()
        frames[direct, :3, 3] = translation[direct]
        authority[direct] = "MANO_GLOBAL_WRIST_ORIENTATION"
    if reconstructed_wrist_pose is not None:
        reconstructed = np.asarray(reconstructed_wrist_pose, dtype=np.float64)
        valid = (authority == "RAW_TRACKING_QUALITY_FAILED") & np.isfinite(reconstructed).all(
            axis=(1, 2)
        )
        frames[valid] = reconstructed[valid]
        authority[valid] = "MANO_LAYER_RECONSTRUCTED_WRIST_FRAME"
    # Only short interior holes bounded by production-authority frames may
    # propagate. The same seconds-domain threshold governs every repair path.
    valid = authority != "RAW_TRACKING_QUALITY_FAILED"
    if timestamps is not None:
        time = np.asarray(timestamps, dtype=np.float64)
        if time.shape != (len(valid),) or not np.all(np.diff(time) > 0):
            raise ValueError("RETARGET_INPUT_WRIST_TIMESTAMPS_INVALID")
        short, _ = _repairable_runs(valid, time, frozen.maximum_repair_gap_seconds)
        for start, end in short:
            keys = Rotation.from_matrix(frames[[start - 1, end], :3, :3])
            frames[start:end, :3, :3] = Slerp(time[[start - 1, end]], keys)(
                time[start:end]
            ).as_matrix()
            frames[start:end, :3, 3] = _linear_fill(frames[:, :3, 3], [(start, end)])[start:end]
            authority[start:end] = "TEMPORALLY_PROPAGATED_VALID_WRIST_FRAME"
    if keypoint_wrist_pose is not None:
        diagnostic = np.asarray(keypoint_wrist_pose, dtype=np.float64)
        valid_diagnostic = (authority == "RAW_TRACKING_QUALITY_FAILED") & np.isfinite(
            diagnostic
        ).all(axis=(1, 2))
        frames[valid_diagnostic] = diagnostic[valid_diagnostic]
        authority[valid_diagnostic] = "KEYPOINT_DERIVED_WRIST_FRAME_DIAGNOSTIC_ONLY"
    return frames, authority


def bone_quality(
    tracked_keypoints: np.ndarray,
    mano_reconstructed_keypoints: np.ndarray,
    bone_parents: np.ndarray,
    bone_children: np.ndarray,
    contract: RetargetInputQualityContractV1 | None = None,
) -> dict[str, np.ndarray]:
    """Classify tracked-bone degeneration and MANO-parametric recovery."""

    frozen = contract or RetargetInputQualityContractV1()
    tracked = np.asarray(tracked_keypoints, dtype=np.float64)
    mano = np.asarray(mano_reconstructed_keypoints, dtype=np.float64)
    parents = np.asarray(bone_parents, dtype=np.int64)
    children = np.asarray(bone_children, dtype=np.int64)
    if tracked.shape != mano.shape or tracked.ndim != 3 or tracked.shape[2:] != (3,):
        raise ValueError("RETARGET_INPUT_BONE_TRACK_SHAPE_INVALID")
    tracked_lengths = np.linalg.norm(tracked[:, children] - tracked[:, parents], axis=-1)
    mano_lengths = np.linalg.norm(mano[:, children] - mano[:, parents], axis=-1)
    tracked_valid = np.isfinite(tracked_lengths).all(axis=1) & np.all(
        tracked_lengths > frozen.near_zero_bone_length_m, axis=1
    )
    mano_valid = np.isfinite(mano_lengths).all(axis=1) & np.all(
        mano_lengths > frozen.near_zero_bone_length_m, axis=1
    )
    return {
        "tracked_min_bone_length_m": np.min(tracked_lengths, axis=1),
        "mano_min_bone_length_m": np.min(mano_lengths, axis=1),
        "tracked_bones_valid": tracked_valid,
        "mano_bones_valid": mano_valid,
        "mano_parametric_recovery": (~tracked_valid) & mano_valid,
        "unrecoverable": ~mano_valid,
    }


def rotation_step_angles(rotation_matrices: np.ndarray) -> np.ndarray:
    """Return geodesic SO(3) step angles with frame zero set to zero."""

    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    result = np.zeros(len(matrices), dtype=np.float64)
    if len(matrices) > 1:
        relative = np.einsum("tji,tjk->tik", matrices[:-1], matrices[1:])
        result[1:] = Rotation.from_matrix(relative).magnitude()
    return result


__all__ = [
    "RetargetInputQualityContractV1",
    "RetargetInputQualityError",
    "bone_quality",
    "invalid_runs",
    "keypoint_frame_diagnostics",
    "repair_mano_pose",
    "repair_object_pose_qxyzw",
    "rotation_step_angles",
    "select_mano_primary_wrist_frames",
]
