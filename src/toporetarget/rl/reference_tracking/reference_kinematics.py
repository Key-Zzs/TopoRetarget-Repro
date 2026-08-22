"""Versioned pose/time-primary kinematics for the Stage 16-D factor-8 reference.

The original PPO reference interpolated stored source twists independently from
the retimed pose.  This module deliberately reverses that authority: a source
pose plus timestamp sequence is materialized first and all runtime twists are
then derived from that resulting trajectory.  It is NumPy-only so it can be
audited and tested without Isaac Lab.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ppo26d_reference import (
    CONTROL_HZ,
    REFERENCE_FIELDS,
    REFERENCE_TIME_SCALE,
    RUNTIME_FRAMES,
    SOURCE_FRAMES,
    inspect_source_reference,
)

V2_IDENTIFIER = "Stage16DReferenceKinematicsV2"
RUNTIME_DT_S = 1.0 / CONTROL_HZ
QUATERNION_CONVENTION = "wxyz_active_right_handed_shortest_rotation"
ANGULAR_VELOCITY_CONVENTION = "world: [omega]_x = R_dot @ R_T"

_RUNTIME_FIELD_SHAPES = {
    "wrist_pose_translation_world_ref": (3,),
    "wrist_pose_quaternion_world_ref_wxyz": (4,),
    "wrist_twist_world_ref": (6,),
    "q_finger_ref": (20,),
    "qdot_finger_ref": (20,),
    "object_pose_translation_world_ref": (3,),
    "object_pose_quaternion_world_ref_wxyz": (4,),
    "object_twist_world_ref": (6,),
    "object_axis_points_world_ref": (6, 3),
    "tracked_link_positions_world_ref": (16, 3),
    "object_axis_points_wrist_ref": (6, 3),
    "tracked_link_positions_wrist_ref": (16, 3),
}


def sha256_file(path: Path) -> str:
    """Return a content hash without normalizing or mutating an artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _series(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "final": float(values[-1]),
    }


def _time_denominator(timestamps: np.ndarray, ndim: int) -> np.ndarray:
    return timestamps.reshape((timestamps.size,) + (1,) * (ndim - 1))


def normalize_quaternions_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a batch of WXYZ quaternions and reject degenerate entries."""

    values = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1.0e-12):
        raise ValueError("REFERENCE_QUATERNION_INVALID")
    return values / norms


def enforce_quaternion_sign_continuity_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Use shortest-arc signs while treating q and -q as one rotation."""

    result = normalize_quaternions_wxyz(quaternion).copy()
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternions_wxyz(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Return the WXYZ-independent rotation vector for matrices close to SO(3)."""

    matrices = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrices, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    skew_vector = np.stack(
        (
            matrices[..., 2, 1] - matrices[..., 1, 2],
            matrices[..., 0, 2] - matrices[..., 2, 0],
            matrices[..., 1, 0] - matrices[..., 0, 1],
        ),
        axis=-1,
    )
    result = np.zeros_like(skew_vector)
    ordinary = np.abs(np.sin(angle)) > 1.0e-8
    result[ordinary] = (
        skew_vector[ordinary] * (angle[ordinary] / (2.0 * np.sin(angle[ordinary])))[:, None]
    )
    # All Stage 16-D relative rotations stay far from pi.  Keeping the small
    # angle branch explicit makes identity/hold segments numerically stable.
    small = ~ordinary & (angle <= 1.0e-8)
    result[small] = 0.5 * skew_vector[small]
    if np.any(~ordinary & ~small):
        raise ValueError("REFERENCE_SO3_LOG_NEAR_PI_AMBIGUOUS")
    return result


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Exponentiate world-frame rotation vectors into rotation matrices."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(vector, axis=-1)
    skew = np.zeros(vector.shape[:-1] + (3, 3), dtype=np.float64)
    skew[..., 0, 1] = -vector[..., 2]
    skew[..., 0, 2] = vector[..., 1]
    skew[..., 1, 0] = vector[..., 2]
    skew[..., 1, 2] = -vector[..., 0]
    skew[..., 2, 0] = -vector[..., 1]
    skew[..., 2, 1] = vector[..., 0]
    angle2 = angle * angle
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(angle > 1.0e-8, np.sin(angle) / angle, 1.0 - angle2 / 6.0)
        b = np.where(
            angle > 1.0e-8,
            (1.0 - np.cos(angle)) / angle2,
            0.5 - angle2 / 24.0,
        )
    identity = np.broadcast_to(np.eye(3), skew.shape)
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def rotation_geodesic_error_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> np.ndarray:
    """Return q/-q-invariant SO(3) errors."""

    first = normalize_quaternions_wxyz(first_wxyz)
    second = normalize_quaternions_wxyz(second_wxyz)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def derive_linear_velocity(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Second-order centered / one-sided derivative along the time axis."""

    array = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if array.shape[0] != times.size or times.size < 3 or not np.all(np.diff(times) > 0.0):
        raise ValueError("REFERENCE_TIMESTAMP_DERIVATIVE_CONTRACT_INVALID")
    result = np.empty_like(array, dtype=np.float64)
    result[1:-1] = (array[2:] - array[:-2]) / _time_denominator(times[2:] - times[:-2], array.ndim)
    first_step = times[1] - times[0]
    last_step = times[-1] - times[-2]
    result[0] = (-3.0 * array[0] + 4.0 * array[1] - array[2]) / (2.0 * first_step)
    result[-1] = (3.0 * array[-1] - 4.0 * array[-2] + array[-3]) / (2.0 * last_step)
    return result


def derive_angular_velocity_world_wxyz(
    quaternion_wxyz: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    """Use SO(3) logs for world angular velocity, never quaternion component FD."""

    quaternion = enforce_quaternion_sign_continuity_wxyz(quaternion_wxyz)
    rotation = quaternion_to_matrix_wxyz(quaternion)
    times = np.asarray(timestamps, dtype=np.float64)
    if rotation.shape[0] != times.size or times.size < 3 or not np.all(np.diff(times) > 0.0):
        raise ValueError("REFERENCE_TIMESTAMP_ANGULAR_DERIVATIVE_CONTRACT_INVALID")
    result = np.empty((times.size, 3), dtype=np.float64)
    result[1:-1] = (
        so3_log(rotation[2:] @ np.swapaxes(rotation[:-2], -1, -2))
        / (times[2:] - times[:-2])[:, None]
    )
    result[0] = so3_log(rotation[1:2] @ np.swapaxes(rotation[:1], -1, -2))[0] / (
        times[1] - times[0]
    )
    result[-1] = so3_log(rotation[-1:] @ np.swapaxes(rotation[-2:-1], -1, -2))[0] / (
        times[-1] - times[-2]
    )
    return result


def angular_velocity_body_from_world(
    quaternion_wxyz: np.ndarray, angular_velocity_world: np.ndarray
) -> np.ndarray:
    """Make the only allowed world→body conversion explicit."""

    rotation = quaternion_to_matrix_wxyz(quaternion_wxyz)
    return np.einsum("tji,tj->ti", rotation, np.asarray(angular_velocity_world))


def _interpolate_linear(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    left = np.floor(coordinates).astype(np.int64).clip(0, values.shape[0] - 2)
    alpha = (coordinates - left).reshape((coordinates.size,) + (1,) * (values.ndim - 1))
    alpha[-1] = 1.0
    return (1.0 - alpha) * values[left] + alpha * values[left + 1]


def _interpolate_hermite(
    values: np.ndarray, derivatives: np.ndarray, coordinates: np.ndarray, *, source_dt_s: float
) -> np.ndarray:
    left = np.floor(coordinates).astype(np.int64).clip(0, values.shape[0] - 2)
    alpha = (coordinates - left).reshape((coordinates.size,) + (1,) * (values.ndim - 1))
    alpha[-1] = 1.0
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    return (
        (2.0 * alpha3 - 3.0 * alpha2 + 1.0) * values[left]
        + (alpha3 - 2.0 * alpha2 + alpha) * source_dt_s * derivatives[left]
        + (-2.0 * alpha3 + 3.0 * alpha2) * values[left + 1]
        + (alpha3 - alpha2) * source_dt_s * derivatives[left + 1]
    )


def _interpolate_quaternion(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    values = enforce_quaternion_sign_continuity_wxyz(values)
    left = np.floor(coordinates).astype(np.int64).clip(0, values.shape[0] - 2)
    alpha = (coordinates - left)[:, None]
    alpha[-1] = 1.0
    start, end = values[left], values[left + 1].copy()
    end[np.sum(start * end, axis=-1) < 0.0] *= -1.0
    return enforce_quaternion_sign_continuity_wxyz((1.0 - alpha) * start + alpha * end)


def _load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name != "metadata"
        }
        metadata = json.loads(str(archive["metadata"].item()))
    return arrays, metadata


def _check_source_shapes(arrays: dict[str, np.ndarray]) -> int:
    fields = set(REFERENCE_FIELDS) | {
        "timestamps",
        "object_axis_points_wrist_ref",
        "tracked_link_positions_wrist_ref",
    }
    missing = sorted(fields - set(arrays))
    if missing:
        raise ValueError(f"REFERENCE_V2_SOURCE_FIELDS_MISSING: {missing}")
    timestamps = np.asarray(arrays["timestamps"])
    if timestamps.ndim != 1 or timestamps.size < 3:
        raise ValueError("REFERENCE_V2_SOURCE_TIMESTAMP_SHAPE_INVALID")
    source_frames = int(timestamps.size)
    for field, shape in _RUNTIME_FIELD_SHAPES.items():
        if arrays[field].shape != (source_frames, *shape):
            raise ValueError(f"REFERENCE_V2_SOURCE_SHAPE_INVALID: {field}")
    return source_frames


def source_contract_timestamps(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    """Normalize the declared Nx20 Hz source timeline without changing its NPZ.

    The frozen source has one historical final timestamp of 1.966666... s even
    though its declared sampling contract is 41 samples at 20 Hz.  V2 records
    that immutable discrepancy and uses the explicit source contract requested
    for this stage: ``t_source[k] = k * 0.05``.
    """

    stored = np.asarray(arrays["timestamps"], dtype=np.float64)
    expected = np.arange(stored.size, dtype=np.float64) * RUNTIME_DT_S
    if stored.ndim != 1 or stored.size < 3 or not np.all(np.diff(stored) > 0.0):
        raise ValueError("REFERENCE_SOURCE_TIMESTAMP_NONMONOTONIC")
    discrepancy = np.abs(stored - expected)
    return expected, {
        "stored_input_timestamps_immutable": True,
        "stored_final_timestamp_s": float(stored[-1]),
        "declared_final_timestamp_s": float(expected[-1]),
        "max_abs_discrepancy_s": float(discrepancy.max()),
        "normalized_for_v2": bool(discrepancy.max() > 1.0e-7),
        "source_time_definition": "t_source[k] = k * 0.05 s",
    }


def _twist_from_pose(
    position: np.ndarray, quaternion: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        (
            derive_linear_velocity(position, timestamps),
            derive_angular_velocity_world_wxyz(quaternion, timestamps),
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class Stage16DReferenceTimeV2:
    identifier: str = "Stage16DReferenceTimeV2"
    runtime_samples: int = RUNTIME_FRAMES
    runtime_dt_s: float = RUNTIME_DT_S
    runtime_duration_s: float = 16.0
    source_samples: int = SOURCE_FRAMES
    source_dt_s: float = RUNTIME_DT_S
    time_scale: int = REFERENCE_TIME_SCALE

    def __post_init__(self) -> None:
        expected_samples = (self.source_samples - 1) * self.time_scale + 1
        expected_duration = (self.runtime_samples - 1) * self.runtime_dt_s
        if self.source_samples < 3 or self.runtime_samples != expected_samples:
            raise ValueError("REFERENCE_V2_TIME_DOMAIN_INVALID")
        if not np.isclose(self.runtime_duration_s, expected_duration, atol=1.0e-12):
            raise ValueError("REFERENCE_V2_RUNTIME_DURATION_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_v1_reference(source_path: Path, v1_path: Path) -> dict[str, Any]:
    """Audit V1 in-place; this function never writes V1 or its source."""

    source = inspect_source_reference(source_path)
    source_frames = int(source["source_frames"])
    runtime_frames = (source_frames - 1) * REFERENCE_TIME_SCALE + 1
    source_arrays, _ = _load_archive(source_path)
    v1_arrays, v1_metadata = _load_archive(v1_path)
    if v1_metadata.get("source_sha256") != source["sha256"]:
        raise RuntimeError("REFERENCE_SOURCE_HASH_DRIFT")
    times = np.asarray(v1_arrays["timestamps"], dtype=np.float64)
    if times.shape != (runtime_frames,):
        raise ValueError("REFERENCE_V1_RUNTIME_LENGTH_INVALID")
    indices = np.arange(source_frames) * REFERENCE_TIME_SCALE
    source_times, source_time_audit = source_contract_timestamps(source_arrays)
    object_source_twist = _twist_from_pose(
        source_arrays["object_pose_translation_world_ref"],
        source_arrays["object_pose_quaternion_world_ref_wxyz"],
        source_times,
    )
    object_v1_twist = _twist_from_pose(
        v1_arrays["object_pose_translation_world_ref"],
        v1_arrays["object_pose_quaternion_world_ref_wxyz"],
        times,
    )
    stored_source = source_arrays["object_twist_world_ref"]
    stored_v1 = v1_arrays["object_twist_world_ref"]
    source_linear_error = np.linalg.norm(stored_source[:, :3] - object_source_twist[:, :3], axis=-1)
    source_angular_error = np.linalg.norm(
        stored_source[:, 3:] - object_source_twist[:, 3:], axis=-1
    )
    v1_linear_error = np.linalg.norm(stored_v1[:, :3] - object_v1_twist[:, :3], axis=-1)
    v1_angular_error = np.linalg.norm(stored_v1[:, 3:] - object_v1_twist[:, 3:], axis=-1)
    key_position_error = np.linalg.norm(
        v1_arrays["object_pose_translation_world_ref"][indices]
        - source_arrays["object_pose_translation_world_ref"],
        axis=-1,
    )
    key_rotation_error = rotation_geodesic_error_rad(
        v1_arrays["object_pose_quaternion_world_ref_wxyz"][indices],
        source_arrays["object_pose_quaternion_world_ref_wxyz"],
    )
    return {
        "schema_version": "Stage16DReferenceV1AuditV2",
        "source": source,
        "v1": {"path": str(v1_path.resolve()), "sha256": sha256_file(v1_path)},
        "source_key_preservation": {
            "object_position_max_m": float(key_position_error.max()),
            "object_rotation_max_rad": float(key_rotation_error.max()),
        },
        "source_timestamp_contract": source_time_audit,
        "source_stored_twist_vs_pose_time": {
            "linear_error_mps": _series(source_linear_error),
            "angular_error_radps": _series(source_angular_error),
            "terminal_linear_derivative_inconsistent": bool(source_linear_error[-1] > 1.0e-5),
        },
        "v1_stored_twist_vs_v1_pose_time": {
            "linear_error_mps": _series(v1_linear_error),
            "angular_error_radps": _series(v1_angular_error),
        },
        "pose_rebuild_required": bool(source_linear_error.max() > 1.0e-5),
        "root_cause": (
            "V1 used source stored linear twist as Hermite interpolation tangents and then "
            "linearly interpolated that independent twist.  The source terminal linear tangent "
            "is inconsistent with source pose/time, so V2 rebuilds the position interpolation "
            "from pose-derived tangents and derives runtime twist from final pose/time."
        ),
    }


def materialize_reference_kinematics_v2(
    source_path: Path, v1_path: Path, destination: Path
) -> dict[str, Any]:
    """Build a new V2 artifact while preserving V1/source bytes unchanged."""

    v1_audit = inspect_v1_reference(source_path, v1_path)
    source_arrays, source_metadata = _load_archive(source_path)
    source_frames = _check_source_shapes(source_arrays)
    runtime_frames = (source_frames - 1) * REFERENCE_TIME_SCALE + 1
    source_times, source_time_audit = source_contract_timestamps(source_arrays)
    coordinates = np.arange(runtime_frames, dtype=np.float64) / REFERENCE_TIME_SCALE
    runtime_times = np.arange(runtime_frames, dtype=np.float64) * RUNTIME_DT_S

    def hermite_from_pose(field: str) -> np.ndarray:
        values = source_arrays[field]
        return _interpolate_hermite(
            values,
            derive_linear_velocity(values, source_times),
            coordinates,
            source_dt_s=RUNTIME_DT_S,
        )

    wrist_position = hermite_from_pose("wrist_pose_translation_world_ref")
    wrist_quaternion = _interpolate_quaternion(
        source_arrays["wrist_pose_quaternion_world_ref_wxyz"], coordinates
    )
    object_position = hermite_from_pose("object_pose_translation_world_ref")
    object_quaternion = _interpolate_quaternion(
        source_arrays["object_pose_quaternion_world_ref_wxyz"], coordinates
    )
    finger_q = hermite_from_pose("q_finger_ref")
    payload: dict[str, np.ndarray] = {
        "timestamps": runtime_times,
        "wrist_pose_translation_world_ref": wrist_position,
        "wrist_pose_quaternion_world_ref_wxyz": wrist_quaternion,
        "wrist_twist_world_ref": _twist_from_pose(wrist_position, wrist_quaternion, runtime_times),
        "q_finger_ref": finger_q,
        "qdot_finger_ref": derive_linear_velocity(finger_q, runtime_times),
        "object_pose_translation_world_ref": object_position,
        "object_pose_quaternion_world_ref_wxyz": object_quaternion,
        "object_twist_world_ref": _twist_from_pose(
            object_position, object_quaternion, runtime_times
        ),
        "object_axis_points_world_ref": hermite_from_pose("object_axis_points_world_ref"),
        "tracked_link_positions_world_ref": hermite_from_pose("tracked_link_positions_world_ref"),
        "object_axis_points_wrist_ref": hermite_from_pose("object_axis_points_wrist_ref"),
        "tracked_link_positions_wrist_ref": hermite_from_pose("tracked_link_positions_wrist_ref"),
    }
    payload["object_angular_velocity_body_ref"] = angular_velocity_body_from_world(
        object_quaternion, payload["object_twist_world_ref"][:, 3:]
    )
    if not all(np.isfinite(value).all() for value in payload.values()):
        raise FloatingPointError("REFERENCE_TWIST_NONFINITE")
    metadata = {
        **source_metadata,
        "schema_version": V2_IDENTIFIER,
        "reference_kinematics_version": 2,
        "parent_v1_sha256": sha256_file(v1_path),
        "source_sha256": sha256_file(source_path),
        "source_frames": source_frames,
        "runtime_samples": runtime_frames,
        "control_hz": CONTROL_HZ,
        "time_scale": REFERENCE_TIME_SCALE,
        "time_contract": Stage16DReferenceTimeV2(
            runtime_samples=runtime_frames,
            runtime_duration_s=(runtime_frames - 1) * RUNTIME_DT_S,
            source_samples=source_frames,
        ).as_dict(),
        "source_timestamp_normalization": source_time_audit,
        "interpolation": {
            "translation": "pose-derived-tangent cubic Hermite",
            "rotation": "shortest-arc normalized linear quaternion interpolation",
            "finger_q": "pose-derived-tangent cubic Hermite",
            "auxiliary_positions": "pose-derived-tangent cubic Hermite",
        },
        "derivative_scheme": {
            "linear": "second-order centered with second-order one-sided endpoints",
            "angular_world": "SO3 log centered with one-sided Lie-group endpoints",
        },
        "quaternion_convention": QUATERNION_CONVENTION,
        "angular_velocity_convention": ANGULAR_VELOCITY_CONVENTION,
        "parent_v1_pose_rebuild_required": v1_audit["pose_rebuild_required"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    archive_payload: dict[str, Any] = {
        key: value.astype(np.float64) for key, value in payload.items()
    }
    archive_payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(destination, **archive_payload)
    return {
        "identifier": V2_IDENTIFIER,
        "destination": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
        "source_sha256": sha256_file(source_path),
        "parent_v1_sha256": sha256_file(v1_path),
        "pose_rebuild_required": v1_audit["pose_rebuild_required"],
    }


def _integral_consistency(
    position: np.ndarray, quaternion: np.ndarray, twist: np.ndarray, timestamps: np.ndarray
) -> dict[str, object]:
    reconstructed_position = np.empty_like(position)
    reconstructed_position[0] = position[0]
    reconstructed_position[1:] = position[0] + np.cumsum(
        0.5 * (twist[:-1, :3] + twist[1:, :3]) * np.diff(timestamps)[:, None], axis=0
    )
    position_error = np.linalg.norm(reconstructed_position - position, axis=-1)
    rotation = quaternion_to_matrix_wxyz(quaternion)
    reconstructed_rotation = np.empty_like(rotation)
    reconstructed_rotation[0] = rotation[0]
    for index, dt in enumerate(np.diff(timestamps)):
        omega = 0.5 * (twist[index, 3:] + twist[index + 1, 3:])
        reconstructed_rotation[index + 1] = (
            so3_exp((omega * dt)[None])[0] @ reconstructed_rotation[index]
        )
    rotation_error = np.linalg.norm(
        so3_log(reconstructed_rotation @ np.swapaxes(rotation, -1, -2)), axis=-1
    )
    return {
        "linear_reconstruction_error_m": _series(position_error),
        "rotation_reconstruction_error_rad": _series(rotation_error),
        "pass": bool(
            position_error[-1] <= 1.0e-3
            and position_error.mean() <= 2.0e-4
            and rotation_error[-1] <= 1.0e-2
            and rotation_error.mean() <= 5.0e-4
        ),
        "frozen_gate": {
            "final_translation_m_max": 1.0e-3,
            "mean_translation_m_max": 2.0e-4,
            "final_rotation_rad_max": 1.0e-2,
            "mean_rotation_rad_max": 5.0e-4,
        },
    }


def _factor8_scaling(
    source_arrays: dict[str, np.ndarray], v2_arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    source_times, _ = source_contract_timestamps(source_arrays)
    native_twist = _twist_from_pose(
        source_arrays["object_pose_translation_world_ref"],
        source_arrays["object_pose_quaternion_world_ref_wxyz"],
        source_times,
    )
    runtime_twist = v2_arrays["object_twist_world_ref"]
    source_frames = int(source_arrays["timestamps"].size)
    indices = np.arange(1, source_frames - 1) * REFERENCE_TIME_SCALE
    result: dict[str, object] = {
        "definition": (
            "runtime pose/time derived twist at runtime index 8*k vs native pose/time twist[k] / 8"
        ),
        "excluded_source_indices": [0, source_frames - 1],
        "frozen_gate": {"median_relative_error_max": 0.05, "p95_relative_error_max": 0.15},
    }
    passes: list[bool] = []
    for name, slice_ in (("linear", slice(0, 3)), ("angular_world", slice(3, 6))):
        expected = native_twist[1:-1, slice_] / REFERENCE_TIME_SCALE
        observed = runtime_twist[indices, slice_]
        magnitude = np.linalg.norm(expected, axis=-1)
        valid = magnitude > 1.0e-3
        relative = np.linalg.norm(observed[valid] - expected[valid], axis=-1) / magnitude[valid]
        if not relative.size:
            raise RuntimeError("REFERENCE_FACTOR8_SCALE_NO_NONZERO_SAMPLES")
        metric = _series(relative)
        passed = bool(metric["median"] <= 0.05 and metric["p95"] <= 0.15)
        result[name] = {
            "valid_sample_count": int(relative.size),
            "relative_error": metric,
            "pass": passed,
        }
        passes.append(passed)
    result["pass"] = all(passes)
    return result


def _terminal_semantics(
    source_arrays: dict[str, np.ndarray], v2_arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    source_position = source_arrays["object_pose_translation_world_ref"]
    source_quaternion = source_arrays["object_pose_quaternion_world_ref_wxyz"]
    final_translation = float(np.linalg.norm(source_position[-1] - source_position[-2]))
    final_rotation = float(
        rotation_geodesic_error_rad(source_quaternion[-2:-1], source_quaternion[-1:])[0]
    )
    final_twist = v2_arrays["object_twist_world_ref"][-1]
    moving = final_translation > 1.0e-3 or final_rotation > 3.0e-3
    return {
        "classification": "TERMINAL_POSE_STILL_MOVING" if moving else "TERMINAL_POSE_HOLD",
        "source_final_key_displacement_m": final_translation,
        "source_final_key_rotation_rad": final_rotation,
        "v_ref_final_world_mps": final_twist[:3].tolist(),
        "v_ref_final_norm_mps": float(np.linalg.norm(final_twist[:3])),
        "omega_ref_final_world_radps": final_twist[3:].tolist(),
        "omega_ref_final_norm_radps": float(np.linalg.norm(final_twist[3:])),
        "decision_rule": "STILL_MOVING if final source translation >1 mm or rotation >3 mrad",
    }


def qualify_reference_kinematics_v2(
    source_path: Path, v1_path: Path, v2_path: Path
) -> dict[str, Any]:
    """Return a fail-closed V2 qualification report from immutable inputs."""

    v1_audit = inspect_v1_reference(source_path, v1_path)
    source_arrays, _ = _load_archive(source_path)
    v1_arrays, _ = _load_archive(v1_path)
    v2_arrays, metadata = _load_archive(v2_path)
    source_frames = _check_source_shapes(source_arrays)
    runtime_frames = (source_frames - 1) * REFERENCE_TIME_SCALE + 1
    if (
        metadata.get("schema_version") != V2_IDENTIFIER
        or metadata.get("reference_kinematics_version") != 2
    ):
        raise ValueError("REFERENCE_V2_VERSION_METADATA_INVALID")
    if metadata.get("source_sha256") != sha256_file(source_path):
        raise RuntimeError("REFERENCE_SOURCE_HASH_DRIFT")
    if metadata.get("parent_v1_sha256") != sha256_file(v1_path):
        raise RuntimeError("REFERENCE_V2_PARENT_HASH_MISMATCH")
    timestamps = v2_arrays["timestamps"]
    expected_timestamps = np.arange(runtime_frames, dtype=np.float64) * RUNTIME_DT_S
    timestamp_pass = bool(
        timestamps.shape == (runtime_frames,)
        and np.all(np.diff(timestamps) > 0.0)
        and np.array_equal(timestamps, expected_timestamps)
    )
    indices = np.arange(source_frames) * REFERENCE_TIME_SCALE
    key_checks: dict[str, dict[str, float | bool]] = {}
    for name in (
        "wrist_pose_translation_world_ref",
        "object_pose_translation_world_ref",
        "q_finger_ref",
    ):
        error = np.max(np.abs(v2_arrays[name][indices] - source_arrays[name]))
        key_checks[name] = {"max_abs_error": float(error), "pass": bool(error <= 1.0e-7)}
    for name in (
        "wrist_pose_quaternion_world_ref_wxyz",
        "object_pose_quaternion_world_ref_wxyz",
    ):
        error = rotation_geodesic_error_rad(v2_arrays[name][indices], source_arrays[name])
        key_checks[name] = {
            "max_geodesic_error_rad": float(error.max()),
            "pass": bool(error.max() <= 1.0e-5),
        }
    quaternion_checks: dict[str, dict[str, float | bool]] = {}
    for name in (
        "wrist_pose_quaternion_world_ref_wxyz",
        "object_pose_quaternion_world_ref_wxyz",
    ):
        quaternion = v2_arrays[name]
        norms = np.linalg.norm(quaternion, axis=-1)
        dots = np.sum(quaternion[:-1] * quaternion[1:], axis=-1)
        quaternion_checks[name] = {
            "max_norm_error": float(np.max(np.abs(norms - 1.0))),
            "minimum_adjacent_dot": float(dots.min()),
            "pass": bool(np.max(np.abs(norms - 1.0)) <= 1.0e-10 and dots.min() >= 0.0),
        }
    position = v2_arrays["object_pose_translation_world_ref"]
    quaternion = v2_arrays["object_pose_quaternion_world_ref_wxyz"]
    stored_twist = v2_arrays["object_twist_world_ref"]
    derived_twist = _twist_from_pose(position, quaternion, timestamps)
    linear_error = np.linalg.norm(stored_twist[:, :3] - derived_twist[:, :3], axis=-1)
    angular_error = np.linalg.norm(stored_twist[:, 3:] - derived_twist[:, 3:], axis=-1)
    linear_scale = np.maximum(np.linalg.norm(derived_twist[:, :3], axis=-1), 1.0e-6)
    angular_scale = np.maximum(np.linalg.norm(derived_twist[:, 3:], axis=-1), 1.0e-6)
    linear_pass = bool(np.all(linear_error <= 1.0e-8 + 1.0e-6 * linear_scale))
    angular_pass = bool(np.all(angular_error <= 1.0e-8 + 1.0e-6 * angular_scale))
    integral = _integral_consistency(position, quaternion, stored_twist, timestamps)
    factor8 = _factor8_scaling(source_arrays, v2_arrays)
    terminal = _terminal_semantics(source_arrays, v2_arrays)
    pose_change = {
        "object_position_max_m": float(
            np.max(
                np.abs(
                    v2_arrays["object_pose_translation_world_ref"]
                    - v1_arrays["object_pose_translation_world_ref"]
                )
            )
        ),
        "object_rotation_max_rad": float(
            rotation_geodesic_error_rad(
                v2_arrays["object_pose_quaternion_world_ref_wxyz"],
                v1_arrays["object_pose_quaternion_world_ref_wxyz"],
            ).max()
        ),
    }
    pose_change["pose_changed"] = bool(
        pose_change["object_position_max_m"] > 1.0e-10
        or pose_change["object_rotation_max_rad"] > 1.0e-10
    )
    checks = {
        "source_key_preservation": all(bool(value["pass"]) for value in key_checks.values()),
        "timestamps": timestamp_pass,
        "quaternion": all(bool(value["pass"]) for value in quaternion_checks.values()),
        "finite": all(np.isfinite(value).all() for value in v2_arrays.values()),
        "linear_fd_consistency": linear_pass,
        "angular_so3_consistency": angular_pass,
        "factor8_scaling": bool(factor8["pass"]),
        "integral_consistency": bool(integral["pass"]),
        "world_angular_convention": metadata.get("angular_velocity_convention")
        == ANGULAR_VELOCITY_CONVENTION,
    }
    return {
        "schema_version": "ReferenceKinematicsQualificationV2",
        "clip": source_path.stem.removesuffix(".world_wrist.stage16"),
        "reference_kinematics_version": 2,
        "source": {"path": str(source_path.resolve()), "sha256": sha256_file(source_path)},
        "parent_v1": {"path": str(v1_path.resolve()), "sha256": sha256_file(v1_path)},
        "v2": {"path": str(v2_path.resolve()), "sha256": sha256_file(v2_path)},
        "v1_audit": v1_audit,
        "checks": checks,
        "source_key_preservation": key_checks,
        "timestamp_contract": {
            **Stage16DReferenceTimeV2(
                runtime_samples=runtime_frames,
                runtime_duration_s=(runtime_frames - 1) * RUNTIME_DT_S,
                source_samples=source_frames,
            ).as_dict(),
            "strictly_monotonic": bool(np.all(np.diff(timestamps) > 0.0)),
            "exact_runtime_mapping": timestamp_pass,
            "runtime_index_for_source_k": "8*k",
        },
        "quaternion_contract": quaternion_checks,
        "linear_velocity_contract": {
            "convention": "world linear velocity",
            "finite_difference_error_mps": _series(linear_error),
            "frozen_tolerance": "absolute 1e-8 m/s plus relative 1e-6",
            "pass": linear_pass,
        },
        "angular_velocity_contract": {
            "convention": ANGULAR_VELOCITY_CONVENTION,
            "finite_difference_error_radps": _series(angular_error),
            "frozen_tolerance": "absolute 1e-8 rad/s plus relative 1e-6",
            "pass": angular_pass,
        },
        "factor8_scaling": factor8,
        "integral_consistency": integral,
        "terminal_reference_semantics": terminal,
        "v1_to_v2_pose_change": pose_change,
        "status": "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED"
        if all(checks.values())
        else "STAGE16D_REFERENCE_KINEMATICS_V2_BLOCKED",
    }


__all__ = [
    "ANGULAR_VELOCITY_CONVENTION",
    "QUATERNION_CONVENTION",
    "RUNTIME_DT_S",
    "Stage16DReferenceTimeV2",
    "V2_IDENTIFIER",
    "angular_velocity_body_from_world",
    "derive_angular_velocity_world_wxyz",
    "derive_linear_velocity",
    "enforce_quaternion_sign_continuity_wxyz",
    "inspect_v1_reference",
    "materialize_reference_kinematics_v2",
    "qualify_reference_kinematics_v2",
    "rotation_geodesic_error_rad",
    "sha256_file",
    "source_contract_timestamps",
    "so3_exp",
    "so3_log",
]
