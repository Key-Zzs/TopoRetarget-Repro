"""Deterministic 20 Hz interpolation for dynamic reference clips."""

from __future__ import annotations

import numpy as np

from .axis_points import object_axis_points_from_poses
from .contracts import Stage16ReferenceClip

REFERENCE_RESAMPLER_ID = "reference_resampler_20hz_v1"


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert one rotation matrix to a normalized xyzw quaternion."""

    value = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return np.asarray(
            [
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    index = int(np.argmax(np.diag(value)))
    if index == 0:
        scale = 2.0 * np.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2])
        return np.asarray(
            [
                0.25 * scale,
                (value[0, 1] + value[1, 0]) / scale,
                (value[0, 2] + value[2, 0]) / scale,
                (value[2, 1] - value[1, 2]) / scale,
            ]
        )
    if index == 1:
        scale = 2.0 * np.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2])
        return np.asarray(
            [
                (value[0, 1] + value[1, 0]) / scale,
                0.25 * scale,
                (value[1, 2] + value[2, 1]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
            ]
        )
    scale = 2.0 * np.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1])
    return np.asarray(
        [
            (value[0, 2] + value[2, 0]) / scale,
            (value[1, 2] + value[2, 1]) / scale,
            0.25 * scale,
            (value[1, 0] - value[0, 1]) / scale,
        ]
    )


def _matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def shortest_arc_slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc normalized quaternion SLERP; inputs and output are xyzw."""

    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        value = q0 + fraction * (q1 - q0)
        return value / np.linalg.norm(value)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sine = np.sin(theta)
    value = np.sin((1.0 - fraction) * theta) / sine * q0 + np.sin(fraction * theta) / sine * q1
    return value / np.linalg.norm(value)


def _interpolate_linear(times: np.ndarray, values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    result = np.stack(
        [np.interp(new_times, times, flat[:, index]) for index in range(flat.shape[1])], axis=1
    )
    return result.reshape((new_times.size, *values.shape[1:]))


def _interpolate_joints(
    times: np.ndarray,
    values: np.ndarray,
    new_times: np.ndarray,
    periodic_joint_indices: tuple[int, ...],
) -> np.ndarray:
    """Interpolate bounded joints directly and unwrap only declared periodic joints."""

    source = np.asarray(values, dtype=np.float64).copy()
    for index in periodic_joint_indices:
        if index < 0 or index >= source.shape[1]:
            raise ValueError(f"periodic joint index out of range: {index}")
        source[:, index] = np.unwrap(source[:, index])
    return _interpolate_linear(times, source, new_times)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm vector for a proper rotation matrix."""

    value = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vector = np.asarray(
        [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]]
    )
    if angle < 1e-8:
        return 0.5 * vector
    if np.pi - angle < 1e-6:
        # The usual ``angle / sin(angle)`` form is singular at 180 degrees.
        # Pick a deterministic axis from the symmetric part instead.
        axis = np.sqrt(np.maximum((np.diag(value) + 1.0) * 0.5, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-8:
            for other in range(3):
                if other != largest:
                    axis[other] = (value[largest, other] + value[other, largest]) / (
                        4.0 * axis[largest]
                    )
        return angle * axis / np.linalg.norm(axis)
    return angle / (2.0 * np.sin(angle)) * vector


def _angular_velocity_from_poses(poses: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Central-difference base-frame angular velocity with one-sided endpoints."""

    rotations = np.asarray(poses, dtype=np.float64)[:, :3, :3]
    times = np.asarray(timestamps, dtype=np.float64)
    result = np.empty((rotations.shape[0], 3), dtype=np.float64)
    result[0] = _rotation_vector(rotations[1] @ rotations[0].T) / (times[1] - times[0])
    result[-1] = _rotation_vector(rotations[-1] @ rotations[-2].T) / (times[-1] - times[-2])
    for index in range(1, rotations.shape[0] - 1):
        result[index] = _rotation_vector(rotations[index + 1] @ rotations[index - 1].T) / (
            times[index + 1] - times[index - 1]
        )
    return result


def resample_reference_20hz(
    clip: Stage16ReferenceClip,
    *,
    target_hz: float = 20.0,
    periodic_joint_indices: tuple[int, ...] = (),
    axis_length_m: float = 0.05,
) -> Stage16ReferenceClip:
    """Resample a dynamic clip without duplicated timestamps and keep the final time."""

    clip.validate()
    if target_hz <= 0.0:
        raise ValueError("target_hz must be positive")
    interval = 1.0 / target_hz
    start, finish = float(clip.timestamps[0]), float(clip.timestamps[-1])
    new_times = np.arange(start, finish, interval, dtype=np.float64)
    if new_times.size == 0 or not np.isclose(new_times[-1], finish, atol=1e-12):
        new_times = np.append(new_times, finish)
    q = _interpolate_joints(clip.timestamps, clip.q_finger_ref, new_times, periodic_joint_indices)
    links = _interpolate_linear(clip.timestamps, clip.tracked_link_positions_base_ref, new_times)
    poses = np.empty((new_times.size, 4, 4), dtype=np.float64)
    rotations = np.asarray(
        [_quaternion_from_matrix(value[:3, :3]) for value in clip.object_pose_base_ref]
    )
    translations = _interpolate_linear(
        clip.timestamps, clip.object_pose_base_ref[:, :3, 3], new_times
    )
    for index, time_value in enumerate(new_times):
        upper = min(
            int(np.searchsorted(clip.timestamps, time_value, side="right")), clip.frame_count - 1
        )
        lower = max(upper - 1, 0)
        denom = clip.timestamps[upper] - clip.timestamps[lower]
        fraction = 0.0 if denom <= 0.0 else float((time_value - clip.timestamps[lower]) / denom)
        poses[index] = np.eye(4)
        poses[index, :3, :3] = _matrix_from_quaternion(
            shortest_arc_slerp(rotations[lower], rotations[upper], fraction)
        )
        poses[index, :3, 3] = translations[index]
    axes = object_axis_points_from_poses(poses, axis_length_m=axis_length_m)
    qdot = np.gradient(q, new_times, axis=0, edge_order=1)
    linear_velocity = np.gradient(translations, new_times, axis=0, edge_order=1)
    result = Stage16ReferenceClip(
        timestamps=new_times,
        q_finger_ref=q,
        object_pose_base_ref=poses,
        object_axis_points_base_ref=axes,
        tracked_link_positions_base_ref=links,
        joint_order=clip.joint_order,
        tracked_link_names=clip.tracked_link_names,
        provenance={
            **clip.provenance,
            "resampler": REFERENCE_RESAMPLER_ID,
            "source_hash": clip.content_hash(),
        },
        qdot_ref=qdot,
        object_velocity_ref=np.concatenate(
            [linear_velocity, _angular_velocity_from_poses(poses, new_times)], axis=1
        ),
        reference_indices=np.arange(new_times.size, dtype=np.int64),
        metadata={
            **clip.metadata,
            "target_hz": target_hz,
            "periodic_joint_indices": list(periodic_joint_indices),
            "axis_resampling": "recomputed_from_resampled_pose",
            "link_resampling": "linear_fallback_no_fk_callable",
        },
    )
    result.validate(expected_hz=target_hz if new_times.size > 2 else None)
    return result


__all__ = ["REFERENCE_RESAMPLER_ID", "resample_reference_20hz", "shortest_arc_slerp"]
