"""World-frame wrist-and-finger Stage-16B reference contract.

This module is deliberately separate from :mod:`toporetarget.rl.contracts`.
The latter remains the preserved paper-oriented, base-relative, finger-only
profile.  ``WorldWristFingerReferenceV1`` is an explicitly labelled
engineering extension that keeps the Stage-12 world wrist trajectory needed
to make an approach phase dynamically actionable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.contracts.reference import RobotReferenceV2
from toporetarget.geometry.se3 import (
    invert_transform,
    relative_transform,
    rotation_geodesic_error,
    transform_points,
    validate_transform,
)

from .axis_points import OBJECT_AXIS_PROFILE_ID, object_axis_points_from_poses
from .references import central_difference
from .tracked_links import TRACKED_LINK_PROFILE_ID, TRACKED_LINKS_WUJI_RH, select_tracked_links

WORLD_WRIST_REFERENCE_SCHEMA = "toporetarget.stage16.world_wrist_finger_reference.v1"
WORLD_WRIST_REFERENCE_PROFILE = "world_wrist_finger_residual_v1"
WORLD_WRIST_QUATERNION_CONVENTION = "wxyz_active_right_handed_shortest_rotation"
WORLD_WRIST_RESAMPLER_ID = "world_wrist_reference_resampler_20hz_v1"


class WorldWristFingerReferenceValidationError(ValueError):
    """Raised when the Stage-16B world-reference contract is invalid."""


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _json_scalar(value: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True), dtype=np.str_)


def quaternion_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized WXYZ quaternion with deterministic sign (w >= 0)."""

    matrix = _array(rotation)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3,3]")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        value = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            value = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            value = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            value = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    value /= np.linalg.norm(value)
    return value if value[0] >= 0.0 else -value


def matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return a rotation matrix from one normalized WXYZ quaternion."""

    value = _array(quaternion).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("quaternion must be non-zero and finite")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Principal SO(3) logarithm, including a deterministic near-pi branch."""

    matrix = _array(rotation)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )
    if angle < 1e-10:
        return 0.5 * skew
    if np.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        return angle * axis / max(float(np.linalg.norm(axis)), 1e-12)
    return angle * skew / (2.0 * np.sin(angle))


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """SO(3) exponential map using the right-handed Rodrigues convention."""

    vector = _array(rotation_vector).reshape(3)
    angle = float(np.linalg.norm(vector))
    cross = np.asarray(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )
    if angle < 1e-10:
        return np.eye(3) + cross
    return (
        np.eye(3)
        + np.sin(angle) / angle * cross
        + (1.0 - np.cos(angle)) / angle**2 * (cross @ cross)
    )


def se3_exp_local(translation: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
    """Residual transform with translation and rotation expressed in local frame."""

    result = np.eye(4)
    result[:3, :3] = so3_exp(rotation_vector)
    result[:3, 3] = _array(translation).reshape(3)
    return result


def _angular_velocity_world(poses: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    rotations = _array(poses)[:, :3, :3]
    times = _array(timestamps).reshape(-1)
    result = np.empty((times.size, 3), dtype=np.float64)
    for index in range(times.size):
        lower, upper = (
            (0, 1)
            if index == 0
            else (index - 1, index)
            if index == times.size - 1
            else (index - 1, index + 1)
        )
        result[index] = so3_log(rotations[upper] @ rotations[lower].T) / (
            times[upper] - times[lower]
        )
    return result


def _interpolate_linear(times: np.ndarray, values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    flat = _array(values).reshape(values.shape[0], -1)
    result = np.stack(
        [np.interp(new_times, times, flat[:, index]) for index in range(flat.shape[1])], axis=1
    )
    return result.reshape((new_times.size, *values.shape[1:]))


def _slerp_wxyz(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _array(left).reshape(4)
    q1 = _array(right).reshape(4)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    result = (
        np.sin((1.0 - fraction) * theta) / np.sin(theta) * q0
        + np.sin(fraction * theta) / np.sin(theta) * q1
    )
    return result / np.linalg.norm(result)


def _interpolate_poses(poses: np.ndarray, times: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    source = _array(poses)
    translations = _interpolate_linear(times, source[:, :3, 3], new_times)
    quaternions = np.asarray([quaternion_wxyz_from_matrix(pose[:3, :3]) for pose in source])
    result = np.broadcast_to(np.eye(4), (new_times.size, 4, 4)).copy()
    for index, time_value in enumerate(new_times):
        upper = min(int(np.searchsorted(times, time_value, side="right")), times.size - 1)
        lower = max(upper - 1, 0)
        duration = times[upper] - times[lower]
        fraction = 0.0 if duration <= 0.0 else float((time_value - times[lower]) / duration)
        result[index, :3, :3] = matrix_from_quaternion_wxyz(
            _slerp_wxyz(quaternions[lower], quaternions[upper], fraction)
        )
        result[index, :3, 3] = translations[index]
    return result


@dataclass
class WorldWristFingerReferenceV1:
    """Validated 20 Hz world-frame reference for the Stage-16B extension."""

    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    wrist_pose_world_ref: np.ndarray
    wrist_twist_world_ref: np.ndarray
    q_finger_ref: np.ndarray
    qdot_finger_ref: np.ndarray
    object_pose_world_ref: np.ndarray
    object_twist_world_ref: np.ndarray
    object_axis_points_world_ref: np.ndarray
    tracked_link_positions_world_ref: np.ndarray
    object_pose_wrist_ref: np.ndarray
    object_axis_points_wrist_ref: np.ndarray
    tracked_link_positions_wrist_ref: np.ndarray
    joint_order: tuple[str, ...]
    tracked_link_names: tuple[str, ...]
    provenance: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version = WORLD_WRIST_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        self.timestamps = _array(self.timestamps).reshape(-1)
        self.source_frame_indices = _array(self.source_frame_indices, dtype=np.int64).reshape(-1)
        for name in (
            "wrist_pose_world_ref",
            "wrist_twist_world_ref",
            "q_finger_ref",
            "qdot_finger_ref",
            "object_pose_world_ref",
            "object_twist_world_ref",
            "object_axis_points_world_ref",
            "tracked_link_positions_world_ref",
            "object_pose_wrist_ref",
            "object_axis_points_wrist_ref",
            "tracked_link_positions_wrist_ref",
        ):
            setattr(self, name, _array(getattr(self, name)))
        self.joint_order = tuple(str(value) for value in self.joint_order)
        self.tracked_link_names = tuple(str(value) for value in self.tracked_link_names)
        self.provenance = dict(self.provenance)
        self.metadata = dict(self.metadata)

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.size)

    @property
    def dof_count(self) -> int:
        return int(self.q_finger_ref.shape[1])

    @property
    def control_hz(self) -> float:
        return 0.0 if self.frame_count < 2 else float(1.0 / np.median(np.diff(self.timestamps)))

    @property
    def wrist_translation_world_ref(self) -> np.ndarray:
        return self.wrist_pose_world_ref[:, :3, 3]

    @property
    def wrist_quaternion_world_ref_wxyz(self) -> np.ndarray:
        return np.asarray(
            [quaternion_wxyz_from_matrix(pose[:3, :3]) for pose in self.wrist_pose_world_ref]
        )

    @property
    def object_translation_world_ref(self) -> np.ndarray:
        return self.object_pose_world_ref[:, :3, 3]

    @property
    def object_quaternion_world_ref_wxyz(self) -> np.ndarray:
        return np.asarray(
            [quaternion_wxyz_from_matrix(pose[:3, :3]) for pose in self.object_pose_world_ref]
        )

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.timestamps,
            self.source_frame_indices,
            self.wrist_pose_world_ref,
            self.q_finger_ref,
            self.object_pose_world_ref,
            self.tracked_link_positions_world_ref,
        ):
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        digest.update(json.dumps(self.provenance, sort_keys=True).encode())
        return digest.hexdigest()

    def validate(
        self,
        *,
        expected_hz: float | None = 20.0,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        frames = self.frame_count
        dof = self.dof_count if self.q_finger_ref.ndim == 2 else -1
        links = len(self.tracked_link_names)
        expected = {
            "source_frame_indices": (frames,),
            "wrist_pose_world_ref": (frames, 4, 4),
            "wrist_twist_world_ref": (frames, 6),
            "q_finger_ref": (frames, dof),
            "qdot_finger_ref": (frames, dof),
            "object_pose_world_ref": (frames, 4, 4),
            "object_twist_world_ref": (frames, 6),
            "object_axis_points_world_ref": (frames, 6, 3),
            "tracked_link_positions_world_ref": (frames, links, 3),
            "object_pose_wrist_ref": (frames, 4, 4),
            "object_axis_points_wrist_ref": (frames, 6, 3),
            "tracked_link_positions_wrist_ref": (frames, links, 3),
        }
        if frames < 2:
            errors.append("world reference requires at least two frames")
        if dof != len(self.joint_order) or dof != 20:
            errors.append("world reference must preserve exactly the 20-D Wuji joint order")
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                errors.append(f"{name} must have shape {shape}")
        values = [self.timestamps, *(getattr(self, name) for name in expected)]
        if any(not np.isfinite(value).all() for value in values):
            errors.append("world reference contains NaN or Inf")
        if frames > 1 and not np.all(np.diff(self.timestamps) > 0.0):
            errors.append("timestamps must be strictly increasing without duplicates")
        if not self.provenance:
            errors.append("provenance is required")
        for name in ("wrist_pose_world_ref", "object_pose_world_ref", "object_pose_wrist_ref"):
            try:
                validate_transform(getattr(self, name))
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
        quaternions = np.concatenate(
            [self.wrist_quaternion_world_ref_wxyz, self.object_quaternion_world_ref_wxyz], axis=0
        )
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-8):
            errors.append("world/reference quaternions are not normalized")
        if (
            expected_hz is not None
            and frames > 2
            and not np.isclose(self.control_hz, expected_hz, rtol=0.0, atol=1e-8)
        ):
            errors.append(f"cadence {self.control_hz} Hz does not equal {expected_hz} Hz")
        reconstructed_object = relative_transform(
            self.wrist_pose_world_ref, self.object_pose_world_ref
        )
        reconstructed_axes = transform_points(
            invert_transform(self.wrist_pose_world_ref), self.object_axis_points_world_ref
        )
        reconstructed_links = transform_points(
            invert_transform(self.wrist_pose_world_ref), self.tracked_link_positions_world_ref
        )
        position_error = float(
            np.max(
                np.linalg.norm(
                    reconstructed_object[:, :3, 3] - self.object_pose_wrist_ref[:, :3, 3], axis=1
                )
            )
        )
        rotation_error_deg = float(
            np.degrees(
                np.max(rotation_geodesic_error(reconstructed_object, self.object_pose_wrist_ref))
            )
        )
        axis_error = float(np.max(np.abs(reconstructed_axes - self.object_axis_points_wrist_ref)))
        link_error = float(
            np.max(np.abs(reconstructed_links - self.tracked_link_positions_wrist_ref))
        )
        if (
            position_error > 1e-6
            or rotation_error_deg > 1e-5
            or axis_error > 1e-6
            or link_error > 1e-6
        ):
            errors.append("world-to-wrist reconstruction exceeds fixed Stage-16B tolerance")
        if joint_lower is not None or joint_upper is not None:
            if joint_lower is None or joint_upper is None:
                errors.append("both joint bounds are required for joint-limit validation")
            else:
                lower, upper = _array(joint_lower), _array(joint_upper)
                if lower.shape != (dof,) or upper.shape != (dof,):
                    errors.append("joint bounds do not match the 20-D reference")
                elif np.any(self.q_finger_ref < lower - 1e-9) or np.any(
                    self.q_finger_ref > upper + 1e-9
                ):
                    errors.append("finger reference violates MuJoCo joint limits")
        report = {
            "schema_version": self.schema_version,
            "reference_profile": WORLD_WRIST_REFERENCE_PROFILE,
            "valid": not errors,
            "errors": errors,
            "frames": frames,
            "dof_count": dof,
            "link_count": links,
            "control_hz": self.control_hz,
            "quaternion_convention": WORLD_WRIST_QUATERNION_CONVENTION,
            "relative_reconstruction": {
                "translation_max_error_m": position_error,
                "rotation_max_error_deg": rotation_error_deg,
                "axis_max_error_m": axis_error,
                "link_max_error_m": link_error,
            },
            "content_hash": self.content_hash(),
        }
        if errors:
            raise WorldWristFingerReferenceValidationError("; ".join(errors))
        return report

    def to_npz(self, path: str | Path) -> Path:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.schema_version,
            "reference_profile": WORLD_WRIST_REFERENCE_PROFILE,
            "reference_frame": "world_scene",
            "joint_order": list(self.joint_order),
            "tracked_link_names": list(self.tracked_link_names),
            "quaternion_convention": WORLD_WRIST_QUATERNION_CONVENTION,
            "units": {"translation": "m", "angles": "rad", "time": "s"},
            "provenance": self.provenance,
            "metadata": self.metadata,
            "content_hash": self.content_hash(),
        }
        np.savez_compressed(
            destination,
            timestamps=self.timestamps,
            source_frame_indices=self.source_frame_indices,
            T_world_wrist_ref=self.wrist_pose_world_ref,
            wrist_pose_translation_world_ref=self.wrist_translation_world_ref,
            wrist_pose_quaternion_world_ref_wxyz=self.wrist_quaternion_world_ref_wxyz,
            wrist_twist_world_ref=self.wrist_twist_world_ref,
            q_finger_ref=self.q_finger_ref,
            qdot_finger_ref=self.qdot_finger_ref,
            T_world_object_ref=self.object_pose_world_ref,
            object_pose_translation_world_ref=self.object_translation_world_ref,
            object_pose_quaternion_world_ref_wxyz=self.object_quaternion_world_ref_wxyz,
            object_twist_world_ref=self.object_twist_world_ref,
            object_axis_points_world_ref=self.object_axis_points_world_ref,
            tracked_link_positions_world_ref=self.tracked_link_positions_world_ref,
            T_wrist_object_ref=self.object_pose_wrist_ref,
            object_axis_points_wrist_ref=self.object_axis_points_wrist_ref,
            tracked_link_positions_wrist_ref=self.tracked_link_positions_wrist_ref,
            metadata=_json_scalar(metadata),
        )
        return destination

    @classmethod
    def from_npz(cls, path: str | Path) -> WorldWristFingerReferenceV1:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            result = cls(
                timestamps=payload["timestamps"],
                source_frame_indices=payload["source_frame_indices"],
                wrist_pose_world_ref=payload["T_world_wrist_ref"],
                wrist_twist_world_ref=payload["wrist_twist_world_ref"],
                q_finger_ref=payload["q_finger_ref"],
                qdot_finger_ref=payload["qdot_finger_ref"],
                object_pose_world_ref=payload["T_world_object_ref"],
                object_twist_world_ref=payload["object_twist_world_ref"],
                object_axis_points_world_ref=payload["object_axis_points_world_ref"],
                tracked_link_positions_world_ref=payload["tracked_link_positions_world_ref"],
                object_pose_wrist_ref=payload["T_wrist_object_ref"],
                object_axis_points_wrist_ref=payload["object_axis_points_wrist_ref"],
                tracked_link_positions_wrist_ref=payload["tracked_link_positions_wrist_ref"],
                joint_order=tuple(metadata["joint_order"]),
                tracked_link_names=tuple(metadata["tracked_link_names"]),
                provenance=dict(metadata["provenance"]),
                metadata=dict(metadata.get("metadata", {})),
            )
        result.validate()
        return result


def resample_world_wrist_reference(
    reference: WorldWristFingerReferenceV1, *, target_hz: float = 20.0
) -> WorldWristFingerReferenceV1:
    """Resample all world quantities together, retaining the final source time."""

    reference.validate(expected_hz=None)
    if target_hz <= 0.0:
        raise ValueError("target_hz must be positive")
    interval = 1.0 / target_hz
    start, finish = float(reference.timestamps[0]), float(reference.timestamps[-1])
    timestamps = np.arange(start, finish, interval, dtype=np.float64)
    if timestamps.size == 0 or not np.isclose(timestamps[-1], finish, atol=1e-12):
        timestamps = np.append(timestamps, finish)
    wrist = _interpolate_poses(reference.wrist_pose_world_ref, reference.timestamps, timestamps)
    obj = _interpolate_poses(reference.object_pose_world_ref, reference.timestamps, timestamps)
    q = _interpolate_linear(reference.timestamps, reference.q_finger_ref, timestamps)
    links = _interpolate_linear(
        reference.timestamps, reference.tracked_link_positions_world_ref, timestamps
    )
    nearest = np.abs(reference.timestamps[:, None] - timestamps[None, :]).argmin(axis=0)
    qdot = central_difference(q, timestamps)
    wrist_twist = np.concatenate(
        [
            central_difference(wrist[:, :3, 3], timestamps),
            _angular_velocity_world(wrist, timestamps),
        ],
        axis=1,
    )
    object_twist = np.concatenate(
        [central_difference(obj[:, :3, 3], timestamps), _angular_velocity_world(obj, timestamps)],
        axis=1,
    )
    axes_world = object_axis_points_from_poses(obj)
    wrist_inverse = invert_transform(wrist)
    result = WorldWristFingerReferenceV1(
        timestamps=timestamps,
        source_frame_indices=reference.source_frame_indices[nearest],
        wrist_pose_world_ref=wrist,
        wrist_twist_world_ref=wrist_twist,
        q_finger_ref=q,
        qdot_finger_ref=qdot,
        object_pose_world_ref=obj,
        object_twist_world_ref=object_twist,
        object_axis_points_world_ref=axes_world,
        tracked_link_positions_world_ref=links,
        object_pose_wrist_ref=relative_transform(wrist, obj),
        object_axis_points_wrist_ref=transform_points(wrist_inverse, axes_world),
        tracked_link_positions_wrist_ref=transform_points(wrist_inverse, links),
        joint_order=reference.joint_order,
        tracked_link_names=reference.tracked_link_names,
        provenance={
            **reference.provenance,
            "resampler": WORLD_WRIST_RESAMPLER_ID,
            "source_world_reference_hash": reference.content_hash(),
        },
        metadata={**reference.metadata, "target_hz": target_hz},
    )
    result.validate(expected_hz=target_hz)
    return result


def export_world_wrist_reference(
    reference: RobotReferenceV2,
    *,
    source_hashes: dict[str, str],
    engineering_assumptions: list[str] | None = None,
    tracked_link_profile: tuple[str, ...] = TRACKED_LINKS_WUJI_RH,
    resample_to_hz: float = 20.0,
) -> WorldWristFingerReferenceV1:
    """Convert an accepted Stage-12 reference directly into the v1 world contract."""

    reference.validate()
    wrist = _array(reference.base_pose)
    object_world = wrist @ _array(reference.object_pose_base)
    links_wrist = select_tracked_links(
        reference.tracked_link_positions, reference.tracked_link_names, profile=tracked_link_profile
    )
    links_world = transform_points(wrist, links_wrist)
    axes_world = object_axis_points_from_poses(object_world)
    wrist_twist = np.concatenate(
        [
            central_difference(wrist[:, :3, 3], reference.timestamps),
            _angular_velocity_world(wrist, reference.timestamps),
        ],
        axis=1,
    )
    object_twist = np.concatenate(
        [
            central_difference(object_world[:, :3, 3], reference.timestamps),
            _angular_velocity_world(object_world, reference.timestamps),
        ],
        axis=1,
    )
    source = WorldWristFingerReferenceV1(
        timestamps=reference.timestamps,
        source_frame_indices=np.asarray(reference.frame_indices, dtype=np.int64),
        wrist_pose_world_ref=wrist,
        wrist_twist_world_ref=wrist_twist,
        q_finger_ref=reference.qpos_reference,
        qdot_finger_ref=central_difference(reference.qpos_reference, reference.timestamps),
        object_pose_world_ref=object_world,
        object_twist_world_ref=object_twist,
        object_axis_points_world_ref=axes_world,
        tracked_link_positions_world_ref=links_world,
        object_pose_wrist_ref=reference.object_pose_base,
        object_axis_points_wrist_ref=transform_points(invert_transform(wrist), axes_world),
        tracked_link_positions_wrist_ref=links_wrist,
        joint_order=reference.joint_order,
        tracked_link_names=tracked_link_profile,
        provenance={
            "source_schema": reference.schema_version,
            "source_stage12_path": reference.dataset_provenance.get("source_final_artifact"),
            "source_hashes": dict(source_hashes),
            "robot_hash": reference.robot_hash,
            "dataset_provenance": reference.dataset_provenance,
            "reference_profile": WORLD_WRIST_REFERENCE_PROFILE,
            "reference_frame": "world_scene",
            "axis_profile": OBJECT_AXIS_PROFILE_ID,
            "tracked_link_profile": TRACKED_LINK_PROFILE_ID,
            "quaternion_convention": WORLD_WRIST_QUATERNION_CONVENTION,
            "engineering_assumptions": engineering_assumptions
            or ["ENGINEERING_ASSUMPTION_ZERO_GRAVITY_NO_GROUND"],
        },
        metadata={
            "frame_conventions": {
                "T_world_wrist_ref": "Stage12 base_pose_scene",
                "T_world_object_ref": "canonical scene object pose",
                "relative_features": "inverse(T_world_wrist_ref) times world feature",
            },
            "units": "m_rad_s",
            "accepted_stage12_final": True,
        },
    )
    source.validate(expected_hz=None)
    return resample_world_wrist_reference(source, target_hz=resample_to_hz)


__all__ = [
    "WORLD_WRIST_QUATERNION_CONVENTION",
    "WORLD_WRIST_REFERENCE_PROFILE",
    "WORLD_WRIST_REFERENCE_SCHEMA",
    "WorldWristFingerReferenceV1",
    "WorldWristFingerReferenceValidationError",
    "export_world_wrist_reference",
    "matrix_from_quaternion_wxyz",
    "quaternion_wxyz_from_matrix",
    "resample_world_wrist_reference",
    "se3_exp_local",
    "so3_exp",
    "so3_log",
]
