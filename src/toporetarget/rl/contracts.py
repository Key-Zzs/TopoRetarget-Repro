"""Versioned, validated reference clips for Stage 16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

STAGE16_REFERENCE_SCHEMA = "toporetarget.stage16_reference_clip.v1"


class Stage16ReferenceValidationError(ValueError):
    """Raised when a Stage16ReferenceClip fails its fail-closed contract."""


def _as_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _json_scalar(mapping: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(mapping, sort_keys=True), dtype=np.str_)


@dataclass
class Stage16ReferenceClip:
    """A base-frame robot/object reference at an explicitly recorded cadence."""

    timestamps: np.ndarray
    q_finger_ref: np.ndarray
    object_pose_base_ref: np.ndarray
    object_axis_points_base_ref: np.ndarray
    tracked_link_positions_base_ref: np.ndarray
    joint_order: tuple[str, ...]
    tracked_link_names: tuple[str, ...]
    provenance: dict[str, Any]
    qdot_ref: np.ndarray | None = None
    object_velocity_ref: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version = STAGE16_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        self.timestamps = _as_array(self.timestamps).reshape(-1)
        self.q_finger_ref = _as_array(self.q_finger_ref)
        self.object_pose_base_ref = _as_array(self.object_pose_base_ref)
        self.object_axis_points_base_ref = _as_array(self.object_axis_points_base_ref)
        self.tracked_link_positions_base_ref = _as_array(self.tracked_link_positions_base_ref)
        self.qdot_ref = None if self.qdot_ref is None else _as_array(self.qdot_ref)
        self.object_velocity_ref = (
            None if self.object_velocity_ref is None else _as_array(self.object_velocity_ref)
        )
        self.reference_indices = (
            np.arange(self.timestamps.size, dtype=np.int64)
            if self.reference_indices is None
            else _as_array(self.reference_indices, dtype=np.int64).reshape(-1)
        )
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
        if self.frame_count < 2:
            return 0.0
        return float(1.0 / np.median(np.diff(self.timestamps)))

    def _validated_reference_indices(self) -> np.ndarray:
        """Return post-init indices with a type-safe guard for static analysis."""

        if self.reference_indices is None:  # pragma: no cover - __post_init__ establishes it
            raise RuntimeError("reference_indices were not initialized")
        return cast(np.ndarray, self.reference_indices)

    def validate(self, *, expected_hz: float | None = None) -> dict[str, Any]:
        errors: list[str] = []
        count = self.frame_count
        if count < 2:
            errors.append("STATIC_REFERENCE_NOT_RL_ELIGIBLE: dynamic frames must be >= 2")
        if self.q_finger_ref.shape != (count, len(self.joint_order)):
            errors.append("q_finger_ref shape does not match frame or joint order")
        if self.object_pose_base_ref.shape != (count, 4, 4):
            errors.append("object_pose_base_ref must have shape [T,4,4]")
        if self.object_axis_points_base_ref.shape != (count, 6, 3):
            errors.append("object_axis_points_base_ref must have shape [T,6,3]")
        expected_link_shape = (count, len(self.tracked_link_names), 3)
        if self.tracked_link_positions_base_ref.shape != expected_link_shape:
            errors.append("tracked_link_positions_base_ref shape does not match link profile")
        if self.qdot_ref is not None and self.qdot_ref.shape != self.q_finger_ref.shape:
            errors.append("qdot_ref must match q_finger_ref")
        if self.object_velocity_ref is not None and self.object_velocity_ref.shape != (count, 6):
            errors.append("object_velocity_ref must have shape [T,6]")
        if self._validated_reference_indices().shape != (count,):
            errors.append("reference_indices must have shape [T]")
        if not self.provenance:
            errors.append("provenance is required")
        if count > 1 and not np.all(np.diff(self.timestamps) > 0.0):
            errors.append("timestamps must be strictly increasing")
        arrays = [
            self.timestamps,
            self.q_finger_ref,
            self.object_pose_base_ref,
            self.object_axis_points_base_ref,
            self.tracked_link_positions_base_ref,
        ]
        arrays.extend(
            value for value in (self.qdot_ref, self.object_velocity_ref) if value is not None
        )
        if any(not np.isfinite(value).all() for value in arrays):
            errors.append("all numeric values must be finite")
        if self.object_pose_base_ref.shape == (count, 4, 4):
            rotation = self.object_pose_base_ref[:, :3, :3]
            identity = np.eye(3)
            if not np.allclose(rotation @ np.swapaxes(rotation, 1, 2), identity, atol=2e-5):
                errors.append("object rotations are not orthonormal")
            if not np.allclose(np.linalg.det(rotation), 1.0, atol=2e-5):
                errors.append("object rotations do not have determinant +1")
        hz = self.control_hz
        if (
            expected_hz is not None
            and count > 1
            and not np.isclose(hz, expected_hz, rtol=0, atol=1e-8)
        ):
            errors.append(f"reference cadence {hz} Hz does not equal {expected_hz} Hz")
        report = {
            "schema_version": self.schema_version,
            "valid": not errors,
            "errors": errors,
            "frames": count,
            "dof_count": self.dof_count if self.q_finger_ref.ndim == 2 else None,
            "link_count": len(self.tracked_link_names),
            "control_hz": hz,
            "source_hash": self.content_hash(),
        }
        if errors:
            raise Stage16ReferenceValidationError("; ".join(errors))
        return report

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.timestamps,
            self.q_finger_ref,
            self.object_pose_base_ref,
            self.object_axis_points_base_ref,
            self.tracked_link_positions_base_ref,
            self._validated_reference_indices(),
        ):
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        digest.update(json.dumps(self.provenance, sort_keys=True).encode())
        return digest.hexdigest()

    def to_npz(self, path: str | Path) -> Path:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.schema_version,
            "joint_order": list(self.joint_order),
            "tracked_link_names": list(self.tracked_link_names),
            "provenance": self.provenance,
            "metadata": self.metadata,
            "content_hash": self.content_hash(),
        }
        payload: dict[str, np.ndarray] = {
            "timestamps": self.timestamps,
            "q_finger_ref": self.q_finger_ref,
            "object_pose_base_ref": self.object_pose_base_ref,
            "object_axis_points_base_ref": self.object_axis_points_base_ref,
            "tracked_link_positions_base_ref": self.tracked_link_positions_base_ref,
            "reference_indices": self._validated_reference_indices(),
            "metadata": _json_scalar(metadata),
        }
        if self.qdot_ref is not None:
            payload["qdot_ref"] = self.qdot_ref
        if self.object_velocity_ref is not None:
            payload["object_velocity_ref"] = self.object_velocity_ref
        savez_compressed: Any = np.savez_compressed
        savez_compressed(destination, **payload)
        return destination

    @classmethod
    def from_npz(cls, path: str | Path) -> Stage16ReferenceClip:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            result = cls(
                timestamps=payload["timestamps"],
                q_finger_ref=payload["q_finger_ref"],
                object_pose_base_ref=payload["object_pose_base_ref"],
                object_axis_points_base_ref=payload["object_axis_points_base_ref"],
                tracked_link_positions_base_ref=payload["tracked_link_positions_base_ref"],
                joint_order=tuple(metadata["joint_order"]),
                tracked_link_names=tuple(metadata["tracked_link_names"]),
                provenance=dict(metadata["provenance"]),
                qdot_ref=payload["qdot_ref"] if "qdot_ref" in payload.files else None,
                object_velocity_ref=(
                    payload["object_velocity_ref"]
                    if "object_velocity_ref" in payload.files
                    else None
                ),
                reference_indices=payload["reference_indices"],
                metadata=dict(metadata.get("metadata", {})),
            )
        result.validate()
        return result


__all__ = [
    "STAGE16_REFERENCE_SCHEMA",
    "Stage16ReferenceClip",
    "Stage16ReferenceValidationError",
]
