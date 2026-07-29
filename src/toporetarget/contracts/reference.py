"""RobotReference v2 for playback and future reference-tracking consumers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import direct_zarr3_arrays, write_zarr3_group_direct
from toporetarget.geometry.se3 import (
    invert_transform,
    relative_transform,
    transform_points,
    validate_transform,
)

from .version import ROBOT_REFERENCE_V1, ROBOT_REFERENCE_V2


class RobotReferenceValidationError(ValueError):
    """Raised when a reference artifact does not satisfy the v2 contract."""


def _array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass
class RobotReferenceV2:
    """A frame-aligned robot reference in explicit robot-base coordinates."""

    qpos_reference: np.ndarray
    base_pose: np.ndarray
    object_pose_base: np.ndarray
    tracked_link_positions: np.ndarray
    timestamps: np.ndarray
    fps: float
    joint_order: tuple[str, ...]
    robot_hash: str
    dataset_provenance: dict[str, Any] = field(default_factory=dict)
    frame_indices: np.ndarray | None = None
    tracked_link_names: tuple[str, ...] = ()
    robot_keypoints_base: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    schema_version = ROBOT_REFERENCE_V2

    def __post_init__(self) -> None:
        self.qpos_reference = _array(self.qpos_reference)
        self.base_pose = _array(self.base_pose)
        self.object_pose_base = _array(self.object_pose_base)
        self.tracked_link_positions = _array(self.tracked_link_positions)
        self.timestamps = _array(self.timestamps).reshape(-1)
        if self.frame_indices is not None:
            self.frame_indices = _array(self.frame_indices, dtype=np.int64).reshape(-1)
        if self.robot_keypoints_base is not None:
            self.robot_keypoints_base = _array(self.robot_keypoints_base)
        self.joint_order = tuple(str(value) for value in self.joint_order)
        self.tracked_link_names = tuple(str(value) for value in self.tracked_link_names)
        self.dataset_provenance = dict(self.dataset_provenance)
        self.metadata = dict(self.metadata)

    @property
    def num_frames(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def q_reference(self) -> np.ndarray:
        """Short alias used by playback-oriented callers."""

        return self.qpos_reference

    @property
    def base_pose_scene(self) -> np.ndarray:
        return self.base_pose

    @property
    def object_pose_robot_base(self) -> np.ndarray:
        return self.object_pose_base

    @property
    def tracked_link_positions_base(self) -> np.ndarray:
        return self.tracked_link_positions

    def validate(self, *, raise_on_error: bool = True) -> dict[str, Any]:
        return validate_reference(self, raise_on_error=raise_on_error)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ROBOT_REFERENCE_V2,
            "fps": float(self.fps),
            "joint_order": list(self.joint_order),
            "robot_hash": self.robot_hash,
            "tracked_link_names": list(self.tracked_link_names),
            "dataset_provenance": self.dataset_provenance,
            "coordinate_frames": {
                "base_pose": "scene",
                "object_pose_base": "robot_base",
                "tracked_link_positions": "robot_base",
            },
            **self.metadata,
        }


def validate_reference(
    reference: RobotReferenceV2, *, raise_on_error: bool = True
) -> dict[str, Any]:
    """Validate shape, frame alignment, transforms, and provenance fields."""

    errors: list[str] = []
    t = reference.num_frames

    def shape(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    shape(reference.qpos_reference.ndim == 2, "qpos_reference must have shape [T,D]")
    shape(reference.base_pose.shape == (t, 4, 4), "base_pose must have shape [T,4,4]")
    shape(
        reference.object_pose_base.shape == (t, 4, 4),
        "object_pose_base must have shape [T,4,4]",
    )
    shape(
        reference.tracked_link_positions.ndim == 3
        and reference.tracked_link_positions.shape[0] == t
        and reference.tracked_link_positions.shape[2] == 3,
        "tracked_link_positions must have shape [T,L,3]",
    )
    shape(reference.timestamps.shape == (t,), "timestamps must have shape [T]")
    shape(bool(np.isfinite(reference.fps) and reference.fps > 0), "fps must be positive")
    shape(bool(reference.joint_order), "joint_order must not be empty")
    if reference.qpos_reference.ndim == 2:
        shape(
            reference.qpos_reference.shape[1] == len(reference.joint_order),
            "joint_order length must match qpos_reference DoF count",
        )
    if reference.frame_indices is not None:
        shape(reference.frame_indices.shape == (t,), "frame_indices must have shape [T]")
    if reference.tracked_link_names:
        shape(
            reference.tracked_link_positions.ndim == 3
            and len(reference.tracked_link_names) == reference.tracked_link_positions.shape[1],
            "tracked_link_names length must match link positions",
        )
    if reference.robot_keypoints_base is not None:
        shape(
            reference.robot_keypoints_base.ndim == 3
            and reference.robot_keypoints_base.shape[0] == t
            and reference.robot_keypoints_base.shape[2] == 3,
            "robot_keypoints_base must have shape [T,K,3]",
        )
    for name, value in (
        ("qpos_reference", reference.qpos_reference),
        ("base_pose", reference.base_pose),
        ("object_pose_base", reference.object_pose_base),
        ("tracked_link_positions", reference.tracked_link_positions),
        ("timestamps", reference.timestamps),
    ):
        if not np.all(np.isfinite(value)):
            errors.append(f"{name} contains NaN or Inf")
    if reference.robot_keypoints_base is not None and not np.all(
        np.isfinite(reference.robot_keypoints_base)
    ):
        errors.append("robot_keypoints_base contains NaN or Inf")
    if reference.timestamps.size > 1 and not np.all(np.diff(reference.timestamps) > 0):
        errors.append("timestamps must be strictly increasing")
    for name, value in (
        ("base_pose", reference.base_pose),
        ("object_pose_base", reference.object_pose_base),
    ):
        if value.shape == (t, 4, 4):
            try:
                validate_transform(value)
            except ValueError as exc:
                errors.append(f"{name}: {exc}")
    if not reference.robot_hash:
        errors.append("robot_hash must not be empty")
    if not isinstance(reference.dataset_provenance, dict) or not reference.dataset_provenance:
        errors.append("dataset_provenance must be a non-empty mapping")

    report = {
        "schema_version": ROBOT_REFERENCE_V2,
        "valid": not errors,
        "errors": errors,
        "num_frames": t,
        "dof_count": int(reference.qpos_reference.shape[1])
        if reference.qpos_reference.ndim == 2
        else None,
        "link_count": int(reference.tracked_link_positions.shape[1])
        if reference.tracked_link_positions.ndim == 3
        else None,
    }
    if errors and raise_on_error:
        raise RobotReferenceValidationError("; ".join(errors))
    return report


def _metadata_from_npz(value: np.ndarray) -> dict[str, Any]:
    if value.ndim != 0:
        raise RobotReferenceValidationError("metadata must be a scalar JSON string")
    parsed = json.loads(str(value.item()))
    if not isinstance(parsed, dict):
        raise RobotReferenceValidationError("metadata must decode to a mapping")
    return parsed


def from_v1_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    robot_hash: str | None = None,
    joint_order: tuple[str, ...] | None = None,
) -> RobotReferenceV2:
    """Convert the repository's v1 export arrays without changing qpos."""

    link_poses = arrays.get("robot_link_poses_scene", arrays.get("robot_link_poses"))
    if link_poses is None:
        raise RobotReferenceValidationError("v1 reference has no robot link poses")
    base_pose = _array(arrays["base_pose_scene"])
    object_pose_scene = _array(arrays["object_pose_scene"])
    link_poses = _array(link_poses)
    link_scene = link_poses[..., :3, 3]
    link_base = transform_points(invert_transform(base_pose), link_scene)
    object_base = relative_transform(base_pose, object_pose_scene)
    keypoints = arrays.get("robot_keypoints_scene")
    keypoints_base = (
        None
        if keypoints is None
        else transform_points(invert_transform(base_pose), _array(keypoints))
    )
    robot_name = str(metadata.get("robot", ""))
    resolved_hash = robot_hash or str(metadata.get("robot_hash", ""))
    if not resolved_hash:
        # A reference exported before Stage 11 may not carry a spec hash.  The
        # provenance remains explicit and validation fails closed until a hash
        # is supplied by the caller.
        resolved_hash = "unresolved-v1-robot-hash"
    fps = metadata.get("fps", metadata.get("native_fps"))
    if fps is None:
        fps = metadata.get("provenance", {}).get("native_fps", 1.0)
    qpos = _array(arrays["qpos"])
    order = joint_order or tuple(str(item) for item in metadata.get("joint_order", ()))
    if not order:
        order = tuple(f"joint_{index}" for index in range(qpos.shape[1]))
    provenance = dict(metadata.get("dataset_provenance", {}))
    provenance.setdefault("source_sequence", metadata.get("source_sequence"))
    provenance.setdefault("source_hash", metadata.get("source_hash"))
    provenance.setdefault("robot", robot_name)
    result = RobotReferenceV2(
        qpos_reference=qpos,
        base_pose=base_pose,
        object_pose_base=object_base,
        tracked_link_positions=link_base,
        timestamps=_array(arrays["timestamps"]),
        fps=float(fps),
        joint_order=order,
        robot_hash=resolved_hash,
        dataset_provenance=provenance,
        frame_indices=arrays.get("frame_indices"),
        tracked_link_names=tuple(str(item) for item in metadata.get("tracked_link_names", ())),
        robot_keypoints_base=keypoints_base,
        metadata={
            "source_schema_version": metadata.get("schema_version", ROBOT_REFERENCE_V1),
            "source_robot": robot_name,
            "source_metadata": metadata,
        },
    )
    result.validate()
    return result


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files if name != "metadata"}
        metadata = _metadata_from_npz(payload["metadata"]) if "metadata" in payload else {}
    return arrays, metadata


def _load_zarr(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    zarr_json = path / "zarr.json"
    if not zarr_json.is_file():
        raise RobotReferenceValidationError(f"reference Zarr group is missing zarr.json: {path}")
    group = json.loads(zarr_json.read_text(encoding="utf-8"))
    attributes = group.get("attributes", {})
    metadata_raw = attributes.get("metadata_json", "{}")
    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw)
    prefixed_root = path / "arrays"
    if prefixed_root.is_dir():
        names = [
            item.name
            for item in prefixed_root.iterdir()
            if item.is_dir() and (item / "zarr.json").is_file()
        ]
        arrays = direct_zarr3_arrays(path, names, array_prefix="arrays")
    else:
        names = [
            item.name for item in path.iterdir() if item.is_dir() and (item / "zarr.json").is_file()
        ]
        arrays = direct_zarr3_arrays(path, names, array_prefix="")
    return arrays, metadata


def load_robot_reference(path: str | Path) -> RobotReferenceV2:
    """Load v2 or losslessly convert a v1 NPZ/Zarr reference."""

    source = Path(path)
    arrays, metadata = _load_npz(source) if source.suffix == ".npz" else _load_zarr(source)
    if metadata.get("schema_version") == ROBOT_REFERENCE_V2:
        result = RobotReferenceV2(
            qpos_reference=arrays["qpos_reference"],
            base_pose=arrays["base_pose"],
            object_pose_base=arrays["object_pose_base"],
            tracked_link_positions=arrays["tracked_link_positions"],
            timestamps=arrays["timestamps"],
            fps=float(metadata["fps"]),
            joint_order=tuple(metadata["joint_order"]),
            robot_hash=str(metadata["robot_hash"]),
            dataset_provenance=dict(metadata["dataset_provenance"]),
            frame_indices=arrays.get("frame_indices"),
            tracked_link_names=tuple(metadata.get("tracked_link_names", ())),
            robot_keypoints_base=arrays.get("robot_keypoints_base"),
            metadata=metadata,
        )
        result.validate()
        return result
    if metadata.get("schema_version") not in {None, ROBOT_REFERENCE_V1}:
        raise RobotReferenceValidationError(
            f"unsupported robot reference schema: {metadata.get('schema_version')!r}"
        )
    return from_v1_arrays(arrays, metadata)


def save_robot_reference(
    reference: RobotReferenceV2,
    path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Write a validated v2 reference as NPZ or repository-compatible Zarr."""

    reference.validate()
    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"reference output exists; pass force=True: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "qpos_reference": reference.qpos_reference,
        "base_pose": reference.base_pose,
        "object_pose_base": reference.object_pose_base,
        "tracked_link_positions": reference.tracked_link_positions,
        "timestamps": reference.timestamps,
    }
    if reference.frame_indices is not None:
        arrays["frame_indices"] = reference.frame_indices
    if reference.robot_keypoints_base is not None:
        arrays["robot_keypoints_base"] = reference.robot_keypoints_base
    metadata = reference.as_metadata()
    if destination.suffix == ".npz":
        np.savez_compressed(
            destination,
            **arrays,  # type: ignore[arg-type]
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    else:
        write_zarr3_group_direct(
            destination,
            {
                "schema_version": ROBOT_REFERENCE_V2,
                "metadata_json": json.dumps(metadata, sort_keys=True),
            },
            arrays,
        )
    return destination


def migrate_reference_v1_to_v2(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    robot_hash: str | None = None,
    joint_order: tuple[str, ...] | None = None,
    force: bool = False,
) -> RobotReferenceV2:
    """Convert an existing v1 reference and optionally write a v2 copy."""

    arrays, metadata = (
        _load_npz(Path(source)) if Path(source).suffix == ".npz" else _load_zarr(Path(source))
    )
    reference = from_v1_arrays(
        arrays,
        metadata,
        robot_hash=robot_hash,
        joint_order=joint_order,
    )
    if destination is not None:
        save_robot_reference(reference, destination, force=force)
    return reference


__all__ = [
    "ROBOT_REFERENCE_V1",
    "ROBOT_REFERENCE_V2",
    "RobotReferenceV2",
    "RobotReferenceValidationError",
    "from_v1_arrays",
    "load_robot_reference",
    "migrate_reference_v1_to_v2",
    "save_robot_reference",
    "validate_reference",
]
