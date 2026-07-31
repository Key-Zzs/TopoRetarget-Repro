"""Robot-independent canonical HOI data structures.

The schema deliberately stores scene-frame geometry as the lossless primary
representation.  Wrist-relative and object-relative values are derived by
the frame helpers instead of replacing the global trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from toporetarget.geometry.se3 import (
    object_to_scene,
    scene_to_object,
    scene_to_wrist,
    validate_transform,
    wrist_to_scene,
)

SCHEMA_VERSION = "toporetarget.hoi.v1"
_EPS = 1e-6


class HOIValidationError(ValueError):
    """Raised when a canonical HOI sequence violates the schema contract."""


def _array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _finite_with_mask(values: np.ndarray, valid: np.ndarray | None, name: str) -> None:
    if valid is None:
        if not np.all(np.isfinite(values)):
            raise HOIValidationError(f"{name} contains NaN or Inf")
        return
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim == 1:
        if values.ndim < 1 or values.shape[0] != mask.shape[0]:
            raise HOIValidationError(f"{name} valid mask shape {mask.shape} is incompatible")
        mask = mask.reshape((mask.shape[0],) + (1,) * (values.ndim - 1))
    elif mask.shape != values.shape[: mask.ndim]:
        raise HOIValidationError(f"{name} valid mask shape {mask.shape} is incompatible")
    else:
        mask = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
    expanded = np.broadcast_to(mask, values.shape)
    if not np.all(np.isfinite(values[expanded])):
        raise HOIValidationError(f"{name} contains NaN or Inf in valid entries")


@dataclass
class ProvenanceRecord:
    source_dataset: str = ""
    source_sequence: str = ""
    source_file: str | None = None
    source_hash: str | None = None
    source_size: int | None = None
    source_mtime_ns: int | None = None
    adapter_name: str = ""
    adapter_version: str = ""
    conversion_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    conversion_options: dict[str, Any] = field(default_factory=dict)
    source_coordinate_convention: str = "unknown"
    no_temporal_resampling: bool = True
    no_spatial_sampling: bool = True


@dataclass
class SequenceMetadata:
    schema_version: str = SCHEMA_VERSION
    dataset_name: str = ""
    sequence_id: str = ""
    native_fps: float | None = None
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    num_frames: int | None = None
    source_frame_name: str = "source"
    scene_frame_name: str = "S"
    source_to_scene: np.ndarray = field(default_factory=_identity)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = _array(self.timestamps, dtype=np.float64).reshape(-1)
        self.source_to_scene = _array(self.source_to_scene, dtype=np.float64)
        if self.num_frames is None:
            self.num_frames = int(self.timestamps.shape[0])


@dataclass
class PoseTrack:
    pose_scene: np.ndarray
    valid: np.ndarray | None = None
    frame_name: str = "S"
    child_frame_name: str = "child"
    orientation_available: bool = True

    def __post_init__(self) -> None:
        self.pose_scene = _array(self.pose_scene, dtype=np.float64)
        if self.valid is None:
            self.valid = np.ones(self.pose_scene.shape[0], dtype=bool)
        else:
            self.valid = _array(self.valid, dtype=bool).reshape(-1)


@dataclass
class MeshDefinition:
    vertices_local: np.ndarray
    faces: np.ndarray
    mesh_frame_name: str = "O"
    mesh_id: str = "mesh"
    mesh_hash: str | None = None
    units: str = "m"

    def __post_init__(self) -> None:
        self.vertices_local = _array(self.vertices_local, dtype=np.float64)
        self.faces = _array(self.faces)


@dataclass
class KeypointTrack:
    positions_scene: np.ndarray
    layout_name: str
    valid: np.ndarray | None = None
    semantic_names: list[str] | None = None
    confidence: np.ndarray | None = None
    frame_name: str = "S"
    units: str = "m"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions_scene = _array(self.positions_scene, dtype=np.float64)
        if self.valid is not None:
            self.valid = _array(self.valid, dtype=bool)
        if self.confidence is not None:
            self.confidence = _array(self.confidence, dtype=np.float64)


@dataclass
class ManoParameterTrack:
    global_orient_aa: np.ndarray | None = None
    hand_pose_aa: np.ndarray | None = None
    transl: np.ndarray | None = None
    betas: np.ndarray | None = None
    personalized_v_template_reference: str | None = None
    model_profile: str | None = None

    def __post_init__(self) -> None:
        for name in ("global_orient_aa", "hand_pose_aa", "transl", "betas"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _array(value, dtype=np.float64))


@dataclass
class HandTrack:
    hand_id: str
    side: str
    wrist_pose_scene: PoseTrack
    valid: np.ndarray | None = None
    keypoint_tracks: dict[str, KeypointTrack] = field(default_factory=dict)
    mesh: MeshDefinition | None = None
    vertices_scene: np.ndarray | None = None
    mano_parameters: ManoParameterTrack | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid is not None:
            self.valid = _array(self.valid, dtype=bool).reshape(-1)
        if self.vertices_scene is not None:
            self.vertices_scene = _array(self.vertices_scene, dtype=np.float64)


@dataclass
class RigidObjectTrack:
    object_id: str
    mesh: MeshDefinition
    pose_scene: PoseTrack
    valid: np.ndarray | None = None
    scale: float | np.ndarray = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid is not None:
            self.valid = _array(self.valid, dtype=bool).reshape(-1)
        if not np.isscalar(self.scale):
            self.scale = _array(self.scale, dtype=np.float64)


@dataclass
class ArticulatedPartTrack:
    part_id: str
    mesh: MeshDefinition
    pose_scene: PoseTrack
    parent_id: str | None = None
    valid: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArticulatedObjectTrack:
    object_id: str
    parts: list[ArticulatedPartTrack] = field(default_factory=list)
    parent_child_structure: dict[str, str | None] = field(default_factory=dict)
    articulation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactTrack:
    hand_id: str
    object_id: str
    source_contact_representation: str
    valid: np.ndarray
    labels: np.ndarray | None = None
    vertex_associations: np.ndarray | None = None
    binary: np.ndarray | None = None
    semantic_labels: np.ndarray | None = None
    semantic_mapping: dict[int, dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.valid = _array(self.valid, dtype=bool)
        if self.labels is not None:
            self.labels = _array(self.labels)
        if self.vertex_associations is not None:
            self.vertex_associations = _array(self.vertex_associations)
        if self.binary is not None:
            self.binary = _array(self.binary, dtype=bool)
        if self.semantic_labels is not None:
            self.semantic_labels = _array(self.semantic_labels, dtype=np.int64)
        if self.semantic_mapping is not None:
            self.semantic_mapping = {
                int(key): dict(value) for key, value in self.semantic_mapping.items()
            }

    @property
    def semantic_ids(self) -> np.ndarray | None:
        """Compatibility name for the integer semantic label track."""

        return self.semantic_labels

    @semantic_ids.setter
    def semantic_ids(self, value: np.ndarray | None) -> None:
        self.semantic_labels = None if value is None else _array(value, dtype=np.int64)


@dataclass
class HOISequence:
    metadata: SequenceMetadata
    hands: list[HandTrack] = field(default_factory=list)
    rigid_objects: list[RigidObjectTrack] = field(default_factory=list)
    articulated_objects: list[ArticulatedObjectTrack] = field(default_factory=list)
    contacts: list[ContactTrack] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.metadata.num_frames or 0)

    @property
    def timestamps(self) -> np.ndarray:
        return self.metadata.timestamps

    def hand(self, hand_id: str) -> HandTrack:
        for item in self.hands:
            if item.hand_id == hand_id:
                return item
        raise KeyError(f"Unknown hand_id: {hand_id}")

    def rigid_object(self, object_id: str) -> RigidObjectTrack:
        for item in self.rigid_objects:
            if item.object_id == object_id:
                return item
        raise KeyError(f"Unknown object_id: {object_id}")

    def validate(self, *, raise_on_error: bool = True) -> list[str]:
        errors: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        metadata = self.metadata
        check(metadata.schema_version == SCHEMA_VERSION, "unsupported schema_version")
        check(metadata.timestamps.ndim == 1, "timestamps must have shape [T]")
        check(bool(np.all(np.isfinite(metadata.timestamps))), "timestamps contain NaN or Inf")
        if metadata.timestamps.size > 1:
            check(
                bool(np.all(np.diff(metadata.timestamps) > 0)),
                "timestamps must be strictly increasing",
            )
        check(metadata.num_frames == metadata.timestamps.shape[0], "num_frames mismatch")
        check(bool(metadata.scene_frame_name), "scene_frame_name must be explicit")
        try:
            validate_transform(metadata.source_to_scene)
        except ValueError as exc:
            errors.append(f"source_to_scene: {exc}")
        if metadata.native_fps is not None:
            check(
                bool(np.isfinite(metadata.native_fps) and metadata.native_fps > 0),
                "invalid native_fps",
            )

        t = self.num_frames

        def check_pose(pose: PoseTrack, prefix: str) -> None:
            check(
                pose.pose_scene.shape == (t, 4, 4), f"{prefix}.pose_scene must have shape [T,4,4]"
            )
            valid = np.ones(t, dtype=bool) if pose.valid is None else pose.valid
            check(valid.shape == (t,), f"{prefix}.valid must have shape [T]")
            if pose.pose_scene.shape == (t, 4, 4):
                try:
                    validate_transform(pose.pose_scene[valid])
                except ValueError as exc:
                    errors.append(f"{prefix}.pose_scene: {exc}")
                _finite_with_mask(pose.pose_scene, valid, f"{prefix}.pose_scene")

        def check_mesh(mesh: MeshDefinition, prefix: str) -> None:
            check(mesh.units == "m", f"{prefix}.units must be 'm'")
            check(
                mesh.vertices_local.ndim == 2 and mesh.vertices_local.shape[1:] == (3,),
                f"{prefix}.vertices_local must have shape [V,3]",
            )
            check(
                mesh.faces.ndim == 2 and mesh.faces.shape[1:] == (3,),
                f"{prefix}.faces must have shape [F,3]",
            )
            check(np.issubdtype(mesh.faces.dtype, np.integer), f"{prefix}.faces must be integer")
            if mesh.vertices_local.ndim == 2:
                _finite_with_mask(mesh.vertices_local, None, f"{prefix}.vertices_local")
            if mesh.faces.size and mesh.vertices_local.ndim == 2:
                check(int(mesh.faces.min()) >= 0, f"{prefix}.faces has negative index")
                check(
                    int(mesh.faces.max()) < mesh.vertices_local.shape[0],
                    f"{prefix}.faces index out of bounds",
                )

        for hand in self.hands:
            prefix = f"hand[{hand.hand_id}]"
            check(hand.side in {"left", "right"}, f"{prefix}.side must be left or right")
            check_pose(hand.wrist_pose_scene, f"{prefix}.wrist_pose_scene")
            if hand.valid is not None:
                check(hand.valid.shape == (t,), f"{prefix}.valid must have shape [T]")
            if hand.mesh is not None:
                check_mesh(hand.mesh, f"{prefix}.mesh")
            if hand.vertices_scene is not None:
                check(
                    hand.vertices_scene.ndim == 3
                    and hand.vertices_scene.shape[0] == t
                    and hand.vertices_scene.shape[2] == 3,
                    f"{prefix}.vertices_scene must have shape [T,V,3]",
                )
                if hand.vertices_scene.ndim == 3 and hand.vertices_scene.shape[0] == t:
                    _finite_with_mask(hand.vertices_scene, hand.valid, f"{prefix}.vertices_scene")
            for layout, track in hand.keypoint_tracks.items():
                prefix_k = f"{prefix}.keypoints[{layout}]"
                check(track.layout_name == layout, f"{prefix_k}.layout_name mismatch")
                check(
                    track.positions_scene.ndim == 3
                    and track.positions_scene.shape[0] == t
                    and track.positions_scene.shape[2] == 3,
                    f"{prefix_k}.positions_scene must have shape [T,K,3]",
                )
                if track.valid is not None:
                    check(
                        track.valid.shape in {(t,), track.positions_scene.shape[:2]},
                        f"{prefix_k}.valid has invalid shape",
                    )
                if track.positions_scene.ndim == 3 and track.positions_scene.shape[0] == t:
                    _finite_with_mask(track.positions_scene, track.valid, prefix_k)
                if track.semantic_names is not None:
                    check(
                        len(track.semantic_names) == track.positions_scene.shape[1],
                        f"{prefix_k}.semantic_names length mismatch",
                    )
                check(bool(track.frame_name), f"{prefix_k}.frame_name must be explicit")
                check(track.units == "m", f"{prefix_k}.units must be 'm'")

        for obj in self.rigid_objects:
            prefix = f"object[{obj.object_id}]"
            check_pose(obj.pose_scene, f"{prefix}.pose_scene")
            check_mesh(obj.mesh, f"{prefix}.mesh")
            if obj.valid is not None:
                check(obj.valid.shape == (t,), f"{prefix}.valid must have shape [T]")
        for articulated_obj in self.articulated_objects:
            for part in articulated_obj.parts:
                check_pose(
                    part.pose_scene,
                    f"articulated[{articulated_obj.object_id}].part[{part.part_id}]",
                )
                check_mesh(
                    part.mesh,
                    f"articulated[{articulated_obj.object_id}].part[{part.part_id}].mesh",
                )
        for contact in self.contacts:
            check(
                contact.valid.shape[0] == t,
                f"contact[{contact.hand_id},{contact.object_id}] time mismatch",
            )
            if contact.labels is not None:
                check(
                    contact.labels.ndim >= 1 and contact.labels.shape[0] == t,
                    f"contact[{contact.hand_id},{contact.object_id}].labels time mismatch",
                )
            if contact.binary is not None:
                check(
                    contact.binary.shape[0] == t,
                    f"contact[{contact.hand_id},{contact.object_id}].binary time mismatch",
                )
                if contact.labels is not None and contact.binary.shape == contact.labels.shape:
                    no_contact = int(contact.metadata.get("no_contact_label", 0))
                    check(
                        bool(np.array_equal(contact.binary, contact.labels != no_contact)),
                        f"contact[{contact.hand_id},{contact.object_id}].binary is not "
                        "derived from labels",
                    )
            if contact.semantic_labels is not None:
                check(
                    contact.semantic_labels.shape[0] == t,
                    f"contact[{contact.hand_id},{contact.object_id}].semantic_labels time mismatch",
                )
                if (
                    contact.labels is not None
                    and contact.semantic_labels.shape != contact.labels.shape
                ):
                    errors.append(
                        f"contact[{contact.hand_id},{contact.object_id}].semantic_labels "
                        "shape does not match labels"
                    )
                check(
                    np.issubdtype(contact.semantic_labels.dtype, np.integer),
                    f"contact[{contact.hand_id},{contact.object_id}].semantic_labels must "
                    "be integer",
                )
                if contact.semantic_mapping is None:
                    errors.append(
                        f"contact[{contact.hand_id},{contact.object_id}].semantic_mapping "
                        "is missing"
                    )

        if errors and raise_on_error:
            raise HOIValidationError("; ".join(errors))
        return errors

    def scene_to_wrist(self, hand_id: str, points_scene: np.ndarray) -> np.ndarray:
        wrist = self.hand(hand_id).wrist_pose_scene
        if not wrist.orientation_available:
            raise HOIValidationError(
                f"hand {hand_id} has no source wrist orientation; derive one explicitly before use"
            )
        return scene_to_wrist(wrist.pose_scene, points_scene)

    def wrist_to_scene(self, hand_id: str, points_wrist: np.ndarray) -> np.ndarray:
        wrist = self.hand(hand_id).wrist_pose_scene
        if not wrist.orientation_available:
            raise HOIValidationError(
                f"hand {hand_id} has no source wrist orientation; derive one explicitly before use"
            )
        return wrist_to_scene(wrist.pose_scene, points_wrist)

    def scene_to_object(self, object_id: str, points_scene: np.ndarray) -> np.ndarray:
        return scene_to_object(self.rigid_object(object_id).pose_scene.pose_scene, points_scene)

    def object_to_scene(self, object_id: str, points_object: np.ndarray) -> np.ndarray:
        return object_to_scene(self.rigid_object(object_id).pose_scene.pose_scene, points_object)


__all__ = [
    "SCHEMA_VERSION",
    "ArticulatedObjectTrack",
    "ArticulatedPartTrack",
    "ContactTrack",
    "HandTrack",
    "HOISequence",
    "HOIValidationError",
    "KeypointTrack",
    "ManoParameterTrack",
    "MeshDefinition",
    "PoseTrack",
    "ProvenanceRecord",
    "RigidObjectTrack",
    "SequenceMetadata",
]
