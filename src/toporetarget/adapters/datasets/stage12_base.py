"""Shared, lazy primitives for the Stage 12 dataset adapters.

The adapters in this module deliberately stop at one explicitly requested
sequence.  They do not create a dataset-wide cache, download data, or infer a
dataset-specific retargeting objective.  Raw geometry is converted to the
existing MANO -> MediaPipe21 contract and then wrapped by Canonical HOI v2.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.contracts.canonical import CanonicalHOIv2
from toporetarget.contracts.dataset import DatasetAdapter
from toporetarget.data.adapters.base import FrameRange, HOIDatasetAdapter
from toporetarget.data.mano_backends.contracts import (
    ManoJointSource,
    ManoPoseRepresentation,
    ManoReconstructionRequest,
    ManoReconstructionResult,
)
from toporetarget.data.mano_backends.smplx_backend import SmplxManoBackend
from toporetarget.data.schema import (
    HandTrack,
    HOISequence,
    KeypointTrack,
    ManoParameterTrack,
    MeshDefinition,
    PoseTrack,
    ProvenanceRecord,
    RigidObjectTrack,
    SequenceMetadata,
)
from toporetarget.keypoints.mano_to_mediapipe import (
    ManoToMediaPipe21Converter,
)
from toporetarget.keypoints.registry import get_layout

DEFAULT_STORAGE_ROOT = Path(
    os.environ.get("REF2DEX_STORAGE_ROOT", "/mnt/nas/storage/Ref2Dex_storage")
).expanduser()
DEFAULT_MANO_ROOT = DEFAULT_STORAGE_ROOT / "shared_assets" / "body_models" / "mano"


class Stage12AdapterError(RuntimeError):
    """Raised when one selected sequence cannot satisfy the canonical contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one mesh without preprocessing, repair, or unit normalization."""

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Stage12AdapterError(
            "Stage 12 mesh loading needs trimesh; install the future/geometry extra"
        ) from exc
    source = Path(path)
    try:
        mesh: Any = trimesh.load(source, process=False, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    except Exception as exc:
        raise Stage12AdapterError(f"could not load mesh {source}: {exc}") from exc
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise Stage12AdapterError(f"mesh {source} has invalid vertices shape {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise Stage12AdapterError(f"mesh {source} has invalid faces shape {faces.shape}")
    if not np.isfinite(vertices).all():
        raise Stage12AdapterError(f"mesh {source} contains non-finite vertices")
    return vertices, faces


def pose_json_wxyz(value: dict[str, Any]) -> np.ndarray:
    """ContactPose JSON convention: transforms3d [w,x,y,z] plus translation."""

    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(value["rotation"], dtype=np.float64)
    if quaternion.shape != (4,):
        raise Stage12AdapterError(f"ContactPose quaternion has shape {quaternion.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        np.asarray([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    ).as_matrix()
    matrix[:3, 3] = np.asarray(value["translation"], dtype=np.float64)
    return matrix


def pose_hocap_qxyzw(value: np.ndarray) -> np.ndarray:
    """HOCap convention: scipy [qx,qy,qz,qw] plus translation."""

    from scipy.spatial.transform import Rotation

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (7,):
        raise Stage12AdapterError(f"HOCap object pose has shape {array.shape}, expected [7]")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(array[:4]).as_matrix()
    matrix[:3, 3] = array[4:]
    return matrix


def identity_poses(frame_count: int) -> np.ndarray:
    return np.broadcast_to(np.eye(4, dtype=np.float64), (frame_count, 4, 4)).copy()


def transform_points(poses: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(poses, dtype=np.float64)
    return np.einsum("tij,tvj->tvi", matrix[:, :3, :3], value) + matrix[:, None, :3, 3]


def render_mano_pca45(
    pose_51: np.ndarray,
    *,
    side: str,
    mano_model_root: Path,
    betas: np.ndarray,
    dataset_name: str,
    source_annotation_path: Path,
    source_annotation_hash: str,
) -> ManoReconstructionResult:
    """Render the explicit AA3 + PCA45 + translation3 dataset convention."""

    pose = np.asarray(pose_51, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1:] != (51,):
        raise Stage12AdapterError(f"MANO pose must have shape [T,51], got {pose.shape}")
    backend = SmplxManoBackend(mano_model_root)
    provenance = backend.model_provenance(
        side=side,
        dataset_name=dataset_name,
        source_annotation_path=source_annotation_path,
        source_annotation_hash=source_annotation_hash,
    )
    return backend.reconstruct(
        ManoReconstructionRequest(
            side=side,
            pose_representation=ManoPoseRepresentation.PCA,
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:48],
            num_pca_components=45,
            translation=pose[:, 48:51],
            betas=np.asarray(betas, dtype=np.float64),
            flat_hand_mean=False,
            units="metre",
            dtype=np.float64,
            model_path=provenance.model_path,
            model_hash=provenance.model_hash,
            model_version=provenance.model_version,
            dataset_name=provenance.dataset_name,
            source_annotation_path=provenance.source_annotation_path,
            source_annotation_hash=provenance.source_annotation_hash,
        )
    )


def render_mano_pca(
    pose_18: np.ndarray,
    *,
    side: str,
    mano_model_root: Path,
    betas: np.ndarray,
    dataset_name: str,
    source_annotation_path: Path,
    source_annotation_hash: str,
) -> ManoReconstructionResult:
    """Render ContactPose's [global axis-angle, PCA-15] MANO fit."""

    pose = np.asarray(pose_18, dtype=np.float64).reshape(1, -1)
    if pose.shape[1] != 18:
        raise Stage12AdapterError(
            f"ContactPose MANO fit pose must have 18 values, got {pose.shape}"
        )
    backend = SmplxManoBackend(mano_model_root)
    provenance = backend.model_provenance(
        side=side,
        dataset_name=dataset_name,
        source_annotation_path=source_annotation_path,
        source_annotation_hash=source_annotation_hash,
    )
    return backend.reconstruct(
        ManoReconstructionRequest(
            side=side,
            pose_representation=ManoPoseRepresentation.PCA,
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:18],
            num_pca_components=15,
            translation=np.zeros((1, 3), dtype=np.float64),
            betas=np.asarray(betas, dtype=np.float64),
            flat_hand_mean=False,
            units="metre",
            dtype=np.float64,
            model_path=provenance.model_path,
            model_hash=provenance.model_hash,
            model_version=provenance.model_version,
            dataset_name=provenance.dataset_name,
            source_annotation_path=provenance.source_annotation_path,
            source_annotation_hash=provenance.source_annotation_hash,
        )
    )


def backend_posed_joint_track(
    result: ManoReconstructionResult, *, valid: np.ndarray
) -> KeypointTrack:
    """Preserve actual backend posed joints; never rest-regress posed vertices."""

    positions = np.asarray(result.posed_joints_native, dtype=np.float64)
    if result.posed_joint_layout != "mano16_smplx" or positions.shape[1] != 16:
        raise Stage12AdapterError(
            "Stage12 backend joint mapping requires explicit mano16_smplx posed joints, got "
            f"{result.posed_joint_layout!r} with shape {positions.shape}"
        )
    names = list(get_layout("mano16_smplx").semantic_names)
    point_valid = np.broadcast_to(valid[:, None], positions.shape[:2]).copy()
    point_valid &= np.isfinite(positions).all(axis=-1)
    return KeypointTrack(
        positions,
        layout_name="mano16_smplx",
        valid=point_valid,
        semantic_names=names,
        frame_name="S",
        units="m",
        provenance={
            "source": ManoJointSource.BACKEND_POSED.value,
            "mapping_mode": "backend_forward_posed_joints",
            "mano_model_hash": result.model_provenance.model_hash,
            "mano_reconstruction_manifest": result.reconstruction_manifest,
        },
    )


def native_mano21_track(
    positions: np.ndarray,
    *,
    valid: np.ndarray,
    source_name: str,
    source_path: str | None = None,
) -> KeypointTrack:
    """Keep an audited dataset-native 21-joint array with semantic labels."""

    value = np.asarray(positions, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (21, 3):
        raise Stage12AdapterError(
            f"{source_name} native joints must have shape [T,21,3], got {value.shape}"
        )
    names = list(get_layout("mano21_named").semantic_names)
    point_valid = np.broadcast_to(np.asarray(valid, dtype=bool)[:, None], value.shape[:2]).copy()
    point_valid &= np.isfinite(value).all(axis=-1)
    if not point_valid.all():
        raise Stage12AdapterError(f"{source_name} native joints contain invalid values")
    return KeypointTrack(
        value,
        layout_name="mano21_named",
        valid=point_valid,
        semantic_names=names,
        frame_name="S",
        units="m",
        provenance={
            "source": ManoJointSource.DATASET_NATIVE.value,
            "source_name": source_name,
            "source_path": source_path,
            "layout_contract": "wrist, thumb, index, middle, ring, pinky",
        },
    )


def contactpose_official_mano21_track(
    posed_joints: np.ndarray,
    vertices: np.ndarray,
    *,
    valid: np.ndarray,
) -> KeypointTrack:
    """Build ContactPose's documented MANO-21 convention from posed data.

    ContactPose uses MANO's 16 posed joints plus its own fingertip vertices:
    index/middle/pinky/ring/thumb = 333/444/672/555/745.  They are not the
    repository-wide SMPL-X anchors, so this path deliberately does not reuse
    the generic MANO-to-MediaPipe profile.
    """

    joints = np.asarray(posed_joints, dtype=np.float64)
    mesh = np.asarray(vertices, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (16, 3):
        raise Stage12AdapterError(
            f"ContactPose posed joints must have shape [T,16,3], got {joints.shape}"
        )
    if mesh.shape != (joints.shape[0], 778, 3):
        raise Stage12AdapterError(
            f"ContactPose vertices must have shape [{joints.shape[0]},778,3], got {mesh.shape}"
        )
    mano_names = list(get_layout("mano16_smplx").semantic_names)
    target_names = list(get_layout("mediapipe21").semantic_names)
    mano_index = {name: index for index, name in enumerate(mano_names)}
    output = np.empty((joints.shape[0], 21, 3), dtype=np.float64)
    tip_vertices = {
        "index_tip": 333,
        "middle_tip": 444,
        "pinky_tip": 672,
        "ring_tip": 555,
        "thumb_tip": 745,
    }
    for target_index, semantic_name in enumerate(target_names):
        if semantic_name in tip_vertices:
            output[:, target_index] = mesh[:, tip_vertices[semantic_name]]
        else:
            output[:, target_index] = joints[:, mano_index[semantic_name]]
    point_valid = np.broadcast_to(np.asarray(valid, dtype=bool)[:, None], output.shape[:2]).copy()
    point_valid &= np.isfinite(output).all(axis=-1)
    if not point_valid.all():
        raise Stage12AdapterError("ContactPose official MANO21 joints contain invalid values")
    return KeypointTrack(
        output,
        layout_name="mano21_named",
        valid=point_valid,
        semantic_names=target_names,
        frame_name="S",
        units="m",
        provenance={
            "source": ManoJointSource.CONTACTPOSE_OFFICIAL.value,
            "mapping_mode": "contactpose_official_mano16_plus_tip_vertices",
            "mano_internal_layout": "mano16_smplx",
            "target_layout": "mediapipe21",
            "fingertip_vertex_ids": tip_vertices,
            "no_generic_smplx_tip_anchor": True,
        },
    )


def _native_mano21_to_mediapipe21(track: KeypointTrack) -> KeypointTrack:
    """Semantically reorder a named native 21-joint track, never shape-cast it."""

    source_names = list(track.semantic_names or [])
    target_names = list(get_layout("mediapipe21").semantic_names)
    if len(source_names) != 21 or len(set(source_names)) != 21:
        raise Stage12AdapterError("native MANO21 source has no complete unique semantic layout")
    source_index = {name: index for index, name in enumerate(source_names)}
    if set(source_index) != set(target_names):
        raise Stage12AdapterError("native MANO21 semantic names do not exactly cover MediaPipe21")
    indices = [source_index[name] for name in target_names]
    return KeypointTrack(
        track.positions_scene[:, indices],
        layout_name="mediapipe21",
        valid=None if track.valid is None else track.valid[:, indices],
        semantic_names=target_names,
        frame_name="S",
        units="m",
        provenance={
            "source": track.provenance.get("source", ManoJointSource.DATASET_NATIVE.value),
            "mapping_mode": "explicit_semantic_named_reorder",
            "source_layout": track.layout_name,
            "target_layout": "mediapipe21",
            "no_vertex_regression": True,
        },
    )


def attach_mediapipe21(
    hand: HandTrack,
    *,
    frame_count: int,
    mano_model_root: Path,
    valid: np.ndarray,
    native_joint_track: KeypointTrack,
) -> None:
    """Attach canonical joints from an explicit native source only."""

    if hand.vertices_scene is None:
        raise Stage12AdapterError(f"hand {hand.hand_id} has no scene vertices")
    native = native_joint_track
    hand.keypoint_tracks[native.layout_name] = native
    if native.layout_name == "mano21_named":
        hand.keypoint_tracks["mediapipe21"] = _native_mano21_to_mediapipe21(native)
        return
    if native.layout_name != "mano16_smplx":
        raise Stage12AdapterError(
            f"unsupported explicit native joint layout {native.layout_name!r}; refusing fallback"
        )
    converter = ManoToMediaPipe21Converter("mano_v1_2_smplx_to_mediapipe21")
    hand.keypoint_tracks["mediapipe21"] = converter.convert_hand_track(
        hand, frame_count=frame_count, mano_model_root=mano_model_root
    )


def make_hand(
    *,
    hand_id: str,
    side: str,
    vertices_scene: np.ndarray,
    faces: np.ndarray,
    wrist_pose_scene: np.ndarray,
    valid: np.ndarray,
    mano_parameters: ManoParameterTrack | None,
    mano_model_root: Path,
    metadata: dict[str, Any],
    native_joint_track: KeypointTrack,
    wrist_orientation_available: bool = True,
) -> HandTrack:
    vertices = np.asarray(vertices_scene, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool).reshape(-1)
    local = np.asarray(vertices[np.flatnonzero(valid_mask)[0]], dtype=np.float64)
    hand = HandTrack(
        hand_id=hand_id,
        side=side,
        wrist_pose_scene=PoseTrack(
            np.asarray(wrist_pose_scene, dtype=np.float64),
            valid=valid_mask,
            frame_name="S",
            child_frame_name=f"W_{side}",
            orientation_available=wrist_orientation_available,
        ),
        valid=valid_mask,
        mesh=MeshDefinition(
            local,
            np.asarray(faces, dtype=np.int64),
            mesh_frame_name="S",
            mesh_id=f"mano_{side}_v1_2",
            units="m",
        ),
        vertices_scene=vertices,
        mano_parameters=mano_parameters,
        metadata=metadata,
    )
    attach_mediapipe21(
        hand,
        frame_count=len(valid_mask),
        mano_model_root=mano_model_root,
        valid=valid_mask,
        native_joint_track=native_joint_track,
    )
    return hand


def make_object(
    *,
    object_id: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    poses_scene: np.ndarray,
    valid: np.ndarray,
    mesh_hash: str | None,
    metadata: dict[str, Any],
) -> RigidObjectTrack:
    return RigidObjectTrack(
        object_id=object_id,
        mesh=MeshDefinition(
            vertices,
            faces,
            mesh_frame_name="O",
            mesh_id=f"{object_id}_source_mesh",
            mesh_hash=mesh_hash,
            units="m",
        ),
        pose_scene=PoseTrack(
            np.asarray(poses_scene), valid=np.asarray(valid, dtype=bool), child_frame_name="O"
        ),
        valid=np.asarray(valid, dtype=bool),
        metadata=metadata,
    )


class Stage12AdapterBase(HOIDatasetAdapter, DatasetAdapter):
    """Base implementing the common Stage 11 facade and lazy bookkeeping."""

    contract_version = "toporetarget.dataset_adapter.v1"
    adapter_version = "1.0.0"
    descriptor: Any

    def __init__(
        self, *, data_root: str | Path | None = None, mano_model_root: str | Path | None = None
    ) -> None:
        self.data_root = Path(data_root or DEFAULT_STORAGE_ROOT).expanduser()
        self.mano_model_root = Path(mano_model_root or DEFAULT_MANO_ROOT).expanduser()

    @abstractmethod
    def _discover_rows(self) -> list[dict[str, Any]]:
        """Read only index/metadata rows; never frame geometry."""

    def discover(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._discover_rows()
        limit = kwargs.get("limit")
        return rows if limit is None else rows[: int(limit)]

    def index(self, **kwargs: Any) -> dict[str, Any]:
        rows = self._discover_rows()
        payload = {
            "schema_version": "toporetarget.stage12.dataset_index.v1",
            "dataset": self.descriptor.as_dict(),
            "data_root": str(self.data_root.resolve()),
            "sequence_count": len(rows),
            "rows": rows,
        }
        output = kwargs.get("output")
        if output is not None:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload["output"] = str(destination)
        return payload

    @abstractmethod
    def describe_sequence(self, sequence: str, **kwargs: Any) -> dict[str, Any]:
        """Describe one sequence without loading frame geometry."""

    def describe(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.describe_sequence(sequence, **kwargs)

    @abstractmethod
    def load_sequence(
        self, sequence: str = "", *, frame_range: FrameRange | None = None, **kwargs: Any
    ) -> HOISequence:
        """Load one explicit raw sequence or contiguous clip."""

    def canonicalize(self, sequence: HOISequence, **kwargs: Any) -> HOISequence:
        sequence.validate()
        return sequence

    def convert_to_canonical(
        self, sequence: HOISequence | CanonicalHOIv2, **kwargs: Any
    ) -> CanonicalHOIv2:
        if isinstance(sequence, CanonicalHOIv2):
            sequence.validate()
            return sequence
        return CanonicalHOIv2.from_v1(self.canonicalize(sequence, **kwargs), copy_arrays=False)

    def validate(self, sequence: HOISequence | CanonicalHOIv2, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        errors = sequence.validate(raise_on_error=False)
        return {
            "dataset": self.descriptor.name,
            "schema_version": sequence.metadata.schema_version,
            "sequence_id": sequence.metadata.sequence_id,
            "num_frames": sequence.num_frames,
            "errors": errors,
            "status": "ok" if not errors else "invalid",
        }

    def visualize(
        self,
        sequence: HOISequence | CanonicalHOIv2,
        *,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        canonical = self.convert_to_canonical(sequence)
        hand = canonical.hands[0]
        payload = {
            "schema_version": "toporetarget.stage12.viewer_handle.v1",
            "status": "ready",
            "dataset": self.descriptor.name,
            "sequence_id": canonical.metadata.sequence_id,
            "num_frames": canonical.num_frames,
            "source_mesh": bool(hand.mesh is not None and hand.vertices_scene is not None),
            "object_meshes": len(canonical.rigid_objects),
            "viewer": "toporetarget.quality.html.render_clip_html",
            "layers": ["source MANO", "warm Wuji", "final Wuji"],
        }
        if output is not None:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload["output"] = str(destination)
        return payload

    def supported_fields(self) -> tuple[str, ...]:
        return ("canonical_hoi_v2", "mediapipe21", "mano_mesh", "rigid_object_pose", "provenance")


def sequence_metadata(
    *,
    dataset: str,
    sequence_id: str,
    frame_count: int,
    fps: float,
    source_file: str | Path,
    source_hash: str,
    adapter_name: str,
    coordinate_convention: str,
    conversion_options: dict[str, Any],
    metadata: dict[str, Any],
) -> SequenceMetadata:
    timestamps = np.arange(frame_count, dtype=np.float64) / float(fps)
    provenance = ProvenanceRecord(
        source_dataset=dataset,
        source_sequence=sequence_id,
        source_file=str(source_file),
        source_hash=source_hash,
        adapter_name=adapter_name,
        adapter_version="1.0.0",
        source_coordinate_convention=coordinate_convention,
        conversion_options={
            **conversion_options,
            "no_temporal_resampling": True,
            "no_spatial_sampling": True,
        },
    )
    return SequenceMetadata(
        dataset_name=dataset,
        sequence_id=sequence_id,
        native_fps=float(fps),
        timestamps=timestamps,
        source_frame_name=f"{dataset}_native",
        scene_frame_name="S",
        provenance=provenance,
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_MANO_ROOT",
    "DEFAULT_STORAGE_ROOT",
    "Stage12AdapterBase",
    "Stage12AdapterError",
    "attach_mediapipe21",
    "identity_poses",
    "load_mesh",
    "load_pickle",
    "make_hand",
    "make_object",
    "pose_hocap_qxyzw",
    "pose_json_wxyz",
    "backend_posed_joint_track",
    "contactpose_official_mano21_track",
    "native_mano21_track",
    "render_mano_pca45",
    "render_mano_pca",
    "sequence_metadata",
    "sha256_file",
    "sha256_paths",
    "transform_points",
]
