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
from toporetarget.data.mano_backends.base import ManoRenderResult
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
    load_mano_model_geometry,
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


def render_mano_fullpose(
    pose_51: np.ndarray,
    *,
    side: str,
    mano_model_root: Path,
    v_template: np.ndarray | None = None,
) -> ManoRenderResult:
    """Render the native 51-vector convention used by DexYCB and HOCap."""

    pose = np.asarray(pose_51, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1:] != (51,):
        raise Stage12AdapterError(f"MANO pose must have shape [T,51], got {pose.shape}")
    geometry = load_mano_model_geometry(mano_model_root, side=side, expected_vertex_count=778)
    backend = SmplxManoBackend(mano_model_root)
    return backend.render(
        params={
            "global_orient": pose[:, :3],
            "fullpose": pose[:, 3:48],
            "transl": pose[:, 48:51],
        },
        v_template=geometry.v_template if v_template is None else v_template,
        side=side,
        frame_count=pose.shape[0],
    )


def render_mano_pca(
    pose_18: np.ndarray,
    *,
    side: str,
    mano_model_root: Path,
    v_template: np.ndarray | None = None,
) -> ManoRenderResult:
    """Render ContactPose's [global axis-angle, PCA-15] MANO fit."""

    pose = np.asarray(pose_18, dtype=np.float64).reshape(1, -1)
    if pose.shape[1] != 18:
        raise Stage12AdapterError(
            f"ContactPose MANO fit pose must have 18 values, got {pose.shape}"
        )
    geometry = load_mano_model_geometry(mano_model_root, side=side, expected_vertex_count=778)
    backend = SmplxManoBackend(mano_model_root)
    return backend.render(
        params={
            "global_orient": np.repeat(pose[:, :3], 1, axis=0),
            "hand_pose": pose[:, 3:18],
            "transl": np.zeros((1, 3), dtype=np.float64),
        },
        v_template=geometry.v_template if v_template is None else v_template,
        side=side,
        frame_count=1,
    )


def _mano_native_track(
    vertices: np.ndarray, *, side: str, mano_model_root: Path, valid: np.ndarray
) -> KeypointTrack:
    geometry = load_mano_model_geometry(mano_model_root, side=side, expected_vertex_count=778)
    positions = np.einsum("jv,tvc->tjc", geometry.joint_regressor, np.asarray(vertices))
    if positions.shape[1] != 16:
        raise Stage12AdapterError(
            f"MANO regressor for {side} has {positions.shape[1]} joints; "
            "expected the audited 16-joint layout"
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
            "source": "MANO J_regressor applied to source MANO vertices",
            "mapping_mode": "explicit_joint_regression",
            "mano_model_hash": geometry.model_hash,
        },
    )


def attach_mediapipe21(
    hand: HandTrack, *, frame_count: int, mano_model_root: Path, valid: np.ndarray
) -> None:
    """Attach the existing audited semantic converter; no shape-only reorder."""

    if hand.vertices_scene is None:
        raise Stage12AdapterError(f"hand {hand.hand_id} has no scene vertices")
    native = _mano_native_track(
        hand.vertices_scene, side=hand.side, mano_model_root=mano_model_root, valid=valid
    )
    hand.keypoint_tracks[native.layout_name] = native
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
        hand, frame_count=len(valid_mask), mano_model_root=mano_model_root, valid=valid_mask
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
    "render_mano_fullpose",
    "render_mano_pca",
    "sequence_metadata",
    "sha256_file",
    "sha256_paths",
    "transform_points",
]
