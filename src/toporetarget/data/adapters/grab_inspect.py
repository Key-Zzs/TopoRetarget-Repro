"""Minimal GRAB inspection adapter for one explicit sequence path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange, HOIDatasetAdapter
from toporetarget.data.mano_backends.base import (
    ManoBackend,
    ManoBackendError,
    ManoRenderResult,
    axis_angle_to_matrix,
)
from toporetarget.data.readers.grab import (
    GrabParseError,
    load_ply_mesh,
    read_grab_npz,
    resolve_grab_resource,
    resolve_grab_root,
)
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


class GrabAdapterError(RuntimeError):
    """Raised when an explicit GRAB inspection cannot be completed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _local_config_value(key: str) -> Path | None:
    config_path = _repo_root() / ".local" / "config.yaml"
    if not config_path.is_file():
        return None
    try:
        import yaml

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, ValueError):
        return None
    value = loaded.get(key) if isinstance(loaded, dict) else None
    return Path(value).expanduser() if isinstance(value, str) and value else None


def resolve_mano_model_root(explicit_root: Path | None = None) -> Path:
    """Resolve MANO root in CLI, environment, then local-config order."""

    candidate = explicit_root
    if candidate is None:
        env_value = os.environ.get("MANO_MODEL_ROOT")
        candidate = Path(env_value).expanduser() if env_value else None
    if candidate is None:
        candidate = _local_config_value("mano_model_root")
    if candidate is None:
        raise GrabAdapterError(
            "MANO model root is required for hand reconstruction; pass --mano-model-root, "
            "set MANO_MODEL_ROOT, or set mano_model_root in .local/config.yaml"
        )
    return candidate


def resolve_adapter_grab_root(sequence_path: Path, explicit_root: Path | None = None) -> Path:
    if explicit_root is not None:
        return resolve_grab_root(sequence_path, explicit_root)
    env_value = os.environ.get("GRAB_ROOT")
    if env_value:
        return resolve_grab_root(sequence_path, Path(env_value))
    local_value = _local_config_value("grab_root")
    if local_value is not None:
        return resolve_grab_root(sequence_path, local_value)
    return resolve_grab_root(sequence_path)


def _object_pose_scene(params: dict[str, np.ndarray]) -> np.ndarray:
    """Convert official GRAB row-vector object motion to canonical column-vector SE(3).

    The official GRAB ``ObjectModel`` computes ``v_template @ R``.  With this
    package's column-vector convention the equivalent rotation block is ``R.T``.
    """

    if "global_orient" not in params or "transl" not in params:
        raise GrabAdapterError("GRAB object params require global_orient and transl")
    rotations = axis_angle_to_matrix(params["global_orient"])
    translations = np.asarray(params["transl"], dtype=np.float64)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], translations.shape[0], axis=0)
    poses[:, :3, :3] = np.swapaxes(rotations, -1, -2)
    poses[:, :3, 3] = translations
    return poses


def _validate_render_result(render: ManoRenderResult, frame_count: int) -> None:
    if render.vertices_scene.ndim != 3 or render.vertices_scene.shape[0] != frame_count:
        raise GrabAdapterError(
            f"MANO backend vertices must have shape [T,V,3], got {render.vertices_scene.shape}"
        )
    if render.vertices_scene.shape[-1] != 3:
        raise GrabAdapterError("MANO backend vertices must end in 3 coordinates")
    if render.faces.ndim != 2 or render.faces.shape[1:] != (3,):
        raise GrabAdapterError("MANO backend faces must have shape [F,3]")
    if render.wrist_pose_scene.shape != (frame_count, 4, 4):
        raise GrabAdapterError("MANO backend wrist pose must have shape [T,4,4]")
    if render.joints_scene is not None and render.joints_scene.shape[0] != frame_count:
        raise GrabAdapterError("MANO backend joints frame count mismatch")


class GrabInspectionAdapter(HOIDatasetAdapter):
    """Load one GRAB ``.npz`` and one selected hand without dataset enumeration."""

    adapter_name = "grab_inspect"
    adapter_version = "1"

    def __init__(
        self,
        *,
        sequence_path: str | Path,
        hand: str = "right",
        grab_root: str | Path | None = None,
        mano_model_root: str | Path | None = None,
        backend: ManoBackend | None = None,
    ) -> None:
        if hand not in {"left", "right"}:
            raise GrabAdapterError("hand must be left or right")
        self.sequence_path = Path(sequence_path).expanduser()
        self.hand_side = hand
        self.grab_root_override = None if grab_root is None else Path(grab_root).expanduser()
        self.mano_model_root = (
            None if mano_model_root is None else Path(mano_model_root).expanduser()
        )
        self.backend = backend

    def _record(self):
        try:
            return read_grab_npz(self.sequence_path)
        except (GrabParseError, OSError) as exc:
            raise GrabAdapterError(str(exc)) from exc

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        record = self._record()
        root = resolve_adapter_grab_root(self.sequence_path, self.grab_root_override)
        hand = record.hand(self.hand_side)
        object_mesh = resolve_grab_resource(root, record.object.mesh_relative, "object mesh")
        hand_vtemp = resolve_grab_resource(
            root, hand.vtemp_relative, f"{self.hand_side} personalized vtemp"
        )
        return {
            "dataset_name": "grab",
            "sequence_id": self.sequence_path.stem,
            "source_path": str(self.sequence_path),
            "subject_id": record.subject_id,
            "object_name": record.object_name,
            "motion_intent": record.motion_intent,
            "gender": record.gender,
            "hand": self.hand_side,
            "native_fps": record.native_fps,
            "num_frames": record.num_frames,
            "n_comps": record.n_comps,
            "hand_vtemp": str(hand_vtemp),
            "object_mesh": str(object_mesh),
            "contact": record.contact_metadata,
            "table": record.table_metadata,
            "source_hash": record.source_hash,
            "no_temporal_resampling": True,
            "no_spatial_sampling": True,
        }

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        record = self._record().clip(frame_range)
        root = resolve_adapter_grab_root(self.sequence_path, self.grab_root_override)
        hand_record = record.hand(self.hand_side)
        hand_vtemp_path = resolve_grab_resource(
            root, hand_record.vtemp_relative, f"{self.hand_side} personalized vtemp"
        )
        object_mesh_path = resolve_grab_resource(root, record.object.mesh_relative, "object mesh")
        hand_vtemp, hand_vtemp_faces = load_ply_mesh(hand_vtemp_path)
        object_vertices, object_faces = load_ply_mesh(object_mesh_path)
        backend = self.backend
        if hand_record.vertices_scene is not None:
            rotations = axis_angle_to_matrix(hand_record.params["global_orient"])
            pose = np.repeat(np.eye(4, dtype=np.float64)[None, ...], record.num_frames, axis=0)
            pose[:, :3, :3] = rotations
            pose[:, :3, 3] = hand_record.params["transl"]
            render = ManoRenderResult(
                vertices_scene=hand_record.vertices_scene,
                faces=hand_vtemp_faces,
                wrist_pose_scene=pose,
                model_profile="source_vertices",
            )
        else:
            if backend is None:
                try:
                    from toporetarget.data.mano_backends.smplx_backend import SmplxManoBackend

                    backend = SmplxManoBackend(resolve_mano_model_root(self.mano_model_root))
                except (GrabAdapterError, ManoBackendError, ImportError) as exc:
                    raise GrabAdapterError(str(exc)) from exc
            try:
                assert backend is not None
                render = backend.render(
                    params=hand_record.params,
                    v_template=hand_vtemp,
                    side=self.hand_side,
                    frame_count=record.num_frames,
                )
            except (ManoBackendError, ValueError, KeyError) as exc:
                raise GrabAdapterError(str(exc)) from exc
        _validate_render_result(render, record.num_frames)
        if render.faces.size and int(render.faces.max()) >= hand_vtemp.shape[0]:
            raise GrabAdapterError(
                "MANO backend face index exceeds personalized vtemp vertex count"
            )

        object_poses = _object_pose_scene(record.object.params)
        timestamps = (
            record.start_frame + np.arange(record.num_frames, dtype=np.float64)
        ) / record.native_fps
        provenance = ProvenanceRecord(
            source_dataset="grab",
            source_sequence=self.sequence_path.stem,
            source_file=str(self.sequence_path),
            source_hash=record.source_hash,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            source_coordinate_convention=(
                "native GRAB scene frame; hand axis-angle uses MANO active column convention; "
                "official object row-vector v@R converted to column R.T"
            ),
            conversion_options={
                "hand": self.hand_side,
                "frame_range": [record.start_frame, record.start_frame + record.num_frames],
                "object_mesh_path": str(object_mesh_path),
                "hand_vtemp_path": str(hand_vtemp_path),
                "mano_backend": render.model_profile,
                "dtype": "float64 canonical arrays",
            },
        )
        metadata = SequenceMetadata(
            dataset_name="grab",
            sequence_id=self.sequence_path.stem,
            native_fps=record.native_fps,
            timestamps=timestamps,
            source_frame_name="GRAB_native",
            scene_frame_name="S_GRAB_native",
            provenance=provenance,
            metadata={
                "gender": record.gender,
                "subject_id": record.subject_id,
                "object_name": record.object_name,
                "motion_intent": record.motion_intent,
                "n_comps": record.n_comps,
                "contact": record.contact_metadata,
                "table": record.table_metadata,
                "axis_semantics": "unknown in source package",
            },
        )
        mano_hand_pose = hand_record.params.get("fullpose", hand_record.params.get("hand_pose"))
        mano_parameters = ManoParameterTrack(
            global_orient_aa=hand_record.params.get("global_orient"),
            hand_pose_aa=mano_hand_pose,
            transl=hand_record.params.get("transl"),
            model_profile=render.model_profile,
        )
        keypoints: dict[str, KeypointTrack] = {}
        if render.joints_scene is not None:
            layout = render.keypoint_layout or "mano_native"
            keypoints[layout] = KeypointTrack(
                render.joints_scene,
                layout_name=layout,
                valid=np.ones(render.joints_scene.shape[:2], dtype=bool),
            )
        hand = HandTrack(
            hand_id=f"hand_{self.hand_side[0]}",
            side=self.hand_side,
            wrist_pose_scene=PoseTrack(
                render.wrist_pose_scene,
                child_frame_name=f"H_{self.hand_side[0].upper()}",
            ),
            valid=np.ones(record.num_frames, dtype=bool),
            keypoint_tracks=keypoints,
            mesh=MeshDefinition(
                hand_vtemp,
                render.faces,
                mesh_frame_name=f"H_{self.hand_side[0].upper()}",
                mesh_id=f"grab_{self.hand_side}_personalized_vtemp",
            ),
            vertices_scene=render.vertices_scene,
            mano_parameters=mano_parameters,
            metadata={
                "source_vtemp": hand_record.vtemp_relative,
                "source_pose_fields": sorted(hand_record.params),
                "selected_hand": self.hand_side,
                "no_mediapipe_mapping": True,
            },
        )
        obj = RigidObjectTrack(
            object_id=record.object_name,
            mesh=MeshDefinition(
                object_vertices,
                object_faces,
                mesh_frame_name="O",
                mesh_id=f"grab_{record.object_name}",
            ),
            pose_scene=PoseTrack(object_poses, child_frame_name="O"),
            valid=np.ones(record.num_frames, dtype=bool),
            metadata={
                "source_mesh": record.object.mesh_relative,
                "official_rotation_convention": "v@R",
            },
        )
        sequence_result = HOISequence(metadata=metadata, hands=[hand], rigid_objects=[obj])
        sequence_result.validate()
        return sequence_result

    def canonicalize(self, sequence: HOISequence, **kwargs: Any) -> HOISequence:
        sequence.validate()
        return sequence

    def supported_fields(self) -> tuple[str, ...]:
        return (
            "gender",
            "sbj_id",
            "obj_name",
            "motion_intent",
            "framerate",
            "n_frames",
            "selected_hand.params",
            "selected_hand.vtemp",
            "object.params",
            "object.object_mesh",
            "table.metadata",
            "contact.metadata",
        )


__all__ = [
    "GrabAdapterError",
    "GrabInspectionAdapter",
    "resolve_adapter_grab_root",
    "resolve_mano_model_root",
]
