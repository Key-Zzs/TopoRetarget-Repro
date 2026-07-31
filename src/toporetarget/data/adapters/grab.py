"""Production GRAB dataset adapter built on the Stage 2B reader/backend.

The adapter is sequence-scoped by construction.  It can resolve a selected
sequence through the disposable JSONL index, but loading MANO, contacts,
meshes, and canonical geometry only happens after ``load_sequence`` is
explicitly called.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange, HOIDatasetAdapter
from toporetarget.data.contacts.grab import ContactLoadError, build_grab_contacts
from toporetarget.data.indexes.grab import (
    load_grab_index,
    resolve_grab_dataset_root,
)
from toporetarget.data.mano_backends.base import (
    ManoBackend,
    ManoBackendError,
    ManoRenderResult,
    axis_angle_to_matrix,
)
from toporetarget.data.readers.grab import (
    GrabParseError,
    GrabSequenceRecord,
    load_grab_auxiliary,
    load_ply_mesh,
    object_pose_scene,
    read_grab_npz,
    resolve_grab_resource,
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
from toporetarget.data.storage import StorageError, load_hoi_sequence, save_hoi_sequence
from toporetarget.keypoints.mano_to_mediapipe import (
    ManoToMediaPipe21Converter,
    load_mano_model_geometry,
)
from toporetarget.utils.hashing import sha256_file


class GrabAdapterError(RuntimeError):
    """Raised when a selected GRAB sequence cannot satisfy adapter options."""


@dataclass(frozen=True)
class GrabLoadOptions:
    hands: str = "auto"
    start_frame: int = 0
    end_frame: int | None = None
    include_table: bool = True
    contact_mode: str = "source"
    include_mediapipe21: bool = True
    mediapipe_mapping_profile: str = "mano_v1_2_smplx_to_mediapipe21"
    include_vertices: bool = True
    include_native_joints: bool = True
    cache_dtype: str = "float64"
    source_hash_mode: str = "sha256"
    strict: bool = True

    def validate(self) -> None:
        if self.hands not in {"auto", "right", "left", "both"}:
            raise GrabAdapterError("hands must be auto, right, left, or both")
        if self.contact_mode not in {"none", "source", "binary", "semantic"}:
            raise GrabAdapterError("contact_mode must be none, source, binary, or semantic")
        if self.cache_dtype not in {"float32", "float64"}:
            raise GrabAdapterError("cache_dtype must be float32 or float64")
        if self.source_hash_mode not in {"none", "stat", "sha256"}:
            raise GrabAdapterError("source_hash_mode must be none, stat, or sha256")
        if self.start_frame < 0 or (
            self.end_frame is not None and self.end_frame <= self.start_frame
        ):
            raise GrabAdapterError("start_frame is inclusive and end_frame is exclusive")


def _hash_options(options: GrabLoadOptions) -> str:
    return hashlib.sha256(json.dumps(asdict(options), sort_keys=True).encode("utf-8")).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _git_commit() -> str:
    configured = os.environ.get("TOPORETARGET_CODE_COMMIT")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _cast_float_arrays(value: Any, dtype: str) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return value.astype(dtype, copy=False)
    if isinstance(value, dict):
        return {key: _cast_float_arrays(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_float_arrays(item, dtype) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            setattr(value, name, _cast_float_arrays(getattr(value, name), dtype))
    return value


def _model_hash(root: Path | None, side: str, vertex_count: int) -> str | None:
    if root is None:
        return None
    try:
        return load_mano_model_geometry(
            root, side=side, expected_vertex_count=vertex_count
        ).model_hash
    except Exception as exc:  # the backend provides the actionable model error later
        raise GrabAdapterError(f"could not inspect MANO {side} model: {exc}") from exc


def _validate_render(render: ManoRenderResult, frame_count: int) -> None:
    if render.vertices_scene.ndim != 3 or render.vertices_scene.shape[0] != frame_count:
        raise GrabAdapterError(
            f"MANO vertices must have shape [T,V,3], got {render.vertices_scene.shape}"
        )
    if render.vertices_scene.shape[-1] != 3 or not np.all(np.isfinite(render.vertices_scene)):
        raise GrabAdapterError("MANO vertices are not finite [T,V,3]")
    if render.faces.ndim != 2 or render.faces.shape[1:] != (3,):
        raise GrabAdapterError("MANO faces must have shape [F,3]")
    if render.wrist_pose_scene.shape != (frame_count, 4, 4):
        raise GrabAdapterError("MANO wrist pose must have shape [T,4,4]")
    if render.joints_scene is not None and render.joints_scene.shape[0] != frame_count:
        raise GrabAdapterError("MANO native joints frame count mismatch")


class GrabDatasetAdapter(HOIDatasetAdapter):
    """Load exactly one GRAB sequence/clip into the canonical HOI schema."""

    adapter_name = "grab_dataset_adapter"
    adapter_version = "1.0.0"

    def __init__(
        self,
        *,
        sequence_path: str | Path | None = None,
        grab_root: str | Path | None = None,
        index: str | Path | None = None,
        mano_model_root: str | Path | None = None,
        backend: ManoBackend | None = None,
        options: GrabLoadOptions | None = None,
    ) -> None:
        self.sequence_path = None if sequence_path is None else Path(sequence_path).expanduser()
        self.grab_root_override = None if grab_root is None else Path(grab_root).expanduser()
        self.index_path = None if index is None else Path(index).expanduser()
        self.mano_model_root = (
            None if mano_model_root is None else Path(mano_model_root).expanduser()
        )
        self.backend = backend
        self.options = options or GrabLoadOptions()
        self.options.validate()

    def _resolve_path(self, sequence: str | Path | None = None) -> Path:
        value: Path | None = self.sequence_path
        if self.sequence_path is not None and sequence not in {None, ""}:
            candidate = Path(sequence).expanduser()
            value = (
                candidate
                if candidate.suffix.lower() == ".npz" and candidate.is_file()
                else self.sequence_path
            )
        if value is not None and value.suffix.lower() == ".npz" and value.is_file():
            return value
        sequence_id = str(sequence or "")
        index_path = self.index_path or Path(".local/index/grab")
        if sequence_id:
            entries = {item["sequence_id"]: item for item in load_grab_index(index_path)}
            item = entries.get(sequence_id)
            if item is None:
                raise GrabAdapterError(
                    f"sequence {sequence_id!r} is not in {index_path}; build the index "
                    "or pass --sequence-path"
                )
            root = (
                Path(index_path / "manifest.json").resolve().parent
                if (index_path / "manifest.json").is_file()
                else None
            )
            if root is not None:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                return Path(manifest["grab_root"]) / item["relative_path"]
        if value is not None:
            raise GrabAdapterError(f"--sequence-path must be an existing .npz: {value}")
        raise GrabAdapterError("a GRAB --sequence or --sequence-path is required")

    def resolve_sequence(self, sequence: str | Path | None = None) -> dict[str, Any]:
        path = self._resolve_path(sequence)
        root = resolve_grab_dataset_root(self.grab_root_override, sequence_path=path)
        return {
            "sequence_id": f"{path.parent.name}/{path.stem}",
            "path": path,
            "grab_root": root,
        }

    def _record(self, sequence: str | Path | None = None) -> tuple[Path, Path, GrabSequenceRecord]:
        resolved = self.resolve_sequence(sequence)
        try:
            return resolved["path"], resolved["grab_root"], read_grab_npz(resolved["path"])
        except (GrabParseError, OSError, KeyError) as exc:
            raise GrabAdapterError(str(exc)) from exc

    def _options_for(self, options: GrabLoadOptions | None, **overrides: Any) -> GrabLoadOptions:
        result = options or self.options
        values = {key: value for key, value in overrides.items() if value is not None}
        if "frame_range" in values:
            frame_range = values.pop("frame_range")
            values["start_frame"] = frame_range.start
            values["end_frame"] = frame_range.end
        result = replace(result, **values)
        result.validate()
        return result

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        path, root, record = self._record(sequence or None)
        hands = sorted(record.hands)
        descriptions: dict[str, Any] = {}
        for side in hands:
            hand = record.hands[side]
            vtemp = resolve_grab_resource(root, hand.vtemp_relative, f"{side} personalized vtemp")
            descriptions[side] = {
                "parameter_keys": sorted(hand.params),
                "vtemp_reference": hand.vtemp_relative,
                "vtemp_path": str(vtemp),
            }
        object_mesh = resolve_grab_resource(root, record.object.mesh_relative, "object mesh")
        table = record.table_metadata
        table_mesh = None
        if table.get("mesh_relative"):
            table_mesh = resolve_grab_resource(root, str(table["mesh_relative"]), "table mesh")
        stat = path.stat()
        return {
            "dataset_name": "grab",
            "sequence_id": f"{path.parent.name}/{path.stem}",
            "source_path": str(path),
            "subject_id": record.subject_id,
            "object": record.object_name,
            "object_name": record.object_name,
            "action": record.motion_intent,
            "motion_intent": record.motion_intent,
            "gender": record.gender,
            "frame_count": record.num_frames,
            "num_frames": record.num_frames,
            "native_fps": record.native_fps,
            "available_hands": hands,
            "mano_parameters": descriptions,
            "personalized_vtemp_references": {
                side: item["vtemp_reference"] for side, item in descriptions.items()
            },
            "object_mesh": str(object_mesh),
            "table_mesh": None if table_mesh is None else str(table_mesh),
            "table": table,
            "contact": record.contact_metadata,
            "source_coordinate_metadata": {
                "scene_frame": "native GRAB scene frame",
                "hand_global_pose": "MANO global_orient axis-angle and transl",
                "object_pose": "official GRAB row-vector v@R; canonical column block R.T",
            },
            "source_file_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "source_hash": sha256_file(path),
            "canonical_conversion_performed": False,
            "no_temporal_resampling": True,
            "no_spatial_sampling": True,
        }

    def _selected_sides(self, record: GrabSequenceRecord, options: GrabLoadOptions) -> list[str]:
        available: list[str] = [side for side in ("right", "left") if side in record.hands]
        if options.hands == "auto":
            return available
        requested = ["right", "left"] if options.hands == "both" else [options.hands]
        missing = [side for side in requested if side not in record.hands]
        if missing and (options.strict or options.hands != "auto"):
            raise GrabAdapterError(f"requested GRAB hand(s) are missing: {', '.join(missing)}")
        return [side for side in requested if side in record.hands]

    def _render_hand(
        self,
        record: GrabSequenceRecord,
        root: Path,
        source_path: Path,
        side: str,
        options: GrabLoadOptions,
    ) -> tuple[HandTrack, dict[str, Any]]:
        item = record.hands[side]
        vtemp_path = resolve_grab_resource(root, item.vtemp_relative, f"{side} personalized vtemp")
        vtemp, vtemp_faces = load_ply_mesh(vtemp_path)
        backend = self.backend
        if item.vertices_scene is not None:
            rotation = axis_angle_to_matrix(item.params["global_orient"])
            pose = np.repeat(np.eye(4, dtype=np.float64)[None, ...], record.num_frames, axis=0)
            pose[:, :3, :3] = rotation
            pose[:, :3, 3] = item.params["transl"]
            render = ManoRenderResult(
                item.vertices_scene, vtemp_faces, pose, model_profile="source_vertices"
            )
        else:
            if backend is None:
                try:
                    from toporetarget.data.adapters.grab_inspect import resolve_mano_model_root
                    from toporetarget.data.mano_backends.smplx_backend import SmplxManoBackend

                    backend = SmplxManoBackend(resolve_mano_model_root(self.mano_model_root))
                except (GrabAdapterError, ManoBackendError, ImportError) as exc:
                    raise GrabAdapterError(str(exc)) from exc
            try:
                from toporetarget.data.mano_backends.smplx_backend import SmplxManoBackend

                if isinstance(backend, SmplxManoBackend):
                    render = backend.render_axis_angle(
                        params=item.params,
                        v_template=vtemp,
                        side=side,
                        frame_count=record.num_frames,
                        flat_hand_mean=True,
                        dataset_name="grab",
                        source_annotation_path=source_path,
                        source_annotation_hash=sha256_file(source_path),
                    )
                else:
                    render = backend.render(
                        params=item.params,
                        v_template=vtemp,
                        side=side,
                        frame_count=record.num_frames,
                    )
            except (ManoBackendError, ValueError, KeyError) as exc:
                raise GrabAdapterError(str(exc)) from exc
        _validate_render(render, record.num_frames)
        if render.faces.size and int(render.faces.max()) >= vtemp.shape[0]:
            raise GrabAdapterError(
                f"MANO {side} face index exceeds personalized vtemp vertex count"
            )
        native_layout = render.keypoint_layout or "mano_native"
        keypoints: dict[str, KeypointTrack] = {}
        if options.include_native_joints and render.joints_scene is not None:
            semantic_names = None
            if native_layout in {"mano16_smplx", "mano16"}:
                from toporetarget.keypoints.registry import get_layout

                semantic_names = list(get_layout("mano16_smplx").semantic_names)
            keypoints[native_layout] = KeypointTrack(
                render.joints_scene,
                layout_name=native_layout,
                valid=np.ones(render.joints_scene.shape[:2], dtype=bool),
                semantic_names=semantic_names,
                provenance={"source": "GRAB MANO backend", "side": side},
            )
        fullpose = item.params.get("fullpose")
        hand = HandTrack(
            hand_id=f"{side}_hand",
            side=side,
            wrist_pose_scene=PoseTrack(render.wrist_pose_scene, child_frame_name=f"W_{side}"),
            valid=np.ones(record.num_frames, dtype=bool),
            keypoint_tracks=keypoints,
            mesh=MeshDefinition(
                vtemp,
                render.faces,
                mesh_frame_name=f"W_{side}",
                mesh_id=f"grab_{side}_personalized_vtemp",
                mesh_hash=sha256_file(vtemp_path),
            ),
            vertices_scene=render.vertices_scene if options.include_vertices else None,
            mano_parameters=ManoParameterTrack(
                global_orient_aa=item.params.get("global_orient"),
                hand_pose_aa=fullpose if fullpose is not None else item.params.get("hand_pose"),
                transl=item.params.get("transl"),
                personalized_v_template_reference=str(vtemp_path),
                model_profile=render.model_profile,
            ),
            metadata={
                "side": side,
                "source_vtemp": item.vtemp_relative,
                "source_pose_fields": sorted(item.params),
                "source_parameters": {key: value.copy() for key, value in item.params.items()},
                "vtemp_hash": sha256_file(vtemp_path),
                "mano_model_hash": _model_hash(
                    self.mano_model_root, side, render.vertices_scene.shape[1]
                )
                if self.mano_model_root
                else None,
                "backend_version": render.model_profile,
            },
        )
        return hand, {
            "vtemp_path": vtemp_path,
            "vtemp_hash": sha256_file(vtemp_path),
            "render": render,
        }

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        options: GrabLoadOptions | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        started = time.perf_counter()
        path, root, full_record = self._record(sequence or None)
        selected = self._options_for(options, frame_range=frame_range, **kwargs)
        selected_range = FrameRange(selected.start_frame, selected.end_frame)
        record = full_record.clip(selected_range)
        sides = self._selected_sides(record, selected)
        if not sides:
            raise GrabAdapterError("no usable GRAB hands were selected")
        backend_started = time.perf_counter()
        rendered: list[HandTrack] = []
        hand_meta: dict[str, Any] = {}
        for side in sides:
            hand, metadata = self._render_hand(record, root, path, side, selected)
            rendered.append(hand)
            hand_meta[side] = metadata
        object_mesh_path = resolve_grab_resource(root, record.object.mesh_relative, "object mesh")
        object_vertices, object_faces = load_ply_mesh(object_mesh_path)
        object_track = RigidObjectTrack(
            object_id=record.object_name,
            mesh=MeshDefinition(
                object_vertices,
                object_faces,
                mesh_frame_name="O",
                mesh_id=f"grab_{record.object_name}",
                mesh_hash=sha256_file(object_mesh_path),
            ),
            pose_scene=PoseTrack(object_pose_scene(record.object.params), child_frame_name="O"),
            valid=np.ones(record.num_frames, dtype=bool),
            metadata={
                "role": "primary_manipulation_object",
                "object_name": record.object_name,
                "source_mesh": record.object.mesh_relative,
                "source_mesh_hash": sha256_file(object_mesh_path),
                "official_rotation_convention": "v@R",
            },
        )
        rigid_objects = [object_track]
        auxiliary = load_grab_auxiliary(
            path,
            frame_range=selected_range,
            include_table=selected.include_table,
            contact_mode=selected.contact_mode,
        )
        table_mesh_path: Path | None = None
        if selected.include_table and isinstance(auxiliary.get("table"), dict):
            table_data = auxiliary["table"]
            relative = table_data.get("table_mesh")
            params = table_data.get("params", {})
            if relative and params:
                table_mesh_path = resolve_grab_resource(root, str(relative), "table mesh")
                table_vertices, table_faces = load_ply_mesh(table_mesh_path)
                rigid_objects.append(
                    RigidObjectTrack(
                        object_id="table",
                        mesh=MeshDefinition(
                            table_vertices,
                            table_faces,
                            mesh_frame_name="T",
                            mesh_id="grab_table",
                            mesh_hash=sha256_file(table_mesh_path),
                        ),
                        pose_scene=PoseTrack(object_pose_scene(params), child_frame_name="T"),
                        valid=np.ones(record.num_frames, dtype=bool),
                        metadata={
                            "role": "support_surface",
                            "source_mesh": str(relative),
                            "source_mesh_hash": sha256_file(table_mesh_path),
                        },
                    )
                )
        contacts = []
        if selected.contact_mode != "none" and full_record.contact_metadata.get("present"):
            try:
                contacts = build_grab_contacts(
                    auxiliary,
                    hand_ids=[hand.hand_id for hand in rendered],
                    object_id=record.object_name,
                    object_vertex_count=object_vertices.shape[0],
                    frame_count=record.num_frames,
                    mode=selected.contact_mode,
                    strict=selected.strict,
                    mapping_config=None,
                )
            except ContactLoadError as exc:
                raise GrabAdapterError(str(exc)) from exc
        source_hash = None
        source_stat = path.stat()
        if selected.source_hash_mode == "sha256":
            source_hash = sha256_file(path)
        elif selected.source_hash_mode == "stat":
            stat = path.stat()
            source_hash = f"stat:{stat.st_size}:{stat.st_mtime_ns}"
        timestamps = (
            record.start_frame + np.arange(record.num_frames, dtype=np.float64)
        ) / record.native_fps
        conversion_options = {
            **asdict(selected),
            "sequence_id": f"{path.parent.name}/{path.stem}",
            "frame_range": [record.start_frame, record.start_frame + record.num_frames],
            "selected_hands": sides,
            "object_mesh_path": str(object_mesh_path),
            "table_included": any(item.object_id == "table" for item in rigid_objects),
            "source_size": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "mano_model_hashes": {
                side: rendered[index].metadata.get("mano_model_hash")
                for index, side in enumerate(sides)
            },
            "personalized_vtemp_hashes": {
                side: rendered[index].metadata.get("vtemp_hash") for index, side in enumerate(sides)
            },
            "object_mesh_hash": object_track.mesh.mesh_hash,
            "table_mesh_hash": next(
                (item.mesh.mesh_hash for item in rigid_objects if item.object_id == "table"), None
            ),
            "mapping_profile_hash": next(
                (
                    hand.keypoint_tracks["mediapipe21"].provenance.get("mapping_profile_hash")
                    for hand in rendered
                    if "mediapipe21" in hand.keypoint_tracks
                ),
                None,
            ),
            "code_commit": _git_commit(),
            "contact_mode": selected.contact_mode,
            "conversion_options_hash": _hash_options(selected),
            "mano_reconstruction_seconds": time.perf_counter() - backend_started,
        }
        provenance = ProvenanceRecord(
            source_dataset="grab",
            source_sequence=f"{path.parent.name}/{path.stem}",
            source_file=str(path),
            source_hash=source_hash,
            source_size=int(source_stat.st_size),
            source_mtime_ns=int(source_stat.st_mtime_ns),
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            source_coordinate_convention=(
                "native GRAB scene; MANO global axis-angle/transl; official object row-vector v@R "
                "converted to canonical column-vector R.T; table uses the same GRAB pose convention"
            ),
            conversion_options=conversion_options,
        )
        sequence_metadata = SequenceMetadata(
            dataset_name="grab",
            sequence_id=f"{path.parent.name}/{path.stem}",
            native_fps=record.native_fps,
            timestamps=timestamps,
            source_frame_name="GRAB_native",
            scene_frame_name="S_GRAB_native",
            provenance=provenance,
            metadata={
                "subject_id": record.subject_id,
                "gender": record.gender,
                "object_name": record.object_name,
                "action": record.motion_intent,
                "motion_intent": record.motion_intent,
                "n_comps": record.n_comps,
                "available_hands": sorted(full_record.hands),
                "selected_hands": sides,
                "table": record.table_metadata,
                "table_excluded": not selected.include_table,
                "contact": record.contact_metadata,
                "native_fps_source": record.native_fps,
                "no_temporal_resampling": True,
                "no_spatial_sampling": True,
                "source_tracks_preserved": True,
            },
        )
        result = HOISequence(
            metadata=sequence_metadata,
            hands=rendered,
            rigid_objects=rigid_objects,
            contacts=contacts,
        )
        if selected.include_mediapipe21:
            converter = ManoToMediaPipe21Converter(selected.mediapipe_mapping_profile)
            for hand in result.hands:
                try:
                    track = converter.convert_hand_track(
                        hand,
                        frame_count=result.num_frames,
                        mano_model_root=self.mano_model_root,
                    )
                except Exception as exc:
                    if selected.strict:
                        raise GrabAdapterError(
                            f"MediaPipe-21 conversion failed for {hand.side}: {exc}"
                        ) from exc
                    hand.metadata["mediapipe_mapping_blocker"] = str(exc)
                else:
                    hand.keypoint_tracks["mediapipe21"] = track
        result.metadata.provenance.conversion_options["conversion_seconds"] = (
            time.perf_counter() - started
        )
        result.validate()
        return result

    def canonicalize(self, sequence: HOISequence, **kwargs: Any) -> HOISequence:
        sequence.validate()
        return sequence

    def validate_sequence(self, sequence: HOISequence, **kwargs: Any) -> dict[str, Any]:
        from toporetarget.data.validation.grab import validate_grab_sequence

        return validate_grab_sequence(sequence, **kwargs)

    def load_raw_renderable(
        self, sequence: str, *, frame_range: FrameRange | None = None, **kwargs: Any
    ) -> HOISequence:
        return self.load_sequence(sequence, frame_range=frame_range, **kwargs)

    def load_raw_renderable_for_path(self, sequence_path: str | Path, **kwargs: Any) -> HOISequence:
        return GrabDatasetAdapter(
            sequence_path=sequence_path,
            grab_root=self.grab_root_override,
            mano_model_root=self.mano_model_root,
            backend=self.backend,
            options=self.options,
        ).load_sequence(frame_range=kwargs.pop("frame_range", None), **kwargs)

    def create_cache(
        self,
        sequence: str = "",
        *,
        output: str | Path,
        options: GrabLoadOptions | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Convert one selected sequence and atomically publish an explicit cache."""

        destination = Path(output).expanduser()
        source_path = self.resolve_sequence(sequence or None)["path"]
        if destination.resolve() == source_path.resolve():
            raise GrabAdapterError("input sequence and output cache must be different paths")
        if destination.exists() and not force:
            raise GrabAdapterError(
                f"cache already exists; pass --force to replace it: {destination}"
            )
        sequence_data = self.load_sequence(sequence, options=options, **kwargs)
        cache_dtype = sequence_data.metadata.provenance.conversion_options.get(
            "cache_dtype", "float64"
        )
        if cache_dtype != "float64":
            sequence_data = _cast_float_arrays(copy.deepcopy(sequence_data), cache_dtype)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.tmp-", dir=str(destination.parent or Path("."))
            )
        )
        try:
            save_hoi_sequence(sequence_data, temporary)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
        except (OSError, StorageError, ValueError) as exc:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise GrabAdapterError(f"could not publish GRAB cache {destination}: {exc}") from exc
        return destination

    def compare_raw_canonical(
        self,
        canonical: str | Path | HOISequence,
        *,
        sequence: str = "",
        options: GrabLoadOptions | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from toporetarget.viz.errors import ComparisonMetrics

        canonical_sequence = (
            load_hoi_sequence(canonical) if isinstance(canonical, (str, Path)) else canonical
        )
        raw = self.load_sequence(sequence, options=options, **kwargs)
        return ComparisonMetrics.compute(raw, canonical_sequence).as_dict()

    def supported_fields(self) -> tuple[str, ...]:
        return (
            "gender",
            "sbj_id",
            "obj_name",
            "motion_intent",
            "framerate",
            "n_frames",
            "body.metadata",
            "lhand.params",
            "rhand.params",
            "lhand.vtemp",
            "rhand.vtemp",
            "object.params",
            "object.object_mesh",
            "table.params",
            "table.table_mesh",
            "contact.body",
            "contact.object",
            "contact.threshold",
        )


__all__ = ["GrabAdapterError", "GrabDatasetAdapter", "GrabLoadOptions"]
