"""Minimal, explicit-path GRAB NPZ and mesh reader.

This module reads one NPZ only. It does not enumerate subjects, sequences, or
contact arrays across the GRAB tree.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange
from toporetarget.utils.hashing import sha256_file


class GrabParseError(ValueError):
    """Raised when one selected GRAB file is not in a supported structure."""


@dataclass
class GrabHandRecord:
    side: str
    params: dict[str, np.ndarray]
    vtemp_relative: str
    vertices_scene: np.ndarray | None = None


@dataclass
class GrabObjectRecord:
    params: dict[str, np.ndarray]
    mesh_relative: str


@dataclass
class GrabSequenceRecord:
    source_path: Path
    gender: str
    subject_id: str
    object_name: str
    motion_intent: str
    native_fps: float
    num_frames: int
    n_comps: int
    hands: dict[str, GrabHandRecord]
    object: GrabObjectRecord
    table_metadata: dict[str, Any]
    contact_metadata: dict[str, Any]
    source_hash: str
    start_frame: int = 0

    def hand(self, side: str) -> GrabHandRecord:
        if side not in {"left", "right"}:
            raise GrabParseError("hand must be left or right")
        try:
            return self.hands[side]
        except KeyError as exc:
            raise GrabParseError(f"GRAB file has no {side} hand record") from exc

    def clip(self, frame_range: FrameRange | None) -> GrabSequenceRecord:
        if frame_range is None:
            return self
        start, end = frame_range.resolve(self.num_frames)

        def slice_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            result: dict[str, np.ndarray] = {}
            for key, value in params.items():
                if value.ndim >= 1 and value.shape[0] == self.num_frames:
                    result[key] = value[start:end].copy()
                else:
                    result[key] = value.copy()
            return result

        hands = {
            side: GrabHandRecord(
                item.side,
                slice_params(item.params),
                item.vtemp_relative,
                None if item.vertices_scene is None else item.vertices_scene[start:end].copy(),
            )
            for side, item in self.hands.items()
        }
        return GrabSequenceRecord(
            source_path=self.source_path,
            gender=self.gender,
            subject_id=self.subject_id,
            object_name=self.object_name,
            motion_intent=self.motion_intent,
            native_fps=self.native_fps,
            num_frames=end - start,
            n_comps=self.n_comps,
            hands=hands,
            object=GrabObjectRecord(slice_params(self.object.params), self.object.mesh_relative),
            table_metadata=self.table_metadata,
            contact_metadata=self.contact_metadata,
            source_hash=self.source_hash,
            start_frame=self.start_frame + start,
        )


def _slice_frames(value: Any, start: int, end: int, num_frames: int) -> np.ndarray:
    """Select a temporal field without changing non-temporal metadata."""

    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == num_frames:
        return array[start:end].copy()
    return array.copy()


def load_grab_auxiliary(
    sequence_path: str | Path,
    *,
    frame_range: FrameRange | None = None,
    include_table: bool = True,
    contact_mode: str = "none",
) -> dict[str, Any]:
    """Load selected table/contact fields for one explicit sequence.

    GRAB stores these values inside pickled dictionaries in an NPZ.  This
    function deliberately loads them only after a sequence and frame range
    have been selected.  ``allow_pickle=True`` is therefore restricted to
    trusted local GRAB files; callers must not use it for arbitrary downloads.
    """

    if contact_mode not in {"none", "source", "binary", "semantic"}:
        raise GrabParseError(f"unsupported contact mode: {contact_mode}")
    source = Path(sequence_path).expanduser()
    with np.load(source, allow_pickle=True) as data:
        num_frames = int(_scalar(data, "n_frames"))
        start, end = (0, num_frames) if frame_range is None else frame_range.resolve(num_frames)
        result: dict[str, Any] = {"frame_range": [start, end]}
        if include_table and "table" in data.files:
            raw_table = _mapping(_scalar(data, "table"), "table")
            params_value = raw_table.get("params")
            params: dict[str, np.ndarray] = {}
            if isinstance(params_value, dict):
                for key, value in params_value.items():
                    array = np.asarray(value)
                    if array.ndim == 0 or array.shape[0] != num_frames:
                        raise GrabParseError(
                            f"GRAB table.params.{key} must have first dimension "
                            f"{num_frames}, got {array.shape}"
                        )
                    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                        raise GrabParseError(f"GRAB table.params.{key} must be finite numeric data")
                    params[key] = _slice_frames(array, start, end, num_frames)
            result["table"] = {
                "params": params,
                "table_mesh": raw_table.get("table_mesh"),
            }
        if contact_mode != "none" and "contact" in data.files:
            raw_contact = _mapping(_scalar(data, "contact"), "contact")
            labels: dict[str, np.ndarray] = {}
            for key, value in raw_contact.items():
                if key == "threshold":
                    labels[key] = np.asarray(value).copy()
                else:
                    array = np.asarray(value)
                    if array.ndim == 0 or array.shape[0] != num_frames:
                        raise GrabParseError(
                            f"GRAB contact.{key} must have first dimension {num_frames}, "
                            f"got {array.shape}"
                        )
                    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                        raise GrabParseError(f"GRAB contact.{key} must be finite numeric data")
                    labels[key] = _slice_frames(array, start, end, num_frames)
            result["contact"] = labels
            result["contact_mode"] = contact_mode
        return result


def object_pose_scene(params: dict[str, np.ndarray]) -> np.ndarray:
    """Convert official GRAB row-vector object motion to column-vector SE(3)."""

    if "global_orient" not in params or "transl" not in params:
        raise GrabParseError("GRAB object params require global_orient and transl")
    from toporetarget.data.mano_backends.base import axis_angle_to_matrix

    rotations = axis_angle_to_matrix(params["global_orient"])
    translations = np.asarray(params["transl"], dtype=np.float64)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], translations.shape[0], axis=0)
    poses[:, :3, :3] = np.swapaxes(rotations, -1, -2)
    poses[:, :3, 3] = translations
    return poses


def _scalar(data: Any, key: str) -> Any:
    if key not in data.files:
        raise GrabParseError(f"GRAB NPZ is missing required top-level key: {key}")
    value = data[key]
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GrabParseError(f"GRAB field {name} must contain a dictionary")
    return value


def _params(mapping: dict[str, Any], name: str, num_frames: int) -> dict[str, np.ndarray]:
    value = _mapping(mapping.get("params"), f"{name}.params")
    result: dict[str, np.ndarray] = {}
    for key, array in value.items():
        converted = np.asarray(array)
        if converted.ndim == 0 or converted.shape[0] != num_frames:
            raise GrabParseError(
                f"GRAB field {name}.params.{key} must have first dimension "
                f"{num_frames}, got {converted.shape}"
            )
        if not np.issubdtype(converted.dtype, np.number) or not np.all(np.isfinite(converted)):
            raise GrabParseError(f"GRAB field {name}.params.{key} must be finite numeric data")
        result[key] = converted.copy()
    return result


def _required_text(mapping: dict[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise GrabParseError(f"GRAB field {name}.{key} must be a non-empty relative path")
    return value


def read_grab_npz(
    sequence_path: str | Path, *, compute_source_hash: bool = True
) -> GrabSequenceRecord:
    """Parse one explicit GRAB NPZ and preserve its native temporal contract."""

    source = Path(sequence_path).expanduser()
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise GrabParseError(f"--sequence-path must be one existing .npz file: {source}")
    required = {
        "gender",
        "sbj_id",
        "framerate",
        "obj_name",
        "body",
        "object",
        "n_frames",
        "motion_intent",
    }
    with np.load(source, allow_pickle=True) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise GrabParseError(f"GRAB NPZ is missing keys: {', '.join(missing)}")
        num_frames = int(_scalar(data, "n_frames"))
        native_fps = float(_scalar(data, "framerate"))
        if num_frames <= 0 or not np.isfinite(native_fps) or native_fps <= 0:
            raise GrabParseError("GRAB n_frames and framerate must be positive")
        hands: dict[str, GrabHandRecord] = {}
        for side, field_name in (("left", "lhand"), ("right", "rhand")):
            if field_name not in data.files:
                continue
            raw_hand = _mapping(_scalar(data, field_name), field_name)
            vtemp = _required_text(raw_hand, "vtemp", field_name)
            raw_vertices = raw_hand.get("vertices", raw_hand.get("verts"))
            vertices = None
            if raw_vertices is not None:
                vertices = np.asarray(raw_vertices, dtype=np.float64)
                if vertices.shape[0] != num_frames or vertices.ndim != 3 or vertices.shape[-1] != 3:
                    raise GrabParseError(
                        f"GRAB field {field_name}.vertices must have shape [T,V,3], "
                        f"got {vertices.shape}"
                    )
            hands[side] = GrabHandRecord(
                side,
                _params(raw_hand, field_name, num_frames),
                vtemp,
                vertices,
            )
        raw_object = _mapping(_scalar(data, "object"), "object")
        mesh_relative = _required_text(raw_object, "object_mesh", "object")
        object_record = GrabObjectRecord(_params(raw_object, "object", num_frames), mesh_relative)
        raw_table = _mapping(_scalar(data, "table"), "table") if "table" in data.files else {}
        contact_metadata: dict[str, Any] = {"present": "contact" in data.files}
        if "contact" in data.files:
            contact_metadata.update(
                {
                    "fields": ["body", "object"],
                    "source_representation": "GRAB per-vertex contact labels",
                    "loaded": False,
                }
            )
        table_metadata = {
            "fields": sorted(raw_table.keys()),
            "mesh_relative": raw_table.get("table_mesh"),
            "params_present": "params" in raw_table,
        }
        return GrabSequenceRecord(
            source_path=source,
            gender=str(_scalar(data, "gender")),
            subject_id=str(_scalar(data, "sbj_id")),
            object_name=str(_scalar(data, "obj_name")),
            motion_intent=str(_scalar(data, "motion_intent")),
            native_fps=native_fps,
            num_frames=num_frames,
            n_comps=int(_scalar(data, "n_comps")) if "n_comps" in data.files else 0,
            hands=hands,
            object=object_record,
            table_metadata=table_metadata,
            contact_metadata=contact_metadata,
            source_hash=(
                sha256_file(source)
                if compute_source_hash
                else f"stat:{source.stat().st_size}:{source.stat().st_mtime_ns}"
            ),
            start_frame=0,
        )


def resolve_grab_root(sequence_path: Path, explicit_root: Path | None = None) -> Path:
    """Resolve an installation root without scanning outside the selected path ancestry."""

    if explicit_root is not None:
        root = explicit_root.expanduser()
        if not (root / "tools" / "object_meshes").is_dir():
            raise GrabParseError(f"--grab-root has no tools/object_meshes directory: {root}")
        return root
    for candidate in (sequence_path.parent, *sequence_path.parents):
        if (candidate / "tools" / "object_meshes").is_dir() and (candidate / "grab").is_dir():
            return candidate
    raise GrabParseError(
        "could not infer GRAB root; pass --grab-root pointing to the directory "
        "containing grab/ and tools/"
    )


def resolve_grab_resource(root: Path, relative_path: str, label: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise GrabParseError(f"{label} escapes GRAB root: {relative_path}") from exc
    if not path.is_file():
        raise GrabParseError(f"{label} does not exist: {path}")
    return path


def load_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ASCII or binary little-endian triangle PLY files used by GRAB."""

    source = Path(path)
    raw = source.read_bytes()
    marker = b"end_header"
    if marker not in raw:
        raise GrabParseError(f"PLY has no end_header: {source}")
    header_end = raw.index(marker) + len(marker)
    newline_end = raw.find(b"\n", header_end)
    if newline_end < 0:
        raise GrabParseError(f"PLY header has no terminating newline: {source}")
    header = raw[:newline_end].decode("ascii", errors="strict")
    elements: dict[str, int] = {}
    format_name = ""
    for line in header.splitlines():
        words = line.split()
        if words[:1] == ["format"]:
            format_name = words[1]
        elif words[:1] == ["element"] and len(words) >= 3:
            elements[words[1]] = int(words[2])
    vertex_count = elements.get("vertex")
    face_count = elements.get("face")
    if (
        vertex_count is None
        or face_count is None
        or format_name not in {"ascii", "binary_little_endian"}
    ):
        raise GrabParseError(f"unsupported PLY structure: {source}")
    if format_name == "ascii":
        lines = raw[newline_end + 1 :].decode("ascii").splitlines()
        vertices = np.asarray(
            [[float(item) for item in line.split()[:3]] for line in lines[:vertex_count]],
            dtype=np.float64,
        )
        face_lines = lines[vertex_count : vertex_count + face_count]
        faces = np.asarray(
            [[int(item) for item in line.split()[1:4]] for line in face_lines], dtype=np.int64
        )
    else:
        offset = newline_end + 1
        vertices = (
            np.frombuffer(raw, dtype="<f4", count=vertex_count * 3, offset=offset)
            .reshape(vertex_count, 3)
            .astype(np.float64)
        )
        offset += vertex_count * 3 * 4
        faces = np.empty((face_count, 3), dtype=np.int64)
        for index in range(face_count):
            count = raw[offset]
            offset += 1
            if count != 3:
                raise GrabParseError("only triangle PLY faces are supported")
            faces[index] = struct.unpack_from("<3i", raw, offset)
            offset += 12
    if vertices.shape != (vertex_count, 3) or faces.shape != (face_count, 3):
        raise GrabParseError(f"invalid PLY vertex/face shape: {source}")
    return vertices, faces


__all__ = [
    "GrabHandRecord",
    "GrabObjectRecord",
    "GrabParseError",
    "GrabSequenceRecord",
    "load_ply_mesh",
    "load_grab_auxiliary",
    "object_pose_scene",
    "read_grab_npz",
    "resolve_grab_resource",
    "resolve_grab_root",
]
