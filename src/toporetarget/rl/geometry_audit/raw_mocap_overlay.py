"""Read-only raw HOCap MANO/object overlay resolver for Stage16 replay.

This module deliberately resolves every display frame through the frozen
world-reference provenance.  It never writes a trace, a reference, or source
data, and it has no Isaac/PhysX dependency so its coordinate and timing
contracts can be tested deterministically.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

from toporetarget.adapters.datasets.stage12_base import (
    load_mesh,
    pose_hocap_qxyzw,
    render_mano_pca45,
)
from toporetarget.data.storage import load_hoi_sequence

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
MANO_TIP_VERTEX_INDICES = (745, 333, 444, 555, 672)
RAW_ALIGNMENT_TRANSLATION_TOLERANCE_M = 2.0e-8
RAW_ALIGNMENT_ROTATION_TOLERANCE_RAD = 2.0e-6


class RawMocapOverlayUnavailable(RuntimeError):
    """A fail-closed raw-source error which leaves actual replay available."""


@dataclass(frozen=True)
class RawMocapOverlay:
    """World-frame, trace-time-aligned raw source geometry for one replay."""

    clip: str
    runtime_reference_indices: np.ndarray
    runtime_timestamps_s: np.ndarray
    raw_frame_float: np.ndarray
    raw_mano_vertices_world: np.ndarray
    raw_mano_faces: np.ndarray
    raw_mano_fingertips_world: np.ndarray
    raw_object_vertices_local: np.ndarray
    raw_object_faces: np.ndarray
    raw_object_pose_world_wxyz: np.ndarray
    source_provenance: dict[str, object]
    coordinate_alignment: dict[str, object]
    time_alignment: dict[str, object]

    @property
    def frame_count(self) -> int:
        return int(self.runtime_timestamps_s.size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pose_matrix_to_wxyz(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    quaternion_xyzw = Rotation.from_matrix(matrix[..., :3, :3]).as_quat()
    return np.concatenate(
        (matrix[..., :3, 3], quaternion_xyzw[..., 3:4], quaternion_xyzw[..., :3]), axis=-1
    )


def pose_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert replay ``[xyz,qw,qx,qy,qz]`` poses to homogeneous matrices."""

    value = np.asarray(pose, dtype=np.float64)
    if value.shape[-1] != 7:
        raise ValueError("RAW_MOCAP_POSE_WXYZ_SHAPE_INVALID")
    norm = np.linalg.norm(value[..., 3:], axis=-1)
    if np.any(norm < 1.0e-8) or not np.isfinite(value).all():
        raise ValueError("RAW_MOCAP_POSE_WXYZ_INVALID")
    xyzw = np.concatenate((value[..., 4:], value[..., 3:4]), axis=-1)
    result = np.broadcast_to(np.eye(4, dtype=np.float64), (*value.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = Rotation.from_quat(xyzw / norm[..., None]).as_matrix()
    result[..., :3, 3] = value[..., :3]
    return result


def interpolate_mano_pca_pose(
    timestamps: np.ndarray, pose: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    """Use the source audit's SO(3)+linear-PCA continuous-time contract."""

    source_t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    source = np.asarray(pose, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if source.shape != (source_t.size, 51) or np.any(np.diff(source_t) <= 0.0):
        raise ValueError("RAW_MOCAP_MANO_TIME_SERIES_INVALID")
    if target.size == 0 or target[0] < source_t[0] - 1.0e-12 or target[-1] > source_t[-1] + 1.0e-12:
        raise ValueError("RAW_MOCAP_MANO_RESAMPLE_OUT_OF_RANGE")
    target = np.clip(target, source_t[0], source_t[-1])
    result = np.empty((target.size, 51), dtype=np.float64)
    result[:, :3] = Slerp(source_t, Rotation.from_rotvec(source[:, :3]))(target).as_rotvec()
    for column in range(3, source.shape[1]):
        result[:, column] = np.interp(target, source_t, source[:, column])
    return result


def interpolate_object_pose(
    timestamps: np.ndarray, poses_qxyzw: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    """Resample raw HOCap translation linearly and rotation by shortest-arc SLERP."""

    source_t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    source = np.asarray(poses_qxyzw, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if source.shape != (source_t.size, 7) or np.any(np.diff(source_t) <= 0.0):
        raise ValueError("RAW_MOCAP_OBJECT_TIME_SERIES_INVALID")
    if target.size == 0 or target[0] < source_t[0] - 1.0e-12 or target[-1] > source_t[-1] + 1.0e-12:
        raise ValueError("RAW_MOCAP_OBJECT_RESAMPLE_OUT_OF_RANGE")
    target = np.clip(target, source_t[0], source_t[-1])
    matrices = np.stack([pose_hocap_qxyzw(row) for row in source], axis=0)
    result = np.broadcast_to(np.eye(4, dtype=np.float64), (target.size, 4, 4)).copy()
    result[:, :3, :3] = Slerp(source_t, Rotation.from_matrix(matrices[:, :3, :3]))(
        target
    ).as_matrix()
    for axis in range(3):
        result[:, axis, 3] = np.interp(target, source_t, matrices[:, axis, 3])
    return result


def _metadata(reference_path: Path) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as source:
        if "metadata" not in source.files:
            raise RawMocapOverlayUnavailable("RAW_MOCAP_REFERENCE_PROVENANCE_MISSING")
        value = json.loads(str(source["metadata"].item()))
    if not isinstance(value, dict):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_REFERENCE_PROVENANCE_INVALID")
    return value


def _reference_arrays(reference_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(reference_path, allow_pickle=False) as source:
        required = {"timestamps", "source_frame_indices", "T_world_object_ref"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise RawMocapOverlayUnavailable(
                f"RAW_MOCAP_REFERENCE_FIELDS_MISSING:{','.join(missing)}"
            )
        timestamps = np.asarray(source["timestamps"], dtype=np.float64)
        source_indices = np.asarray(source["source_frame_indices"], dtype=np.int64)
        object_poses = np.asarray(source["T_world_object_ref"], dtype=np.float64)
    if (
        timestamps.ndim != 1
        or source_indices.shape != timestamps.shape
        or object_poses.shape != (timestamps.size, 4, 4)
        or timestamps.size < 2
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_REFERENCE_TIME_CONTRACT_INVALID")
    return timestamps, source_indices, object_poses


def _runtime_reference_indices(trace_path: Path, frame_count: int) -> np.ndarray:
    with np.load(trace_path, allow_pickle=False) as trace:
        if "reference_index" not in trace.files:
            raise RawMocapOverlayUnavailable("RAW_MOCAP_RUNTIME_REFERENCE_INDEX_MISSING")
        indices = np.asarray(trace["reference_index"], dtype=np.int64)
    if indices.shape != (frame_count,) or np.any(indices < 0):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_RUNTIME_REFERENCE_INDEX_INVALID")
    if np.any(np.diff(indices) < 0):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_RUNTIME_REFERENCE_INDEX_NONMONOTONIC")
    return indices


def _selected_raw_slice(summary: dict[str, Any], raw_frames: int) -> tuple[int, int]:
    options = summary["conversion_options"]
    selected = options.get("selected_frame_range") if isinstance(options, dict) else None
    if not isinstance(selected, list) or len(selected) != 2:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_SELECTED_FRAME_RANGE_MISSING")
    start, stop = (int(selected[0]), int(selected[1]))
    if (
        start < 0
        or stop <= start
        or stop - start != len(summary["timestamps"])
        or stop > raw_frames
    ):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_SELECTED_FRAME_RANGE_INVALID")
    return start, stop


def _raw_assets(summary: dict[str, Any], object_id: str) -> dict[str, Path]:
    if summary["source_dataset"] != "hocap" or not summary["no_temporal_resampling"]:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_CANONICAL_PROVENANCE_NOT_IDENTIFIABLE")
    meta = Path(summary["source_file"])
    if meta.name != "meta.yaml" or not meta.is_file():
        raise RawMocapOverlayUnavailable("RAW_MOCAP_SOURCE_META_MISSING")
    data_root = meta.parents[2]
    assets = {
        "meta": meta,
        "poses_m": meta.parent / "poses_m.npy",
        "poses_o": meta.parent / "poses_o.npy",
        "betas": data_root / "calibration/mano" / f"{meta.parent.parent.name}.yaml",
        "object_mesh": data_root / "models" / object_id / "textured_mesh.obj",
        "mano_root": data_root.parents[1] / "shared_assets/body_models/mano",
    }
    missing = [name for name, path in assets.items() if not path.is_file() and name != "mano_root"]
    if missing:
        raise RawMocapOverlayUnavailable(f"RAW_MOCAP_SOURCE_ASSET_MISSING:{','.join(missing)}")
    if not assets["mano_root"].is_dir():
        raise RawMocapOverlayUnavailable("RAW_MANO_ASSET_MISSING")
    return assets


def _canonical_summary(canonical_path: Path) -> dict[str, Any]:
    """Read only the canonical provenance/arrays needed by the overlay.

    IsaacLab intentionally omits optional Zarr support.  When that is the
    active interpreter, a local project-environment subprocess reads the same
    immutable canonical artifact and returns a short-lived array payload.
    """

    def summary(sequence: Any) -> dict[str, Any]:
        primary = sequence.primary_rigid_object()
        hand = sequence.hands[0].mano_parameters
        if hand is None or hand.global_orient_aa is None or hand.transl is None:
            raise RawMocapOverlayUnavailable("RAW_MOCAP_CANONICAL_MANO_PARAMETERS_MISSING")
        provenance = sequence.metadata.provenance
        return {
            "timestamps": np.asarray(sequence.timestamps, dtype=np.float64),
            "source_dataset": provenance.source_dataset,
            "source_file": provenance.source_file,
            "source_sequence": provenance.source_sequence,
            "source_coordinate_convention": provenance.source_coordinate_convention,
            "no_temporal_resampling": bool(provenance.no_temporal_resampling),
            "conversion_options": provenance.conversion_options,
            "object_id": primary.object_id,
            "object_pose": np.asarray(primary.pose_scene.pose_scene, dtype=np.float64),
            "mano_global_orient": np.asarray(hand.global_orient_aa, dtype=np.float64),
            "mano_translation": np.asarray(hand.transl, dtype=np.float64),
        }

    try:
        return summary(load_hoi_sequence(canonical_path))
    except Exception as initial_error:
        base_python = Path("/home/deepcybo/miniconda3/envs/toporetarget-rl/bin/python")
        if not base_python.is_file():
            raise RawMocapOverlayUnavailable(
                f"RAW_MOCAP_CANONICAL_READ_UNAVAILABLE:{initial_error}"
            ) from initial_error
        with tempfile.TemporaryDirectory(prefix="raw_mocap_canonical_") as temporary:
            output = Path(temporary) / "canonical.npz"
            code = f"""
import json, sys
import numpy as np
sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
from toporetarget.data.storage import load_hoi_sequence
sequence = load_hoi_sequence({str(canonical_path)!r})
primary = sequence.primary_rigid_object()
hand = sequence.hands[0].mano_parameters
provenance = sequence.metadata.provenance
np.savez_compressed(
    {str(output)!r}, timestamps=sequence.timestamps,
    object_pose=primary.pose_scene.pose_scene,
    mano_global_orient=hand.global_orient_aa, mano_translation=hand.transl,
    metadata=np.asarray(json.dumps({{
        'source_dataset': provenance.source_dataset, 'source_file': provenance.source_file,
        'source_sequence': provenance.source_sequence,
        'source_coordinate_convention': provenance.source_coordinate_convention,
        'no_temporal_resampling': provenance.no_temporal_resampling,
        'conversion_options': provenance.conversion_options, 'object_id': primary.object_id,
    }})),
)
"""
            try:
                subprocess.run([str(base_python), "-c", code], check=True, capture_output=True)
                with np.load(output, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata"].item()))
                    return {
                        **metadata,
                        "timestamps": np.asarray(archive["timestamps"], dtype=np.float64),
                        "object_pose": np.asarray(archive["object_pose"], dtype=np.float64),
                        "mano_global_orient": np.asarray(
                            archive["mano_global_orient"], dtype=np.float64
                        ),
                        "mano_translation": np.asarray(
                            archive["mano_translation"], dtype=np.float64
                        ),
                    }
            except (OSError, subprocess.SubprocessError, FileNotFoundError) as fallback_error:
                detail = f"{initial_error}; fallback={fallback_error}"
                raise RawMocapOverlayUnavailable(
                    f"RAW_MOCAP_CANONICAL_READ_UNAVAILABLE:{detail}"
                ) from fallback_error


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(np.asarray(first) * np.asarray(second), axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _render_source_mano(
    pose: np.ndarray, *, betas: np.ndarray, mano_root: Path, annotation: Path, source_hash: str
) -> tuple[np.ndarray, np.ndarray]:
    """Use the installed backend, or the existing local base interpreter fallback.

    The fallback is deliberately local-only; it is the same PCA45 renderer used
    by the completed source-contact audit and never downloads a MANO asset.
    """

    try:
        rendered = render_mano_pca45(
            pose,
            side="right",
            mano_model_root=mano_root,
            betas=betas,
            dataset_name="hocap",
            source_annotation_path=annotation,
            source_annotation_hash=source_hash,
        )
        return np.asarray(rendered.vertices, dtype=np.float64), np.asarray(
            rendered.faces, dtype=np.int64
        )
    except Exception as initial_error:
        base_python = Path("/home/deepcybo/miniconda3/bin/python")
        if not base_python.is_file():
            raise RawMocapOverlayUnavailable(
                f"RAW_MANO_ASSET_MISSING:{initial_error}"
            ) from initial_error
        with tempfile.TemporaryDirectory(prefix="raw_mocap_overlay_") as temporary:
            root = Path(temporary)
            pose_path = root / "pose.npz"
            betas_path = root / "betas.npz"
            output_path = root / "render.npz"
            np.savez_compressed(pose_path, pose=pose)
            np.savez_compressed(betas_path, betas=betas)
            code = f"""
import inspect, sys
from collections import namedtuple
import numpy as np
if not hasattr(inspect, 'getargspec'):
    _ArgSpec = namedtuple('arg_spec', 'args varargs keywords defaults')
    def getargspec(function):
        value = inspect.getfullargspec(function)
        return _ArgSpec(value.args, value.varargs, value.varkw, value.defaults)
    inspect.getargspec = getargspec
compatibility = {{
    'bool': bool, 'int': int, 'float': float, 'complex': complex,
    'object': object, 'unicode': str, 'str': str,
}}
for name, value in compatibility.items():
    if name not in np.__dict__:
        setattr(np, name, value)
sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
from pathlib import Path
from toporetarget.adapters.datasets.stage12_base import render_mano_pca45
pose = np.load({str(pose_path)!r})['pose']
betas = np.load({str(betas_path)!r})['betas']
result = render_mano_pca45(
    pose, side='right', mano_model_root=Path({str(mano_root)!r}), betas=betas,
    dataset_name='hocap', source_annotation_path=Path({str(annotation)!r}),
    source_annotation_hash={source_hash!r},
)
np.savez_compressed({str(output_path)!r}, vertices=result.vertices, faces=result.faces)
"""
            try:
                subprocess.run(
                    [str(base_python), "-c", code], check=True, capture_output=True, text=True
                )
                with np.load(output_path, allow_pickle=False) as result:
                    return (
                        np.asarray(result["vertices"], dtype=np.float64),
                        np.asarray(result["faces"], dtype=np.int64),
                    )
            except (OSError, subprocess.SubprocessError, FileNotFoundError) as fallback_error:
                raise RawMocapOverlayUnavailable(
                    f"RAW_MANO_ASSET_MISSING:{initial_error}; fallback={fallback_error}"
                ) from fallback_error


def resolve_raw_mocap_overlay(
    *, trace_path: Path, frame_count: int, clip: str, reference_path: Path
) -> RawMocapOverlay:
    """Resolve raw MANO/object geometry for the exact frames of one replay trace."""

    timestamps, source_indices, reference_object = _reference_arrays(reference_path)
    runtime_indices = _runtime_reference_indices(trace_path, frame_count)
    if not np.array_equal(runtime_indices, np.arange(frame_count, dtype=np.int64)):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_RUNTIME_REFERENCE_INDEX_NONIDENTITY")
    intervals = timestamps.size - 1
    if intervals <= 0 or (frame_count - 1) % intervals:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_RUNTIME_RETIMING_NOT_IDENTIFIABLE")
    time_scale = (frame_count - 1) // intervals
    if time_scale < 1:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_RUNTIME_RETIMING_INVALID")
    metadata = _metadata(reference_path)
    provenance = metadata.get("provenance", {})
    dataset_provenance = (
        provenance.get("dataset_provenance", {}) if isinstance(provenance, dict) else {}
    )
    canonical_path_value = dataset_provenance.get("source_canonical_artifact")
    if not isinstance(canonical_path_value, str):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_CANONICAL_PATH_MISSING")
    canonical_path = Path(canonical_path_value)
    if not canonical_path.exists():
        raise RawMocapOverlayUnavailable("RAW_MOCAP_CANONICAL_ARTIFACT_MISSING")
    canonical = _canonical_summary(canonical_path)
    source_file = canonical["source_file"]
    if not source_file:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_CANONICAL_SOURCE_MISSING")
    object_id = str(canonical["object_id"])
    assets = _raw_assets(canonical, object_id)
    raw_mano = np.asarray(np.load(assets["poses_m"], mmap_mode="r"), dtype=np.float64)
    raw_object = np.asarray(np.load(assets["poses_o"], mmap_mode="r"), dtype=np.float64)
    if raw_mano.ndim != 3 or raw_mano.shape[2:] != (51,):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_MANO_SHAPE_INVALID")
    if raw_object.ndim != 3 or raw_object.shape[2:] != (7,):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_OBJECT_SHAPE_INVALID")
    meta = yaml.safe_load(assets["meta"].read_text(encoding="utf-8")) or {}
    sides = [str(value).lower() for value in meta.get("mano_sides", ["right"])]
    if "right" not in sides:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_RIGHT_MANO_MISSING")
    hand_index = sides.index("right")
    object_ids = [str(value) for value in meta.get("object_ids", [])]
    if object_id not in object_ids:
        raise RawMocapOverlayUnavailable("RAW_MOCAP_PRIMARY_OBJECT_INDEX_MISSING")
    object_index = object_ids.index(object_id)
    raw_start, raw_stop = _selected_raw_slice(canonical, raw_mano.shape[1])
    if (
        raw_stop > raw_object.shape[1]
        or object_index >= raw_object.shape[0]
        or hand_index >= raw_mano.shape[0]
    ):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_SOURCE_RANGE_INVALID")
    if source_indices.size and (
        source_indices.min() < 0 or source_indices.max() >= len(canonical["timestamps"])
    ):
        raise RawMocapOverlayUnavailable("RAW_MOCAP_REFERENCE_SOURCE_INDEX_INVALID")
    source_times = np.asarray(canonical["timestamps"], dtype=np.float64)
    # Derived from the trace's complete recorded reference-index sequence and
    # this reference's actual key count; no historical rate is assumed.
    runtime_times = np.interp(
        runtime_indices / float(time_scale),
        np.arange(timestamps.size, dtype=np.float64),
        timestamps,
    )
    raw_mano_selected = raw_mano[hand_index, raw_start:raw_stop]
    raw_object_selected = raw_object[object_index, raw_start:raw_stop]
    mano_pose = interpolate_mano_pca_pose(source_times, raw_mano_selected, runtime_times)
    object_matrix = interpolate_object_pose(source_times, raw_object_selected, runtime_times)
    betas_value = yaml.safe_load(assets["betas"].read_text(encoding="utf-8")) or {}
    betas = np.asarray(betas_value.get("betas"), dtype=np.float64)
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise RawMocapOverlayUnavailable("RAW_MANO_BETAS_INVALID")
    source_hash = _sha256(assets["poses_m"])
    vertices, faces = _render_source_mano(
        mano_pose,
        betas=betas,
        mano_root=assets["mano_root"],
        annotation=assets["poses_m"],
        source_hash=source_hash,
    )
    if vertices.shape[1:] != (778, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise RawMocapOverlayUnavailable("RAW_MANO_RECONSTRUCTION_SHAPE_INVALID")
    object_vertices, object_faces = load_mesh(assets["object_mesh"])
    canonical_pose = np.asarray(canonical["object_pose"], dtype=np.float64)[source_indices]
    raw_keys = np.stack(
        [pose_hocap_qxyzw(row) for row in raw_object_selected[source_indices]], axis=0
    )
    # The source-key correspondence is the authoritative coordinate audit.  It
    # compares raw native frames with their exact canonical entries, while the
    # reference object's own 20 Hz transform identifies Stage16 world.
    raw_key_positions = raw_keys[:, :3, 3]
    canonical_key_positions = canonical_pose[:, :3, 3]
    raw_key_quat = Rotation.from_matrix(raw_keys[:, :3, :3]).as_quat()
    canonical_key_quat = Rotation.from_matrix(canonical_pose[:, :3, :3]).as_quat()
    key_times = source_times[source_indices]
    key_interpolated = interpolate_object_pose(source_times, raw_object_selected, key_times)
    runtime_key_indices = np.arange(timestamps.size, dtype=np.int64) * time_scale
    reference_key_positions = reference_object
    reference_key_quat = Rotation.from_matrix(reference_key_positions[:, :3, :3]).as_quat()
    interpolated_key_quat = Rotation.from_matrix(key_interpolated[:, :3, :3]).as_quat()
    raw_object_translation_error = float(
        np.linalg.norm(raw_key_positions - canonical_key_positions, axis=1).max()
    )
    raw_object_rotation_error = float(_rotation_error_rad(raw_key_quat, canonical_key_quat).max())
    raw_mano_orientation_error = float(
        np.max(
            np.abs(
                raw_mano_selected[source_indices, :3]
                - canonical["mano_global_orient"][source_indices]
            )
        )
    )
    raw_mano_translation_error = float(
        np.max(
            np.abs(
                raw_mano_selected[source_indices, 48:51]
                - canonical["mano_translation"][source_indices]
            )
        )
    )
    alignment = {
        "schema_version": "RawMocapToStage16AlignmentV1",
        "status": "PASS",
        "transform": (
            "identity: raw HOCap world equals canonical Scene, "
            "per frozen source_coordinate_convention"
        ),
        "raw_to_canonical_object_translation_max_m": raw_object_translation_error,
        "raw_to_canonical_object_rotation_max_rad": raw_object_rotation_error,
        "raw_to_canonical_mano_global_orient_max_abs_rad": raw_mano_orientation_error,
        "raw_to_canonical_mano_translation_max_m": raw_mano_translation_error,
        "raw_vs_geometric_reference_object_translation_max_m": float(
            np.linalg.norm(
                key_interpolated[:, :3, 3] - reference_key_positions[:, :3, 3], axis=1
            ).max()
        ),
        "raw_vs_geometric_reference_object_rotation_max_rad": float(
            _rotation_error_rad(interpolated_key_quat, reference_key_quat).max()
        ),
    }
    if (
        raw_object_translation_error > RAW_ALIGNMENT_TRANSLATION_TOLERANCE_M
        or raw_object_rotation_error > RAW_ALIGNMENT_ROTATION_TOLERANCE_RAD
        or raw_mano_orientation_error > RAW_ALIGNMENT_TRANSLATION_TOLERANCE_M
        or raw_mano_translation_error > RAW_ALIGNMENT_TRANSLATION_TOLERANCE_M
    ):
        alignment["status"] = "FAIL"
        raise RawMocapOverlayUnavailable("RAW_MOCAP_WORLD_TRANSFORM_NOT_IDENTIFIABLE")
    raw_frame_float = (
        np.interp(runtime_times, source_times, np.arange(source_times.size, dtype=np.float64))
        + raw_start
    )
    time_alignment = {
        "schema_version": "RawMocapRuntimeTimeAlignmentV1",
        "status": "PASS",
        "runtime_reference_index_source": "trace.reference_index",
        "runtime_timestamp_source": "frozen world-reference timestamps",
        "raw_timestamp_source": "canonical timestamps with no_temporal_resampling=True",
        "interpolation": {
            "translation": "linear",
            "rotation": "shortest_arc_slerp",
            "mano_global": "SO3_slerp",
        },
        "runtime_frame_count": int(frame_count),
        "reference_source_key_count": int(timestamps.size),
        "runtime_time_scale": int(time_scale),
        "runtime_key_indices": runtime_key_indices.tolist(),
        "raw_frame_float_min": float(raw_frame_float.min()),
        "raw_frame_float_max": float(raw_frame_float.max()),
    }
    return RawMocapOverlay(
        clip=clip,
        runtime_reference_indices=runtime_indices,
        runtime_timestamps_s=runtime_times,
        raw_frame_float=raw_frame_float,
        raw_mano_vertices_world=vertices,
        raw_mano_faces=faces,
        raw_mano_fingertips_world=vertices[:, MANO_TIP_VERTEX_INDICES],
        raw_object_vertices_local=np.asarray(object_vertices, dtype=np.float64),
        raw_object_faces=np.asarray(object_faces, dtype=np.int64),
        raw_object_pose_world_wxyz=_pose_matrix_to_wxyz(object_matrix),
        source_provenance={
            "clip": clip,
            "canonical_path": str(canonical_path),
            "raw_meta": str(assets["meta"]),
            "raw_mano_pose": str(assets["poses_m"]),
            "raw_object_pose": str(assets["poses_o"]),
            "raw_object_mesh": str(assets["object_mesh"]),
            "raw_object_mesh_sha256": _sha256(assets["object_mesh"]),
            "mano_model_root": str(assets["mano_root"]),
            "source_sequence": canonical["source_sequence"],
            "source_coordinate_convention": canonical["source_coordinate_convention"],
            "object_id": object_id,
        },
        coordinate_alignment=alignment,
        time_alignment=time_alignment,
    )
