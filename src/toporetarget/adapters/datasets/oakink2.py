"""OakInk-v2 source authority reader for raw-to-physical preparation.

The local OakInk-v2 hub is annotation-only: official primitive boundaries live
in ``program/program_info`` while per-mocap-frame MANO and object tracks live
in the ``anno_preview`` symlink.  This module keeps both authorities explicit
and never rewrites the dataset.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.adapters.datasets.stage12_base import (
    load_mesh,
    make_hand,
    make_object,
    native_mano21_track,
    sequence_metadata,
)
from toporetarget.contracts.canonical import CanonicalHOIv2
from toporetarget.data.schema import ManoParameterTrack


class OakInk2AdapterError(RuntimeError):
    """Raised when the local source snapshot cannot satisfy an authority check."""


@dataclass(frozen=True)
class OakInk2PrimitiveTask:
    """One official PrimitiveTask row; intervals are source ``[start, end)``."""

    sequence_id: str
    ordinal: int
    primitive_key: str
    lh_interval: tuple[int, int] | None
    rh_interval: tuple[int, int] | None
    primitive: str
    interaction_mode: str
    obj_list: tuple[str, ...]
    obj_list_lh: tuple[str, ...]
    obj_list_rh: tuple[str, ...]
    source_path: Path

    @property
    def record_id(self) -> str:
        return f"oakink2:{self.sequence_id}:{self.ordinal:05d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _parse_intervals(key: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    try:
        value = ast.literal_eval(key)
    except (SyntaxError, ValueError):
        return None, None
    if not isinstance(value, tuple) or len(value) != 2:
        return None, None

    def parse(value: object) -> tuple[int, int] | None:
        if not isinstance(value, tuple) or len(value) != 2:
            return None
        if not all(isinstance(item, int) for item in value):
            return None
        start, end = value
        return (start, end) if start < end else None

    return parse(value[0]), parse(value[1])


def _tensor_array(value: Any) -> np.ndarray:
    detached = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    return np.asarray(detached, dtype=np.float64)


class OakInk2CanonicalAdapterV1:
    """Dataset-specific source/canonical bridge preserving the raw global frame.

    OakInk-v2 preview annotations expose an unlabelled common global frame for
    MANO translations and object matrices.  Version 1 uses that as canonical
    world with an explicit identity conversion; it does not borrow HOCap's
    mocap-frame or quaternion conventions.
    """

    schema_version = "OakInk2CanonicalAdapterV1"
    dataset_name = "OakInk2"
    source_interval_semantics = "[start,end)"
    source_fps = 30.0

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.hub_root = self.dataset_root / "data" / "OakInk-v2-hub"
        self.program_root = self.hub_root / "program" / "program_info"
        self.annotation_root = self.hub_root / "anno_preview"
        self.raw_mesh_root = self.hub_root / "object_raw" / "align_ds"
        self.repaired_mesh_root = self.hub_root / "object_repair" / "align_ds"
        if not self.program_root.is_dir() or not self.annotation_root.is_dir():
            raise OakInk2AdapterError("OAKINK2_REQUIRED_PROGRAM_OR_ANNOTATION_ROOT_MISSING")

    def program_paths(self) -> list[Path]:
        return sorted(self.program_root.glob("*.json"))

    def annotation_path(self, sequence_id: str) -> Path:
        return self.annotation_root / f"{sequence_id}.pkl"

    def asset_path(self, object_id: str) -> Path | None:
        for root, names in (
            (self.repaired_mesh_root / object_id, ("model.obj", "model.ply", "scan.ply")),
            (self.raw_mesh_root / object_id, ("model_align.obj", "scan.ply", "model.obj")),
        ):
            for name in names:
                candidate = root / name
                if candidate.is_file():
                    return candidate
        return None

    def primitives(self) -> list[OakInk2PrimitiveTask]:
        rows: list[OakInk2PrimitiveTask] = []
        for path in self.program_paths():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise OakInk2AdapterError(f"OAKINK2_PROGRAM_NOT_MAPPING:{path}")
            for ordinal, (key, value) in enumerate(sorted(payload.items())):
                if not isinstance(value, dict):
                    continue
                lh, rh = _parse_intervals(str(key))
                rows.append(
                    OakInk2PrimitiveTask(
                        sequence_id=path.stem,
                        ordinal=ordinal,
                        primitive_key=str(key),
                        lh_interval=lh,
                        rh_interval=rh,
                        primitive=str(value.get("primitive", "")),
                        interaction_mode=str(value.get("interaction_mode", "")),
                        obj_list=_as_string_tuple(value.get("obj_list")),
                        obj_list_lh=_as_string_tuple(value.get("obj_list_lh")),
                        obj_list_rh=_as_string_tuple(value.get("obj_list_rh")),
                        source_path=path,
                    )
                )
        return rows

    def load_annotation(self, sequence_id: str) -> dict[str, Any]:
        path = self.annotation_path(sequence_id)
        if not path.is_file():
            raise OakInk2AdapterError(f"OAKINK2_ANNOTATION_MISSING:{path}")
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, dict):
            raise OakInk2AdapterError(f"OAKINK2_ANNOTATION_NOT_MAPPING:{path}")
        required = {"raw_mano", "obj_transf", "obj_list", "mocap_frame_id_list"}
        missing = sorted(required - value.keys())
        if missing:
            raise OakInk2AdapterError(f"OAKINK2_ANNOTATION_FIELDS_MISSING:{path}:{missing}")
        return value

    @staticmethod
    def available_frames(annotation: dict[str, Any]) -> np.ndarray:
        values = np.asarray(annotation["mocap_frame_id_list"], dtype=np.int64)
        if values.ndim != 1 or not np.all(np.diff(values) > 0):
            raise OakInk2AdapterError("OAKINK2_MOCAP_FRAME_INDEX_INVALID")
        return values

    def hand_track(
        self, annotation: dict[str, Any], side: str, frames: np.ndarray
    ) -> dict[str, np.ndarray]:
        prefix = "rh" if side == "right" else "lh"
        raw = annotation["raw_mano"]
        if not isinstance(raw, dict):
            raise OakInk2AdapterError("OAKINK2_RAW_MANO_NOT_MAPPING")
        values = []
        for frame in frames.tolist():
            item = raw.get(int(frame))
            if not isinstance(item, dict):
                raise OakInk2AdapterError(f"OAKINK2_MANO_FRAME_MISSING:{frame}")
            values.append(item)
        pose = np.concatenate(
            [_tensor_array(item[f"{prefix}__pose_coeffs"]) for item in values], axis=0
        )
        translation = np.concatenate(
            [_tensor_array(item[f"{prefix}__tsl"]) for item in values], axis=0
        )
        betas = np.concatenate([_tensor_array(item[f"{prefix}__betas"]) for item in values], axis=0)
        if pose.shape[1:] != (16, 4) or translation.shape[1:] != (3,) or betas.shape[1:] != (10,):
            raise OakInk2AdapterError(
                f"OAKINK2_MANO_SCHEMA_INVALID:{side}:{pose.shape}:{translation.shape}:{betas.shape}"
            )
        if not all(np.isfinite(value).all() for value in (pose, translation, betas)):
            raise OakInk2AdapterError(f"OAKINK2_MANO_NONFINITE:{side}")
        norms = np.linalg.norm(pose, axis=-1)
        if np.max(np.abs(norms - 1.0)) > 2e-3:
            raise OakInk2AdapterError(f"OAKINK2_MANO_QUATERNION_NORMALIZATION_INVALID:{side}")
        # OakInk2 documents MANO quaternions in scalar-first [w, x, y, z]
        # order.  Keep that source convention explicit until reconstruction;
        # scipy's scalar-last conversion happens in exactly one place below.
        return {"pose_quat_wxyz": pose, "translation_world": translation, "betas": betas}

    def object_track(
        self, annotation: dict[str, Any], object_id: str, frames: np.ndarray
    ) -> np.ndarray:
        raw = annotation["obj_transf"]
        if not isinstance(raw, dict) or not isinstance(raw.get(object_id), dict):
            raise OakInk2AdapterError(f"OAKINK2_OBJECT_TRACK_MISSING:{object_id}")
        values = np.stack(
            [np.asarray(raw[object_id][int(frame)], dtype=np.float64) for frame in frames], axis=0
        )
        if values.shape[1:] != (4, 4) or not np.isfinite(values).all():
            raise OakInk2AdapterError(
                f"OAKINK2_OBJECT_TRANSFORM_INVALID:{object_id}:{values.shape}"
            )
        det = np.linalg.det(values[:, :3, :3])
        if not np.allclose(det, 1.0, atol=2e-3):
            raise OakInk2AdapterError(f"OAKINK2_OBJECT_ROTATION_NOT_SO3:{object_id}")
        if not np.allclose(values[:, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
            raise OakInk2AdapterError(f"OAKINK2_OBJECT_HOMOGENEOUS_ROW_INVALID:{object_id}")
        return values

    @staticmethod
    def select_interval(interval: tuple[int, int], available: np.ndarray) -> np.ndarray:
        start, end = interval
        values = available[(available >= start) & (available < end)]
        if len(values) < 2:
            raise OakInk2AdapterError(f"OAKINK2_INTERVAL_UNAVAILABLE:{start}:{end}")
        return values

    @staticmethod
    def quaternion_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
        """Convert OakInk2's scalar-first MANO quaternions to rotation matrices."""
        values = np.asarray(quaternions, dtype=np.float64)
        if values.shape[-1] != 4:
            raise OakInk2AdapterError(f"OAKINK2_MANO_QUATERNION_SHAPE_INVALID:{values.shape}")
        return Rotation.from_quat(values[..., [1, 2, 3, 0]]).as_matrix()


class _ChumpyPlaceholder:
    """Minimal unpickling target for MANO's legacy ``chumpy`` shapedirs."""

    def __setstate__(self, state: object) -> None:
        self.state = state


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module.startswith("chumpy."):
            return _ChumpyPlaceholder
        return super().find_class(module, name)


@lru_cache(maxsize=2)
def _mano_model(model_path: str) -> dict[str, np.ndarray]:
    """Load the numerical MANO fields without installing legacy chumpy."""
    with Path(model_path).open("rb") as handle:
        raw = _ManoUnpickler(handle, encoding="latin1").load()
    if not isinstance(raw, dict):
        raise OakInk2AdapterError(f"OAKINK2_MANO_MODEL_NOT_MAPPING:{model_path}")
    shapedirs_value = raw["shapedirs"]
    try:
        shapedirs = np.asarray(shapedirs_value.state["a"].state["x"], dtype=np.float64)
    except (AttributeError, KeyError, TypeError) as exc:
        raise OakInk2AdapterError("OAKINK2_MANO_SHAPEDIRS_DECODE_FAILED") from exc
    if shapedirs.shape != (778, 3, 20):
        raise OakInk2AdapterError(f"OAKINK2_MANO_SHAPEDIRS_INVALID:{shapedirs.shape}")
    return {
        "v_template": np.asarray(raw["v_template"], dtype=np.float64),
        "shapedirs": shapedirs[:, :, :10],
        "posedirs": np.asarray(raw["posedirs"], dtype=np.float64),
        "weights": np.asarray(raw["weights"], dtype=np.float64),
        "j_regressor": np.asarray(raw["J_regressor"].todense(), dtype=np.float64),
        "kintree": np.asarray(raw["kintree_table"], dtype=np.int64),
        "faces": np.asarray(raw["f"], dtype=np.int64),
    }


MANO_RIGHT_JOINT_NAMES = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)


def reconstruct_mano_geometry(
    pose_quat_wxyz: np.ndarray,
    translation_world: np.ndarray,
    betas: np.ndarray,
    model_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct official-style MANO vertices and 21 joints in source space.

    OakInk2 stores sixteen scalar-first ``[w, x, y, z]`` quaternions (global
    plus 15 hand joints), a wrist-root translation, and ten betas.  Its
    official segmented viewer evaluates ``ManoLayer(rot_mode="quat",
    center_idx=0)`` and then adds that translation.  This CPU LBS mirrors
    those semantics: it centres the mesh at MANO joint 0 before adding the
    source translation.
    """
    pose = np.asarray(pose_quat_wxyz, dtype=np.float64)
    translation = np.asarray(translation_world, dtype=np.float64)
    shape = np.asarray(betas, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (16, 4):
        raise OakInk2AdapterError(f"OAKINK2_MANO_POSE_FOR_RECONSTRUCTION_INVALID:{pose.shape}")
    if translation.shape != (len(pose), 3) or shape.shape != (len(pose), 10):
        raise OakInk2AdapterError("OAKINK2_MANO_TRANSLATION_OR_BETAS_INVALID")
    model = _mano_model(str(Path(model_path).resolve()))
    rotations = (
        Rotation.from_quat(pose[..., [1, 2, 3, 0]].reshape(-1, 4))
        .as_matrix()
        .reshape(len(pose), 16, 3, 3)
    )
    vertices_shaped = model["v_template"][None] + np.einsum(
        "vck,tk->tvc", model["shapedirs"], shape
    )
    rest_joints = np.einsum("jv,tvc->tjc", model["j_regressor"], vertices_shaped)
    pose_feature = (rotations[:, 1:] - np.eye(3)).reshape(len(pose), -1)
    vertices_posed = vertices_shaped + np.einsum("vcp,tp->tvc", model["posedirs"], pose_feature)
    ids, parent_ids = model["kintree"][1], model["kintree"][0]
    index_by_id = {int(identifier): index for index, identifier in enumerate(ids.tolist())}
    transforms = np.zeros((len(pose), 16, 4, 4), dtype=np.float64)
    transforms[:, :, 3, 3] = 1.0
    transforms[:, 0, :3, :3] = rotations[:, 0]
    transforms[:, 0, :3, 3] = rest_joints[:, 0]
    for index in range(1, 16):
        parent = index_by_id.get(int(parent_ids[index]))
        if parent is None:
            raise OakInk2AdapterError("OAKINK2_MANO_KINTREE_INVALID")
        local = np.zeros((len(pose), 4, 4), dtype=np.float64)
        local[:, 3, 3] = 1.0
        local[:, :3, :3] = rotations[:, index]
        local[:, :3, 3] = rest_joints[:, index] - rest_joints[:, parent]
        transforms[:, index] = transforms[:, parent] @ local
    joint16 = transforms[:, :, :3, 3].copy()
    transforms[:, :, :3, 3] -= np.einsum("tjab,tjb->tja", transforms[:, :, :3, :3], rest_joints)
    blended = np.einsum("vj,tjab->tvab", model["weights"], transforms)
    homogeneous = np.concatenate((vertices_posed, np.ones((len(pose), 778, 1))), axis=-1)
    vertices = np.einsum("tvab,tvb->tva", blended, homogeneous)[..., :3]
    # ``ManoLayer(center_idx=0)`` returns vertices centred at the root joint;
    # OakInk2's ``rh__tsl``/``lh__tsl`` is applied afterwards.
    center = joint16[:, 0]
    vertices = vertices - center[:, None, :] + translation[:, None, :]

    # Match manotorch's right-hand SNAP joint order exactly.  The first 16
    # entries are transformed MANO joints; the remaining five are fingertip
    # vertices in thumb/index/middle/ring/little order before reordering.
    tips = vertices[:, [745, 317, 444, 556, 673]]
    joint21 = np.concatenate((joint16 - center[:, None, :] + translation[:, None, :], tips), axis=1)
    joint21 = joint21[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    if not np.isfinite(vertices).all() or not np.isfinite(joint21).all():
        raise OakInk2AdapterError("OAKINK2_MANO_RECONSTRUCTION_NONFINITE")
    return vertices, joint21, model["faces"]


def reconstruct_mano_vertices(
    pose_quat_wxyz: np.ndarray,
    translation_world: np.ndarray,
    betas: np.ndarray,
    model_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible vertex-only wrapper around the audited geometry path."""
    vertices, _, faces = reconstruct_mano_geometry(
        pose_quat_wxyz, translation_world, betas, model_path
    )
    return vertices, faces


def materialize_manifest_record_v2(
    adapter: OakInk2CanonicalAdapterV1,
    record: dict[str, Any],
    *,
    mano_model_path: str | Path,
    admitted_split: str,
) -> tuple[CanonicalHOIv2, dict[str, Any]]:
    """Materialize one exact Manifest V2 row without sampling or reselection.

    The record supplies the authoritative half-open source interval, active
    hand, target object, and asset hashes. OakInk2 annotations supply every
    frame in that interval. No Certification or Heldout row is discovered by
    this function; callers must pass the already-admitted development row.
    """

    if admitted_split != "DEVELOPMENT":
        raise OakInk2AdapterError("OAKINK2_O5_REQUIRES_DEVELOPMENT_RECORD")
    if str(record.get("active_hand", "")).upper() != "RIGHT":
        raise OakInk2AdapterError("OAKINK2_O5_REQUIRES_RIGHT_HAND_RECORD")
    interval_value = record.get("source_interval")
    if not isinstance(interval_value, list | tuple) or len(interval_value) != 2:
        raise OakInk2AdapterError("OAKINK2_MANIFEST_INTERVAL_INVALID")
    interval = (int(interval_value[0]), int(interval_value[1]))
    sequence_id = str(record["sequence_id"])
    object_id = str(record.get("target_object") or record["canonical_target_object"])
    source_started = time.perf_counter()
    annotation = adapter.load_annotation(sequence_id)
    frames = adapter.select_interval(interval, adapter.available_frames(annotation))
    if len(frames) != interval[1] - interval[0]:
        raise OakInk2AdapterError(
            f"OAKINK2_MANIFEST_INTERVAL_NOT_CONTIGUOUS:{record['record_id']}:{len(frames)}"
        )
    if not np.array_equal(frames, np.arange(interval[0], interval[1], dtype=np.int64)):
        raise OakInk2AdapterError("OAKINK2_MANIFEST_FRAME_IDS_NOT_CONTIGUOUS")

    hand_source = adapter.hand_track(annotation, "right", frames)
    source_load_sec = time.perf_counter() - source_started
    reconstruction_started = time.perf_counter()
    vertices, joints, faces = reconstruct_mano_geometry(
        hand_source["pose_quat_wxyz"],
        hand_source["translation_world"],
        hand_source["betas"],
        mano_model_path,
    )
    mano_reconstruct_sec = time.perf_counter() - reconstruction_started
    assembly_started = time.perf_counter()
    wrist = np.broadcast_to(np.eye(4), (len(frames), 4, 4)).copy()
    wrist[:, :3, :3] = adapter.quaternion_matrices_wxyz(hand_source["pose_quat_wxyz"][:, 0])
    wrist[:, :3, 3] = hand_source["translation_world"]
    rotvec = (
        Rotation.from_matrix(
            adapter.quaternion_matrices_wxyz(hand_source["pose_quat_wxyz"]).reshape(-1, 3, 3)
        )
        .as_rotvec()
        .reshape(len(frames), 16, 3)
    )
    valid = np.ones(len(frames), dtype=bool)
    annotation_path = adapter.annotation_path(sequence_id)
    source_hash = str(record.get("source_annotation_sha256") or sha256_file(annotation_path))
    mano_track = ManoParameterTrack(
        global_orient_aa=rotvec[:, 0],
        hand_pose_aa=rotvec[:, 1:].reshape(len(frames), 45),
        transl=hand_source["translation_world"],
        betas=hand_source["betas"],
        model_profile="oakink2_raw_mano_quaternion_wxyz_flat_hand_mean_v1",
    )
    hand = make_hand(
        hand_id="right_hand",
        side="right",
        vertices_scene=vertices,
        faces=faces,
        wrist_pose_scene=wrist,
        valid=valid,
        mano_parameters=mano_track,
        mano_model_root=Path(mano_model_path).resolve().parent,
        metadata={
            "source": "OakInk2 raw_mano official quaternion reconstruction",
            "quaternion_order": "SCALAR_FIRST_WXYZ",
            "mano_center_idx": 0,
            "use_pca": False,
            "flat_hand_mean": True,
            "manifest_record_sha256": record.get("canonical_record_sha256"),
            "source_frame_ids": frames.tolist(),
        },
        native_joint_track=native_mano21_track(
            joints,
            valid=valid,
            source_name="OakInk2 official MANO21",
            source_path=str(annotation_path.resolve()),
        ),
    )
    assembly_before_object_sec = time.perf_counter() - assembly_started
    object_started = time.perf_counter()
    object_path = Path(str(record["object_asset"]))
    object_vertices, object_faces = load_mesh(object_path)
    object_poses = adapter.object_track(annotation, object_id, frames)
    object_load_sec = time.perf_counter() - object_started
    assembly_started = time.perf_counter()
    object_track = make_object(
        object_id=object_id,
        vertices=object_vertices,
        faces=object_faces,
        poses_scene=object_poses,
        valid=valid,
        mesh_hash=str(record.get("object_asset_sha256") or sha256_file(object_path)),
        metadata={
            "role": "primary_manipulation_object",
            "source_path": str(object_path.resolve()),
            "source_sha256": record.get("object_asset_sha256"),
        },
    )
    metadata = sequence_metadata(
        dataset="OakInk2",
        sequence_id=str(record["record_id"]),
        frame_count=len(frames),
        fps=float(record.get("source_fps", 30.0)),
        source_file=annotation_path,
        source_hash=source_hash,
        adapter_name=adapter.schema_version,
        coordinate_convention="OakInk2 common global scene; identity source-to-scene",
        conversion_options={
            "selected_frame_range": list(interval),
            "source_frame_ids": frames.tolist(),
            "active_hand": "RIGHT",
            "target_object": object_id,
        },
        metadata={
            "manifest_record": record,
            "manifest_record_sha256": record.get("canonical_record_sha256"),
            "primitive": record.get("primitive"),
            "interaction_mode": record.get("interaction_mode"),
            "source_frame_ids": frames.tolist(),
            "retiming": "NONE",
        },
    )
    canonical = CanonicalHOIv2(metadata=metadata, hands=[hand], rigid_objects=[object_track])
    canonical.validate()
    canonical_assembly_sec = assembly_before_object_sec + time.perf_counter() - assembly_started
    return canonical, {
        "schema_version": "OakInk2ManifestV2MaterializationReceiptV1",
        "record_id": record["record_id"],
        "record_sha256": record.get("canonical_record_sha256"),
        "source_interval": list(interval),
        "source_frame_ids": frames.tolist(),
        "frame_count": int(len(frames)),
        "no_temporal_subsampling": True,
        "no_retiming": True,
        "active_hand": "RIGHT",
        "target_object": object_id,
        "mano_model_path": str(Path(mano_model_path).resolve()),
        "mano_model_sha256": sha256_file(Path(mano_model_path)),
        "source_annotation_path": str(annotation_path.resolve()),
        "source_annotation_sha256": source_hash,
        "object_asset_path": str(object_path.resolve()),
        "object_asset_sha256": record.get("object_asset_sha256"),
        "timing": {
            "source_load_sec": source_load_sec,
            "object_load_sec": object_load_sec,
            "mano_reconstruct_sec": mano_reconstruct_sec,
            "canonical_assembly_sec": canonical_assembly_sec,
        },
    }


__all__ = [
    "OakInk2AdapterError",
    "OakInk2CanonicalAdapterV1",
    "OakInk2PrimitiveTask",
    "MANO_RIGHT_JOINT_NAMES",
    "reconstruct_mano_geometry",
    "reconstruct_mano_vertices",
    "materialize_manifest_record_v2",
    "sha256_file",
]
