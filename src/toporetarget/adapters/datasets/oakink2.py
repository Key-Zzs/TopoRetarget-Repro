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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


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


__all__ = [
    "OakInk2AdapterError",
    "OakInk2CanonicalAdapterV1",
    "OakInk2PrimitiveTask",
    "MANO_RIGHT_JOINT_NAMES",
    "reconstruct_mano_geometry",
    "reconstruct_mano_vertices",
    "sha256_file",
]
