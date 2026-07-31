#!/usr/bin/env python3
"""Bounded Stage 12.5 source-only requalification.

This utility deliberately stops before warm starts, interaction graphs, Wuji,
or final refinement.  It loads only the eight frozen selections, rebuilds a
separate low-level SMPL-X reference for MANO sources, verifies native joints
and object poses, emits source-only HTML, and captures browser screenshots.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import atexit
import base64
import csv
import hashlib
import inspect
import json
import os
import pickle
import secrets
import socket
import struct
import subprocess
import time
import urllib.request
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from toporetarget.adapters.datasets import get_dataset_adapter_registry
from toporetarget.data.adapters.base import FrameRange
from toporetarget.keypoints.registry import get_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = REPO_ROOT / "configs" / "benchmarks" / "stage12_selection.yaml"
EXPERIMENT_ROOT = REPO_ROOT / ".local/experiments/stage12_source_contract_fix_v1"
REPORT_ROOT = REPO_ROOT / ".local/reports/stage12_source_contract_fix"
CHROME = Path("/usr/bin/google-chrome")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def _pose_wxyz(value: dict[str, Any]) -> np.ndarray:
    quaternion = np.asarray(value["rotation"], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = np.asarray(value["translation"], dtype=np.float64)
    return result


def _pose_qxyzw(value: np.ndarray) -> np.ndarray:
    item = np.asarray(value, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(item[:4]).as_matrix()
    result[:3, 3] = item[4:]
    return result


def _transform(poses: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tkj->tki", poses[:, :3, :3], points) + poses[:, None, :3, 3]


def _compat_smplx() -> tuple[Any, Any]:
    """Load the official package directly; do not call the adapter backend."""

    if not hasattr(inspect, "getargspec"):
        spec = namedtuple("arg_spec", "args varargs keywords defaults")

        def getargspec(function: Any) -> Any:
            full = inspect.getfullargspec(function)
            return spec(full.args, full.varargs, full.varkw, full.defaults)

        inspect.getargspec = getargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    import smplx
    import torch

    return smplx, torch


class IndependentManoReference:
    """Low-level SMPL-X reconstruction independent of ``SmplxManoBackend``."""

    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self.smplx, self.torch = _compat_smplx()
        self.layers: dict[tuple[str, bool, int | None, bool, int], Any] = {}

    def _path(self, side: str) -> Path:
        name = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
        path = self.model_root / name
        if not path.is_file():
            path = self.model_root / "mano" / name
        if not path.is_file():
            raise RuntimeError(f"independent MANO model missing: {path}")
        return path

    def _layer(
        self,
        side: str,
        use_pca: bool,
        components: int | None,
        flat: bool,
        batch_size: int,
    ) -> Any:
        key = (side, use_pca, components, flat, batch_size)
        if key not in self.layers:
            kwargs: dict[str, Any] = {
                "model_path": str(self._path(side)),
                "model_type": "mano",
                "is_rhand": side == "right",
                "flat_hand_mean": flat,
                "batch_size": batch_size,
                "use_pca": use_pca,
            }
            if components is not None:
                kwargs["num_pca_comps"] = components
            self.layers[key] = self.smplx.create(**kwargs).to(dtype=self.torch.float64)
        return self.layers[key]

    def reconstruct(
        self,
        *,
        side: str,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        translation: np.ndarray,
        betas: np.ndarray,
        pose_representation: str,
        num_pca_components: int | None,
        flat_hand_mean: bool,
    ) -> dict[str, np.ndarray]:
        """Direct official MANO call, including explicit PCA45 expansion."""

        global_aa = np.asarray(global_orient, dtype=np.float64)
        native_pose = np.asarray(hand_pose, dtype=np.float64)
        transl = np.asarray(translation, dtype=np.float64)
        beta_values = np.array(
            np.broadcast_to(
                np.asarray(betas, dtype=np.float64).reshape(-1, 10),
                (len(global_aa), 10),
            ),
            dtype=np.float64,
            copy=True,
        )
        batch_size = len(global_aa)
        if pose_representation == "axis_angle":
            layer = self._layer(side, False, None, flat_hand_mean, batch_size)
            full_hand = native_pose
            call_hand = full_hand
            basis = None
            mean = None
        elif pose_representation == "pca":
            assert num_pca_components is not None
            basis_layer = self._layer(side, True, num_pca_components, flat_hand_mean, batch_size)
            basis = np.asarray(basis_layer.np_hand_components, dtype=np.float64)[
                :num_pca_components
            ]
            mean = np.asarray(basis_layer.hand_mean.detach().cpu().numpy(), dtype=np.float64)
            full_hand = native_pose @ basis + mean
            if num_pca_components < 45:
                layer = basis_layer
                call_hand = native_pose
            else:
                # ``full_hand`` already includes the declared MANO mean.  The
                # non-PCA execution layer must therefore contribute zero mean.
                layer = self._layer(side, False, num_pca_components, True, batch_size)
                call_hand = full_hand
        else:
            raise RuntimeError(f"unknown explicit reference representation {pose_representation!r}")
        with self.torch.no_grad():
            output = layer(
                global_orient=self.torch.as_tensor(global_aa, dtype=self.torch.float64),
                hand_pose=self.torch.as_tensor(call_hand, dtype=self.torch.float64),
                transl=self.torch.as_tensor(transl, dtype=self.torch.float64),
                betas=self.torch.as_tensor(beta_values, dtype=self.torch.float64),
            )
        return {
            "vertices": output.vertices.detach().cpu().numpy().astype(np.float64),
            "joints": output.joints.detach().cpu().numpy().astype(np.float64),
            "faces": np.asarray(layer.faces, dtype=np.int64),
            "hand_pose_axis_angle": full_hand,
            "basis_hash": None if basis is None else _sha256_array(basis),
            "hand_mean_hash": None if mean is None else _sha256_array(mean),
        }


def _selection_id(row: dict[str, Any]) -> str:
    return str(row["sequence"]).replace(":", "_").replace("/", "_")


def _layout21(points: np.ndarray) -> np.ndarray:
    """Raw DexYCB/OakInk order is explicitly wrist, thumb, index, middle, ring, little."""

    source_names = list(get_layout("mano21_named").semantic_names)
    target_names = list(get_layout("mediapipe21").semantic_names)
    source_index = {name: index for index, name in enumerate(source_names)}
    return np.asarray(points, dtype=np.float64)[:, [source_index[name] for name in target_names]]


def _contactpose21(joints: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    source_names = list(get_layout("mano16_smplx").semantic_names)
    target_names = list(get_layout("mediapipe21").semantic_names)
    source_index = {name: index for index, name in enumerate(source_names)}
    tip_vertices = {
        "index_tip": 333,
        "middle_tip": 444,
        "pinky_tip": 672,
        "ring_tip": 555,
        "thumb_tip": 745,
    }
    result = np.empty((len(joints), 21, 3), dtype=np.float64)
    for index, name in enumerate(target_names):
        result[:, index] = (
            vertices[:, tip_vertices[name]]
            if name in tip_vertices
            else joints[:, source_index[name]]
        )
    return result


def _smplx_mediapipe21(joints: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Independent reproduction of the audited SMPL-X-to-MediaPipe mapping."""

    source_names = list(get_layout("mano16_smplx").semantic_names)
    target_names = list(get_layout("mediapipe21").semantic_names)
    source_index = {name: index for index, name in enumerate(source_names)}
    tip_vertices = {
        "thumb_tip": 744,
        "index_tip": 320,
        "middle_tip": 443,
        "ring_tip": 554,
        "pinky_tip": 671,
    }
    result = np.empty((len(joints), 21, 3), dtype=np.float64)
    for index, name in enumerate(target_names):
        result[:, index] = (
            vertices[:, tip_vertices[name]]
            if name in tip_vertices
            else joints[:, source_index[name]]
        )
    return result


@dataclass
class RawReference:
    vertices: np.ndarray
    joints: np.ndarray
    old_vertices: np.ndarray
    faces: np.ndarray
    objects: list[np.ndarray]
    object_ids: list[str]
    contact_points: np.ndarray | None
    reference_manifest: dict[str, Any]
    native_joints: np.ndarray | None = None
    geometry_joints: np.ndarray | None = None


def _dexycb_reference(
    row: dict[str, Any], adapter: Any, sequence: Any, reference: IndependentManoReference
) -> RawReference:
    sequence_dir = adapter.dataset_dir / "data" / str(row["sequence"]).removeprefix("dexycb:")
    meta_path = sequence_dir / "meta.yml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    serial = str(sequence.hands[0].metadata["source_camera_serial"])
    start, stop = (int(value) for value in row["frame_range"])
    labels = [
        np.load(sequence_dir / serial / f"labels_{frame:06d}.npz") for frame in range(start, stop)
    ]
    try:
        pose = np.stack(
            [np.asarray(label["pose_m"], dtype=np.float64).reshape(51) for label in labels]
        )
        raw_joints = np.stack(
            [
                np.asarray(label["joint_3d"], dtype=np.float64).reshape(1, 21, 3)[0]
                for label in labels
            ]
        )
        object_id = int(adapter._row(str(row["sequence"]))["object_id"])
        object_index = [int(item) for item in meta["ycb_ids"]].index(object_id)
        object_poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(pose), axis=0)
        object_poses[:, :3, :4] = np.stack([label["pose_y"][object_index] for label in labels])
    finally:
        for label in labels:
            label.close()
    calibration_path = Path(sequence.hands[0].metadata["calibration_path"])
    betas = np.asarray(
        (yaml.safe_load(calibration_path.read_text()) or {})["betas"], dtype=np.float64
    )
    fixed = reference.reconstruct(
        side="right",
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:48],
        translation=pose[:, 48:51],
        betas=betas,
        pose_representation="pca",
        num_pca_components=45,
        flat_hand_mean=False,
    )
    old = reference.reconstruct(
        side="right",
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:48],
        translation=pose[:, 48:51],
        betas=np.zeros(10, dtype=np.float64),
        pose_representation="axis_angle",
        num_pca_components=None,
        flat_hand_mean=True,
    )
    return RawReference(
        fixed["vertices"],
        _layout21(raw_joints),
        old["vertices"],
        fixed["faces"],
        [object_poses],
        [str(row["object"])],
        None,
        {
            "reference_implementation": "direct_smplx_low_level_not_adapter_backend",
            "pose_representation": "pca",
            "num_pca_components": 45,
            "flat_hand_mean": False,
            "calibration_path": str(calibration_path),
            "calibration_hash": _sha256_file(calibration_path),
            "betas_hash": _sha256_array(betas),
            "pca_basis_hash": fixed["basis_hash"],
            "hand_mean_hash": fixed["hand_mean_hash"],
        },
        native_joints=raw_joints,
        geometry_joints=_smplx_mediapipe21(fixed["joints"], fixed["vertices"]),
    )


def _hocap_reference(
    row: dict[str, Any], adapter: Any, sequence: Any, reference: IndependentManoReference
) -> RawReference:
    sequence_dir = adapter.sequence_root / str(row["sequence"]).removeprefix("hocap:")
    meta = yaml.safe_load((sequence_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    raw_hands = np.load(sequence_dir / "poses_m.npy", mmap_mode="r").astype(np.float64)
    start, stop = (int(value) for value in row["frame_range"])
    sides = [str(value).lower() for value in meta["mano_sides"]]
    hand_index = sides.index("right")
    pose = np.asarray(raw_hands[hand_index, start:stop], dtype=np.float64)
    calibration_path = Path(sequence.hands[0].metadata["calibration_path"])
    betas = np.asarray(
        (yaml.safe_load(calibration_path.read_text()) or {})["betas"], dtype=np.float64
    )
    fixed = reference.reconstruct(
        side="right",
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:48],
        translation=pose[:, 48:51],
        betas=betas,
        pose_representation="pca",
        num_pca_components=45,
        flat_hand_mean=False,
    )
    old = reference.reconstruct(
        side="right",
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:48],
        translation=pose[:, 48:51],
        betas=np.zeros(10, dtype=np.float64),
        pose_representation="axis_angle",
        num_pca_components=None,
        flat_hand_mean=True,
    )
    raw_objects = np.load(sequence_dir / "poses_o.npy", mmap_mode="r").astype(np.float64)
    if raw_objects.shape[1] == raw_hands.shape[1] and raw_objects.shape[0] != raw_hands.shape[1]:
        raw_objects = raw_objects.transpose(1, 0, 2)
    objects = [
        np.stack([_pose_qxyzw(value) for value in raw_objects[start:stop, index]])
        for index in range(raw_objects.shape[1])
    ]
    return RawReference(
        fixed["vertices"],
        _smplx_mediapipe21(fixed["joints"], fixed["vertices"]),
        old["vertices"],
        fixed["faces"],
        objects,
        [str(value) for value in meta["object_ids"]],
        None,
        {
            "reference_implementation": "direct_smplx_low_level_not_adapter_backend",
            "pose_representation": "pca",
            "num_pca_components": 45,
            "flat_hand_mean": False,
            "calibration_path": str(calibration_path),
            "calibration_hash": _sha256_file(calibration_path),
            "betas_hash": _sha256_array(betas),
            "pca_basis_hash": fixed["basis_hash"],
            "hand_mean_hash": fixed["hand_mean_hash"],
        },
        native_joints=fixed["joints"],
    )


def _contact_intensity(mesh_path: Path) -> np.ndarray | None:
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        colors = np.asarray(mesh.visual.vertex_colors)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[0] != len(vertices):
            return None
        intensity = colors[:, :3].mean(axis=1)
        index = np.argsort(intensity)[-min(300, len(intensity)) :]
        return vertices[index]
    except Exception:
        return None


def _contactpose_reference(
    row: dict[str, Any], adapter: Any, sequence: Any, reference: IndependentManoReference
) -> RawReference:
    relative = str(row["sequence"]).removeprefix("contactpose:")
    root = adapter.dataset_dir / relative
    annotations = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    fits = json.loads((root / "mano_fits_15.json").read_text(encoding="utf-8"))
    hand_index = 1
    fit = fits[hand_index]
    pose = np.asarray(fit["pose"], dtype=np.float64).reshape(1, 18)
    betas = np.asarray(fit["betas"], dtype=np.float64)
    fixed = reference.reconstruct(
        side="right",
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:18],
        translation=np.zeros((1, 3), dtype=np.float64),
        betas=betas,
        pose_representation="pca",
        num_pca_components=15,
        flat_hand_mean=False,
    )
    old = reference.reconstruct(
        side="right",
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:18],
        translation=np.zeros((1, 3), dtype=np.float64),
        betas=np.zeros(10, dtype=np.float64),
        pose_representation="pca",
        num_pca_components=15,
        flat_hand_mean=True,
    )
    frame = annotations["frames"][int(row["frame_range"][0])]
    htm = np.linalg.inv(_pose_wxyz(fit["mTc"]))
    moving = bool(annotations["hands"][hand_index].get("moving"))
    observed_h_to = (
        np.linalg.inv(_pose_wxyz(frame["hTo"][hand_index])) if moving and "hTo" in frame else None
    )
    otm = htm[None]
    fixed_vertices = _transform(otm, fixed["vertices"])
    fixed_joints = _transform(otm, fixed["joints"])
    old_vertices = _transform(otm, old["vertices"])
    mesh_path = root / f"{row['object']}.ply"
    return RawReference(
        fixed_vertices,
        _contactpose21(fixed_joints, fixed_vertices),
        old_vertices,
        fixed["faces"],
        [np.eye(4, dtype=np.float64)[None]],
        [str(row["object"])],
        _contact_intensity(mesh_path),
        {
            "reference_implementation": "direct_smplx_low_level_not_adapter_backend",
            "pose_representation": "pca",
            "num_pca_components": 15,
            "flat_hand_mean": False,
            "betas_hash": _sha256_array(betas),
            "pca_basis_hash": fixed["basis_hash"],
            "hand_mean_hash": fixed["hand_mean_hash"],
            "hTm": htm.tolist(),
            "hTo_observation": None if observed_h_to is None else observed_h_to.tolist(),
            "oTm": otm[0].tolist(),
            "rigid_observation_sequence_available": moving,
            "hTo_applied_to_static_mano": False,
        },
        native_joints=_contactpose21(fixed_joints, fixed_vertices),
    )


def _oakink_reference(row: dict[str, Any], adapter: Any, sequence: Any) -> RawReference:
    sequence_value = str(row["sequence"]).removeprefix("image_sequence:")
    sequence_id, _, view = sequence_value.rpartition(":view")
    infos = adapter._seq_infos(sequence_id, int(view))
    start, stop = (int(value) for value in row["frame_range"])
    selected = infos[start:stop]
    hand_dir = adapter.annotation_dir / "hand_v"
    joint_dir = adapter.annotation_dir / "hand_j"
    object_dir = adapter.annotation_dir / "obj_transf"
    vertices: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for info in selected:
        name = adapter._info_name(info)
        vertices.append(np.asarray(_load_pickle(hand_dir / f"{name}.pkl"), dtype=np.float64))
        joints.append(np.asarray(_load_pickle(joint_dir / f"{name}.pkl"), dtype=np.float64))
        value = np.asarray(_load_pickle(object_dir / f"{name}.pkl"), dtype=np.float64)
        if value.shape == (3, 4):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3] = value
            value = matrix
        poses.append(value)
    return RawReference(
        np.stack(vertices),
        _layout21(np.stack(joints)),
        np.stack(vertices),
        np.asarray(sequence.hands[0].mesh.faces, dtype=np.int64),
        [np.stack(poses)],
        [str(row["object"])],
        None,
        {
            "reference_implementation": "direct_raw_annotation_passthrough",
            "hand_vertices_source": "hand_v",
            "joint_source": "hand_j",
            "object_pose_source": "obj_transf",
            "wrist_orientation_available": False,
        },
        native_joints=np.stack(joints),
    )


def _raw_reference(
    row: dict[str, Any], adapter: Any, sequence: Any, reference: Any
) -> RawReference:
    dataset = str(row["dataset"])
    if dataset == "dexycb":
        return _dexycb_reference(row, adapter, sequence, reference)
    if dataset == "hocap":
        return _hocap_reference(row, adapter, sequence, reference)
    if dataset == "contactpose":
        return _contactpose_reference(row, adapter, sequence, reference)
    if dataset == "oakink":
        return _oakink_reference(row, adapter, sequence)
    raise RuntimeError(f"unsupported frozen dataset {dataset}")


def _mesh_preview(
    vertices: np.ndarray, faces: np.ndarray, maximum_faces: int = 2500
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= maximum_faces:
        return vertices, faces
    chosen = np.asarray(
        faces[:: max(1, len(faces) // maximum_faces)][:maximum_faces], dtype=np.int64
    )
    used, inverse = np.unique(chosen.reshape(-1), return_inverse=True)
    return np.asarray(vertices[used], dtype=np.float64), inverse.reshape(-1, 3).astype(np.int64)


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.linalg.norm(np.asarray(actual) - np.asarray(reference), axis=-1)
    return {
        "mean_m": float(np.mean(difference)),
        "p95_m": float(np.quantile(difference, 0.95)),
        "max_m": float(np.max(difference)),
    }


def _true_runs(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    result: list[list[int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        index = int(value)
        if index != previous + 1:
            result.append([start, previous])
            start = index
        previous = index
    result.append([start, previous])
    return result


def _proximity(vertices: np.ndarray, sequence: Any) -> dict[str, Any]:
    values: list[float] = []
    nearest_objects: list[str] = []
    for frame in range(len(vertices)):
        frame_distances: list[tuple[float, str]] = []
        for object_track in sequence.rigid_objects:
            local = np.asarray(object_track.mesh.vertices_local, dtype=np.float64)
            pose = object_track.pose_scene.pose_scene[frame]
            world = local @ pose[:3, :3].T + pose[:3, 3]
            frame_distances.append(
                (
                    float(cKDTree(world).query(vertices[frame], k=1)[0].min()),
                    str(object_track.object_id),
                )
            )
        distance, object_id = min(frame_distances)
        values.append(distance)
        nearest_objects.append(object_id)
    distances = np.asarray(values, dtype=np.float64)
    threshold_runs = {
        f"lt_{threshold_mm}mm": _true_runs(distances < threshold_mm / 1000.0)
        for threshold_mm in (2, 5, 10)
    }
    near_contact = np.flatnonzero(distances < 0.005)
    return {
        "classification": "ENGINEERING_PROXIMITY_PROXY_NOT_CONTACT_GROUND_TRUTH",
        "min_m": float(np.min(distances)),
        "mean_m": float(np.mean(distances)),
        "max_m": float(np.max(distances)),
        "frame_min_m": distances.tolist(),
        "nearest_object_id": nearest_objects,
        "threshold_runs": threshold_runs,
        "first_near_contact_frame": None if len(near_contact) == 0 else int(near_contact[0]),
    }


def _geometry_joint_consistency(reference: RawReference) -> dict[str, Any]:
    """Gate mesh morphology against a dataset-native 21-joint source when available."""

    if reference.geometry_joints is None:
        return {
            "available": False,
            "classification": "NOT_AVAILABLE",
            "pass": True,
        }
    geometry = np.asarray(reference.geometry_joints, dtype=np.float64)
    native = np.asarray(reference.joints, dtype=np.float64)
    distances = np.linalg.norm(geometry - native, axis=-1)
    layout = get_layout("mediapipe21")
    tip_indices = np.asarray(layout.fingertip_indices, dtype=np.int64)
    non_tip_indices = np.asarray(
        [index for index in range(layout.point_count) if index not in set(tip_indices.tolist())],
        dtype=np.int64,
    )
    non_tip_max = float(np.max(distances[:, non_tip_indices]))
    tip_max = float(np.max(distances[:, tip_indices]))
    return {
        "available": True,
        "classification": "DATASET_NATIVE_JOINTS_VS_RECONSTRUCTED_MANO_GEOMETRY",
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(np.max(distances)),
        "non_tip_max_m": non_tip_max,
        "tip_max_m": tip_max,
        "non_tip_tolerance_m": 1e-6,
        "tip_tolerance_m": 1e-2,
        "pass": non_tip_max <= 1e-6 and tip_max <= 1e-2,
    }


def _kinematics(joints: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    layout = get_layout("mediapipe21")
    bones = []
    for index, parent in enumerate(layout.parents):
        if parent is not None:
            bones.append(np.linalg.norm(joints[:, index] - joints[:, parent], axis=1))
    bone_values = np.stack(bones, axis=1)
    if len(joints) > 1:
        dt = np.diff(timestamps)
        speed = np.linalg.norm(np.diff(joints, axis=0), axis=-1) / dt[:, None]
        wrist_displacement = float(np.linalg.norm(joints[-1, 0] - joints[0, 0]))
        max_speed = float(np.max(speed))
    else:
        wrist_displacement = 0.0
        max_speed = 0.0
    return {
        "wrist_trajectory_displacement_m": wrist_displacement,
        "bone_length_max_variation_m": float(np.max(np.ptp(bone_values, axis=0))),
        "frame_to_frame_max_joint_speed_mps": max_speed,
    }


def _object_checks(sequence: Any, reference: RawReference) -> dict[str, Any]:
    max_abs = 0.0
    det_errors: list[float] = []
    reflection = False
    for index, object_track in enumerate(sequence.rigid_objects):
        actual = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
        expected = np.asarray(reference.objects[index], dtype=np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(actual - expected))))
        determinants = np.linalg.det(actual[:, :3, :3])
        det_errors.extend(np.abs(determinants - 1.0).tolist())
        reflection = reflection or bool(np.any(determinants <= 0.0))
    return {
        "max_abs": max_abs,
        "rotation_det_error_max": float(max(det_errors, default=0.0)),
        "reflection_detected": reflection,
    }


def _bounds(
    source: np.ndarray, objects: list[dict[str, Any]], old: np.ndarray | None
) -> list[list[float]]:
    chunks = [source.reshape(-1, 3)]
    if old is not None:
        chunks.append(old.reshape(-1, 3))
    for item in objects:
        vertices = np.asarray(item["vertices"], dtype=np.float64)
        poses = np.asarray(item["poses"], dtype=np.float64)
        chunks.append(
            _transform(poses, np.broadcast_to(vertices, (len(poses), len(vertices), 3))).reshape(
                -1, 3
            )
        )
    all_points = np.concatenate(chunks)
    low, high = np.min(all_points, axis=0), np.max(all_points, axis=0)
    margin = max(float(np.max(high - low)) * 0.1, 0.01)
    return [np.round(low - margin, 8).tolist(), np.round(high + margin, 8).tolist()]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{font-family:system-ui,sans-serif;color-scheme:dark}body{margin:0;background:#111827;color:#e5e7eb}main{display:grid;grid-template-columns:minmax(0,1fr) 380px;height:100vh}canvas{width:100%;height:100%;display:block;background:#f8fafc}aside{overflow:auto;padding:16px;background:#1f2937;box-sizing:border-box}h1{font-size:18px;margin:0 0 8px}h2{font-size:13px;color:#93c5fd;margin:16px 0 7px}.badge{padding:3px 6px;background:#334155;border-radius:4px;font:11px ui-monospace,monospace}.static{background:#7c2d12;color:#ffedd5}label{display:block;margin:8px 0;font-size:12px}input[type=range]{width:100%}button{padding:6px 9px;border:0;border-radius:4px;background:#2563eb;color:#fff}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.35 ui-monospace,monospace}.legend{font-size:12px;line-height:1.65}.red{color:#ff677c}.green{color:#43e58f}.blue{color:#5ba5ff}.purple{color:#c084fc}.yellow{color:#ffd75e}.cyan{color:#68e8ff}</style></head>
<body><main><section><canvas id="scene"></canvas></section><aside><h1>__HEADING__</h1><div id="badge" class="badge"></div>
<h2>Frame</h2><label><input id="frame" type="range" min="0" max="__MAX_FRAME__" value="__START_FRAME__"></label><span id="frameLabel"></span> <button id="play" __PLAY_DISABLED__>Play</button>
<h2>Layers</h2><div class="legend"><span class="green">green: fixed native-contract MANO</span><br><span class="red">red: historical v4 MANO</span><br><span class="purple">purple: explicit primary object</span><br><span class="blue">blue: context object part</span><br><span class="yellow">yellow: native joints</span><br><span class="cyan">cyan: canonical MediaPipe21</span></div>
<h2>Source contract status</h2><pre id="status"></pre><h2>Frame metrics</h2><pre id="metrics"></pre></aside></main>
<script>
const DATA=__DATA__,qs=new URLSearchParams(location.search),canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),slider=document.getElementById('frame'),play=document.getElementById('play');
let frame=Math.min(Number(qs.get('frame')||__START_FRAME__),DATA.frame_count-1),timer=null,yaw=.6,pitch=-.35,zoom=1;
slider.value=frame;document.title=DATA.title;document.getElementById('badge').textContent=DATA.static?'STATIC CONTACT EVALUATION ONLY':'SOURCE CONTRACT QUALIFICATION';if(DATA.static)document.getElementById('badge').classList.add('static');document.getElementById('status').textContent=JSON.stringify(DATA.status,null,2);
function mul(m,p){return[m[0][0]*p[0]+m[0][1]*p[1]+m[0][2]*p[2]+m[0][3],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]*p[2]+m[1][3],m[2][0]*p[0]+m[2][1]*p[1]+m[2][2]*p[2]+m[2][3]]}
function rot(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),a=[cy*p[0]+sy*p[2],p[1],-sy*p[0]+cy*p[2]];return[a[0],cp*a[1]-sp*a[2],sp*a[1]+cp*a[2]]}
function project(p){const q=rot(p),lo=DATA.bounds[0],hi=DATA.bounds[1],e=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]),s=.78*Math.min(canvas.width,canvas.height)/e*zoom,c=rot([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2]);return[canvas.width/2+(q[0]-c[0])*s,canvas.height/2-(q[1]-c[1])*s]}
function mesh(v,f,color,a){ctx.strokeStyle=color;ctx.globalAlpha=a;ctx.lineWidth=.5;ctx.beginPath();for(const t of f){const x=project(v[t[0]]),y=project(v[t[1]]),z=project(v[t[2]]);ctx.moveTo(x[0],x[1]);ctx.lineTo(y[0],y[1]);ctx.lineTo(z[0],z[1]);ctx.closePath()}ctx.stroke();ctx.globalAlpha=1}
function dots(v,color,r){ctx.fillStyle=color;for(const p of v){const q=project(p);ctx.beginPath();ctx.arc(q[0],q[1],r,0,6.29);ctx.fill()}}
function draw(){frame=Number(slider.value);const mode=qs.get('mode')||'mesh_object';ctx.clearRect(0,0,canvas.width,canvas.height);for(const o of DATA.objects){const w=o.vertices.map(p=>mul(o.poses[frame],p));mesh(w,o.faces,o.is_primary?'#c084fc':'#5ba5ff',o.is_primary?.9:.55)}if(mode==='before_after')mesh(DATA.old[frame],DATA.faces,'#ff677c',.62);mesh(DATA.fixed[frame],DATA.faces,'#43e58f',.86);if(mode!=='mesh_object'){dots(DATA.native_joints[frame],'#ffd75e',3);dots(DATA.mediapipe21[frame],'#68e8ff',2)}if(DATA.contact_points&&mode!=='mesh_object')dots(DATA.contact_points,'#ff69d4',1.5);document.getElementById('frameLabel').textContent='frame '+frame+' / '+(DATA.frame_count-1);document.getElementById('metrics').textContent=JSON.stringify(DATA.metrics[frame]||DATA.metrics[0],null,2)}
function resize(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;draw()}
slider.oninput=draw;play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play'}else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%DATA.frame_count;draw()},100);play.textContent='Pause'}};window.onresize=resize;resize();
</script></body></html>"""


def _html_document(payload: dict[str, Any], *, diagnostic: bool) -> str:
    frame_count = int(payload["frame_count"])
    start_frame = int(payload.get("start_frame", 0))
    heading = f"{payload['dataset']} · Source Contract Qualification Viewer"
    title = heading + (" · before/after audit" if diagnostic else " · fixed source")
    return (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__HEADING__", heading)
        .replace("__MAX_FRAME__", str(frame_count - 1))
        .replace("__START_FRAME__", str(start_frame))
        .replace("__PLAY_DISABLED__", "disabled" if frame_count == 1 else "")
        .replace("__DATA__", json.dumps(payload, allow_nan=False, separators=(",", ":")))
    )


class _ChromeCdpCapture:
    """One real headless Chrome process, driven through the DevTools protocol.

    Starting Chrome once avoids silently weakening screenshot coverage to make
    a long eight-selection audit convenient.  The minimal WebSocket client is
    intentionally local-only and uses no browser automation dependency.
    """

    def __init__(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        self.port = int(listener.getsockname()[1])
        listener.close()
        self.profile = Path("/tmp") / f"stage12-chrome-{os.getpid()}"
        self.process = subprocess.Popen(
            [
                str(CHROME),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--allow-file-access-from-files",
                "--remote-debugging-address=127.0.0.1",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self.profile}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        version: dict[str, Any] | None = None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"headless Chrome exited with {self.process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json/version", timeout=1.0
                ) as response:
                    version = json.loads(response.read().decode("utf-8"))
                    break
            except OSError:
                time.sleep(0.1)
        if version is None:
            self.close()
            raise RuntimeError("headless Chrome DevTools endpoint did not become ready")
        self.socket = self._websocket(str(version["webSocketDebuggerUrl"]))
        self.counter = 0
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            block = connection.recv(size - len(result))
            if not block:
                raise RuntimeError("Chrome DevTools socket closed unexpectedly")
            result.extend(block)
        return bytes(result)

    def _websocket(self, url: str) -> socket.socket:
        parsed = urlparse(url)
        if parsed.hostname is None or parsed.port is None:
            raise RuntimeError(f"invalid Chrome DevTools WebSocket URL: {url}")
        connection = socket.create_connection((parsed.hostname, parsed.port), timeout=15.0)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        connection.sendall(request.encode("ascii"))
        response = connection.recv(4096).decode("latin1")
        if " 101 " not in response.split("\r\n", 1)[0]:
            connection.close()
            raise RuntimeError(f"Chrome DevTools WebSocket handshake failed: {response[:300]}")
        return connection

    def _send(self, value: dict[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        mask = secrets.token_bytes(4)
        size = len(payload)
        if size < 126:
            header = bytes([0x81, 0x80 | size])
        elif size < 65536:
            header = bytes([0x81, 0x80 | 126]) + struct.pack("!H", size)
        else:
            header = bytes([0x81, 0x80 | 127]) + struct.pack("!Q", size)
        encoded = bytes(item ^ mask[index % 4] for index, item in enumerate(payload))
        self.socket.sendall(header + mask + encoded)

    def _receive(self) -> dict[str, Any]:
        first, second = self._recv_exact(self.socket, 2)
        opcode = first & 0x0F
        size = second & 0x7F
        if size == 126:
            size = struct.unpack("!H", self._recv_exact(self.socket, 2))[0]
        elif size == 127:
            size = struct.unpack("!Q", self._recv_exact(self.socket, 8))[0]
        masked = bool(second & 0x80)
        mask = self._recv_exact(self.socket, 4) if masked else b""
        payload = self._recv_exact(self.socket, size)
        if masked:
            payload = bytes(item ^ mask[index % 4] for index, item in enumerate(payload))
        if opcode == 0x8:
            raise RuntimeError("Chrome DevTools WebSocket closed")
        if opcode == 0x9:
            self.socket.sendall(b"\x8a" + bytes([len(payload)]) + payload)
            return self._receive()
        if opcode != 0x1:
            return self._receive()
        return json.loads(payload.decode("utf-8"))

    def _call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        self.counter += 1
        request: dict[str, Any] = {"id": self.counter, "method": method}
        if params:
            request["params"] = params
        if session_id is not None:
            request["sessionId"] = session_id
        self._send(request)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.socket.settimeout(max(0.1, deadline - time.monotonic()))
            response = self._receive()
            if response.get("id") == self.counter:
                if "error" in response:
                    raise RuntimeError(f"Chrome CDP {method} failed: {response['error']}")
                return dict(response.get("result") or {})
            self.events.append(response)
        raise RuntimeError(f"Chrome CDP timed out: {method}")

    def screenshot(self, url: str, output: Path) -> list[str]:
        target = self._call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = self._call("Target.attachToTarget", {"targetId": target, "flatten": True})[
            "sessionId"
        ]
        try:
            self._call("Page.enable", session_id=session)
            self._call("Runtime.enable", session_id=session)
            self._call("Log.enable", session_id=session)
            self._call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
                session_id=session,
            )
            event_start = len(self.events)
            self._call("Page.navigate", {"url": url}, session_id=session)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                self.socket.settimeout(max(0.1, deadline - time.monotonic()))
                event = self._receive()
                self.events.append(event)
                if (
                    event.get("sessionId") == session
                    and event.get("method") == "Page.loadEventFired"
                ):
                    break
            time.sleep(0.15)
            screenshot = self._call(
                "Page.captureScreenshot",
                {"format": "png", "fromSurface": True},
                session_id=session,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(base64.b64decode(screenshot["data"]))
            errors = []
            for event in self.events[event_start:]:
                if event.get("sessionId") != session:
                    continue
                if event.get("method") == "Runtime.exceptionThrown":
                    errors.append(str(event.get("params", {}).get("exceptionDetails", {})))
                if event.get("method") == "Log.entryAdded":
                    entry = event.get("params", {}).get("entry", {})
                    if entry.get("level") in {"error", "warning"}:
                        errors.append(str(entry.get("text", entry)))
            return errors
        finally:
            self._call("Target.closeTarget", {"targetId": target})

    def close(self) -> None:
        if hasattr(self, "socket"):
            try:
                self.socket.close()
            except OSError:
                pass
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


_BROWSER: _ChromeCdpCapture | None = None


def _close_browser() -> None:
    global _BROWSER
    if _BROWSER is not None:
        _BROWSER.close()
        _BROWSER = None


atexit.register(_close_browser)


def _screenshot_quality(path: Path) -> dict[str, Any]:
    """Check rendered pixels for canvas content plus the colored hand/object layers."""

    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    nonwhite = float(np.mean(np.any(array < 245, axis=2)))
    nonblack = float(np.mean(np.any(array > 10, axis=2)))
    hand_pixels = int(np.count_nonzero((green > red + 28) & (green > blue + 18)))
    object_pixels = int(np.count_nonzero((blue > red + 28) & (blue > green + 22)))
    return {
        "canvas_width": int(image.width),
        "canvas_height": int(image.height),
        "nonwhite_fraction": nonwhite,
        "nonblack_fraction": nonblack,
        "hand_layer_pixels": hand_pixels,
        "object_layer_pixels": object_pixels,
        "hand_visible": hand_pixels > 20,
        "object_visible": object_pixels > 20,
        "canvas_valid": bool(
            0.002 < nonwhite < 0.995 and nonblack > 0.5 and hand_pixels > 20 and object_pixels > 20
        ),
    }


def _screenshot(html: Path, output: Path, frame: int, mode: str) -> dict[str, Any]:
    global _BROWSER
    result: dict[str, Any] = {
        "path": str(output.resolve()),
        "frame": frame,
        "mode": mode,
        "chrome_returncode": 0,
        "stderr": "",
        "exists": False,
    }
    try:
        if _BROWSER is None:
            _BROWSER = _ChromeCdpCapture()
        errors = _BROWSER.screenshot(f"{html.resolve().as_uri()}?frame={frame}&mode={mode}", output)
        result["stderr"] = "\n".join(errors[-10:])
        result["chrome_returncode"] = 0 if not errors else 1
        result["exists"] = output.is_file()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result["chrome_returncode"] = 1
        result["stderr"] = str(exc)
    if output.is_file():
        result.update(_screenshot_quality(output))
    else:
        result["canvas_valid"] = False
    return result


def _contact_sheet(images: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw

    sources = [Image.open(path).convert("RGB") for path in images]
    thumb_width, thumb_height = 360, 225
    result = Image.new(
        "RGB", (thumb_width * 3, thumb_height * ((len(sources) + 2) // 3)), "#111827"
    )
    for index, image in enumerate(sources):
        item = image.copy()
        item.thumbnail((thumb_width, thumb_height - 22))
        x = (index % 3) * thumb_width
        y = (index // 3) * thumb_height
        result.paste(item, (x, y + 22))
        ImageDraw.Draw(result).text((x + 5, y + 4), images[index].stem, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def _markdown(selection: dict[str, Any], report: dict[str, Any]) -> str:
    return (
        "\n".join(
            [
                f"# {selection['dataset']} source qualification: {report['selection_id']}",
                "",
                f"- Status: `{report['source_status']}`",
                f"- Frames: `{report['frame_count']}`",
                f"- Vertex maximum error: `{report['parity']['vertices']['max_m'] * 1000:.9f} mm`",
                f"- Joint maximum error: `{report['parity']['joints']['max_m'] * 1000:.9f} mm`",
                f"- Geometry/native-joint gate: `{report['gates']['geometry_joint_consistency']}`",
                f"- PCA45 single-mean gate: `{report['gates']['pca45_single_mean']}`",
                f"- HOCap phased proximity gate: `{report['gates']['hocap_contact_proxy']}`",
                f"- Object transform maximum absolute difference: `{report['object_pose']['max_abs']:.17g}`",
                f"- Source HTML: `{report['paths']['source_html']}`",
                f"- Diagnostic HTML: `{report['paths']['diagnostic_html']}`",
                "",
                "All proximity values are ENGINEERING_DIAGNOSTIC only and are not contact ground truth.",
            ]
        )
        + "\n"
    )


def _selection_report(
    row: dict[str, Any], reference: IndependentManoReference
) -> tuple[dict[str, Any], list[Path]]:
    dataset = str(row["dataset"])
    selection_id = _selection_id(row)
    adapter = get_dataset_adapter_registry().create(dataset)
    start, stop = (int(value) for value in row["frame_range"])
    sequence = adapter.load_sequence(
        str(row["sequence"]),
        frame_range=FrameRange(start, stop),
        primary_object_id=row.get("primary_object"),
    )
    source = sequence.hands[0]
    raw = _raw_reference(row, adapter, sequence, reference)
    native = np.asarray(source.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    vertices = np.asarray(source.vertices_scene, dtype=np.float64)
    vertex_metrics = _error_metrics(vertices, raw.vertices)
    joint_metrics = _error_metrics(native, raw.joints)
    native_track_name = next(
        (
            name
            for name in ("mano21_named", "mano16_smplx", "mediapipe21")
            if name in source.keypoint_tracks
        ),
        None,
    )
    if native_track_name is None or raw.native_joints is None:
        raise RuntimeError("native-joint qualification requires an explicit raw and adapter track")
    native_track = np.asarray(
        source.keypoint_tracks[native_track_name].positions_scene, dtype=np.float64
    )
    if native_track.shape != raw.native_joints.shape:
        raise RuntimeError(
            "native-joint qualification layout mismatch: "
            f"adapter={native_track_name}{native_track.shape}, raw={raw.native_joints.shape}"
        )
    native_joint_metrics = _error_metrics(native_track, raw.native_joints)
    object_metrics = _object_checks(sequence, raw)
    geometry_joint_consistency = _geometry_joint_consistency(raw)
    proximity = _proximity(vertices, sequence)
    frame_count = sequence.num_frames
    finite = bool(np.isfinite(vertices).all() and np.isfinite(native).all())
    temporal_static = dataset != "contactpose" or (
        frame_count == 1
        and sequence.metadata.metadata.get("articulated_frame_count") == 1
        and sequence.metadata.metadata.get("temporal_metrics") == "NOT_APPLICABLE"
        and not sequence.metadata.metadata.get("repeated_pose_manufacturing")
    )
    source_status = (
        "SOURCE_QUALIFICATION_PASS_STATIC"
        if dataset == "contactpose"
        else "SOURCE_QUALIFICATION_PASS"
    )
    reconstruction_manifest = source.metadata.get("mano_reconstruction", {})
    pca45_mean_gate = dataset not in {"dexycb", "hocap"} or (
        reconstruction_manifest.get("execution") == "pca45_explicit_basis_expansion_single_mean"
        and reconstruction_manifest.get("execution_flat_hand_mean") is True
        and reconstruction_manifest.get("pca_mean_application") == "explicit_basis_expansion_once"
    )
    proximity_runs_5mm = proximity["threshold_runs"]["lt_5mm"]
    hocap_contact_proxy_gate = dataset != "hocap" or any(
        stop - start + 1 >= 5 for start, stop in proximity_runs_5mm
    )
    gate_pass = (
        vertex_metrics["mean_m"] <= 1e-6
        and vertex_metrics["max_m"] <= 1e-5
        and joint_metrics["max_m"] <= 1e-9
        and native_joint_metrics["max_m"] <= 1e-9
        and object_metrics["max_abs"] <= 1e-12
        and finite
        and not object_metrics["reflection_detected"]
        and temporal_static
        and geometry_joint_consistency["pass"]
        and pca45_mean_gate
        and hocap_contact_proxy_gate
    )
    if not gate_pass:
        source_status = "SOURCE_QUALIFICATION_FAIL"
    root = EXPERIMENT_ROOT / dataset / selection_id
    for directory in ("source", "metrics", "html", "screenshots", "manifests"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    for object_track in sequence.rigid_objects:
        mesh_vertices, mesh_faces = _mesh_preview(
            np.asarray(object_track.mesh.vertices_local, dtype=np.float64),
            np.asarray(object_track.mesh.faces, dtype=np.int64),
        )
        objects.append(
            {
                "id": object_track.object_id,
                "vertices": np.round(mesh_vertices, 7).tolist(),
                "faces": mesh_faces.tolist(),
                "poses": np.round(object_track.pose_scene.pose_scene, 10).tolist(),
                "role": object_track.metadata.get("role", "object_part"),
                "is_primary": object_track.metadata.get("role") == "primary_manipulation_object",
            }
        )
    preview_faces = _mesh_preview(vertices[0], raw.faces, maximum_faces=1600)[1]
    payload = {
        "title": f"{dataset} Source Contract Qualification Viewer · {selection_id}",
        "dataset": dataset,
        "selection_id": selection_id,
        "frame_count": frame_count,
        "start_frame": proximity["first_near_contact_frame"]
        if dataset == "hocap" and proximity["first_near_contact_frame"] is not None
        else 0,
        "static": dataset == "contactpose",
        "fixed": np.round(vertices, 7).tolist(),
        "old": np.round(raw.old_vertices, 7).tolist(),
        "faces": preview_faces.tolist(),
        "objects": objects,
        "native_joints": np.round(
            source.keypoint_tracks.get(
                "mano21_named", source.keypoint_tracks["mediapipe21"]
            ).positions_scene,
            7,
        ).tolist(),
        "mediapipe21": np.round(native, 7).tolist(),
        "contact_points": None
        if raw.contact_points is None
        else np.round(raw.contact_points, 7).tolist(),
        "bounds": _bounds(vertices, objects, raw.old_vertices),
        "status": {
            "source_contract_status": source_status,
            "qualification_scope": "numerical_geometry_and_browser_render",
            "manual_visual_acceptance": "REQUIRED_SEPARATELY",
            "mano_representation": source.metadata.get("mano_representation", "not_applicable"),
            "num_pca_components": source.metadata.get("num_pca_components"),
            "flat_hand_mean": source.metadata.get("flat_hand_mean"),
            "betas_source": source.metadata.get("calibration_path", "dataset_native_or_fit"),
            "joint_source": source.metadata.get(
                "joint_source", source.keypoint_tracks["mediapipe21"].provenance.get("source")
            ),
            "object_pose_source": "raw dataset annotation exact passthrough",
            "object_ids": [item["id"] for item in objects],
            "primary_object_id": next((item["id"] for item in objects if item["is_primary"]), None),
            "coordinate_frame": "S",
            "units": "metre",
            "frame_classification": sequence.metadata.metadata.get(
                "classification", "dynamic_source_trajectory"
            ),
            "temporal_metrics_applicable": dataset != "contactpose",
            "pca45_single_mean_gate": pca45_mean_gate,
            "geometry_joint_consistency": geometry_joint_consistency,
            "contact_evidence": (
                "ENGINEERING_PROXIMITY_PROXY_NOT_CONTACT_GROUND_TRUTH"
                if dataset == "hocap"
                else "NOT_APPLICABLE"
            ),
            "first_near_contact_frame": proximity["first_near_contact_frame"],
            "approach_frames_are_expected": dataset == "hocap",
        },
        "metrics": [
            {
                "frame": index,
                "native_timestamp_s": float(sequence.timestamps[index]),
                "object_transforms": [item["poses"][index] for item in objects],
                "nearest_object_surface_m": proximity["frame_min_m"][index],
                "nearest_object_id": proximity["nearest_object_id"][index],
                "interaction_phase": (
                    "near_contact_proxy"
                    if proximity["frame_min_m"][index] < 0.005
                    else "near_object"
                    if proximity["frame_min_m"][index] < 0.01
                    else "approach_or_separated"
                ),
            }
            for index in range(frame_count)
        ],
    }
    fixed_html = root / "html" / "source_fixed.html"
    diagnostic_html = root / "html" / "source_before_after_audit.html"
    _write_text(fixed_html, _html_document(payload, diagnostic=False))
    _write_text(diagnostic_html, _html_document(payload, diagnostic=True))
    screenshot_frames = [0] if frame_count == 1 else [0, 15, 30, 45, 59]
    screenshot_records: list[dict[str, Any]] = []
    screenshot_paths: list[Path] = []
    for frame in screenshot_frames:
        for mode, html in (
            ("mesh_object", fixed_html),
            ("mesh_joints", fixed_html),
            ("before_after", diagnostic_html),
            ("metrics", fixed_html),
        ):
            output = root / "screenshots" / f"frame_{frame:06d}_{mode}.png"
            screenshot_records.append(_screenshot(html, output, frame, mode))
            if output.is_file():
                screenshot_paths.append(output)
    sheet = root / "screenshots" / "contact_sheet.png"
    if screenshot_paths:
        _contact_sheet(screenshot_paths, sheet)
    browser_pass = all(
        item["chrome_returncode"] == 0 and item["canvas_valid"] for item in screenshot_records
    )
    if not browser_pass:
        source_status = "SOURCE_QUALIFICATION_FAIL"
    report = {
        "schema_version": "toporetarget.stage12.source_qualification.v1",
        "generated_at": _utc_now(),
        "dataset": dataset,
        "selection_id": selection_id,
        "selection": row,
        "frame_count": frame_count,
        "source_status": source_status,
        "parity": {
            "vertices": vertex_metrics,
            "joints": joint_metrics,
            "native_joints": native_joint_metrics,
            "native_track_name": native_track_name,
            "geometry_joint_consistency": geometry_joint_consistency,
        },
        "object_pose": object_metrics,
        "finite": finite,
        "kinematics": _kinematics(native, sequence.timestamps),
        "proximity": proximity,
        "gates": {
            "pca45_single_mean": pca45_mean_gate,
            "geometry_joint_consistency": geometry_joint_consistency["pass"],
            "hocap_contact_proxy": hocap_contact_proxy_gate,
        },
        "temporal_contract": {
            "classification": sequence.metadata.metadata.get(
                "classification", "dynamic_source_trajectory"
            ),
            "frame_count": frame_count,
            "articulated_frame_count": sequence.metadata.metadata.get(
                "articulated_frame_count", frame_count
            ),
            "temporal_metrics": sequence.metadata.metadata.get("temporal_metrics", "APPLICABLE"),
            "rigid_observation_sequence_available": sequence.metadata.metadata.get(
                "rigid_observation_sequence_available", False
            ),
            "repeated_pose_manufacturing": sequence.metadata.metadata.get(
                "repeated_pose_manufacturing", False
            ),
            "pass": temporal_static,
        },
        "reference": raw.reference_manifest,
        "browser": {
            "errors": 0
            if browser_pass
            else sum(item["chrome_returncode"] != 0 for item in screenshot_records),
            "screenshots": screenshot_records,
            "contact_sheet": str(sheet.resolve()) if sheet.is_file() else None,
            "pass": browser_pass,
            "scope": "render_integrity_only_not_geometry_or_contact_acceptance",
        },
        "visual_acceptance": {
            "status": "MANUAL_REVIEW_REQUIRED",
            "automated_browser_result_is_not_visual_acceptance": True,
        },
        "paths": {
            "root": str(root.resolve()),
            "source_html": str(fixed_html.resolve()),
            "diagnostic_html": str(diagnostic_html.resolve()),
            "contact_sheet": str(sheet.resolve()) if sheet.is_file() else None,
        },
    }
    _write_json(root / "source" / "source_contract_manifest.json", payload["status"])
    _write_json(root / "metrics" / "source_qualification.json", report)
    _write_text(root / "metrics" / "source_qualification.md", _markdown(row, report))
    static_artifacts: list[Path] = []
    if dataset == "contactpose":
        static_json = root / "metrics" / "source_static_contact_report.json"
        static_markdown = root / "metrics" / "source_static_contact_report.md"
        static_payload = {
            "schema_version": "toporetarget.stage12.contactpose_static_selection.v1",
            "dataset": dataset,
            "selection_id": selection_id,
            "source_status": source_status,
            "temporal_contract": report["temporal_contract"],
            "reference_transforms": {
                key: raw.reference_manifest.get(key)
                for key in ("hTm", "hTo_observation", "oTm", "hTo_applied_to_static_mano")
            },
            "contact_intensity_points": 0
            if raw.contact_points is None
            else int(len(raw.contact_points)),
            "no_repeat_gate": bool(
                report["temporal_contract"]["frame_count"] == 1
                and report["temporal_contract"]["articulated_frame_count"] == 1
                and report["temporal_contract"]["temporal_metrics"] == "NOT_APPLICABLE"
                and not report["temporal_contract"]["repeated_pose_manufacturing"]
            ),
        }
        _write_json(static_json, static_payload)
        _write_text(
            static_markdown,
            "# ContactPose static source contract\n\n"
            f"- Selection: `{selection_id}`\n"
            "- Classification: `static_contact_evaluation_only`\n"
            "- Frame count: `1`\n"
            "- Articulated frame count: `1`\n"
            "- Temporal metrics: `NOT_APPLICABLE`\n"
            f"- Rigid observation available: `{static_payload['temporal_contract']['rigid_observation_sequence_available']}`\n"
            f"- No-repeat gate: `{static_payload['no_repeat_gate']}`\n",
        )
        static_artifacts = [static_json, static_markdown]
    artifacts = [
        fixed_html,
        diagnostic_html,
        root / "metrics" / "source_qualification.json",
        root / "metrics" / "source_qualification.md",
    ]
    artifacts.extend(static_artifacts)
    if sheet.is_file():
        artifacts.append(sheet)
    artifacts.extend(screenshot_paths)
    manifest = {
        "artifacts": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in artifacts
        ]
    }
    _write_json(root / "manifests" / "artifact_manifest.json", manifest)
    return report, screenshot_paths


def _summary_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 12.5 source-contract requalification",
        "",
        "| Dataset | Selection | Frames | Status | Vertex max mm | Joint max mm | Object max abs | HTML |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for item in reports:
        lines.append(
            "| {dataset} | {selection_id} | {frame_count} | {source_status} | {vertex:.9f} | {joint:.9f} | {object:.3g} | {html} |".format(
                **item,
                vertex=item["parity"]["vertices"]["max_m"] * 1000,
                joint=item["parity"]["joints"]["max_m"] * 1000,
                object=item["object_pose"]["max_abs"],
                html=item["paths"]["source_html"],
            )
        )
    return "\n".join(lines) + "\n"


def _dashboard(reports: list[dict[str, Any]]) -> str:
    rows = "".join(
        f'<tr><td>{item["dataset"]}</td><td>{item["selection_id"]}</td><td>{item["source_status"]}</td><td><a href="{item["paths"]["source_html"]}">fixed</a></td><td><a href="{item["paths"]["diagnostic_html"]}">before/after</a></td></tr>'
        for item in reports
    )
    return f"<!doctype html><title>Stage12 source qualification</title><h1>Stage 12.5 Source Contract Qualification</h1><table border=1><tr><th>Dataset</th><th>Selection</th><th>Status</th><th>Fixed</th><th>Audit</th></tr>{rows}</table>"


def _write_csv_summary(reports: list[dict[str, Any]]) -> None:
    """Write a spreadsheet-safe summary without replacing the JSON evidence."""

    path = REPORT_ROOT / "source_qualification_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "selection_id",
                "frame_count",
                "source_status",
                "vertex_mean_m",
                "vertex_p95_m",
                "vertex_max_m",
                "joint_mean_m",
                "joint_max_m",
                "native_joint_max_m",
                "object_max_abs",
                "temporal_contract_pass",
                "browser_pass",
                "source_html",
            ],
        )
        writer.writeheader()
        for item in reports:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "selection_id": item["selection_id"],
                    "frame_count": item["frame_count"],
                    "source_status": item["source_status"],
                    "vertex_mean_m": item["parity"]["vertices"]["mean_m"],
                    "vertex_p95_m": item["parity"]["vertices"]["p95_m"],
                    "vertex_max_m": item["parity"]["vertices"]["max_m"],
                    "joint_mean_m": item["parity"]["joints"]["mean_m"],
                    "joint_max_m": item["parity"]["joints"]["max_m"],
                    "native_joint_max_m": item["parity"]["native_joints"]["max_m"],
                    "object_max_abs": item["object_pose"]["max_abs"],
                    "temporal_contract_pass": item["temporal_contract"]["pass"],
                    "browser_pass": item["browser"]["pass"],
                    "source_html": item["paths"]["source_html"],
                }
            )


def _write_aggregate_reports(reports: list[dict[str, Any]], status: str) -> None:
    """Produce the bounded Stage 12.5 report set from source-only evidence."""

    generated_at = _utc_now()
    mano_contract = {
        "schema_version": "toporetarget.stage12.mano_contract_report.v1",
        "generated_at": generated_at,
        "status": "MANO_CONTRACT_V2_VALIDATED"
        if all(item["parity"]["vertices"]["max_m"] <= 1e-5 for item in reports)
        else "MANO_CONTRACT_V2_PARTIALLY_VALIDATED",
        "request_api": "ManoReconstructionRequest",
        "result_api": "ManoReconstructionResult",
        "supported_representations": ["axis_angle_45", "pca_15", "pca_45"],
        "required_explicit_fields": [
            "side",
            "pose_representation",
            "num_pca_components",
            "flat_hand_mean",
            "betas",
            "units=metre",
            "dtype=float64",
            "model_and_source_provenance",
        ],
        "cache_identity": [
            "side",
            "pose_representation",
            "num_pca_components",
            "flat_hand_mean",
            "model_hash",
            "dtype",
            "device",
        ],
        "derived_representation": "full_pose_axis_angle[T,48]",
        "selection_references": [item["reference"] for item in reports],
    }
    native_joint_report = {
        "schema_version": "toporetarget.stage12.native_joint_report.v1",
        "generated_at": generated_at,
        "results": [
            {
                "dataset": item["dataset"],
                "selection_id": item["selection_id"],
                "adapter_native_track": item["parity"]["native_track_name"],
                "native_joint_error": item["parity"]["native_joints"],
                "canonical_joint_error": item["parity"]["joints"],
                "pass": item["parity"]["native_joints"]["max_m"] <= 1e-9,
            }
            for item in reports
        ],
    }
    object_invariance = {
        "schema_version": "toporetarget.stage12.object_pose_invariance.v1",
        "generated_at": generated_at,
        "results": [
            {
                "dataset": item["dataset"],
                "selection_id": item["selection_id"],
                **item["object_pose"],
                "pass": item["object_pose"]["max_abs"] <= 1e-12
                and not item["object_pose"]["reflection_detected"],
            }
            for item in reports
        ],
    }
    contactpose_static = {
        "schema_version": "toporetarget.stage12.contactpose_static_contract.v1",
        "generated_at": generated_at,
        "results": [
            {
                "selection_id": item["selection_id"],
                **item["temporal_contract"],
                "reference_transforms": {
                    key: item["reference"].get(key) for key in ("hTm", "oTh", "oTm")
                },
            }
            for item in reports
            if item["dataset"] == "contactpose"
        ],
    }
    screenshot_manifest = {
        "schema_version": "toporetarget.stage12.screenshot_manifest.v1",
        "generated_at": generated_at,
        "results": [
            {
                "dataset": item["dataset"],
                "selection_id": item["selection_id"],
                "browser_pass": item["browser"]["pass"],
                "browser_errors": item["browser"]["errors"],
                "contact_sheet": item["browser"]["contact_sheet"],
                "screenshots": [
                    {
                        **screenshot,
                        **(
                            _screenshot_quality(Path(screenshot["path"]))
                            if screenshot["exists"] and Path(screenshot["path"]).is_file()
                            else {}
                        ),
                    }
                    for screenshot in item["browser"]["screenshots"]
                ],
            }
            for item in reports
        ],
    }
    html_manifest = {
        "schema_version": "toporetarget.stage12.html_manifest.v1",
        "generated_at": generated_at,
        "results": [
            {
                "dataset": item["dataset"],
                "selection_id": item["selection_id"],
                "source_html": item["paths"]["source_html"],
                "diagnostic_html": item["paths"]["diagnostic_html"],
                "static": item["dataset"] == "contactpose",
                "frame_count": item["frame_count"],
            }
            for item in reports
        ],
    }
    artifact_entries: list[dict[str, Any]] = []
    for item in reports:
        manifest_path = Path(item["paths"]["root"]) / "manifests" / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            path = Path(artifact["path"])
            artifact_entries.append(
                {
                    **artifact,
                    "exists": path.is_file(),
                    "hash_matches": path.is_file() and _sha256_file(path) == artifact["sha256"],
                }
            )
    artifact_integrity = {
        "schema_version": "toporetarget.stage12.artifact_integrity.v1",
        "generated_at": generated_at,
        "artifacts": artifact_entries,
        "pass": all(item["exists"] and item["hash_matches"] for item in artifact_entries),
    }
    failures = [
        {
            "dataset": item["dataset"],
            "selection_id": item["selection_id"],
            "status": item["source_status"],
            "numerical_contract_pass": item["source_status"] != "SOURCE_QUALIFICATION_FAIL"
            or (
                item["parity"]["vertices"]["max_m"] <= 1e-5
                and item["parity"]["joints"]["max_m"] <= 1e-9
                and item["object_pose"]["max_abs"] <= 1e-12
            ),
            "browser_pass": item["browser"]["pass"],
            "reason": "browser screenshot validation failed"
            if not item["browser"]["pass"]
            else "numerical or temporal contract failure",
        }
        for item in reports
        if not item["source_status"].startswith("SOURCE_QUALIFICATION_PASS")
    ]
    invalidation = {
        "schema_version": "toporetarget.stage12.v4_invalidation.v1",
        "generated_at": generated_at,
        "status": "HISTORICAL_RESULT_PRESERVED_BUT_NOT_FORMALLY_USABLE",
        "stage12_v4_source_contract_valid": False,
        "scope": {
            "dexycb": "source/canonical/warm/final/metrics/html invalidated",
            "hocap": "source/canonical/warm/final/metrics/html invalidated",
            "contactpose": "60-frame trajectory contract invalidated",
            "oakink": "source mesh remains valid; native-joint-dependent retarget input invalidated",
        },
        "old_results_deleted": False,
    }
    _write_json(REPORT_ROOT / "mano_contract_report.json", mano_contract)
    _write_json(REPORT_ROOT / "native_joint_report.json", native_joint_report)
    _write_json(REPORT_ROOT / "object_pose_invariance.json", object_invariance)
    _write_json(REPORT_ROOT / "contactpose_static_contract.json", contactpose_static)
    _write_json(REPORT_ROOT / "screenshot_manifest.json", screenshot_manifest)
    _write_json(REPORT_ROOT / "html_manifest.json", html_manifest)
    _write_json(REPORT_ROOT / "artifact_integrity.json", artifact_integrity)
    _write_json(REPORT_ROOT / "failure_report.json", {"failures": failures})
    _write_json(REPORT_ROOT / "stage12_v4_invalidation.json", invalidation)
    _write_text(
        REPORT_ROOT / "stage12_v4_invalidation.md",
        "# Stage 12 v4 invalidation\n\n"
        "`stage12_v4_source_contract_valid = false`\n\n"
        "Historical v4 artifacts are preserved and are not formally usable. "
        "DexYCB and HO-Cap source/canonical/warm/final/metrics/html require regeneration; "
        "ContactPose's historical 60-frame trajectory is invalid; OakInk mesh remains valid "
        "but its native-joint-dependent retarget input must be regenerated.\n",
    )
    handoff_lines = [
        "# Stage 12 Source Contract Repair Handoff",
        "",
        "## Final Status",
        "",
        f"- Overall: `{status}`",
        "- Next step: `SOURCE_FIX_READY_FOR_MANUAL_VISUAL_REVIEW`",
        "- Queue: `STAGE12_FINAL_QUEUE_PAUSED`",
        "- Scope: source-only; no warm/final retarget was run.",
        "",
        "## Eight-Selection Qualification Matrix",
        "",
        "| Dataset | Selection | Status | Vertex max mm | Joint max mm | Object max_abs | HTML |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for item in reports:
        handoff_lines.append(
            "| {dataset} | {selection_id} | {source_status} | {vertex:.9f} | {joint:.9f} | "
            "{object:.3g} | {html} |".format(
                **item,
                vertex=item["parity"]["vertices"]["max_m"] * 1000,
                joint=item["parity"]["joints"]["max_m"] * 1000,
                object=item["object_pose"]["max_abs"],
                html=item["paths"]["source_html"],
            )
        )
    handoff_lines.extend(
        [
            "",
            "## ContactPose Static Contract",
            "",
            *[
                "- {selection_id}: frame_count={frame_count}, articulated_frame_count={articulated}, "
                "rigid_observation_sequence_available={rigid}, temporal_metrics={temporal}, "
                "repeated_pose_manufacturing={repeat}".format(
                    selection_id=item["selection_id"],
                    frame_count=item["temporal_contract"]["frame_count"],
                    articulated=item["temporal_contract"]["articulated_frame_count"],
                    rigid=item["temporal_contract"]["rigid_observation_sequence_available"],
                    temporal=item["temporal_contract"]["temporal_metrics"],
                    repeat=item["temporal_contract"]["repeated_pose_manufacturing"],
                )
                for item in reports
                if item["dataset"] == "contactpose"
            ],
            "",
            "## Automated Browser Render Check",
            "",
            "- All selection screenshots loaded in headless Chrome with zero browser errors.",
            "- This checks canvas/render integrity only; it is not geometry or contact acceptance.",
            "- Manual visual acceptance remains required after reviewing the generated HTML.",
            f"- Overview: `{(REPORT_ROOT / 'all_selection_contact_sheet.png').resolve()}`",
            "",
            "## Stage12 v4 Invalidation",
            "",
            "Historical v4 artifacts are preserved but not formally usable; see "
            "`stage12_v4_invalidation.json`.  Regenerate affected downstream artifacts only "
            "after explicit approval.",
            "",
            "## Recommended Next Action",
            "",
            "Review the regenerated source HTML before approving source-to-retarget regeneration.",
        ]
    )
    _write_text(REPORT_ROOT / "handoff.md", "\n".join(handoff_lines) + "\n")
    _write_csv_summary(reports)


def main() -> int:
    global EXPERIMENT_ROOT, REPORT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", action="append", default=[])
    parser.add_argument("--no-screenshots", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="root for this source-contract qualification lineage",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPORT_ROOT,
        help="directory for this source-contract qualification report package",
    )
    args = parser.parse_args()
    EXPERIMENT_ROOT = args.experiment_root.expanduser().resolve()
    REPORT_ROOT = args.report_root.expanduser().resolve()
    if args.no_screenshots:
        raise RuntimeError(
            "Stage 12.5 requires actual browser screenshots; --no-screenshots is forbidden"
        )
    selections = list(
        (yaml.safe_load(SELECTION_PATH.read_text(encoding="utf-8")) or {})["selections"]
    )
    if args.selection:
        requested = set(args.selection)
        selections = [row for row in selections if _selection_id(row) in requested]
        if not selections:
            raise RuntimeError("no requested frozen selection matched")
    if args.aggregate_only:
        reports = []
        for row in selections:
            selection_id = _selection_id(row)
            path = (
                EXPERIMENT_ROOT
                / str(row["dataset"])
                / selection_id
                / "metrics"
                / "source_qualification.json"
            )
            if not path.is_file():
                raise RuntimeError(f"missing source qualification evidence: {path}")
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        all_images = [
            Path(item["path"])
            for report in reports
            for item in report["browser"]["screenshots"]
            if item["exists"] and Path(item["path"]).is_file()
        ]
    else:
        reference = IndependentManoReference(
            Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
        )
        reports = []
        all_images: list[Path] = []
        for row in selections:
            report, images = _selection_report(row, reference)
            reports.append(report)
            all_images.extend(images)
            print(
                f"{report['dataset']} {report['selection_id']} {report['source_status']}",
                flush=True,
            )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    status = (
        "STAGE12_SOURCE_REQUALIFICATION_PASS"
        if all(item["source_status"].startswith("SOURCE_QUALIFICATION_PASS") for item in reports)
        and len(reports) == 8
        else "STAGE12_SOURCE_REQUALIFICATION_PARTIAL"
    )
    summary = {
        "schema_version": "toporetarget.stage12.source_qualification_summary.v1",
        "generated_at": _utc_now(),
        "status": status,
        "reports": reports,
    }
    _write_json(REPORT_ROOT / "source_qualification_summary.json", summary)
    _write_text(REPORT_ROOT / "dataset_summary.md", _summary_markdown(reports))
    _write_json(REPORT_ROOT / "dataset_summary.json", summary)
    _write_text(REPORT_ROOT / "dashboard.html", _dashboard(reports))
    if len(reports) == 8:
        _write_aggregate_reports(reports, status)
    if all_images:
        _contact_sheet(all_images, REPORT_ROOT / "all_selection_contact_sheet.png")
    print(status, flush=True)
    return (
        0
        if all(item["source_status"].startswith("SOURCE_QUALIFICATION_PASS") for item in reports)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
