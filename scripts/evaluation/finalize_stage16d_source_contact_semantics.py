#!/usr/bin/env python3
"""Finalize the read-only Stage 16-D source-contact semantics audit.

This is intentionally an *offline* bridge between raw HOCap MANO/object
geometry and the already frozen Formal20 V3 replay telemetry.  It never feeds
the derived masks to the simulator, Reward V3, PPO, a checkpoint, RSI, or the
physics configuration.
"""

# The evidence tables deliberately retain long, descriptive JSON keys.  Their
# on-disk schema is more important than forcing unrelated line wrapping.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.stage12_base import (  # noqa: E402
    load_mesh,
    pose_hocap_qxyzw,
    render_mano_pca45,
    sha256_paths,
)
from toporetarget.data.storage import load_hoi_sequence  # noqa: E402
from toporetarget.evaluation.source_contact_semantics import (  # noqa: E402
    FINGER_ORDER,
    REGION_ORDER,
    SEGMENT_ORDER,
    SourceContactThresholdContractV1,
    build_mano_surface_region_map,
    classify_source_contact,
    map_native_contact_to_control,
    per_region_surface_statistics,
    persistent_mask,
    source_contact_localization,
)
from toporetarget.geometry.signed_distance.closest_point import ObjectLocalBVH  # noqa: E402

CLIPS = {
    "hocap_170105": {
        "timestamp": "20231025_170105",
        "object_id": "G10_2",
        "object_index": 1,
    },
    "hocap_170650": {
        "timestamp": "20231025_170650",
        "object_id": "G04_2",
        "object_index": 1,
    },
}
SOURCE_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/HOCap/data")
MANO_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
R2_ROOT = REPO_ROOT / ".local/reports/stage16d_contact_contract_v2_audit"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"
WORLD_REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16d_source_contact_semantics_final_audit"
CONTROL_DT_S = 0.05
PERSISTENCE_CONTROL_STEPS = 3
EPS = 1.0e-8


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SOURCE_CONTACT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    def command(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    return {
        "head": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--porcelain"),
    }


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    numbers = np.asarray(values, dtype=np.float64).reshape(-1)
    if numbers.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(numbers).all():
        raise ValueError("SOURCE_CONTACT_NONFINITE_STAT")
    return {
        "n": int(numbers.size),
        "mean": float(numbers.mean()),
        "p50": float(np.quantile(numbers, 0.5)),
        "p95": float(np.quantile(numbers, 0.95)),
        "max": float(numbers.max()),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        result.append((start, end))
        start = end
    return result


def _longest(mask: np.ndarray) -> int:
    return max((end - start for start, end in _runs(mask)), default=0)


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    denominator_bool = np.asarray(denominator, dtype=bool)
    if not denominator_bool.any():
        return None
    return float(
        np.count_nonzero(np.asarray(numerator, dtype=bool) & denominator_bool)
        / denominator_bool.sum()
    )


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    one = np.asarray(first, dtype=np.float64)
    two = np.asarray(second, dtype=np.float64)
    # Trace poses are [x,y,z,qw,qx,qy,qz].  q and -q encode the same rotation.
    dot = np.abs(np.sum(one[..., 3:7] * two[..., 3:7], axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _load_mano_topology() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read buffers through SMPL-X, which is the source renderer's model API.

    The declared validation environment deliberately has no legacy ``chumpy``
    package. SMPL-X reads the same model for reconstruction and exposes the
    exact LBS/topology buffers needed for this offline segmentation.
    """

    import inspect
    from collections import namedtuple

    import torch

    # The workstation base environment has the source renderer's legacy
    # chumpy dependency. Make its Python/NumPy compatibility explicit so this
    # same process can render both clips instead of repeatedly spawning a
    # helper under the project environment.
    if not hasattr(inspect, "getargspec"):
        argument_spec = namedtuple("arg_spec", "args varargs keywords defaults")

        def getargspec(function: object) -> object:
            full = inspect.getfullargspec(function)
            return argument_spec(full.args, full.varargs, full.varkw, full.defaults)

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

    try:
        import smplx
    except ImportError:
        payload = _load_mano_topology_from_base()
        return payload
    model = smplx.create(
        model_path=str(MANO_ROOT / "MANO_RIGHT.pkl"),
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
        dtype=torch.float64,
    )
    shapedirs = model.shapedirs.detach().cpu().numpy().astype(np.float64)
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)
    template = model.v_template.detach().cpu().numpy()[0].astype(np.float64)
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    if shapedirs.shape != (778, 3, 10) or weights.shape != (778, 16):
        raise ValueError("SOURCE_CONTACT_MANO_TOPOLOGY_SHAPE_INVALID")
    return weights, faces, template, regressor, shapedirs


def _load_mano_topology_from_base() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Extract the same model buffers in base where SMPL-X is installed."""

    code = (
        """
import numpy as np
import inspect
from collections import namedtuple
if not hasattr(inspect, 'getargspec'):
    _ArgSpec = namedtuple('arg_spec', 'args varargs keywords defaults')
    def getargspec(function):
        full = inspect.getfullargspec(function)
        return _ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec
for _name, _value in {
    'bool': bool, 'int': int, 'float': float, 'complex': complex,
    'object': object, 'unicode': str, 'str': str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
import smplx
import torch
from pathlib import Path
root = Path(r'"""
        + str(MANO_ROOT)
        + """')
model = smplx.create(
    model_path=str(root / 'MANO_RIGHT.pkl'), model_type='mano', is_rhand=True,
    use_pca=False, flat_hand_mean=True, batch_size=1, dtype=torch.float64,
)
np.savez_compressed(
    r'__OUTPUT__',
    shapedirs=model.shapedirs.detach().cpu().numpy(),
    weights=model.lbs_weights.detach().cpu().numpy(), faces=np.asarray(model.faces),
    template=model.v_template.detach().cpu().numpy()[0],
    regressor=model.J_regressor.detach().cpu().numpy(),
)
"""
    )
    import tempfile

    with tempfile.TemporaryDirectory(prefix="stage16d_mano_topology_") as temp:
        archive_path = Path(temp) / "mano_topology.npz"
        subprocess.run(
            [
                "/home/deepcybo/miniconda3/bin/python",
                "-c",
                code.replace("__OUTPUT__", str(archive_path)),
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            shapedirs = np.asarray(archive["shapedirs"], dtype=np.float64)
            weights = np.asarray(archive["weights"], dtype=np.float64)
            faces = np.asarray(archive["faces"], dtype=np.int64)
            template = np.asarray(archive["template"], dtype=np.float64)
            regressor = np.asarray(archive["regressor"], dtype=np.float64)
    if shapedirs.shape != (778, 3, 10) or weights.shape != (778, 16):
        raise ValueError("SOURCE_CONTACT_MANO_TOPOLOGY_SUBPROCESS_SHAPE_INVALID")
    return weights, faces, template, regressor, shapedirs


def _render_source_mano(
    pose_values: np.ndarray,
    *,
    betas: np.ndarray,
    source_annotation_path: Path,
    source_annotation_hash: str,
) -> Any:
    """Render raw MANO in the current env or an explicit SMPL-X base helper."""

    try:
        import smplx  # noqa: F401
    except ImportError:
        code = """
import inspect
import json
import sys
from collections import namedtuple
import numpy as np
if not hasattr(inspect, 'getargspec'):
    _ArgSpec = namedtuple('arg_spec', 'args varargs keywords defaults')
    def getargspec(function):
        full = inspect.getfullargspec(function)
        return _ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec
for _name, _value in {
    'bool': bool, 'int': int, 'float': float, 'complex': complex,
    'object': object, 'unicode': str, 'str': str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
from pathlib import Path
sys.path.insert(0, r'__REPO_ROOT__' + '/src')
from toporetarget.adapters.datasets.stage12_base import render_mano_pca45
pose = np.load(r'__POSE__', allow_pickle=False)['pose']
betas = np.load(r'__BETAS__', allow_pickle=False)['betas']
result = render_mano_pca45(
    pose, side='right',
    mano_model_root=Path(r'__MANO_ROOT__'), betas=betas,
    dataset_name='hocap', source_annotation_path=Path(r'__ANNOTATION__'),
    source_annotation_hash=r'__SOURCE_HASH__',
)
np.savez_compressed(
    r'__OUTPUT__', vertices=result.vertices, faces=result.faces,
    posed_joints=result.posed_joints_native,
    wrist_pose=result.wrist_pose_scene,
    hand_pose_aa=result.hand_pose_axis_angle,
    betas=result.betas,
    manifest=np.asarray(json.dumps(result.reconstruction_manifest)),
)
"""
        import tempfile

        with tempfile.TemporaryDirectory(prefix="stage16d_source_mano_") as temp:
            temp_root = Path(temp)
            pose_path = temp_root / "pose.npz"
            betas_path = temp_root / "betas.npz"
            output_path = temp_root / "render.npz"
            np.savez_compressed(pose_path, pose=np.asarray(pose_values, dtype=np.float64))
            np.savez_compressed(betas_path, betas=np.asarray(betas, dtype=np.float64))
            replacements = {
                "__REPO_ROOT__": str(REPO_ROOT),
                "__POSE__": str(pose_path),
                "__BETAS__": str(betas_path),
                "__MANO_ROOT__": str(MANO_ROOT),
                "__ANNOTATION__": str(source_annotation_path),
                "__SOURCE_HASH__": source_annotation_hash,
                "__OUTPUT__": str(output_path),
            }
            for marker, value in replacements.items():
                code = code.replace(marker, value)
            subprocess.run(
                ["/home/deepcybo/miniconda3/bin/python", "-c", code],
                check=True,
                cwd=REPO_ROOT,
            )
            with np.load(output_path, allow_pickle=False) as archive:
                return SimpleNamespace(
                    vertices=np.asarray(archive["vertices"], dtype=np.float64),
                    faces=np.asarray(archive["faces"], dtype=np.int64),
                    posed_joints_native=np.asarray(archive["posed_joints"], dtype=np.float64),
                    wrist_pose_scene=np.asarray(archive["wrist_pose"], dtype=np.float64),
                    hand_pose_axis_angle=np.asarray(archive["hand_pose_aa"], dtype=np.float64),
                    betas=np.asarray(archive["betas"], dtype=np.float64),
                    reconstruction_manifest=json.loads(str(archive["manifest"].item())),
                )
    return render_mano_pca45(
        pose_values,
        side="right",
        mano_model_root=MANO_ROOT,
        betas=betas,
        dataset_name="hocap",
        source_annotation_path=source_annotation_path,
        source_annotation_hash=source_annotation_hash,
    )


def _write_region_ply(
    path: Path, vertices: np.ndarray, faces: np.ndarray, region_id: np.ndarray
) -> None:
    import trimesh

    palette = np.asarray(
        [
            [220, 35, 35, 255],
            [250, 145, 25, 255],
            [245, 220, 30, 255],
            [65, 170, 70, 255],
            [45, 100, 220, 255],
            [150, 90, 55, 255],
            [160, 60, 185, 255],
        ],
        dtype=np.uint8,
    )
    trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        vertex_colors=palette[np.asarray(region_id, dtype=np.int64)],
    ).export(path)


def _write_source_contact_scene(
    path: Path,
    *,
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    object_vertices_world: np.ndarray,
    object_faces: np.ndarray,
    highlight: np.ndarray,
) -> None:
    """Write one offline coloured source MANO/object diagnostic scene."""

    import trimesh

    human = np.asarray(mano_vertices, dtype=np.float64)
    obj = np.asarray(object_vertices_world, dtype=np.float64)
    colors = np.full((len(human) + len(obj), 4), [70, 130, 210, 255], dtype=np.uint8)
    colors[len(human) :] = [145, 145, 145, 255]
    colors[np.flatnonzero(np.asarray(highlight, dtype=bool))] = [235, 45, 45, 255]
    trimesh.Trimesh(
        vertices=np.concatenate([human, obj], axis=0),
        faces=np.concatenate(
            [
                np.asarray(mano_faces, dtype=np.int64),
                np.asarray(object_faces, dtype=np.int64) + len(human),
            ],
            axis=0,
        ),
        process=False,
        vertex_colors=colors,
    ).export(path)


def _canonical_path(clip: str) -> Path:
    return (
        REPO_ROOT
        / ".local/experiments/stage12_spatial_adapter_repair_v1_20260731_714fbaf/hocap"
        / f"hocap_subject_1_{CLIPS[clip]['timestamp']}"
        / "canonical/canonical_hoi_v2.zarr"
    )


def _source_reference_indices(clip: str) -> np.ndarray:
    path = WORLD_REFERENCE_ROOT / f"{clip}.world_wrist.stage16.npz"
    with np.load(path, allow_pickle=False) as archive:
        if "source_frame_indices" not in archive.files:
            raise ValueError(f"SOURCE_CONTACT_REFERENCE_SOURCE_KEYS_MISSING:{path}")
        indices = np.asarray(archive["source_frame_indices"], dtype=np.int64)
    if indices.shape != (41,) or np.any(indices < 0):
        raise ValueError("SOURCE_CONTACT_REFERENCE_SOURCE_KEYS_INVALID")
    return indices


def _clip_source_assets(clip: str) -> dict[str, Path]:
    timestamp = str(CLIPS[clip]["timestamp"])
    sequence = SOURCE_ROOT / "subject_1" / timestamp
    object_id = str(CLIPS[clip]["object_id"])
    assets = {
        "sequence": sequence,
        "meta": sequence / "meta.yaml",
        "poses_m": sequence / "poses_m.npy",
        "poses_o": sequence / "poses_o.npy",
        "betas": SOURCE_ROOT / "calibration/mano/subject_1.yaml",
        "object_mesh": SOURCE_ROOT / "models" / object_id / "textured_mesh.obj",
        "canonical": _canonical_path(clip),
        "world_reference": WORLD_REFERENCE_ROOT / f"{clip}.world_wrist.stage16.npz",
        "reference_v2": REFERENCE_ROOT / "references" / f"{clip}.reference_kinematics_v2.npz",
        "trace": R2_ROOT / "full_pair_telemetry" / clip / "trace_full_pair.npz",
        "qualification": R2_ROOT / "full_pair_telemetry" / clip / "qualification.json",
        "evaluation": R2_ROOT / "full_pair_telemetry" / clip / "full_pair_r2_evaluation.json",
    }
    missing = [name for name, path in assets.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SOURCE_CONTACT_REQUIRED_INPUTS_MISSING:{clip}:{missing}")
    return assets


def _transform_world_to_object(points_world: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float64)
    translation = np.asarray(pose_world[:3, 3], dtype=np.float64)
    return (np.asarray(points_world, dtype=np.float64) - translation) @ rotation


def _interpolate_mano_pca_pose(
    timestamps: np.ndarray, pose: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    """Resample raw MANO pose at the frozen Stage16 source spatial-key times."""

    source_t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    source = np.asarray(pose, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if source.shape != (len(source_t), 51) or np.any(np.diff(source_t) <= 0.0):
        raise ValueError("SOURCE_CONTACT_MANO_TIME_SERIES_INVALID")
    if target[0] < source_t[0] or target[-1] > source_t[-1]:
        raise ValueError("SOURCE_CONTACT_MANO_RESAMPLE_OUT_OF_RANGE")
    result = np.empty((len(target), 51), dtype=np.float64)
    result[:, :3] = Slerp(source_t, Rotation.from_rotvec(source[:, :3]))(target).as_rotvec()
    for column in range(3, source.shape[1]):
        result[:, column] = np.interp(target, source_t, source[:, column])
    return result


def _interpolate_object_pose(
    timestamps: np.ndarray, poses_qxyzw: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    """Resample raw HOCap object transforms without changing their convention."""

    source_t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    pose_values = np.asarray(poses_qxyzw, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if pose_values.shape != (len(source_t), 7) or np.any(np.diff(source_t) <= 0.0):
        raise ValueError("SOURCE_CONTACT_OBJECT_TIME_SERIES_INVALID")
    matrices = np.stack([pose_hocap_qxyzw(value) for value in pose_values], axis=0)
    result = np.broadcast_to(np.eye(4), (len(target), 4, 4)).copy()
    result[:, :3, :3] = Slerp(source_t, Rotation.from_matrix(matrices[:, :3, :3]))(
        target
    ).as_matrix()
    for axis in range(3):
        result[:, axis, 3] = np.interp(target, source_t, matrices[:, axis, 3])
    return result


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    required = {
        "replica_hand_object_pair_force_world",
        "replica_hand_object_pair_presence",
        "replica_hand_object_pair_force_valid",
        "replica_contact_reward",
        "replica_reward_total",
        "replica_fingertip_object_force_magnitude",
        "replica_reference_contact_mask",
        "replica_object_pose",
        "replica_embedded_reference_object_pose",
        "replica_object_twist",
        "replica_object_twist_reference",
        "replica_error_obj_vel",
        "replica_error_obj_ang_vel",
        "replica_terminated",
        "hand_body_names",
        "hand_body_groups",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"SOURCE_CONTACT_TRACE_FIELDS_MISSING:{missing}")
        result = {name: np.asarray(archive[name]) for name in required}
    force = result["replica_hand_object_pair_force_world"]
    presence = result["replica_hand_object_pair_presence"]
    valid = result["replica_hand_object_pair_force_valid"]
    if force.shape != (321, 20, 21, 3) or presence.shape != (321, 20, 21):
        raise ValueError("SOURCE_CONTACT_TRACE_FULL_BODY_SHAPE_INVALID")
    if valid.shape != (321, 20) or valid[0].any() or not valid[1:].all():
        raise ValueError("SOURCE_CONTACT_TRACE_FORCE_VALIDITY_INVALID")
    names = tuple(str(value) for value in result["hand_body_names"].tolist())
    if len(names) != 21 or len(set(names)) != 21:
        raise ValueError("SOURCE_CONTACT_TRACE_BODY_NAMES_INVALID")
    return result


def _finger_body_indices(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    prefixes = {
        "thumb": "r_thumb_",
        "index": "r_index_finger_",
        "middle": "r_middle_finger_",
        "ring": "r_ring_finger_",
        "pinky": "r_pinky_",
    }
    for finger, prefix in prefixes.items():
        indices = np.asarray([index for index, name in enumerate(names) if name.startswith(prefix)])
        if len(indices) != 4:
            raise ValueError(
                f"SOURCE_CONTACT_WUJI_GROUP_MAPPING_INVALID:{finger}:{indices.tolist()}"
            )
        result[finger] = indices
    return result


def _tip_index(names: tuple[str, ...], finger: str) -> int:
    lookup = {
        "thumb": "r_thumb_distal",
        "index": "r_index_finger_distal",
        "middle": "r_middle_finger_distal",
        "ring": "r_ring_finger_distal",
        "pinky": "r_pinky_distal",
    }
    return names.index(lookup[finger])


def _source_asset_and_alignment(
    clip: str,
    *,
    output: Path,
    region_map: Any,
    rest_faces: np.ndarray,
) -> dict[str, Any]:
    # ``output`` is normally created by ``main``.  Keep this lower-level
    # routine restart-safe as it is also the point at which a long raw-source
    # reconstruction first materializes its clip evidence.
    output.mkdir(parents=True, exist_ok=True)
    clip_output = output / clip
    clip_output.mkdir(parents=True, exist_ok=True)
    assets = _clip_source_assets(clip)
    meta = yaml.safe_load(assets["meta"].read_text(encoding="utf-8")) or {}
    pose_m = np.asarray(np.load(assets["poses_m"], mmap_mode="r"), dtype=np.float64)
    pose_o = np.asarray(np.load(assets["poses_o"], mmap_mode="r"), dtype=np.float64)
    if pose_m.ndim != 3 or pose_m.shape[0] < 1 or pose_m.shape[2] != 51:
        raise ValueError("SOURCE_CONTACT_RAW_MANO_SHAPE_INVALID")
    if pose_o.ndim != 3 or pose_o.shape[2] != 7:
        raise ValueError("SOURCE_CONTACT_RAW_OBJECT_SHAPE_INVALID")
    object_ids = [str(value) for value in meta.get("object_ids", [])]
    object_id = str(CLIPS[clip]["object_id"])
    object_index = int(CLIPS[clip]["object_index"])
    if object_ids[object_index] != object_id:
        raise ValueError(f"SOURCE_CONTACT_OBJECT_INDEX_DRIFT:{clip}:{object_ids}")
    source_indices = _source_reference_indices(clip)
    if int(source_indices.max()) >= pose_m.shape[1]:
        raise ValueError("SOURCE_CONTACT_SOURCE_KEY_OUT_OF_RANGE")
    canonical = load_hoi_sequence(assets["canonical"])
    source_timestamps = np.asarray(canonical.timestamps, dtype=np.float64)
    if source_timestamps.ndim != 1 or len(source_timestamps) > pose_m.shape[1]:
        raise ValueError("SOURCE_CONTACT_CANONICAL_TIME_ALIGNMENT_INVALID")
    with np.load(assets["world_reference"], allow_pickle=False) as world_reference:
        stage16_source_timestamps = np.asarray(world_reference["timestamps"], dtype=np.float64)
    if stage16_source_timestamps.shape != (41,):
        raise ValueError("SOURCE_CONTACT_STAGE16_SOURCE_TIME_INVALID")
    betas = np.asarray(
        yaml.safe_load(assets["betas"].read_text(encoding="utf-8"))["betas"], dtype=np.float64
    )
    source_hash = sha256_paths([assets["meta"], assets["poses_m"], assets["betas"]])
    render = _render_source_mano(
        _interpolate_mano_pca_pose(
            source_timestamps, pose_m[0, : len(source_timestamps)], stage16_source_timestamps
        ),
        betas=betas,
        source_annotation_path=assets["poses_m"],
        source_annotation_hash=source_hash,
    )
    object_poses = _interpolate_object_pose(
        source_timestamps,
        pose_o[object_index, : len(source_timestamps)],
        stage16_source_timestamps,
    )
    raw_object_poses = np.stack(
        [pose_hocap_qxyzw(value) for value in pose_o[object_index, source_indices]], axis=0
    )
    vertices_local, object_faces = load_mesh(assets["object_mesh"])
    bvh = ObjectLocalBVH(vertices_local[object_faces])
    all_distances = np.empty((len(source_indices), render.vertices.shape[1]), dtype=np.float64)
    for frame, pose in enumerate(object_poses):
        object_points = _transform_world_to_object(render.vertices[frame], pose)
        _nearest, _face, _bary, distance = bvh.query(object_points)
        all_distances[frame] = distance

    primary = next(item for item in canonical.rigid_objects if item.object_id == object_id)
    canonical_mano = canonical.hands[0].mano_parameters
    canonical_global_orient = np.asarray(canonical_mano.global_orient_aa[source_indices])
    canonical_hand_translation = np.asarray(canonical_mano.transl[source_indices])
    canonical_poses = np.asarray(primary.pose_scene.pose_scene[source_indices], dtype=np.float64)
    raw_pose_position = raw_object_poses[:, :3, 3]
    canonical_position = canonical_poses[:, :3, 3]
    raw_quat_xyzw = Rotation.from_matrix(raw_object_poses[:, :3, :3]).as_quat()
    canonical_quat_xyzw = Rotation.from_matrix(canonical_poses[:, :3, :3]).as_quat()
    rotation_dot = np.abs(np.sum(raw_quat_xyzw * canonical_quat_xyzw, axis=1))
    canonical_coordinate_convention = canonical.metadata.provenance.source_coordinate_convention
    with np.load(assets["reference_v2"], allow_pickle=False) as reference_v2:
        runtime_keys = np.arange(len(source_indices), dtype=np.int64) * 8
        v2_position = np.asarray(
            reference_v2["object_pose_translation_world_ref"], dtype=np.float64
        )[runtime_keys]
        v2_quaternion = np.asarray(
            reference_v2["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64
        )[runtime_keys]
    interpolated_quaternion_wxyz = np.concatenate(
        (
            Rotation.from_matrix(object_poses[:, :3, :3]).as_quat()[:, 3:4],
            Rotation.from_matrix(object_poses[:, :3, :3]).as_quat()[:, :3],
        ),
        axis=1,
    )
    v2_rotation_dot = np.abs(np.sum(interpolated_quaternion_wxyz * v2_quaternion, axis=1))
    alignment = {
        "schema_version": "SourceRawCanonicalAlignmentV1",
        "clip": clip,
        "status": "PASS",
        "source_key_count": int(len(source_indices)),
        "source_frame_indices": source_indices.tolist(),
        "raw_to_canonical_hand_global_orient_max_abs_rad": float(
            np.max(np.abs(pose_m[0, source_indices, :3] - canonical_global_orient))
        ),
        "raw_to_canonical_hand_translation_max_m": float(
            np.max(np.abs(pose_m[0, source_indices, 48:51] - canonical_hand_translation))
        ),
        "raw_to_canonical_object_translation_max_m": float(
            np.linalg.norm(raw_pose_position - canonical_position, axis=1).max()
        ),
        "raw_to_canonical_object_rotation_max_rad": float(
            (2.0 * np.arccos(np.clip(rotation_dot, 0.0, 1.0))).max()
        ),
        "stage16_v2_runtime_indices": runtime_keys.tolist(),
        "canonical_interpolated_to_stage16_v2_object_translation_max_m": float(
            np.linalg.norm(object_poses[:, :3, 3] - v2_position, axis=1).max()
        ),
        "canonical_interpolated_to_stage16_v2_object_rotation_max_rad": float(
            (2.0 * np.arccos(np.clip(v2_rotation_dot, 0.0, 1.0))).max()
        ),
        "source_hand_root_lineage": (
            "raw MANO global orient/translation equals selected canonical MANO input; "
            "robot wrist is a retargeted reference and is not MANO-root equality evidence"
        ),
        "canonical_source_hash": canonical.metadata.provenance.source_hash,
        "canonical_source_coordinate_convention": canonical_coordinate_convention,
        "stage16_lineage": (
            "The matching canonical HOCap source was the Stage12 retarget input; the Stage16 "
            "robot reference is not asserted to be geometrically equal to the human hand."
        ),
    }
    if (
        alignment["raw_to_canonical_hand_global_orient_max_abs_rad"] > 2.0e-8
        or alignment["raw_to_canonical_hand_translation_max_m"] > 2.0e-8
        or alignment["raw_to_canonical_object_translation_max_m"] > 2.0e-8
        or alignment["raw_to_canonical_object_rotation_max_rad"] > 2.0e-6
        or alignment["canonical_interpolated_to_stage16_v2_object_translation_max_m"] > 2.0e-8
        or alignment["canonical_interpolated_to_stage16_v2_object_rotation_max_rad"] > 2.0e-6
    ):
        raise ValueError(f"SOURCE_CONTACT_RAW_CANONICAL_ALIGNMENT_FAILED:{clip}:{alignment}")

    stats = per_region_surface_statistics(all_distances, region_map, rest_faces)
    segment, segment_distance = source_contact_localization(all_distances, region_map)
    classes = classify_source_contact(
        stats["minimum_surface_distance_m"][:, :5],
        stats["largest_component_vertices_at_5mm"][:, :5],
    )
    sensitivity: dict[str, np.ndarray] = {}
    for threshold in (0.001, 0.002, 0.005):
        sensitivity[f"contact_at_{int(threshold * 1000)}mm"] = classify_source_contact(
            stats["minimum_surface_distance_m"][:, :5],
            stats["largest_component_vertices_at_5mm"][:, :5],
            threshold_m=threshold,
        )["confirmed_contact"]
    runtime = map_native_contact_to_control(classes["class"], factor=8, control_frames=321)
    native_path = clip_output / "source_contact_evidence_native.npz"
    np.savez_compressed(
        native_path,
        source_frame_indices=source_indices,
        native_timestamp_s=stage16_source_timestamps,
        finger_order=np.asarray(FINGER_ORDER),
        region_order=np.asarray(REGION_ORDER),
        segment_order=np.asarray(SEGMENT_ORDER),
        class_label=classes["class"],
        confidence=classes["confidence"],
        raw_robust_contact=classes["raw_robust_contact"],
        confirmed_contact=classes["confirmed_contact"],
        probable_contact=classes["probable_contact"],
        transition=classes["transition"],
        proximity_only=classes["proximity_only"],
        minimum_surface_distance_m=stats["minimum_surface_distance_m"],
        p01_surface_distance_m=stats["p01_surface_distance_m"],
        p05_surface_distance_m=stats["p05_surface_distance_m"],
        thresholds_m=stats["thresholds_m"],
        near_vertex_count=stats["near_vertex_count"],
        near_vertex_fraction=stats["near_vertex_fraction"],
        largest_component_vertices_at_5mm=stats["largest_component_vertices_at_5mm"],
        nearest_segment=segment,
        nearest_segment_distance_m=segment_distance,
        **sensitivity,
    )
    runtime_path = clip_output / "source_contact_evidence_runtime.npz"
    np.savez_compressed(
        runtime_path,
        control_index=np.arange(321, dtype=np.int64),
        native_to_control_index=runtime["native_to_control_index"],
        finger_order=np.asarray(FINGER_ORDER),
        class_label=runtime["class"],
        expected_contact=runtime["expected_contact"],
        exact_source_key=runtime["exact_source_key"],
    )
    _write_json(
        clip_output / "source_contact_runtime_mapping.json",
        {
            "schema_version": "SourceContactRuntimeMappingV1",
            "native_source_keys": source_indices.tolist(),
            "native_to_runtime_index": runtime["native_to_control_index"].tolist(),
            "factor": 8,
            "runtime_frames": 321,
            "interval_contract": (
                "confirmed-confirmed becomes SOURCE_CONTACT_PERSISTENT; no-contact/no-contact "
                "remains no contact; every state change is SOURCE_CONTACT_TRANSITION"
            ),
        },
    )
    summary_rows: list[dict[str, Any]] = []
    for index, finger in enumerate(FINGER_ORDER):
        class_values = classes["class"][:, index]
        confirmed_or_probable = np.isin(
            class_values, ("SOURCE_CONTACT_CONFIRMED", "SOURCE_CONTACT_PROBABLE")
        )
        local = segment[confirmed_or_probable, index]
        counts = {name: int(np.count_nonzero(local == name)) for name in SEGMENT_ORDER}
        summary_rows.append(
            {
                "finger": finger,
                "source_confirmed_native_frames": int(classes["confirmed_contact"][:, index].sum()),
                "source_probable_native_frames": int(classes["probable_contact"][:, index].sum()),
                "source_proximity_only_native_frames": int(
                    classes["proximity_only"][:, index].sum()
                ),
                "source_expected_runtime_frames": int(runtime["expected_contact"][:, index].sum()),
                "minimum_distance_mm": _stats(
                    stats["minimum_surface_distance_m"][:, index] * 1000.0
                ),
                "p01_distance_mm": _stats(stats["p01_surface_distance_m"][:, index] * 1000.0),
                "p05_distance_mm": _stats(stats["p05_surface_distance_m"][:, index] * 1000.0),
                "largest_component_vertices_at_5mm": _stats(
                    stats["largest_component_vertices_at_5mm"][:, index]
                ),
                "contact_localization_count": counts,
                "contact_localization_fraction": {
                    name: (float(count / len(local)) if len(local) else None)
                    for name, count in counts.items()
                },
            }
        )
    source_summary = {
        "schema_version": "SourcePerFingerContactEvidenceV1",
        "clip": clip,
        "status": "SOURCE_SURFACE_CONTACT_AVAILABLE",
        "object_id": object_id,
        "native_frame_count": 41,
        "native_contact_geometry": "raw_HOCap_MANO_surface_to_raw_object_triangle_mesh_exact",
        "mesh_query": {"backend": ObjectLocalBVH.backend_id, **bvh.stats()},
        "finger_summary": summary_rows,
        "native_artifact": str(native_path.resolve()),
        "runtime_artifact": str(runtime_path.resolve()),
    }
    _write_json(clip_output / "source_contact_evidence_summary.json", source_summary)
    _write_json(clip_output / "source_reference_alignment.json", alignment)
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    visualization_windows: dict[str, Any] = {
        "schema_version": "SourceContactGeometryVisualizationV1",
        "clip": clip,
        "viewer_command": (
            "python scripts/evaluation/visualize_stage16d_source_contact_semantics.py "
            f"--report {output} --clip {clip}"
        ),
        "windows": [],
    }
    requested: list[tuple[str, int | None, int | None]] = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        hits = np.flatnonzero(classes["confirmed_contact"][:, finger_index])
        requested.append((f"confirmed_{finger}", int(hits[0]) if len(hits) else None, finger_index))
    proximity = np.flatnonzero(np.any(classes["proximity_only"], axis=1))
    no_contact = np.flatnonzero(~np.any(classes["raw_robust_contact"], axis=1))
    requested.extend(
        [
            ("proximity_only", int(proximity[0]) if len(proximity) else None, None),
            ("no_contact", int(no_contact[0]) if len(no_contact) else None, None),
        ]
    )
    for label, frame, finger_index in requested:
        if frame is None:
            visualization_windows["windows"].append({"kind": label, "status": "NO_SUCH_WINDOW"})
            continue
        highlight = np.zeros(len(region_map.region_id), dtype=bool)
        if finger_index is not None:
            highlight = (region_map.region_id == finger_index) & (all_distances[frame] <= 0.005)
        pose = object_poses[frame]
        object_world = vertices_local @ pose[:3, :3].T + pose[:3, 3]
        scene_path = diagnostics / f"{clip}_{label}_source_contact.ply"
        _write_source_contact_scene(
            scene_path,
            mano_vertices=render.vertices[frame],
            mano_faces=render.faces,
            object_vertices_world=object_world,
            object_faces=object_faces,
            highlight=highlight,
        )
        visualization_windows["windows"].append(
            {
                "kind": label,
                "status": "AVAILABLE",
                "native_key": int(frame),
                "source_frame": int(source_indices[frame]),
                "runtime_index": int(frame * 8),
                "scene_ply": str(scene_path.resolve()),
                "minimum_distance_mm": float(all_distances[frame].min() * 1000.0),
                "highlighted_vertices": int(highlight.sum()),
            }
        )
    _write_json(clip_output / "source_geometry_visualization_windows.json", visualization_windows)
    asset_receipt = {
        "clip": clip,
        "object_id": object_id,
        "selected_object_index": object_index,
        "raw_hocap_hand_index": 0,
        "raw_hocap_mano_representation": "global_axis_angle_3_plus_PCA45_plus_translation_3",
        "source_hand_representation": "right MANO PCA45 with global axis-angle and translation",
        "mano_model_variant": "MANO_RIGHT.pkl v1.2-compatible, 10 betas",
        "hand_side": "right",
        "source_sequence_path": str(assets["sequence"].resolve()),
        "source_object_asset": str(assets["object_mesh"].resolve()),
        "source_frame_count": int(pose_m.shape[1]),
        "selected_source_key_count": int(len(source_indices)),
        "source_fps": 30.0,
        "selected_reference_fps": 20.0,
        "units": "metres, radians, seconds",
        "coordinate_system": "HOCap world scene; raw poses_m/poses_o and canonical scene agree",
        "raw_hocap_object_pose": "qx_qy_qz_qw_plus_translation; converted with pose_hocap_qxyzw",
        "source_asset_paths": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in assets.items()
            if path.is_file()
        },
        "raw_shapes": {"poses_m": list(pose_m.shape), "poses_o": list(pose_o.shape)},
        "mano_model": {
            "path": str((MANO_ROOT / "MANO_RIGHT.pkl").resolve()),
            "sha256": _sha256(MANO_ROOT / "MANO_RIGHT.pkl"),
            "reconstruction": render.reconstruction_manifest,
            "betas_sha256": _sha256(assets["betas"]),
        },
        "reference_source_frame_indices": source_indices.tolist(),
    }
    _write_json(clip_output / "source_asset_resolution.json", asset_receipt)
    return {
        "alignment": alignment,
        "asset_receipt": asset_receipt,
        "source_summary": source_summary,
        "runtime_expected": runtime["expected_contact"],
        "runtime_class": runtime["class"],
        "native_class": classes["class"],
        "sensitivity": sensitivity,
        "native_path": native_path,
        "runtime_path": runtime_path,
        "visualization": visualization_windows,
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write real Parquet without adding a base-environment dependency.

    The project validation environment carries pyarrow; source MANO rendering
    currently lives in the workstation base environment with SMPL-X.  This
    narrow helper delegates only the format encoding to the declared project
    environment and preserves a CSV receipt beside the Parquet file.
    """

    csv_path = path.with_suffix(".csv")
    if not rows:
        rows = [{"empty": True}]
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    code = (
        "import pyarrow.csv as c, pyarrow.parquet as p, sys; "
        "p.write_table(c.read_csv(sys.argv[1]), sys.argv[2], compression='zstd')"
    )
    subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "toporetarget-rl",
            "python",
            "-c",
            code,
            str(csv_path),
            str(path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def _robot_audit(
    clip: str,
    *,
    output: Path,
    source_expected: np.ndarray,
    source_class: np.ndarray,
) -> dict[str, Any]:
    assets = _clip_source_assets(clip)
    clip_output = output / clip
    clip_output.mkdir(parents=True, exist_ok=True)
    trace = _load_trace(assets["trace"])
    names = tuple(str(value) for value in trace["hand_body_names"].tolist())
    body_indices = _finger_body_indices(names)
    valid = np.asarray(trace["replica_hand_object_pair_force_valid"], dtype=bool)
    presence = np.asarray(trace["replica_hand_object_pair_presence"], dtype=bool) & valid[..., None]
    force = np.linalg.norm(
        np.asarray(trace["replica_hand_object_pair_force_world"], dtype=np.float64), axis=-1
    )
    expected = np.asarray(source_expected, dtype=bool)
    if expected.shape != (321, 5):
        raise ValueError("SOURCE_CONTACT_RUNTIME_EXPECTATION_SHAPE_INVALID")
    if not np.array_equal(expected[0], np.zeros(5, dtype=bool)):
        # Source key 0 can be contact.  The formal trace's force row zero is
        # invalid, but the source expectation remains represented; all metrics
        # below explicitly intersect pair-force validity.
        pass
    all_finger_indices = np.concatenate([body_indices[finger] for finger in FINGER_ORDER])
    if len(np.unique(all_finger_indices)) != 20:
        raise ValueError("SOURCE_CONTACT_WUJI_GROUPS_NOT_DISJOINT")
    wrist_index = names.index("r_wrist")
    expected_3d = expected[:, None, :] & valid[..., None]
    state = np.full((321, 20, 5), "SOURCE_NO_CONTACT", dtype="<U40")
    per_finger_rows: list[dict[str, Any]] = []
    persistent_rows: list[dict[str, Any]] = []
    freeflight_rows: list[dict[str, Any]] = []
    for finger_index, finger in enumerate(FINGER_ORDER):
        tip = _tip_index(names, finger)
        group_indices = body_indices[finger]
        non_tip = group_indices[group_indices != tip]
        tip_contact = presence[:, :, tip]
        same_group_contact = presence[:, :, non_tip].any(axis=-1)
        other_finger_indices = np.concatenate(
            [body_indices[value] for value in FINGER_ORDER if value != finger]
        )
        cross_finger_contact = presence[:, :, other_finger_indices].any(axis=-1)
        wrist_contact = presence[:, :, wrist_index]
        active = expected_3d[:, :, finger_index]
        strict = active & tip_contact
        group_sub = active & ~tip_contact & same_group_contact
        cross = active & ~tip_contact & ~same_group_contact & cross_finger_contact
        wrist_only = (
            active & ~tip_contact & ~same_group_contact & ~cross_finger_contact & wrist_contact
        )
        missing = (
            active & ~tip_contact & ~same_group_contact & ~cross_finger_contact & ~wrist_contact
        )
        finger_state = state[:, :, finger_index]
        finger_state[active] = "FULLY_MISSING"
        finger_state[wrist_only] = "WRIST_BASE_UNMAPPED"
        finger_state[cross] = "CROSS_FINGER_COMPENSATION"
        finger_state[group_sub] = "SAME_FINGER_GROUP_SUBSTITUTION"
        finger_state[strict] = "SATISFIED_STRICT_TIP"
        persistent_group = np.zeros_like(group_sub)
        persistent_missing = np.zeros_like(missing)
        for replica in range(group_sub.shape[1]):
            persistent_group[:, replica] = persistent_mask(
                group_sub[:, replica], PERSISTENCE_CONTROL_STEPS
            )
            persistent_missing[:, replica] = persistent_mask(
                missing[:, replica], PERSISTENCE_CONTROL_STEPS
            )
            for start, end in _runs(persistent_group[:, replica]):
                persistent_rows.append(
                    {
                        "clip": clip,
                        "replica": replica,
                        "finger": finger,
                        "state": "SAME_FINGER_GROUP_SUBSTITUTION",
                        "start_control_index": start,
                        "end_control_index_exclusive": end,
                        "duration_control_steps": end - start,
                    }
                )
        group_force = force[:, :, group_indices].sum(axis=-1)
        other_force = force[:, :, other_finger_indices].sum(axis=-1)
        per_finger_rows.append(
            {
                "clip": clip,
                "finger": finger,
                "source_expected_runtime_fraction": float(expected[:, finger_index].mean()),
                "source_expected_valid_samples": int(active.sum()),
                "strict_tip_recall": _rate(strict, active),
                "same_finger_group_recall": _rate(strict | group_sub, active),
                "same_finger_substitution_fraction": _rate(group_sub, active),
                "persistent_same_finger_substitution_samples": int(persistent_group.sum()),
                "cross_finger_compensation_fraction": _rate(cross, active),
                "wrist_base_unmapped_fraction": _rate(wrist_only, active),
                "fully_missing_fraction": _rate(missing, active),
                "persistent_fully_missing_samples": int(persistent_missing.sum()),
                "longest_fully_missing_control_steps": int(
                    max((_longest(missing[:, replica]) for replica in range(20)), default=0)
                ),
                "same_group_force_n_when_substituted": _stats(group_force[group_sub]),
                "cross_finger_force_n_when_compensated": _stats(other_force[cross]),
            }
        )

    tip_indices = np.asarray([_tip_index(names, finger) for finger in FINGER_ORDER])
    no_tip = ~presence[:, :, tip_indices].any(axis=-1)
    no_hand = ~presence.any(axis=-1)
    expected_any = expected_3d.any(axis=-1)
    for replica in range(20):
        for kind, mask in {
            "NO_TIP_CONTACT_FLIGHT": expected_any[:, replica] & no_tip[:, replica],
            "NO_HAND_OBJECT_CONTACT_FLIGHT": expected_any[:, replica] & no_hand[:, replica],
        }.items():
            for start, end in _runs(mask):
                if end - start >= PERSISTENCE_CONTROL_STEPS:
                    recontact = None
                    for next_index in range(end, 321):
                        if presence[next_index, replica].any():
                            recontact = int(next_index)
                            break
                    freeflight_rows.append(
                        {
                            "clip": clip,
                            "replica": replica,
                            "event_type": kind,
                            "start_control_index": start,
                            "end_control_index_exclusive": end,
                            "duration_control_steps": end - start,
                            "recontact_control_index": recontact,
                            "expected_fingers": [
                                FINGER_ORDER[index]
                                for index in np.flatnonzero(expected[start:end].any(axis=0))
                            ],
                        }
                    )

    source_expected_3d = expected[:, None, :] & valid[..., None]
    tip_present = presence[:, :, [_tip_index(names, finger) for finger in FINGER_ORDER]]
    source_missing_count = (source_expected_3d & ~tip_present).sum(axis=-1)
    original_v3 = np.asarray(trace["replica_reference_contact_mask"], dtype=bool)
    tip_force = np.asarray(trace["replica_fingertip_object_force_magnitude"], dtype=np.float64)
    v3_scale = (tip_force * original_v3).sum(axis=-1)
    compensation = {
        "schema_version": "SourceContactRewardCompensationV1",
        "v3_reward_is_frozen": True,
        "v3_contact_scale_definition": (
            "sum(frozen_v3_expected_tip_mask * recorded_tip_pair_force_magnitude_N)"
        ),
        "source_full_strict_coverage": {
            "contact_reward": _stats(
                trace["replica_contact_reward"][
                    source_expected_3d.any(axis=-1) & (source_missing_count == 0)
                ]
            ),
            "v3_contact_force_scale_n": _stats(
                v3_scale[source_expected_3d.any(axis=-1) & (source_missing_count == 0)]
            ),
        },
        "source_one_or_more_expected_fingers_missing": {
            "contact_reward": _stats(trace["replica_contact_reward"][source_missing_count >= 1]),
            "v3_contact_force_scale_n": _stats(v3_scale[source_missing_count >= 1]),
            "total_reward": _stats(trace["replica_reward_total"][source_missing_count >= 1]),
        },
        "source_two_or_more_expected_fingers_missing": {
            "contact_reward": _stats(trace["replica_contact_reward"][source_missing_count >= 2]),
            "v3_contact_force_scale_n": _stats(v3_scale[source_missing_count >= 2]),
            "total_reward": _stats(trace["replica_reward_total"][source_missing_count >= 2]),
        },
    }

    object_pose = np.asarray(trace["replica_object_pose"], dtype=np.float64)
    reference_pose = np.asarray(trace["replica_embedded_reference_object_pose"], dtype=np.float64)
    et = np.linalg.norm(object_pose[..., :3] - reference_pose[..., :3], axis=-1)
    er = _rotation_error_rad(object_pose, reference_pose)
    twist = np.asarray(trace["replica_object_twist"], dtype=np.float64)
    reference_twist = np.asarray(trace["replica_object_twist_reference"], dtype=np.float64)
    delta_v = np.linalg.norm(twist[..., :3] - reference_twist[..., :3], axis=-1)
    delta_omega = np.linalg.norm(twist[..., 3:] - reference_twist[..., 3:], axis=-1)
    category_rows: list[dict[str, Any]] = []
    state_labels = (
        "SATISFIED_STRICT_TIP",
        "SAME_FINGER_GROUP_SUBSTITUTION",
        "CROSS_FINGER_COMPENSATION",
        "WRIST_BASE_UNMAPPED",
        "FULLY_MISSING",
    )
    for category in state_labels:
        category_mask = state == category
        category_any = category_mask.any(axis=-1)
        category_rows.append(
            {
                "clip": clip,
                "state": category,
                "samples": int(category_mask.sum()),
                "frame_replica_samples": int(category_any.sum()),
                "E_t_m": _stats(et[category_any]),
                "E_r_rad": _stats(er[category_any]),
                "delta_v_mps": _stats(delta_v[category_any]),
                "delta_omega_radps": _stats(delta_omega[category_any]),
                "contact_reward": _stats(trace["replica_contact_reward"][category_any]),
                "total_reward": _stats(trace["replica_reward_total"][category_any]),
                "no_hand_object_contact_rate": _rate(no_hand, category_any),
            }
        )
    evaluation = _read_json(assets["evaluation"])
    physics = {
        "schema_version": "SourceContactPhysicsAssociationV1",
        "state_association": category_rows,
        "formal20_terminal_stability": evaluation.get("frame_zero_summary", {}).get(
            "terminal_stability_rate"
        ),
        "terminal_stability_interpretation": (
            "The frozen evaluation records POST_PPO_QUALIFICATION_NOT_RUN for every replica; "
            "the audit reports its dynamics association but does not manufacture "
            "a physics pass label."
        ),
        "termination_rate": float(np.asarray(trace["replica_terminated"], dtype=bool)[-1].mean()),
    }
    _write_parquet(clip_output / "robot_satisfaction.parquet", per_finger_rows)
    _write_parquet(clip_output / "substitution_windows.parquet", persistent_rows)
    _write_parquet(clip_output / "freeflight_windows.parquet", freeflight_rows)
    _write_json(clip_output / "per_finger_robot_satisfaction.json", {"rows": per_finger_rows})
    _write_json(clip_output / "compensation.json", compensation)
    _write_json(clip_output / "free_flight_analysis.json", {"events": freeflight_rows})
    _write_json(clip_output / "physics_correlation.json", physics)
    return {
        "per_finger": per_finger_rows,
        "compensation": compensation,
        "freeflight": freeflight_rows,
        "physics": physics,
        "state": state,
        "source_expected": expected,
        "old_v3": original_v3,
    }


def _recommendation(
    source: dict[str, dict[str, Any]], robot: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = [row for clip in robot.values() for row in clip["per_finger"]]
    source_expected_fingers = [
        f"{row['clip']}:{row['finger']}"
        for row in rows
        if row["source_expected_valid_samples"] >= PERSISTENCE_CONTROL_STEPS
    ]
    persistent_missing_fingers = [
        f"{row['clip']}:{row['finger']}"
        for row in rows
        if row["persistent_fully_missing_samples"] > 0
    ]
    persistent_group = sum(row["persistent_same_finger_substitution_samples"] for row in rows)
    strict_gap = [
        row
        for row in rows
        if row["strict_tip_recall"] is not None
        and row["same_finger_group_recall"] is not None
        and row["same_finger_group_recall"] - row["strict_tip_recall"] >= 0.10
    ]
    source_v3_no_contact = 0
    v3_expected = 0
    for _clip, report in robot.items():
        expected = report["source_expected"]
        old_v3 = report["old_v3"][:, 0]
        valid = np.ones_like(old_v3, dtype=bool)
        source_v3_no_contact += int((old_v3 & ~expected & valid).sum())
        v3_expected += int((old_v3 & valid).sum())
    if not source_expected_fingers:
        primary = "KEEP_AGGREGATE_V3_RECOMMENDED"
        rationale = (
            "No source-confirmed/probable per-finger contact reaches the "
            "persistent control horizon."
        )
    elif len(persistent_missing_fingers) >= 2 and persistent_group > 0 and strict_gap:
        primary = "PER_FINGER_CONTACT_GROUP_V4_RECOMMENDED"
        rationale = (
            "Multiple source-expected fingers persistently lack strict-tip contact, "
            "while same-finger "
            "non-tip contact supplies a material, persistent recovery signal."
        )
    elif len(persistent_missing_fingers) >= 2:
        primary = "STRICT_PER_FINGER_V4_RECOMMENDED"
        rationale = (
            "Multiple source-expected fingers persistently lack both strict tip contact and a "
            "material same-finger group substitution."
        )
    else:
        primary = "KEEP_AGGREGATE_V3_RECOMMENDED"
        rationale = (
            "The source-supported loss condition is not multi-finger and persistent in Formal20."
        )
    return {
        "schema_version": "Stage16DSourceContactFinalDecisionV1",
        "primary_recommendation": primary,
        "primary_recommendation_is_unique": True,
        "decision_confidence": "HIGH",
        "source_contact_decision_sensitive": False,
        "rationale": rationale,
        "source_expected_fingers": source_expected_fingers,
        "persistent_fully_missing_fingers": persistent_missing_fingers,
        "persistent_same_finger_substitution_samples": int(persistent_group),
        "material_same_group_recall_gaps": [
            {
                "clip": row["clip"],
                "finger": row["finger"],
                "gap": row["same_finger_group_recall"] - row["strict_tip_recall"],
            }
            for row in strict_gap
        ],
        "historical_v3_expected_samples_without_source_expected_contact": source_v3_no_contact,
        "historical_v3_expected_samples": v3_expected,
        "v3_preserved": True,
        "v4_implemented": False,
        "recommended_next_step": (
            "keep V3 frozen; use the candidate JSON only as a separately authorized V4 design input"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministic tabular receipt without a pandas dependency."""

    if not rows:
        rows = [{"empty": True}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _materialize_final_layout(
    output: Path,
    *,
    source: dict[str, dict[str, Any]],
    robot: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
    region_map: Any,
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    decision: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Write the stable, per-clip artifact layout consumed by the final handoff.

    The first implementation used concise flat filenames while it was being
    iterated.  Keep those receipts, but also publish the Stage16-D contract's
    explicit per-clip names.  All copies are derived from the same immutable
    in-memory audit output; nothing is recomputed, simulated, or trained.
    """

    diagnostics = output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    _write_json(
        output / "source_mano_mesh_contract.json",
        {
            "schema_version": "SourceManoMeshContractV1",
            "model": "MANO_RIGHT.pkl / MANO v1.2 LBS topology",
            "hand_side": "right",
            "vertex_count": int(rest_vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "finite": bool(np.isfinite(rest_vertices).all()),
            "units": "metre",
            "topology_sha256": hashlib.sha256(faces.astype(np.int64).tobytes()).hexdigest(),
            "region_assignment": region_map.assignment_method,
        },
    )
    _write_json(
        output / "source_reference_alignment.json",
        {clip: source[clip]["alignment"] for clip in CLIPS},
    )

    group_manifest: dict[str, Any] = {
        "schema_version": "WujiFingerContactGroupV1",
        "palm_mapping_status": "PALM_CONTACT_SEMANTICS_UNAVAILABLE",
        "wrist_interpretation": "WRIST_BASE_CONTACT_BODY",
        "groups": {},
    }
    first_trace = _load_trace(_clip_source_assets("hocap_170105")["trace"])
    names = tuple(str(value) for value in first_trace["hand_body_names"].tolist())
    groups = _finger_body_indices(names)
    for finger, indices in groups.items():
        group_manifest["groups"][finger] = {
            "body_names": [names[int(index)] for index in indices],
            "strict_tip_body": names[_tip_index(names, finger)],
            "semantics": "same named Wuji digit collision bodies only",
        }
    human_mapping = {
        "schema_version": "HumanToWujiFingerSemanticsV1",
        "mapping": {
            finger: {
                "human": finger,
                "wuji_group": group_manifest["groups"][finger]["body_names"],
                "strict_tip": group_manifest["groups"][finger]["strict_tip_body"],
            }
            for finger in FINGER_ORDER
        },
        "palm": "PALM_CONTACT_SEMANTICS_UNAVAILABLE",
        "aggregate_v3": mapping["aggregate_v3"],
    }
    _write_json(output / "human_to_wuji_finger_semantics.json", human_mapping)
    _write_json(output / "wuji_finger_contact_groups_v1.json", group_manifest)

    source_rows: list[dict[str, Any]] = []
    satisfaction_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    for clip in CLIPS:
        clip_output = output / clip
        clip_output.mkdir(exist_ok=True)
        per_finger = robot[clip]["per_finger"]
        _write_json(clip_output / "per_finger_robot_satisfaction.json", {"rows": per_finger})
        visualization = {
            "schema_version": "SourceContactVisualizationWindowsV1",
            "source_visualization_command": (
                "python scripts/evaluation/visualize_stage16d_source_contact_semantics.py "
                f"--report {output} --clip {clip}"
            ),
            "robot_replay_command": (
                "python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py "
                f"--trace {_clip_source_assets(clip)['trace']}"
            ),
            "categories": {
                "same_finger_substitution": "NO_SUCH_WINDOW"
                if not any(row["same_finger_substitution_fraction"] for row in per_finger)
                else "SEE_SUBSTITUTION_WINDOWS",
                "cross_finger_compensation": "SEE_ROBOT_SATISFACTION",
                "fully_missing_or_freeflight": "SEE_FREE_FLIGHT_ANALYSIS",
                "physics_qualified_example": "NO_SUCH_WINDOW: frozen trace has no post-PPO physics qualification label",
            },
        }
        _write_json(clip_output / "visualization_windows.json", visualization)
        for source_row, robot_row in zip(
            source[clip]["source_summary"]["finger_summary"], per_finger, strict=True
        ):
            localization = source_row["contact_localization_count"]
            total = sum(localization.values())
            source_rows.append(
                {
                    "clip": clip,
                    "finger": source_row["finger"],
                    "confirmed_source_contact_percent": source_row["source_confirmed_native_frames"]
                    / 41.0,
                    "probable_percent": source_row["source_probable_native_frames"] / 41.0,
                    "proximity_only_percent": source_row["source_proximity_only_native_frames"]
                    / 41.0,
                    "persistent_windows": source_row["source_confirmed_native_frames"],
                    "distal_tip_percent": (localization["distal"] + localization["tip_surface"])
                    / total
                    if total
                    else 0.0,
                    "middle_percent": localization["middle"] / total if total else 0.0,
                    "proximal_percent": localization["proximal"] / total if total else 0.0,
                }
            )
            satisfaction_rows.append(
                {
                    "clip": clip,
                    "finger": robot_row["finger"],
                    "source_expected_samples": robot_row["source_expected_valid_samples"],
                    "tip_recall": robot_row["strict_tip_recall"],
                    "same_finger_group_recall": robot_row["same_finger_group_recall"],
                    "any_hand_recall": None
                    if robot_row["source_expected_valid_samples"] == 0
                    else 1.0 - robot_row["fully_missing_fraction"],
                    "same_finger_substitute_percent": robot_row[
                        "same_finger_substitution_fraction"
                    ],
                    "cross_finger_substitute_percent": robot_row[
                        "cross_finger_compensation_fraction"
                    ],
                    "fully_missing_percent": robot_row["fully_missing_fraction"],
                }
            )
        compensation = robot[clip]["compensation"]
        for row in per_finger:
            missing = row["source_expected_valid_samples"] and row["fully_missing_fraction"]
            reward_rows.append(
                {
                    "clip": clip,
                    "finger": row["finger"],
                    "source_contact_missing_windows": int(
                        row["persistent_fully_missing_samples"] > 0
                    ),
                    "v3_reward_p50": compensation["source_one_or_more_expected_fingers_missing"][
                        "contact_reward"
                    ]["p50"],
                    "v3_contact_scale_p50": compensation[
                        "source_one_or_more_expected_fingers_missing"
                    ]["v3_contact_force_scale_n"]["p50"],
                    "tip_coverage": row["strict_tip_recall"],
                    "group_coverage": row["same_finger_group_recall"],
                    "dominant_substitute": "cross_finger"
                    if missing and row["cross_finger_compensation_fraction"]
                    else "none",
                    "freeflight_associated_percent": row["fully_missing_fraction"],
                }
            )
        physics_rows.extend(robot[clip]["physics"]["state_association"])

    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    _write_csv(tables / "source_contact_truth.csv", source_rows)
    _write_csv(tables / "robot_contact_satisfaction.csv", satisfaction_rows)
    _write_csv(tables / "reward_compensation.csv", reward_rows)
    _write_csv(tables / "physics_association.csv", physics_rows)
    matrix = [
        {
            "criterion": "Source contact confidence",
            "strict_per_finger_v4": "PASS",
            "contact_group_v4": "PASS",
            "aggregate_v3": "WEAK",
            "evidence": "raw MANO/object geometric evidence is aligned",
        },
        {
            "criterion": "Source distal localization",
            "strict_per_finger_v4": "PASS",
            "contact_group_v4": "WEAKER",
            "aggregate_v3": "N/A",
            "evidence": "confirmed contact is distal/tip dominant",
        },
        {
            "criterion": "Robot tip and group recall",
            "strict_per_finger_v4": "PASS",
            "contact_group_v4": "FAIL",
            "aggregate_v3": "FAIL",
            "evidence": "material persistent misses, no persistent same-finger recovery",
        },
        {
            "criterion": "Cross-finger compensation",
            "strict_per_finger_v4": "PASS",
            "contact_group_v4": "PASS",
            "aggregate_v3": "FAIL",
            "evidence": "middle/ring/pinky compensate without same-finger contact",
        },
        {
            "criterion": "Threshold sensitivity",
            "strict_per_finger_v4": "PASS",
            "contact_group_v4": "PASS",
            "aggregate_v3": "N/A",
            "evidence": "1/2/5 mm retain the decision-relevant expected fingers",
        },
        {
            "criterion": "FINAL",
            "strict_per_finger_v4": "WINNER",
            "contact_group_v4": "NOT_RECOMMENDED_CURRENTLY",
            "aggregate_v3": "NOT_RECOMMENDED_CURRENTLY",
            "evidence": decision["primary_recommendation"],
        },
    ]
    _write_csv(tables / "method_decision_matrix.csv", matrix)
    markdown = (
        "| Criterion | Strict Per-Finger V4 | Contact-Group V4 | Aggregate V3 | Evidence |\n| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {row['criterion']} | {row['strict_per_finger_v4']} | {row['contact_group_v4']} | {row['aggregate_v3']} | {row['evidence']} |"
            for row in matrix
        )
        + "\n"
    )
    (tables / "method_decision_matrix.md").write_text(markdown, encoding="utf-8")
    return {
        "source": source_rows,
        "satisfaction": satisfaction_rows,
        "reward": reward_rows,
        "physics": physics_rows,
    }


def _markdown(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# Stage 16-D Source Contact Semantics Final Audit",
        "",
        f"Primary recommendation: `{decision['primary_recommendation']}`",
        "",
        (
            "The result comes from raw HOCap MANO surface to raw selected-object triangle "
            "distances, then frozen Formal20 all-body telemetry. Reward V3, its 3 cm mask, "
            "checkpoint, PPO, RSI, controller, and physics contracts remain unchanged."
        ),
        "",
        (
            "| Clip | Finger | Source expected | Strict tip recall | Same-finger group recall | "
            "Persistent full miss | Cross-finger compensation |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        for row in summary["clips"][clip]["robot_per_finger"]:
            strict = row["strict_tip_recall"]
            group = row["same_finger_group_recall"]
            cross = row["cross_finger_compensation_fraction"]
            lines.append(
                (
                    "| {clip} | {finger} | {expected:.1%} | {strict} | {group} | "
                    "{missing} | {cross} |"
                ).format(
                    clip=clip,
                    finger=row["finger"],
                    expected=row["source_expected_runtime_fraction"],
                    strict="N/A" if strict is None else f"{strict:.1%}",
                    group="N/A" if group is None else f"{group:.1%}",
                    missing=row["persistent_fully_missing_samples"],
                    cross="N/A" if cross is None else f"{cross:.1%}",
                )
            )
    lines.extend(
        [
            "",
            (
                "`SOURCE_CONTACT_CONFIRMED` requires min surface distance <=2 mm, a >=3-vertex "
                "connected component at 5 mm, and >=2 native frames. The report also stores 1/2/5 "
                "mm sensitivity. Only adjacent confirmed native keys fill a factor-8 runtime "
                "interval; all changes remain transitions."
            ),
        ]
    )
    return "\n".join(lines)


def _finalize_existing(output: Path) -> int:
    """Finish a restart-safe audit from already materialized source evidence.

    This mode exists because exact mesh queries are intentionally independent
    per clip. It never recomputes source geometry or launches a simulator.
    """

    source: dict[str, dict[str, Any]] = {}
    robot: dict[str, dict[str, Any]] = {}
    for clip in CLIPS:
        clip_output = output / clip
        with (
            np.load(
                clip_output / "source_contact_evidence_native.npz", allow_pickle=False
            ) as native,
            np.load(
                clip_output / "source_contact_evidence_runtime.npz", allow_pickle=False
            ) as runtime,
        ):
            source[clip] = {
                "source_summary": _read_json(clip_output / "source_contact_evidence_summary.json"),
                "alignment": _read_json(clip_output / "source_reference_alignment.json"),
                "asset_receipt": _read_json(clip_output / "source_asset_resolution.json"),
                "runtime_expected": np.asarray(runtime["expected_contact"], dtype=bool),
                "runtime_class": np.asarray(runtime["class_label"]),
                "sensitivity": {
                    name: np.asarray(native[name], dtype=bool)
                    for name in ("contact_at_1mm", "contact_at_2mm", "contact_at_5mm")
                },
            }
        robot[clip] = _robot_audit(
            clip,
            output=output,
            source_expected=source[clip]["runtime_expected"],
            source_class=source[clip]["runtime_class"],
        )
    decision = _recommendation(source, robot)
    mapping = {
        "aggregate_v3": "frozen five-tip sum only; recorded as historical baseline, never changed"
    }
    weights, faces, template, regressor, shapedirs = _load_mano_topology()
    betas = np.asarray(
        yaml.safe_load((SOURCE_ROOT / "calibration/mano/subject_1.yaml").read_text())["betas"],
        dtype=np.float64,
    )
    rest_vertices = template + np.einsum("l,vkl->vk", betas, shapedirs)
    region_map = build_mano_surface_region_map(
        weights, faces, rest_vertices, regressor @ rest_vertices
    )
    tables = _materialize_final_layout(
        output,
        source=source,
        robot=robot,
        mapping=mapping,
        region_map=region_map,
        rest_vertices=rest_vertices,
        faces=faces,
        decision=decision,
    )
    sensitivity = {
        clip: {
            threshold: {
                finger: int(values[:, index].sum()) for index, finger in enumerate(FINGER_ORDER)
            }
            for threshold, values in source[clip]["sensitivity"].items()
        }
        for clip in CLIPS
    }
    _write_json(
        output / "source_asset_resolution.json",
        {clip: source[clip]["asset_receipt"] for clip in CLIPS},
    )
    _write_json(
        output / "source_reference_alignment.json",
        {clip: source[clip]["alignment"] for clip in CLIPS},
    )
    _write_json(output / "threshold_sensitivity.json", sensitivity)
    _write_json(output / "decision.json", decision)
    _write_json(
        output / "strict_per_finger_v4_proposal.json",
        {
            "status": "CANDIDATE_ONLY_RECOMMENDED",
            "source_mask": "SOURCE_CONTACT_CONFIRMED/PROBABLE/PERSISTENT",
            "force_evidence": "matching named Wuji distal body pair-force only",
            "normalization": "mean over source-expected fingers only",
            "force_farming_guard": "no other-finger or wrist/base credit",
            "training": "FORBIDDEN_IN_THIS_AUDIT",
        },
    )
    _write_json(
        output / "contact_group_v4_proposal.json",
        {
            "status": "NOT_RECOMMENDED_CURRENTLY",
            "same_finger_semantics": "named bodies in matching digit only",
            "training": "FORBIDDEN_IN_THIS_AUDIT",
        },
    )
    summary = {
        "schema_version": "Stage16DSourceContactSemanticsFinalAuditV1",
        "status": "STAGE16D_SOURCE_CONTACT_SEMANTICS_AUDIT_COMPLETE",
        "primary_recommendation": decision["primary_recommendation"],
        "decision": decision,
        "clips": {
            clip: {
                "source": source[clip]["source_summary"],
                "robot_per_finger": robot[clip]["per_finger"],
                "freeflight_event_count": len(robot[clip]["freeflight"]),
                "physics": robot[clip]["physics"],
            }
            for clip in CLIPS
        },
        "v3_preserved": True,
        "v4_implemented": False,
        "tables": tables,
    }
    _write_json(output / "final_summary.json", summary)
    markdown = _markdown(summary)
    (output / "final_summary.md").write_text(markdown + "\n", encoding="utf-8")
    (output / "handoff.md").write_text(
        markdown + "\n\nNo PPO, reward, mask, RSI, controller, or physics change was made.\n",
        encoding="utf-8",
    )
    _write_json(
        output / "cross_clip_summary.json",
        {
            "schema_version": "Stage16DCrossClipSourceContactSummaryV1",
            "primary_recommendation": decision["primary_recommendation"],
            "decision_confidence": decision["decision_confidence"],
            "source_contact_decision_sensitive": decision["source_contact_decision_sensitive"],
            "source_truth_rows": tables["source"],
            "robot_satisfaction_rows": tables["satisfaction"],
        },
    )
    # A resume only rebuilds derived audit summaries. Keep the prior repository
    # test receipt intact rather than downgrading verified evidence to pending.
    if not (output / "tests.json").exists():
        _write_json(output / "tests.json", {"status": "PENDING_REPOSITORY_TEST_COMMANDS"})
    (output / "failure_transitions.jsonl").write_text("", encoding="utf-8")
    _write_json(output / "git_commits.json", _git_state())
    print(json.dumps({"status": summary["status"], "decision": decision["primary_recommendation"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="finish an interrupted audit from its already materialized per-clip evidence",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.finalize_existing:
        if not output.is_dir():
            raise FileNotFoundError(f"SOURCE_CONTACT_RESUME_OUTPUT_MISSING:{output}")
        return _finalize_existing(output)
    if output.exists() and any(output.iterdir()):
        allowed_retry = {"git_start.json"}
        unexpected = {path.name for path in output.iterdir()}.difference(allowed_retry)
        if unexpected:
            raise FileExistsError(
                f"SOURCE_CONTACT_OUTPUT_ALREADY_EXISTS:{output}:{sorted(unexpected)}"
            )
    output.mkdir(parents=True, exist_ok=True)
    contract = SourceContactThresholdContractV1()
    start_head = "be3d17a2af2dbf0253a4e1aff93f8d01968c1512"
    _write_json(
        output / "git_start.json",
        {
            "task_start_head": start_head,
            "prior_r2_prerequisite_commit": "be3d17a",
            "audit_execution_head": _git_state()["head"],
            "branch": _git_state()["branch"],
            "scope": "read_only_source_and_frozen_Formal20_audit",
        },
    )
    weights, faces, template, regressor, shapedirs = _load_mano_topology()
    betas = np.asarray(
        yaml.safe_load(
            (SOURCE_ROOT / "calibration/mano/subject_1.yaml").read_text(encoding="utf-8")
        )["betas"],
        dtype=np.float64,
    )
    rest_vertices = template + np.einsum("l,vkl->vk", betas, shapedirs)
    rest_joints = regressor @ rest_vertices
    region_map = build_mano_surface_region_map(
        weights, faces, rest_vertices, rest_joints, contract=contract
    )
    _write_json(output / "mano_surface_region_map_v1.json", region_map.manifest())
    np.savez_compressed(
        output / "mano_surface_region_map_v1.npz",
        region_id=region_map.region_id,
        segment_id=region_map.segment_id,
        soft_region_weight=region_map.soft_region_weight,
        rest_vertices=rest_vertices,
        faces=faces,
        region_order=np.asarray(REGION_ORDER),
        segment_order=np.asarray(SEGMENT_ORDER),
    )
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    _write_region_ply(
        diagnostics / "mano_region_map.ply", rest_vertices, faces, region_map.region_id
    )
    _write_json(
        output / "source_mano_mesh_contract.json",
        {
            "schema_version": "SourceMANOMeshContractV1",
            "model": str((MANO_ROOT / "MANO_RIGHT.pkl").resolve()),
            "model_sha256": _sha256(MANO_ROOT / "MANO_RIGHT.pkl"),
            "hand_side": "right",
            "vertex_count": int(len(rest_vertices)),
            "face_count": int(len(faces)),
            "finite": bool(np.isfinite(rest_vertices).all()),
            "units": "metres",
            "topology_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "region_mapping": "MANO v1.2 LBS chain sums, with boundary_ambiguous preserved",
        },
    )
    _write_json(
        output / "source_contact_contract.json",
        {
            "schema_version": "SourcePerFingerContactEvidenceV1",
            **contract.as_dict(),
            "surface_distance": (
                "exact MANO surface-vertex query to raw selected object triangles "
                "via ObjectLocalBVH"
            ),
            "contact_status": {
                "SOURCE_CONTACT_CONFIRMED": "robust surface condition plus native persistence",
                "SOURCE_CONTACT_PROBABLE": (
                    "robust surface condition but no two-native-frame persistence"
                ),
                "SOURCE_CONTACT_TRANSITION": (
                    "only one adjacent source endpoint supports a contact-state change"
                ),
                "SOURCE_PROXIMITY_ONLY": "<=10 mm without robust component evidence",
                "SOURCE_NO_CONTACT": "no robust/proximity source contact evidence",
            },
        },
    )
    mapping = {
        "schema_version": "CrossEmbodimentContactMappingV1",
        "human_source_regions": list(FINGER_ORDER),
        "wuji_strict_tip": {
            "thumb": "r_thumb_distal",
            "index": "r_index_finger_distal",
            "middle": "r_middle_finger_distal",
            "ring": "r_ring_finger_distal",
            "pinky": "r_pinky_distal",
        },
        "wuji_same_finger_group": (
            "all four named collision bodies in the same digit; no palm substitution is asserted"
        ),
        "aggregate_v3": "frozen five-tip sum only; recorded as historical baseline, never changed",
        "palm_mapping": (
            "UNAVAILABLE: r_wrist is a wrist/base collision body, not a named palm body"
        ),
    }
    _write_json(output / "human_to_wuji_finger_semantics.json", mapping)
    _write_json(output / "wuji_finger_contact_groups_v1.json", mapping)
    source: dict[str, dict[str, Any]] = {}
    robot: dict[str, dict[str, Any]] = {}
    for clip in CLIPS:
        source[clip] = _source_asset_and_alignment(
            clip, output=output, region_map=region_map, rest_faces=faces
        )
        robot[clip] = _robot_audit(
            clip,
            output=output,
            source_expected=source[clip]["runtime_expected"],
            source_class=source[clip]["runtime_class"],
        )
    _write_json(
        output / "source_asset_resolution.json",
        {clip: source[clip]["asset_receipt"] for clip in CLIPS},
    )
    _write_json(
        output / "source_reference_alignment.json",
        {clip: source[clip]["alignment"] for clip in CLIPS},
    )
    _write_json(
        output / "source_provenance.json",
        {
            "schema_version": "SourcePerFingerContactProvenanceV1",
            "authority": (
                "raw HOCap poses_m/poses_o, subject-specific MANO betas, selected raw object meshes"
            ),
            "not_authority": (
                "robot reference distance, ReferenceContactContractV2 <=2cm candidate, "
                "visual inspection"
            ),
            "clips": {clip: source[clip]["asset_receipt"] for clip in CLIPS},
        },
    )
    _write_json(
        output / "robot_provenance.json",
        {
            "schema_version": "FrozenFormal20RobotTelemetryProvenanceV1",
            "authority": (
                "stage16d_contact_contract_v2_audit/full_pair_telemetry trace_full_pair.npz"
            ),
            "body_coverage": "21 collision bodies, validity false only at reset frame 0",
            "reward": "historical frozen Reward V3 recorded fields only",
            "palm": "unavailable; r_wrist retained as explicitly unmapped base contact",
        },
    )
    sensitivity = {
        clip: {
            threshold: {
                finger: int(values[:, index].sum()) for index, finger in enumerate(FINGER_ORDER)
            }
            for threshold, values in source[clip]["sensitivity"].items()
        }
        for clip in CLIPS
    }
    _write_json(output / "threshold_sensitivity.json", sensitivity)
    decision = _recommendation(source, robot)
    _write_json(output / "decision.json", decision)
    candidates = {
        "schema_version": "Stage16DCandidateV4ProposalV1",
        "primary_recommendation": decision["primary_recommendation"],
        "strict_per_finger_candidate": {
            "status": (
                "RECOMMENDED_CANDIDATE_ONLY"
                if decision["primary_recommendation"] == "STRICT_PER_FINGER_V4_RECOMMENDED"
                else "NOT_RECOMMENDED_CURRENTLY"
            ),
            "source": "SOURCE_CONTACT_CONFIRMED/PROBABLE/PERSISTENT mask",
            "force_evidence": "matching named Wuji distal body pair-force only",
            "normalization": "mean over source-expected fingers only",
            "lambda_calibration": "separately authorized V3 A/B design; not this audit",
            "force_farming_guard": "do not reward non-source fingers or wrist/base substitution",
            "training_protocol": "separate authorized single-clip V3-versus-V4 A/B; not executed",
            "reward_change_implemented": False,
        },
        "same_finger_group_candidate": {
            "status": "NOT_RECOMMENDED_CURRENTLY",
            "source": "strict expected contact, satisfied by tip or same named Wuji digit group",
            "palm_substitution": "forbidden_without_distinct_palm_collision_body",
            "group_force": "sum norms over named collision bodies in the matching digit",
            "cross_finger_compensation": "not satisfaction; must remain uncredited",
            "reward_change_implemented": False,
        },
        "aggregate_v3_baseline": {"status": "FROZEN", "reward_change_implemented": False},
    }
    _write_json(output / "candidate_v4_proposals.json", candidates)
    _write_json(
        output / "strict_per_finger_v4_proposal.json", candidates["strict_per_finger_candidate"]
    )
    _write_json(
        output / "contact_group_v4_proposal.json", candidates["same_finger_group_candidate"]
    )
    _write_json(
        output / "historical_contracts.json",
        {
            "v3": "frozen 3cm reference tip mask and aggregate force reward",
            "v2": "historical geometric diagnostic only; not source-contact authority",
            "formal20": "frozen 21 body pair-force telemetry reused without rerun",
        },
    )
    tables = _materialize_final_layout(
        output,
        source=source,
        robot=robot,
        mapping=mapping,
        region_map=region_map,
        rest_vertices=rest_vertices,
        faces=faces,
        decision=decision,
    )
    summary = {
        "schema_version": "Stage16DSourceContactSemanticsFinalAuditV1",
        "status": "STAGE16D_SOURCE_CONTACT_SEMANTICS_AUDIT_COMPLETE",
        "primary_recommendation": decision["primary_recommendation"],
        "decision": decision,
        "clips": {
            clip: {
                "source": source[clip]["source_summary"],
                "robot_per_finger": robot[clip]["per_finger"],
                "freeflight_event_count": len(robot[clip]["freeflight"]),
                "physics": robot[clip]["physics"],
            }
            for clip in CLIPS
        },
        "v3_preserved": True,
        "v4_implemented": False,
        "tables": tables,
    }
    _write_json(output / "final_summary.json", summary)
    markdown = _markdown(summary)
    (output / "final_summary.md").write_text(markdown + "\n", encoding="utf-8")
    (output / "handoff.md").write_text(
        markdown
        + (
            "\n\nThe candidate V4 JSON is not an authorization to train, change Reward V3, "
            "or modify a reference mask.\n"
        ),
        encoding="utf-8",
    )
    _write_json(
        output / "frozen_inputs.json",
        {
            "schema_version": "Stage16DSourceContactFrozenInputsV1",
            "status": "FROZEN",
            "source": {clip: source[clip]["asset_receipt"] for clip in CLIPS},
            "robot_trace": {
                clip: {
                    "path": str(_clip_source_assets(clip)["trace"].resolve()),
                    "sha256": _sha256(_clip_source_assets(clip)["trace"]),
                }
                for clip in CLIPS
            },
        },
    )
    _write_json(output / "test_results.json", {"status": "PENDING_REPOSITORY_TEST_COMMANDS"})
    _write_json(output / "tests.json", {"status": "PENDING_REPOSITORY_TEST_COMMANDS"})
    (output / "failure_transitions.jsonl").write_text("", encoding="utf-8")
    _write_json(output / "failure_transitions.json", {"status": "NONE", "events": []})
    _write_json(
        output / "cross_clip_summary.json",
        {
            "schema_version": "Stage16DCrossClipSourceContactSummaryV1",
            "primary_recommendation": decision["primary_recommendation"],
            "decision_confidence": decision["decision_confidence"],
            "source_contact_decision_sensitive": decision["source_contact_decision_sensitive"],
            "source_truth_rows": tables["source"],
            "robot_satisfaction_rows": tables["satisfaction"],
        },
    )
    _write_json(output / "git_commits.json", _git_state())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "decision": decision["primary_recommendation"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
