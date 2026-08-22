#!/usr/bin/env python3
"""Materialize manifest-bound source-contact authority for one independent HOCap clip.

The geometric thresholds and Strict V4 force scale are inherited unchanged from
the frozen development contracts.  This command derives only a clip-local raw
MANO/object contact label and its factor-8 runtime mask; it never observes a
policy rollout or recalibrates a reward parameter.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.hocap_primary_object import (  # noqa: E402
    load_primary_object_authority,
    primary_object_from_authority,
)
from toporetarget.evaluation.source_contact_semantics import (  # noqa: E402
    FINGER_ORDER,
    REGION_ORDER,
    SEGMENT_ORDER,
    SourceContactThresholdContractV1,
    build_mano_surface_region_map,
    classify_source_contact,
    map_native_contact_to_control,
    per_region_surface_statistics,
    source_contact_localization,
)
from toporetarget.geometry.signed_distance.closest_point import ObjectLocalBVH  # noqa: E402
from toporetarget.rl.geometry_audit.raw_mocap_overlay import (  # noqa: E402
    RAW_ALIGNMENT_ROTATION_TOLERANCE_RAD,
    RAW_ALIGNMENT_TRANSLATION_TOLERANCE_M,
    pose_wxyz_to_matrix,
    resolve_raw_mocap_overlay,
)
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_manifest,
    atomic_write_json,
)
from toporetarget.rl.reference_tracking.contact_reward_mode import (  # noqa: E402
    ContactRewardMode,
    validate_frozen_contact_contract,
)
from toporetarget.rl.reference_tracking.reference_gated_contact import (  # noqa: E402
    EVALUATION_FINGERTIP_LINKS,
)
from toporetarget.rl.reference_tracking.strict_per_finger_contact import (  # noqa: E402
    SOURCE_CONTACT_REQUIRED_CLASSES,
    strict_source_contact_mask,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-object-authority", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--world-reference", type=Path, required=True)
    parser.add_argument("--reference-v2", type=Path, required=True)
    parser.add_argument("--strict-v4-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_SOURCE_CONTACT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_SOURCE_CONTACT_INPUT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_xyzw = Rotation.from_matrix(np.asarray(first)[..., :3, :3]).as_quat()
    second_xyzw = Rotation.from_matrix(np.asarray(second)[..., :3, :3]).as_quat()
    dot = np.abs(np.sum(first_xyzw * second_xyzw, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _load_reference_contract(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"timestamps", "source_frame_indices", "T_world_object_ref"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"INDEPENDENT_WORLD_REFERENCE_FIELDS_MISSING:{missing}")
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        source_indices = np.asarray(archive["source_frame_indices"], dtype=np.int64)
        object_pose = np.asarray(archive["T_world_object_ref"], dtype=np.float64)
    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or source_indices.shape != timestamps.shape
        or object_pose.shape != (timestamps.size, 4, 4)
        or np.any(np.diff(timestamps) <= 0.0)
        or np.any(source_indices < 0)
    ):
        raise ValueError("INDEPENDENT_WORLD_REFERENCE_CONTRACT_INVALID")
    return timestamps, int(timestamps.size)


def _load_mano_topology(
    mano_root: Path, betas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # MANO v1.2 pickles can import legacy chumpy code.  Apply the same narrow
    # compatibility shim as the project's reconstruction backend immediately
    # before reading that explicit local model.
    if not hasattr(inspect, "getargspec"):
        arg_spec = namedtuple("arg_spec", "args varargs keywords defaults")

        def getargspec(function: Any) -> Any:
            full = inspect.getfullargspec(function)
            return arg_spec(full.args, full.varargs, full.varkw, full.defaults)

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
    model = smplx.create(
        model_path=str(mano_root / "MANO_RIGHT.pkl"),
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
        dtype=torch.float64,
    )
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)
    template = model.v_template.detach().cpu().numpy().astype(np.float64)
    if template.shape == (1, 778, 3):
        template = template[0]
    shapedirs = model.shapedirs.detach().cpu().numpy().astype(np.float64)
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    if (
        weights.shape != (778, 16)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or template.shape != (778, 3)
        or shapedirs.shape != (778, 3, 10)
        or regressor.shape != (16, 778)
    ):
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_MANO_TOPOLOGY_INVALID")
    rest_vertices = template + np.tensordot(shapedirs, betas, axes=([2], [0]))
    rest_joints = regressor @ rest_vertices
    return weights, faces, rest_vertices, rest_joints


def _reference_v2_alignment(
    path: Path, *, native_object_pose: np.ndarray, native_frames: int
) -> dict[str, Any]:
    runtime_frames = (native_frames - 1) * 8 + 1
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "timestamps",
            "object_pose_translation_world_ref",
            "object_pose_quaternion_world_ref_wxyz",
            "metadata",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"INDEPENDENT_REFERENCE_V2_FIELDS_MISSING:{missing}")
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        translation = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
        quaternion_wxyz = np.asarray(
            archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64
        )
        metadata = json.loads(str(archive["metadata"].item()))
    if (
        timestamps.shape != (runtime_frames,)
        or translation.shape != (runtime_frames, 3)
        or quaternion_wxyz.shape != (runtime_frames, 4)
        or int(metadata.get("reference_kinematics_version", -1)) != 2
        or not np.allclose(timestamps, np.arange(runtime_frames) * 0.05, atol=1.0e-12)
    ):
        raise ValueError("INDEPENDENT_REFERENCE_V2_RUNTIME_DOMAIN_INVALID")
    keys = np.arange(native_frames, dtype=np.int64) * 8
    key_pose = np.broadcast_to(np.eye(4), (native_frames, 4, 4)).copy()
    key_pose[:, :3, 3] = translation[keys]
    xyzw = np.concatenate((quaternion_wxyz[keys, 1:], quaternion_wxyz[keys, :1]), axis=1)
    key_pose[:, :3, :3] = Rotation.from_quat(xyzw).as_matrix()
    translation_error = float(
        np.linalg.norm(key_pose[:, :3, 3] - native_object_pose[:, :3, 3], axis=1).max()
    )
    rotation_error = float(_rotation_error_rad(key_pose, native_object_pose).max())
    if (
        translation_error > RAW_ALIGNMENT_TRANSLATION_TOLERANCE_M
        or rotation_error > RAW_ALIGNMENT_ROTATION_TOLERANCE_RAD
    ):
        raise ValueError("INDEPENDENT_REFERENCE_V2_SOURCE_KEY_ALIGNMENT_FAILED")
    return {
        "status": "PASS",
        "native_frames": native_frames,
        "runtime_frames": runtime_frames,
        "factor": 8,
        "translation_max_m": translation_error,
        "rotation_max_rad": rotation_error,
    }


def _reference_robot_tip_distances(
    path: Path, *, object_surface: ObjectLocalBVH
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure the retargeted robot tips against the same object triangle surface.

    Grouped reward/RSE consumes this robot-reference field, whereas the Strict
    V4 mask above deliberately remains raw-MANO semantic authority.  Keeping
    both in one manifest-bound materialization prevents either representation
    from being silently substituted for the other.
    """

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "tracked_link_positions_world_ref",
            "object_pose_translation_world_ref",
            "object_pose_quaternion_world_ref_wxyz",
            "metadata",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"INDEPENDENT_REFERENCE_DISTANCE_FIELDS_MISSING:{missing}")
        tracked = np.asarray(archive["tracked_link_positions_world_ref"], dtype=np.float64)
        translation = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
        quaternion = np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
    names = tuple(str(value) for value in metadata.get("tracked_link_names", ()))
    missing_names = [name for name in EVALUATION_FINGERTIP_LINKS if name not in names]
    if (
        missing_names
        or tracked.ndim != 3
        or tracked.shape[1:] != (len(names), 3)
        or translation.shape != (tracked.shape[0], 3)
        or quaternion.shape != (tracked.shape[0], 4)
        or not np.isfinite(tracked).all()
        or not np.isfinite(translation).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError(f"INDEPENDENT_REFERENCE_DISTANCE_RUNTIME_INVALID:{missing_names}")
    indices = [names.index(name) for name in EVALUATION_FINGERTIP_LINKS]
    tips = tracked[:, indices]
    xyzw = np.concatenate((quaternion[:, 1:], quaternion[:, :1]), axis=1)
    rotation = Rotation.from_quat(xyzw).as_matrix()
    distances = np.empty((tracked.shape[0], len(indices)), dtype=np.float32)
    for frame in range(tracked.shape[0]):
        object_local = (tips[frame] - translation[frame]) @ rotation[frame]
        distances[frame] = object_surface.query(object_local)[-1]
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("INDEPENDENT_REFERENCE_DISTANCE_NONFINITE")
    return distances, {
        "source": "retargeted_robot_fingertip_to_primary_object_triangle_surface",
        "finger_order": list(FINGER_ORDER),
        "tracked_link_names": list(EVALUATION_FINGERTIP_LINKS),
        "runtime_frames": int(distances.shape[0]),
        "heldout_threshold_calibration": False,
    }


def _stats(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> int:
    args = _parser().parse_args()
    manifest = _json(args.manifest.resolve())
    assert_frozen_manifest(manifest)
    authority = load_primary_object_authority(args.primary_object_authority.resolve())
    rows = [row for row in manifest["clips"] if row.get("clip_id") == args.clip_id]
    if len(rows) != 1:
        raise ValueError(f"INDEPENDENT_SOURCE_CONTACT_CLIP_CARDINALITY:{len(rows)}")
    row = rows[0]
    primary = primary_object_from_authority(
        authority,
        sequence=str(row["sequence"]),
        available_object_ids=[str(value) for value in row["object_ids"]],
    )
    if row.get("primary_object_id") != primary or row.get("object_id") != primary:
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_PRIMARY_AUTHORITY_MISMATCH")
    if manifest.get("primary_object_authority_sha256") != authority.get("authority_sha256"):
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_AUTHORITY_HASH_MISMATCH")

    world_reference = args.world_reference.resolve()
    reference_v2 = args.reference_v2.resolve()
    strict_contract_path = args.strict_v4_contract.resolve()
    world_timestamps, native_frames = _load_reference_contract(world_reference)
    runtime_frames = (native_frames - 1) * 8 + 1
    strict_payload = _json(strict_contract_path)
    strict_parameters = validate_frozen_contact_contract(
        ContactRewardMode.STRICT_PER_FINGER_V4, strict_payload
    )

    with tempfile.TemporaryDirectory(prefix=f"{args.clip_id}_source_contact_") as temporary:
        identity_trace = Path(temporary) / "identity_reference_trace.npz"
        np.savez_compressed(
            identity_trace, reference_index=np.arange(native_frames, dtype=np.int64)
        )
        geometry = resolve_raw_mocap_overlay(
            trace_path=identity_trace,
            frame_count=native_frames,
            clip=args.clip_id,
            reference_path=world_reference,
        )
    if geometry.source_provenance.get("object_id") != primary:
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_REFERENCE_PRIMARY_MISMATCH")
    if not np.allclose(geometry.runtime_timestamps_s, world_timestamps, atol=1.0e-12):
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_REFERENCE_TIME_MISMATCH")

    object_pose = pose_wxyz_to_matrix(geometry.raw_object_pose_world_wxyz)
    v2_alignment = _reference_v2_alignment(
        reference_v2, native_object_pose=object_pose, native_frames=native_frames
    )
    mano_root = Path(str(geometry.source_provenance["mano_model_root"]))
    raw_meta = Path(str(geometry.source_provenance["raw_meta"]))
    meta = yaml.safe_load(raw_meta.read_text(encoding="utf-8")) or {}
    subject = str(meta.get("subject_id", ""))
    betas_path = raw_meta.parents[2] / "calibration" / "mano" / f"{subject}.yaml"
    betas_payload = yaml.safe_load(betas_path.read_text(encoding="utf-8")) or {}
    betas = np.asarray(betas_payload.get("betas"), dtype=np.float64)
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_MANO_BETAS_INVALID")
    weights, mano_faces, rest_vertices, rest_joints = _load_mano_topology(mano_root, betas)
    if not np.array_equal(mano_faces, geometry.raw_mano_faces):
        raise ValueError("INDEPENDENT_SOURCE_CONTACT_MANO_FACE_TOPOLOGY_MISMATCH")
    threshold_contract = SourceContactThresholdContractV1()
    region_map = build_mano_surface_region_map(
        weights,
        mano_faces,
        rest_vertices,
        rest_joints,
        contract=threshold_contract,
    )
    triangles = geometry.raw_object_vertices_local[geometry.raw_object_faces]
    bvh = ObjectLocalBVH(triangles)
    distances = np.empty((native_frames, geometry.raw_mano_vertices_world.shape[1]))
    for frame in range(native_frames):
        rotation = object_pose[frame, :3, :3]
        translation = object_pose[frame, :3, 3]
        object_local = (geometry.raw_mano_vertices_world[frame] - translation) @ rotation
        _nearest, _face, _barycentric, distance = bvh.query(object_local)
        distances[frame] = distance

    surface = per_region_surface_statistics(distances, region_map, mano_faces)
    nearest_segment, nearest_segment_distance = source_contact_localization(distances, region_map)
    classes = classify_source_contact(
        surface["minimum_surface_distance_m"][:, :5],
        surface["largest_component_vertices_at_5mm"][:, :5],
        contract=threshold_contract,
    )
    runtime = map_native_contact_to_control(classes["class"], factor=8)
    mask = strict_source_contact_mask(runtime["class"])
    if mask.shape != (runtime_frames, 5):
        raise AssertionError("INDEPENDENT_SOURCE_CONTACT_RUNTIME_MASK_SHAPE_INVALID")
    reference_distances, reference_distance_contract = _reference_robot_tip_distances(
        reference_v2, object_surface=bvh
    )
    if reference_distances.shape != (runtime_frames, 5):
        raise AssertionError("INDEPENDENT_REFERENCE_DISTANCE_SHAPE_INVALID")

    output = args.output_root.resolve()
    clip_output = output / args.clip_id
    reference_distance_path = output / f"reference_contact_mask_{args.clip_id}.npz"
    if (
        clip_output.exists()
        or (output / f"strict_source_contact_mask_{args.clip_id}.npz").exists()
        or reference_distance_path.exists()
    ):
        raise FileExistsError(f"INDEPENDENT_SOURCE_CONTACT_REFUSES_OVERWRITE:{clip_output}")
    clip_output.mkdir(parents=True)
    native_path = clip_output / "source_contact_evidence_native.npz"
    runtime_path = clip_output / "source_contact_evidence_runtime.npz"
    mask_path = output / f"strict_source_contact_mask_{args.clip_id}.npz"
    np.savez_compressed(
        native_path,
        source_frame_indices=geometry.raw_frame_float,
        native_timestamp_s=world_timestamps,
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
        minimum_surface_distance_m=surface["minimum_surface_distance_m"],
        p01_surface_distance_m=surface["p01_surface_distance_m"],
        p05_surface_distance_m=surface["p05_surface_distance_m"],
        thresholds_m=surface["thresholds_m"],
        near_vertex_count=surface["near_vertex_count"],
        near_vertex_fraction=surface["near_vertex_fraction"],
        largest_component_vertices_at_5mm=surface["largest_component_vertices_at_5mm"],
        nearest_segment=nearest_segment,
        nearest_segment_distance_m=nearest_segment_distance,
    )
    np.savez_compressed(
        runtime_path,
        control_index=np.arange(runtime_frames, dtype=np.int64),
        native_to_control_index=runtime["native_to_control_index"],
        finger_order=np.asarray(FINGER_ORDER),
        class_label=runtime["class"],
        expected_contact=runtime["expected_contact"],
        exact_source_key=runtime["exact_source_key"],
    )
    np.savez_compressed(
        mask_path,
        strict_source_contact_mask=mask,
        source_contact_class=runtime["class"],
        finger_names=np.asarray(FINGER_ORDER),
        control_index=np.arange(runtime_frames, dtype=np.int64),
    )
    np.savez_compressed(
        reference_distance_path,
        reference_fingertip_to_object_distance_m=reference_distances,
        finger_order=np.asarray(FINGER_ORDER),
        metadata=np.asarray(json.dumps(reference_distance_contract, sort_keys=True)),
    )
    source_inputs = {
        "manifest": _artifact(args.manifest.resolve()),
        "primary_object_authority": _artifact(args.primary_object_authority.resolve()),
        "world_reference": _artifact(world_reference),
        "reference_v2": _artifact(reference_v2),
        "strict_v4_development_contract": _artifact(strict_contract_path),
        "raw_meta": _artifact(raw_meta),
        "raw_mano_pose": _artifact(Path(str(geometry.source_provenance["raw_mano_pose"]))),
        "raw_object_pose": _artifact(Path(str(geometry.source_provenance["raw_object_pose"]))),
        "raw_object_mesh": _artifact(Path(str(geometry.source_provenance["raw_object_mesh"]))),
        "mano_betas": _artifact(betas_path),
        "mano_model": _artifact(mano_root / "MANO_RIGHT.pkl"),
    }
    receipt = {
        "schema_version": "IndependentHOCapSourceContactAuthorityV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "sequence": row["sequence"],
        "primary_object_id": primary,
        "primary_object_authority_sha256": authority["authority_sha256"],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "native_frame_count": native_frames,
        "runtime_frame_count": runtime_frames,
        "runtime_mapping": "manifest_bound_native_keys_to_factor8_control",
        "source_geometry": "raw_HOCap_MANO_surface_to_primary_object_triangle_mesh_exact",
        "threshold_contract": threshold_contract.as_dict(),
        "strict_v4_method_parameters": {
            "lambda_tip_n": float(strict_parameters["lambda_tip_n"]),
            "numerical_floor_n": float(strict_parameters["numerical_floor_n"]),
            "source_required_classes": list(SOURCE_CONTACT_REQUIRED_CLASSES),
            "heldout_scale_recalibration": False,
        },
        "coordinate_alignment": geometry.coordinate_alignment,
        "reference_v2_alignment": v2_alignment,
        "reference_robot_tip_distance": {
            **reference_distance_contract,
            "statistics_m": {
                finger: _stats(reference_distances[:, index])
                for index, finger in enumerate(FINGER_ORDER)
            },
        },
        "mesh_query": {"backend": ObjectLocalBVH.backend_id, **bvh.stats()},
        "counts_by_finger": {
            finger: int(mask[:, index].sum()) for index, finger in enumerate(FINGER_ORDER)
        },
        "required_runtime_frame_count": int(mask.any(axis=1).sum()),
        "minimum_distance_m": {
            finger: _stats(surface["minimum_surface_distance_m"][:, index])
            for index, finger in enumerate(FINGER_ORDER)
        },
        "artifacts": {
            "native": {"path": str(native_path), "sha256": _sha256(native_path)},
            "runtime": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "strict_mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
            "reference_distance": {
                "path": str(reference_distance_path),
                "sha256": _sha256(reference_distance_path),
            },
        },
        "frozen_inputs": source_inputs,
        "policy_outcomes_observed": False,
        "reward_scale_calibrated_on_heldout": False,
    }
    atomic_write_json(clip_output / "source_contact_authority.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
