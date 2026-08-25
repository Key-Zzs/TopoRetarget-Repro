#!/usr/bin/env python3
"""Run RetargetInputQualityV1 before expensive HOCap retarget solvers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.materialize_hocap_episode import load_episode_row  # noqa: E402
from toporetarget.adapters.datasets.hocap import (  # noqa: E402
    HOCapAdapterV1,
    hocap_mano_storage_index,
)
from toporetarget.adapters.datasets.stage12_base import (  # noqa: E402
    backend_posed_joint_track,
    make_hand,
    render_mano_pca45,
    sha256_paths,
)
from toporetarget.retarget.bones import load_bone_profile  # noqa: E402
from toporetarget.retarget.input_quality import (  # noqa: E402
    RetargetInputQualityContractV1,
    RetargetInputQualityError,
    bone_quality,
    keypoint_frame_diagnostics,
    repair_mano_pose,
    repair_object_pose_qxyzw,
    rotation_step_angles,
    select_mano_primary_wrist_frames,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mano-model-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-frame-csv", type=Path, required=True)
    parser.add_argument("--repaired-output", type=Path, required=True)
    return parser


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _dataset_root(requested: Path) -> Path:
    resolved = requested.resolve()
    storage = resolved.parent if (resolved / "data").is_dir() else resolved
    return storage / "HOCap"


def _load_raw(
    row: dict[str, Any], data_root: Path
) -> tuple[Path, dict[str, Any], Path, Path, np.ndarray, np.ndarray]:
    import yaml

    dataset = _dataset_root(data_root)
    sequence_dir = dataset / "data" / str(row["raw_sequence"])
    meta_path = sequence_dir / "meta.yaml"
    mano_path = sequence_dir / "poses_m.npy"
    object_path = sequence_dir / "poses_o.npy"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    mano = np.load(mano_path, mmap_mode="r")
    objects = np.load(object_path, mmap_mode="r")
    if mano.ndim != 3 or mano.shape[2:] != (51,):
        raise RetargetInputQualityError(f"RAW_TRACKING_QUALITY_FAILED:MANO_SHAPE:{mano.shape}")
    if objects.ndim != 3 or objects.shape[2:] != (7,):
        raise RetargetInputQualityError(
            f"RAW_TRACKING_QUALITY_FAILED:OBJECT_POSE_SHAPE:{objects.shape}"
        )
    if objects.shape[1] == mano.shape[1] and objects.shape[0] != mano.shape[1]:
        objects = np.asarray(objects).transpose(1, 0, 2)
    return sequence_dir, meta, mano_path, object_path, mano, objects


def _pairwise_duplicate_count(points: np.ndarray, threshold: float) -> np.ndarray:
    delta = points[:, :, None] - points[:, None, :]
    distances = np.linalg.norm(delta, axis=-1)
    upper = np.triu(np.ones(distances.shape[1:], dtype=bool), k=1)
    return np.count_nonzero((distances <= threshold) & upper[None], axis=(1, 2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["frame"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scan(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = RetargetInputQualityContractV1()
    index_path = args.episode_index.resolve()
    row = load_episode_row(index_path, args.episode_id)
    hand = str(row.get("active_hand", ""))
    if hand not in {"left", "right"}:
        raise RetargetInputQualityError("RAW_TRACKING_QUALITY_FAILED:SINGLE_HAND_REQUIRED")
    start, end = int(row["start_frame"]), int(row["end_frame"])
    sequence_dir, meta, mano_path, object_path, raw_mano, raw_objects = _load_raw(
        row, args.data_root
    )
    if end > raw_mano.shape[1] or end > raw_objects.shape[0]:
        raise RetargetInputQualityError("RAW_TRACKING_QUALITY_FAILED:MISSING_FRAMES")
    fps = float(meta.get("fps") or meta.get("frame_rate") or 30.0)
    timestamps = np.arange(end - start, dtype=np.float64) / fps
    if not np.all(np.diff(timestamps) > 0):
        raise RetargetInputQualityError("RAW_TRACKING_QUALITY_FAILED:TIMESTAMPS_NOT_MONOTONIC")
    hand_index = hocap_mano_storage_index(hand)
    if hand_index >= raw_mano.shape[0]:
        raise RetargetInputQualityError("RAW_TRACKING_QUALITY_FAILED:MANO_SLOT_MISSING")
    mano_input = np.asarray(raw_mano[hand_index, start:end], dtype=np.float64)
    object_input = np.asarray(raw_objects[start:end], dtype=np.float64)
    repaired_mano, mano_repair = repair_mano_pose(mano_input, timestamps, frozen)
    repaired_objects, object_repair = repair_object_pose_qxyzw(object_input, timestamps, frozen)

    adapter = HOCapAdapterV1(
        data_root=_dataset_root(args.data_root).parent,
        mano_model_root=args.mano_model_root.resolve(),
    )
    betas, calibration_path = adapter._subject_betas(  # noqa: SLF001
        meta=meta, sequence_dir=sequence_dir
    )
    source_hash = sha256_paths([sequence_dir / "meta.yaml", mano_path, calibration_path])
    render = render_mano_pca45(
        repaired_mano,
        side=hand,
        mano_model_root=args.mano_model_root.resolve(),
        betas=betas,
        dataset_name="hocap",
        source_annotation_path=mano_path,
        source_annotation_hash=source_hash,
    )
    valid = np.ones(len(repaired_mano), dtype=bool)
    reconstructed_hand = make_hand(
        hand_id=f"{hand}_hand",
        side=hand,
        vertices_scene=render.vertices,
        faces=render.faces,
        wrist_pose_scene=render.wrist_pose_scene,
        valid=valid,
        mano_parameters=None,
        mano_model_root=args.mano_model_root.resolve(),
        metadata={"quality_scan_only": True},
        native_joint_track=backend_posed_joint_track(render, valid=valid),
    )
    keypoints = np.asarray(
        reconstructed_hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    vertices = np.asarray(render.vertices, dtype=np.float64)
    native_joints = np.asarray(render.posed_joints_native, dtype=np.float64)
    if not np.isfinite(vertices).all() or not np.isfinite(native_joints).all():
        raise RetargetInputQualityError("RAW_TRACKING_QUALITY_FAILED:MANO_RECONSTRUCTION_NONFINITE")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    parents = np.asarray([item.parent_index for item in bone_profile.bones], dtype=np.int64)
    children = np.asarray([item.child_index for item in bone_profile.bones], dtype=np.int64)
    # HOCap exposes parameters rather than a separate raw keypoint track.  The
    # canonical track is therefore explicitly the MANO-parametric authority.
    bones = bone_quality(keypoints, keypoints, parents, children, frozen)
    frame_diag = keypoint_frame_diagnostics(keypoints, frozen)
    wrist_frames, wrist_authority = select_mano_primary_wrist_frames(
        repaired_mano[:, :3],
        repaired_mano[:, 48:51],
        timestamps=timestamps,
        contract=frozen,
        reconstructed_wrist_pose=render.wrist_pose_scene,
    )
    if np.any(wrist_authority == "RAW_TRACKING_QUALITY_FAILED"):
        raise RetargetInputQualityError("RAW_TRACKING_QUALITY_FAILED:WRIST_FRAME_UNRECOVERABLE")
    if np.any(wrist_authority == "KEYPOINT_DERIVED_WRIST_FRAME_DIAGNOSTIC_ONLY"):
        raise RetargetInputQualityError(
            "RAW_TRACKING_QUALITY_FAILED:KEYPOINT_WRIST_FRAME_IS_DIAGNOSTIC_ONLY"
        )
    if np.any(bones["unrecoverable"]):
        bad = np.flatnonzero(bones["unrecoverable"]).tolist()
        raise RetargetInputQualityError(
            f"RAW_TRACKING_QUALITY_FAILED:MANO_SKELETON_DEGENERATE:{bad}"
        )
    duplicate_count = _pairwise_duplicate_count(keypoints, frozen.duplicate_joint_distance_m)
    if np.any(duplicate_count):
        bad = np.flatnonzero(duplicate_count).tolist()
        raise RetargetInputQualityError(f"RAW_TRACKING_QUALITY_FAILED:DUPLICATE_MANO_JOINTS:{bad}")
    orientation_step = rotation_step_angles(wrist_frames[:, :3, :3])
    translation_step = np.zeros(len(wrist_frames), dtype=np.float64)
    translation_step[1:] = np.linalg.norm(np.diff(wrist_frames[:, :3, 3], axis=0), axis=1)
    object_rotation_step = np.zeros((len(repaired_objects), repaired_objects.shape[1]))
    for object_index in range(repaired_objects.shape[1]):
        from scipy.spatial.transform import Rotation

        object_rotation_step[:, object_index] = rotation_step_angles(
            Rotation.from_quat(repaired_objects[:, object_index, :4]).as_matrix()
        )
    object_pose_finite = np.isfinite(repaired_objects).all(axis=(1, 2))
    repaired_frames = set(mano_repair["repaired_frames"])
    for object_row in object_repair["objects"]:
        repaired_frames.update(object_row["repaired_frames"])
    rows: list[dict[str, Any]] = []
    for local in range(len(timestamps)):
        rows.append(
            {
                "episode_id": args.episode_id,
                "local_frame": local,
                "source_frame": start + local,
                "timestamp_seconds": float(timestamps[local]),
                "mano_global_orientation_finite": bool(np.isfinite(repaired_mano[local, :3]).all()),
                "mano_pose_finite": bool(np.isfinite(repaired_mano[local, 3:48]).all()),
                "mano_translation_finite": bool(np.isfinite(repaired_mano[local, 48:51]).all()),
                "mano_shape_finite": bool(np.isfinite(betas).all()),
                "restored_vertices_finite": bool(np.isfinite(vertices[local]).all()),
                "restored_joints_finite": bool(np.isfinite(native_joints[local]).all()),
                "minimum_bone_length_m": float(bones["mano_min_bone_length_m"][local]),
                "duplicate_joint_pairs": int(duplicate_count[local]),
                "wrist_longitudinal_axis_norm_m": float(frame_diag["longitudinal_norm_m"][local]),
                "wrist_lateral_axis_norm_m": float(frame_diag["lateral_norm_m"][local]),
                "wrist_axis_sine": float(frame_diag["axis_sine"][local]),
                "keypoint_wrist_frame_valid_diagnostic": bool(frame_diag["valid"][local]),
                "wrist_authority": str(wrist_authority[local]),
                "orientation_step_rad": float(orientation_step[local]),
                "orientation_discontinuity": bool(
                    orientation_step[local] > frozen.orientation_discontinuity_rad
                ),
                "translation_step_m": float(translation_step[local]),
                "translation_discontinuity": bool(
                    translation_step[local] > frozen.translation_discontinuity_m
                ),
                "object_pose_finite": bool(object_pose_finite[local]),
                "object_max_orientation_step_rad": float(
                    np.max(object_rotation_step[local], initial=0.0)
                ),
                "timestamp_monotonic": bool(
                    local == 0 or timestamps[local] > timestamps[local - 1]
                ),
                "repaired": local in repaired_frames,
            }
        )
    _write_csv(args.per_frame_csv.resolve(), rows)
    repair_path = args.repaired_output.resolve()
    _atomic_npz(
        repair_path,
        {
            "mano_pose_51": repaired_mano,
            "object_pose_qxyzw": repaired_objects,
            "timestamps": timestamps,
            "wrist_pose_scene": wrist_frames,
            "wrist_authority": wrist_authority.astype("S64"),
            "source_frame_range": np.asarray([start, end], dtype=np.int64),
        },
    )
    warnings: list[str] = []
    if np.any(~frame_diag["valid"]):
        warnings.append("KEYPOINT_WRIST_FRAME_DEGENERATE_DIAGNOSTIC_ONLY")
    if np.any(orientation_step > frozen.orientation_discontinuity_rad):
        warnings.append("MANO_WRIST_ORIENTATION_DISCONTINUITY")
    if np.any(translation_step > frozen.translation_discontinuity_m):
        warnings.append("MANO_WRIST_TRANSLATION_DISCONTINUITY")
    receipt = {
        "schema_version": frozen.schema_version,
        "status": "PASS_WITH_WARNINGS" if warnings else "PASS",
        "terminal_classification": "INPUT_QUALITY_PRECHECK_PASS",
        "episode_id": args.episode_id,
        "raw_sequence": row["raw_sequence"],
        "active_hand": hand,
        "target_object": row["target_object"],
        "source_frame_range": [start, end],
        "frames": end - start,
        "fps": fps,
        "contract": frozen.as_dict(),
        "contract_sha256": frozen.contract_sha256,
        "wrist_orientation_authority": "MANO_GLOBAL_WRIST_ORIENTATION",
        "canonical_keypoint_wrist_production_authority": False,
        "keypoint_wrist_diagnostic_invalid_frames": np.flatnonzero(~frame_diag["valid"]).tolist(),
        "mano_repair": mano_repair,
        "object_repair": object_repair,
        "warnings": warnings,
        "checks": {
            "mano_global_orientation_finite": bool(np.isfinite(repaired_mano[:, :3]).all()),
            "mano_pose_finite": bool(np.isfinite(repaired_mano[:, 3:48]).all()),
            "mano_translation_finite": bool(np.isfinite(repaired_mano[:, 48:51]).all()),
            "mano_shape_finite": bool(np.isfinite(betas).all()),
            "restored_mano_vertices_finite": bool(np.isfinite(vertices).all()),
            "restored_mano_joints_finite": bool(np.isfinite(native_joints).all()),
            "mano_bones_nondegenerate": bool(np.all(bones["mano_bones_valid"])),
            "duplicate_mano_joints_absent": bool(not np.any(duplicate_count)),
            "object_pose_finite": bool(np.all(object_pose_finite)),
            "object_rotation_valid": True,
            "timestamps_monotonic": bool(np.all(np.diff(timestamps) > 0)),
            "missing_frames_absent": True,
            "long_invalid_gaps_absent": True,
        },
        "provenance": {
            "episode_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
            "raw_mano": {"path": str(mano_path), "sha256": sha256_file(mano_path)},
            "raw_object": {"path": str(object_path), "sha256": sha256_file(object_path)},
            "mano_calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
        },
        "artifacts": {
            "per_frame_quality_csv": str(args.per_frame_csv.resolve()),
            "repaired_input": str(repair_path),
            "repaired_input_sha256": sha256_file(repair_path),
        },
        "scan_seconds": time.perf_counter() - started,
    }
    receipt["receipt_sha256"] = _stable_hash(receipt)
    return receipt


def main() -> int:
    args = _parser().parse_args()
    report = args.report.resolve()
    try:
        receipt = scan(args)
    except (RetargetInputQualityError, ValueError, OSError, RuntimeError) as error:
        receipt = {
            "schema_version": "RetargetInputQualityV1",
            "status": "FAIL",
            "terminal_classification": (
                "UNRECOVERABLE_TRACKING_GAP"
                if "UNRECOVERABLE_TRACKING_GAP" in str(error)
                else "RAW_TRACKING_QUALITY_FAILED"
            ),
            "episode_id": args.episode_id,
            "reason": f"{type(error).__name__}: {error}",
        }
        _atomic_json(report, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    _atomic_json(report, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
