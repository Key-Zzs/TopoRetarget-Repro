"""Metric registry implementation for the fixed four-clip quality experiment."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory

from .contact import _proxy_metrics
from .schema import QUALITY_SCHEMA_VERSION, ClipSpec, write_json

FINGERS = {"thumb": (1, 4), "index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}


def _finite_mean(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if len(finite) == 0 else float(np.mean(finite))


def _hand(sequence: Any, side: str) -> Any:
    for item in sequence.hands:
        if item.side == side or item.hand_id == side:
            return item
    raise ValueError(f"canonical sequence has no {side} hand")


def _translation_steps(base: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(base, dtype=np.float64)[:, :3, 3]
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    acceleration = np.diff(values, n=2, axis=0)
    jerk = np.diff(values, n=3, axis=0)
    return (
        float(np.max(steps, initial=0.0)),
        float(np.max(np.linalg.norm(acceleration, axis=1), initial=0.0)),
        float(np.max(np.linalg.norm(jerk, axis=1), initial=0.0)),
    )


def _rotation_steps(base: np.ndarray) -> tuple[float, float, float]:
    from toporetarget.geometry.se3 import rotation_geodesic_error

    rotations = np.asarray(base, dtype=np.float64)[:, :3, :3]
    if len(rotations) < 2:
        return 0.0, 0.0, 0.0
    deltas = rotation_geodesic_error(rotations[1:], rotations[:-1])
    acceleration = np.diff(deltas)
    jerk = np.diff(deltas, n=2)
    return (
        float(np.max(deltas, initial=0.0)),
        float(np.max(np.abs(acceleration), initial=0.0)),
        float(np.max(np.abs(jerk), initial=0.0)),
    )


def _q_steps(qpos: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(qpos, dtype=np.float64)
    if len(q) < 2:
        return 0.0, 0.0, 0.0
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    accel = np.diff(q, n=2, axis=0)
    jerk = np.diff(q, n=3, axis=0)
    return (
        float(np.max(steps, initial=0.0)),
        float(np.max(np.linalg.norm(accel, axis=1), initial=0.0)),
        float(np.max(np.linalg.norm(jerk, axis=1), initial=0.0)),
    )


def _metrics_from_states(
    source: np.ndarray, robot: np.ndarray, base: np.ndarray, qpos: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    residual = np.asarray(robot, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    raw = np.linalg.norm(residual, axis=-1)
    scale = np.linalg.norm(source[:, 5] - source[:, 17], axis=-1)
    scale = np.maximum(scale, 1e-6)
    source_rel = source - source[:, :1]
    robot_rel = robot - robot[:, :1]
    morph = np.linalg.norm(robot_rel - source_rel, axis=-1) / scale[:, None]
    per_finger: list[dict[str, Any]] = []
    for finger, (start, end) in FINGERS.items():
        per_finger.append(
            {
                "finger": finger,
                "raw_rmse_mm": float(np.sqrt(np.mean(raw[:, start : end + 1] ** 2)) * 1000.0),
                "morphology_rmse_mm": float(
                    np.sqrt(np.mean(morph[:, start : end + 1] ** 2)) * 1000.0
                ),
                "fingertip_rmse_mm": float(np.sqrt(np.mean(raw[:, end] ** 2)) * 1000.0),
            }
        )
    q_step, q_accel, q_jerk = _q_steps(qpos)
    base_step, base_accel, base_jerk = _translation_steps(base)
    base_rot_step, base_rot_accel, base_rot_jerk = _rotation_steps(base)
    return (
        {
            "raw_keypoint_rmse_mm": float(np.sqrt(np.mean(raw * raw)) * 1000.0),
            "morphology_normalized_keypoint_rmse_mm": float(
                np.sqrt(np.mean(morph * morph)) * 1000.0
            ),
            "whole_hand_raw_rmse_mm": float(np.sqrt(np.mean(raw * raw)) * 1000.0),
            "whole_hand_morphology_rmse_mm": float(np.sqrt(np.mean(morph * morph)) * 1000.0),
            "q_step_max_rad": q_step,
            "q_acceleration_max": q_accel,
            "q_jerk_max": q_jerk,
            "base_translation_step_max_m": base_step,
            "base_translation_acceleration_max": base_accel,
            "base_translation_jerk_max": base_jerk,
            "base_rotation_step_max_rad": base_rot_step,
            "base_rotation_acceleration_max_rad": base_rot_accel,
            "base_rotation_jerk_max_rad": base_rot_jerk,
        },
        per_finger,
    )


def evaluate_profile(
    *,
    clip: ClipSpec,
    canonical_path: str | Path,
    source_path: str | Path,
    artifact_path: str | Path,
    profile_id: str,
    is_warm: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequence = load_hoi_sequence(canonical_path)
    hand = _hand(sequence, clip.hand)
    source = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    if is_warm:
        artifact: Any = load_warm_start(artifact_path)
        arrays = artifact.arrays
        robot = np.asarray(arrays["robot_keypoints_scene"], dtype=np.float64)
        base = np.asarray(arrays["base_pose_scene"], dtype=np.float64)
        qpos = np.asarray(arrays["qpos"], dtype=np.float64)
        status = np.asarray(
            arrays.get("solver_success", np.ones(len(qpos), dtype=bool)), dtype=bool
        )
        accepted = status
        solver_status = np.zeros(len(qpos), dtype=np.int64)
        e_im = np.full(len(qpos), np.nan)
        e_bone = np.asarray(arrays.get("ebone", np.full(len(qpos), np.nan)), dtype=np.float64)
        penetration = np.zeros(len(qpos))
        runtime = np.zeros(len(qpos))
    else:
        artifact = load_final_trajectory(artifact_path)
        arrays = artifact.arrays
        robot = np.asarray(arrays["robot_keypoints_scene"], dtype=np.float64)
        base = np.asarray(arrays["base_pose_scene"], dtype=np.float64)
        qpos = np.asarray(arrays["qpos"], dtype=np.float64)
        status = np.asarray(
            arrays.get("solver_success", np.zeros(len(qpos), dtype=bool)), dtype=bool
        )
        accepted = np.asarray(arrays.get("accepted", status), dtype=bool)
        solver_status = np.asarray(
            arrays.get("solver_status", np.full(len(qpos), -1)), dtype=np.int64
        )
        e_im = np.asarray(arrays.get("e_im", np.full(len(qpos), np.nan)), dtype=np.float64)
        e_bone = np.asarray(arrays.get("e_bone", np.full(len(qpos), np.nan)), dtype=np.float64)
        penetration = np.asarray(
            arrays.get("max_penetration", np.zeros(len(qpos))), dtype=np.float64
        )
        runtime = np.asarray(arrays.get("solve_time_s", np.zeros(len(qpos))), dtype=np.float64)
    state_metrics, per_finger = _metrics_from_states(source, robot, base, qpos)
    contact = _proxy_metrics(sequence, clip, artifact_path, source_path)
    five = contact["retention"]["5mm"]
    row = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "unit_id": clip.unit_id,
        "sequence": clip.sequence,
        "object": clip.object_name,
        "subject": clip.subject,
        "profile": profile_id,
        "paper_method": profile_id
        in {
            "paper_warm",
            "scipy_slsqp_active_set_contact_rich_v2",
            "scipy_slsqp_active_set_contact_rich_v3_fixed",
        },
        "paper_external_extension": profile_id.startswith(("contact", "morphology", "E2_", "E3_")),
        "frame_count": int(len(qpos)),
        "solver_status_mode": int(np.bincount(solver_status + 1).argmax() - 1)
        if len(solver_status)
        else None,
        "solver_success_frames": int(np.count_nonzero(status)),
        "strict_accepted_frames": int(np.count_nonzero(accepted)),
        "complete_60_frames": bool(len(qpos) == clip.length),
        "strict_accepted": bool(len(accepted) == clip.length and np.all(accepted)),
        "full_512_pass": bool(
            np.all(arrays.get("full_surface_hard_audit_pass", np.ones(len(qpos), dtype=bool)))
        ),
        "e_im_mean": float(np.nanmean(e_im)) if np.any(np.isfinite(e_im)) else None,
        "e_bone_mean": float(np.nanmean(e_bone)) if np.any(np.isfinite(e_bone)) else None,
        "e_morph_mean": _finite_mean(arrays.get("e_morph")),
        "weighted_e_morph_mean": _finite_mean(arrays.get("weighted_e_morph")),
        "e_contact_pos_mean": _finite_mean(arrays.get("e_contact_pos")),
        "weighted_e_contact_pos_mean": _finite_mean(arrays.get("weighted_e_contact_pos")),
        "e_contact_dir_mean": _finite_mean(arrays.get("e_contact_dir")),
        "weighted_e_contact_dir_mean": _finite_mean(arrays.get("weighted_e_contact_dir")),
        "penetration_max_mm": float(np.max(penetration, initial=0.0) * 1000.0),
        "penetration_frames_gt_2mm": int(np.count_nonzero(penetration > 0.002)),
        "runtime_total_s": float(np.sum(runtime)),
        "runtime_median_s": float(np.median(runtime)) if len(runtime) else 0.0,
        "runtime_p95_s": float(np.quantile(runtime, 0.95)) if len(runtime) else 0.0,
        **state_metrics,
        **{
            f"contact_{key}": value
            for key, value in contact.items()
            if key in {"precision_proxy", "alignment_proxy", "f1"}
        },
        "contact_precision_proxy": contact["contact_precision_proxy"],
        "contact_recall_proxy": contact["contact_recall_proxy"],
        "contact_alignment_proxy": contact["contact_alignment_proxy"],
        "contact_f1_5mm": five["f1"],
        "retention_precision_2mm": contact["retention"]["2mm"]["precision"],
        "retention_recall_2mm": contact["retention"]["2mm"]["recall"],
        "retention_f1_2mm": contact["retention"]["2mm"]["f1"],
        "retention_precision_3mm": contact["retention"]["3mm"]["precision"],
        "retention_recall_3mm": contact["retention"]["3mm"]["recall"],
        "retention_f1_3mm": contact["retention"]["3mm"]["f1"],
        "retention_precision_5mm": five["precision"],
        "retention_recall_5mm": five["recall"],
        "retention_f1_5mm": five["f1"],
        "retention_precision_8mm": contact["retention"]["8mm"]["precision"],
        "retention_recall_8mm": contact["retention"]["8mm"]["recall"],
        "retention_f1_8mm": contact["retention"]["8mm"]["f1"],
        "retention_precision_10mm": contact["retention"]["10mm"]["precision"],
        "retention_recall_10mm": contact["retention"]["10mm"]["recall"],
        "retention_f1_10mm": contact["retention"]["10mm"]["f1"],
        "per_finger_retention": contact["per_finger_retention"],
        "metric_semantics": "DATASET_PROXY",
        "artifact_path": str(Path(artifact_path).resolve()),
    }
    finger_rows: list[dict[str, Any]] = []
    for item in per_finger:
        finger = str(item["finger"])
        row_item = {"unit_id": clip.unit_id, "profile": profile_id, **item}
        row_item.update(
            {
                f"contact_{threshold}_{metric}": contact["per_finger_retention"][threshold][finger][
                    metric
                ]
                for threshold in ("2mm", "3mm", "5mm", "8mm", "10mm")
                for metric in ("precision", "recall", "f1")
            }
        )
        finger_rows.append(row_item)
    return row, finger_rows


def evaluate_all(
    clips: tuple[ClipSpec, ...],
    clip_records: dict[str, dict[str, Any]],
    output_root: str | Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
    for clip in clips:
        record = clip_records[clip.unit_id]
        for profile_id, path, is_warm in record["profiles"]:
            row, fingers = evaluate_profile(
                clip=clip,
                canonical_path=record["canonical"],
                source_path=record["source"],
                artifact_path=path,
                profile_id=profile_id,
                is_warm=is_warm,
            )
            rows.append(row)
            finger_rows.extend(fingers)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(rows, destination / "per_clip_metrics.json")
    write_json(finger_rows, destination / "per_finger_metrics.json")
    return {"rows": rows, "per_finger": finger_rows}


__all__ = ["evaluate_all", "evaluate_profile"]
