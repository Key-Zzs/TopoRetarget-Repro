#!/usr/bin/env python3
"""Materialize the offline Stage16 contact-timing, angular, PF, and DF audit."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation import object_metric_series  # noqa: E402
from toporetarget.rl.dynamic_physical_qualification import (  # noqa: E402
    PHASE_NAMES,
    DynamicTerminalGate,
    phase_labels_from_reference_index,
)
from toporetarget.rl.geometry_audit.raw_mocap_overlay import (  # noqa: E402
    resolve_raw_mocap_overlay,
)
from toporetarget.rl.stage16_pf_df import (  # noqa: E402
    AngularAuditContract,
    ContactTimingContract,
    DemonstrationFidelityContract,
    PhysicalFunctionalityContract,
    angular_episode_audit,
    angular_root_cause,
    contact_timing_metrics,
    distribution,
    evaluate_demonstration_fidelity,
    evaluate_physical_functionality,
    terminal_threshold_pass,
    timing_attribution,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_contact_timing_angular_twist_pf_df"
DYNAMIC_ROOT = (
    REPO_ROOT / ".local/reports/stage16_dynamic_physical_qualification_and_grasp_diagnostic"
)
STAGE_A_ROOT = REPO_ROOT / ".local/reports/stage16_raw_mocap_replay_overlay"
STRICT_V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
REFERENCE_CONTACT_ROOT = REPO_ROOT / ".local/reports/stage16d_contact_contract_v2_audit"
REFERENCE_KINEMATICS_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
WORLD_WRIST_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
FROZEN_GATES = STRICT_V4_ROOT / "frozen_evaluation_gates.json"
FORMAL_170650_ROOT = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
)
FORMAL_170650_CONFIG = (
    REPO_ROOT
    / ".local/reports/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/frozen_source_training_config.json"
)
SOURCE_170105_ROOT = (
    REPO_ROOT
    / ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4"
)
SOURCE_170105_REPORT = (
    REPO_ROOT
    / ".local/reports/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4"
)
SOURCE_170105_AUTHORITY = SOURCE_170105_REPORT / "sources/v4_hocap_170105.json"
SOURCE_170105_QUALIFICATION = SOURCE_170105_REPORT / "sweep/v4/hocap_170105/c4/qualification.json"
SOURCE_170105_EPISODES = SOURCE_170105_REPORT / "sweep/v4/hocap_170105/c4/per_episode.csv"
START_HEAD = "ac7ef78e9ea8c769f515a8b454d9584c40c7563a"
CLIPS = ("hocap_170105", "hocap_170650")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STAGE16_PF_DF_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"STAGE16_PF_DF_CSV_EMPTY:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"STAGE16_PF_DF_CSV_FIELD_DRIFT:{path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) in {"True", "False"}:
        return str(value) == "True"
    raise ValueError(f"STAGE16_PF_DF_BOOLEAN_INVALID:{value!r}")


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _trace_paths(clip: str) -> list[Path]:
    if clip == "hocap_170105":
        paths = [SOURCE_170105_ROOT / f"episode_{episode:02d}.npz" for episode in range(10)]
    elif clip == "hocap_170650":
        paths = [FORMAL_170650_ROOT / f"episode_{episode:03d}.npz" for episode in range(20)]
    else:
        raise ValueError(f"STAGE16_PF_DF_UNKNOWN_CLIP:{clip}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"STAGE16_PF_DF_TRACE_MISSING:{','.join(missing)}")
    return paths


def _dynamic_rows(clip: str) -> list[dict[str, str]]:
    name = "v4_170105_dynamic.csv" if clip == "hocap_170105" else "v4_170650_formal20_dynamic.csv"
    rows = _read_csv(DYNAMIC_ROOT / "stage_b" / name)
    if len(rows) != len(_trace_paths(clip)):
        raise ValueError(f"STAGE16_PF_DF_DYNAMIC_EPISODE_COUNT_DRIFT:{clip}")
    return rows


def _phase(trace: Mapping[str, np.ndarray]) -> np.ndarray:
    reference_index = np.asarray(trace["reference_index"], dtype=np.int64)
    labels = phase_labels_from_reference_index(reference_index)
    recorded = trace.get("recorded_phase")
    if recorded is not None:
        value = np.asarray(recorded)
        expected_codes = np.clip(
            (reference_index * len(PHASE_NAMES)) // len(reference_index),
            0,
            len(PHASE_NAMES) - 1,
        )
        matches = (
            np.array_equal(value.astype(np.int64), expected_codes)
            if np.issubdtype(value.dtype, np.number)
            else np.array_equal(value.astype("U24"), labels)
        )
        if not matches:
            raise ValueError("STAGE16_PF_DF_PHASE_AUTHORITY_DRIFT")
    return labels


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    aliases = {
        "object_reference": ("object_reference", "embedded_reference_object_pose"),
        "wrist_reference": ("wrist_reference", "embedded_reference_wrist_pose"),
    }
    required = (
        "object_pose",
        "object_twist",
        "object_twist_reference",
        "wrist_pose",
        "wrist_twist_world",
        "hand_object_pair_force_valid",
        "hand_object_pair_presence",
        "tip_pair_presence",
        "source_contact_mask",
        "table_object_contact",
        "action",
        "reference_index",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"STAGE16_PF_DF_TRACE_FIELDS_MISSING:{path}:{','.join(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
        for target, candidates in aliases.items():
            source = next((name for name in candidates if name in archive.files), None)
            if source is None:
                raise ValueError(f"STAGE16_PF_DF_REFERENCE_FIELD_MISSING:{path}:{target}")
            result[target] = np.asarray(archive[source])
        result["recorded_phase"] = (
            np.asarray(archive["phase"]) if "phase" in archive.files else None
        )
        result["metadata"] = (
            np.asarray(archive["metadata"]) if "metadata" in archive.files else None
        )
    frame_count = len(result["object_pose"])
    expected = {
        "object_pose": (frame_count, 7),
        "object_twist": (frame_count, 6),
        "object_twist_reference": (frame_count, 6),
        "wrist_pose": (frame_count, 7),
        "wrist_twist_world": (frame_count, 6),
        "object_reference": (frame_count, 7),
        "wrist_reference": (frame_count, 7),
        "hand_object_pair_force_valid": (frame_count,),
        "hand_object_pair_presence": (frame_count, 21),
        "tip_pair_presence": (frame_count, 5),
        "source_contact_mask": (frame_count, 5),
        "table_object_contact": (frame_count,),
        "action": (frame_count, 26),
        "reference_index": (frame_count,),
    }
    for name, shape in expected.items():
        if result[name].shape != shape:
            raise ValueError(f"STAGE16_PF_DF_TRACE_SHAPE_INVALID:{path}:{name}")
    if not np.array_equal(result["reference_index"], np.arange(frame_count, dtype=np.int64)):
        raise ValueError(f"STAGE16_PF_DF_REFERENCE_INDEX_DRIFT:{path}")
    result["phase"] = _phase(result)
    return result


def _raw_mask(clip: str) -> tuple[np.ndarray, Path]:
    path = STRICT_V4_ROOT / f"strict_source_contact_mask_{clip}.npz"
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["strict_source_contact_mask"], dtype=bool)
        names = tuple(str(value) for value in archive["finger_names"].tolist())
        control = np.asarray(archive["control_index"], dtype=np.int64)
    if names != ("thumb", "index", "middle", "ring", "pinky"):
        raise ValueError(f"STAGE16_PF_DF_RAW_FINGER_ORDER_DRIFT:{clip}")
    if mask.shape != (321, 5) or not np.array_equal(control, np.arange(321)):
        raise ValueError(f"STAGE16_PF_DF_RAW_TIME_AUTHORITY_DRIFT:{clip}")
    return mask, path


def _retarget_mask(clip: str) -> tuple[np.ndarray, np.ndarray, Path]:
    short = clip.removeprefix("hocap_")
    path = REFERENCE_CONTACT_ROOT / f"reference_contact_contract_v2_{short}.npz"
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["strong_contact_expected"], dtype=bool)
        distance = np.asarray(archive["reference_distance_m"], dtype=np.float64)
        names = tuple(str(value) for value in archive["finger_order"].tolist())
    if names != ("thumb", "index", "middle", "ring", "pinky") or mask.shape != (321, 5):
        raise ValueError(f"STAGE16_PF_DF_RETARGET_CONTACT_AUTHORITY_DRIFT:{clip}")
    if not np.array_equal(mask, distance <= ContactTimingContract().retarget_strong_distance_m):
        raise ValueError(f"STAGE16_PF_DF_RETARGET_DISTANCE_MASK_DRIFT:{clip}")
    return mask, distance, path


def _runtime_timestamps(clip: str) -> np.ndarray:
    path = REFERENCE_KINEMATICS_ROOT / f"{clip}.reference_kinematics_v2.npz"
    with np.load(path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    if (
        timestamps.shape != (321,)
        or not np.all(np.diff(timestamps) > 0.0)
        or not np.allclose(np.diff(timestamps), 0.05, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"STAGE16_PF_DF_RUNTIME_TIMESTAMP_AUTHORITY_DRIFT:{clip}")
    return timestamps


def _gate_by_clip() -> dict[str, DynamicTerminalGate]:
    document = _read_json(FROZEN_GATES)
    if document.get("status") != "STRICT_V4_EVALUATION_GATES_FROZEN":
        raise ValueError("STAGE16_PF_DF_FROZEN_GATE_STATUS_DRIFT")
    return {
        clip: DynamicTerminalGate.from_frozen_gate(document["task_gates"]["clips"][clip])
        for clip in CLIPS
    }


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"STAGE16_PF_DF_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _input_manifest() -> dict[str, object]:
    stage_a = _read_json(STAGE_A_ROOT / "final_summary.json")
    if stage_a.get("status") != "PASS":
        raise ValueError("STAGE16_PF_DF_STAGE_A_NOT_PASS")
    source_provenance = _read_json(STAGE_A_ROOT / "source_provenance.json")
    authority_105 = _read_json(SOURCE_170105_AUTHORITY)
    config_650 = _read_json(FORMAL_170650_CONFIG)
    episode_105 = _read_csv(SOURCE_170105_EPISODES)
    traces: list[dict[str, object]] = []
    for clip in CLIPS:
        for episode, path in enumerate(_trace_paths(clip)):
            if clip == "hocap_170105":
                seed = int(episode_105[episode]["seed"])
                actor_hash = authority_105["actor_hash"]
                checkpoint_hash = authority_105["checkpoint_sha256"]
                normalizer_hash = authority_105["normalizer_hash"]
                physics_contract = _artifact(SOURCE_170105_QUALIFICATION)
            else:
                trace = _load_trace(path)
                metadata_raw = trace["metadata"]
                if metadata_raw is None:
                    raise ValueError("STAGE16_PF_DF_170650_METADATA_MISSING")
                metadata = json.loads(str(np.asarray(metadata_raw).item()))
                seed = int(metadata["result"]["seed"])
                source = config_650["source_authority"]
                actor_hash = source["actor_hash"]
                checkpoint_hash = source["checkpoint"]["sha256"]
                normalizer_hash = source["normalizer_hash"]
                physics_contract = {
                    "path": metadata["evaluation"]["path"],
                    "sha256": metadata["evaluation"]["sha256"],
                    "support_contract_hash": metadata["support_contract_hash"],
                    "gravity_friction": metadata["gravity_friction"],
                }
            traces.append(
                {
                    "clip": clip,
                    "reward_mode": "strict_per_finger_v4",
                    "episode": episode,
                    "seed": seed,
                    "trace": _artifact(path),
                    "actor_hash": actor_hash,
                    "checkpoint_sha256": checkpoint_hash,
                    "normalizer_hash": normalizer_hash,
                    "physics_contract": physics_contract,
                }
            )
    return {
        "schema_version": "Stage16ContactTimingAngularPfDfInputManifestV1",
        "traces": traces,
        "raw_mocap_provenance": source_provenance,
        "reference_provenance": {
            clip: _artifact(REFERENCE_KINEMATICS_ROOT / f"{clip}.reference_kinematics_v2.npz")
            for clip in CLIPS
        },
        "raw_contact_authority": {
            clip: _artifact(STRICT_V4_ROOT / f"strict_source_contact_mask_{clip}.npz")
            for clip in CLIPS
        },
        "retarget_contact_authority": {
            clip: _artifact(
                REFERENCE_CONTACT_ROOT
                / f"reference_contact_contract_v2_{clip.removeprefix('hocap_')}.npz"
            )
            for clip in CLIPS
        },
        "phase_authority": "runtime phase labels exactly reconstructed from trace.reference_index",
        "stage_a_receipt": _artifact(STAGE_A_ROOT / "final_summary.json"),
        "dynamic_v1_receipt": _artifact(DYNAMIC_ROOT / "final_summary.json"),
        "frozen_v2_gate": _artifact(FROZEN_GATES),
    }


def _median_int(values: Iterable[object]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return None if not present else int(np.rint(np.median(present)))


def _range(values: Iterable[object]) -> str:
    present = [int(value) for value in values if value is not None]
    return "NOT_IDENTIFIABLE" if not present else f"{min(present)}..{max(present)}"


def _timing_summary(clip: str, episodes: list[Mapping[str, object]]) -> dict[str, object]:
    fields = (
        "raw_ready",
        "retarget_ready",
        "actual_ready",
        "raw_margin_frames",
        "retarget_margin_frames",
        "actual_margin_frames",
        "raw_to_retarget_delay_frames",
        "retarget_to_actual_delay_frames",
    )
    result: dict[str, object] = {"clip": clip, "episodes": len(episodes)}
    for name in fields:
        result[f"{name}_median"] = _median_int(row[name] for row in episodes)
        result[f"{name}_range"] = _range(row[name] for row in episodes)
    result["lift_onset"] = _median_int(row["lift_onset"] for row in episodes)
    result["lift_runtime_time_s"] = float(
        np.median([float(row["lift_runtime_time_s"]) for row in episodes])
    )
    result["lift_raw_time_s"] = float(
        np.median([float(row["lift_raw_time_s"]) for row in episodes])
    )
    result["lift_raw_frame_float"] = float(
        np.median([float(row["lift_raw_frame_float"]) for row in episodes])
    )
    result["prelift_ready_episodes"] = sum(
        bool(row["prelift_multifinger_grasp_ready"]) for row in episodes
    )
    result["named_source_contact_match_at_lift_episodes"] = sum(
        bool(row["named_source_contact_match_at_lift"]) for row in episodes
    )
    return result


def _timing_rows(
    *, clip: str, episodes: list[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episode_rows: list[dict[str, object]] = []
    finger_rows: list[dict[str, object]] = []
    for episode, value in enumerate(episodes):
        episode_rows.append(
            {
                "clip": clip,
                "episode": episode,
                "raw_ready": value["raw_ready"],
                "retarget_ready": value["retarget_ready"],
                "actual_ready": value["actual_ready"],
                "lift": value["lift_onset"],
                "raw_ready_runtime_time_s": value["raw_ready_runtime_time_s"],
                "raw_ready_source_time_s": value["raw_ready_source_time_s"],
                "raw_ready_source_frame_float": value["raw_ready_source_frame_float"],
                "retarget_ready_runtime_time_s": value["retarget_ready_runtime_time_s"],
                "actual_ready_runtime_time_s": value["actual_ready_runtime_time_s"],
                "lift_runtime_time_s": value["lift_runtime_time_s"],
                "lift_raw_time_s": value["lift_raw_time_s"],
                "lift_raw_frame_float": value["lift_raw_frame_float"],
                "raw_margin_frames": value["raw_margin_frames"],
                "retarget_margin_frames": value["retarget_margin_frames"],
                "actual_margin_frames": value["actual_margin_frames"],
                "raw_to_retarget_frames": value["raw_to_retarget_delay_frames"],
                "retarget_to_actual_frames": value["retarget_to_actual_delay_frames"],
                "prelift_grasp_ready": value["prelift_multifinger_grasp_ready"],
                "source_required_fingers_at_lift": ";".join(
                    value["source_required_fingers_at_lift"]
                ),
                "actual_persistent_fingers_at_lift": ";".join(
                    value["actual_persistent_fingers_at_lift"]
                ),
                "named_contact_match_at_lift": value["named_source_contact_match_at_lift"],
                "named_contact_recall_at_lift": value["named_source_contact_recall_at_lift"],
            }
        )
        for row in value["per_finger"]:
            finger_rows.append({"clip": clip, "episode": episode, **row})
    return episode_rows, finger_rows


def _aggregate_finger_rows(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for clip in CLIPS:
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            selected = [row for row in rows if row["clip"] == clip and row["finger"] == finger]
            result.append(
                {
                    "clip": clip,
                    "finger": finger,
                    "raw_onset": _median_int(row["raw_onset"] for row in selected),
                    "raw_onset_runtime_time_s": next(
                        (
                            row["raw_onset_runtime_time_s"]
                            for row in selected
                            if row["raw_onset_runtime_time_s"] is not None
                        ),
                        None,
                    ),
                    "raw_onset_source_time_s": next(
                        (
                            row["raw_onset_source_time_s"]
                            for row in selected
                            if row["raw_onset_source_time_s"] is not None
                        ),
                        None,
                    ),
                    "raw_onset_source_frame_float": next(
                        (
                            row["raw_onset_source_frame_float"]
                            for row in selected
                            if row["raw_onset_source_frame_float"] is not None
                        ),
                        None,
                    ),
                    "retarget_onset": _median_int(row["retarget_onset"] for row in selected),
                    "retarget_onset_runtime_time_s": next(
                        (
                            row["retarget_onset_runtime_time_s"]
                            for row in selected
                            if row["retarget_onset_runtime_time_s"] is not None
                        ),
                        None,
                    ),
                    "actual_first_median": _median_int(row["actual_first"] for row in selected),
                    "actual_first_range": _range(row["actual_first"] for row in selected),
                    "actual_first_runtime_time_s_median": (
                        None
                        if not any(
                            row["actual_first_runtime_time_s"] is not None for row in selected
                        )
                        else float(
                            np.median(
                                [
                                    float(row["actual_first_runtime_time_s"])
                                    for row in selected
                                    if row["actual_first_runtime_time_s"] is not None
                                ]
                            )
                        )
                    ),
                    "actual_persistent_median": _median_int(
                        row["actual_persistent"] for row in selected
                    ),
                    "actual_persistent_range": _range(row["actual_persistent"] for row in selected),
                    "lift": _median_int(row["lift_onset"] for row in selected),
                    "raw_to_retarget_frames": _median_int(
                        row["raw_to_retarget_frames"] for row in selected
                    ),
                    "retarget_to_actual_first_frames_median": _median_int(
                        row["retarget_to_actual_first_frames"] for row in selected
                    ),
                    "retarget_to_actual_persistent_frames_median": _median_int(
                        row["retarget_to_actual_persistent_frames"] for row in selected
                    ),
                    "raw_required_at_lift": all(
                        bool(row["raw_required_at_lift"]) for row in selected
                    ),
                    "actual_persistent_at_lift_episodes": sum(
                        bool(row["actual_persistent_at_lift"]) for row in selected
                    ),
                    "episodes": len(selected),
                }
            )
    return result


def _aggregate_angular(
    audits: list[Mapping[str, object]], traces: list[Mapping[str, np.ndarray]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    keys = (
        "measurement_consistency",
        "reference_estimator_consistency",
        "Delta_omega_trace",
        "Delta_omega_pose",
        "relative_angular_twist_trace",
        "relative_angular_twist_pose",
    )
    series_key = {
        "measurement_consistency": "trace_pose_mismatch",
        "reference_estimator_consistency": "reference_estimator_mismatch",
        "Delta_omega_trace": "Delta_omega_trace",
        "Delta_omega_pose": "Delta_omega_pose",
        "relative_angular_twist_trace": "relative_angular_twist_trace",
        "relative_angular_twist_pose": "relative_angular_twist_pose",
    }
    result: dict[str, object] = {
        key: distribution(
            np.concatenate(
                [
                    np.asarray(audit["series"][series_key[key]], dtype=np.float64)[1:]
                    for audit in audits
                ]
            )
        )
        for key in keys
    }
    result["exceedance"] = {
        "frame_fraction": float(
            np.concatenate(
                [
                    np.asarray(audit["series"]["exceedance_trace"], dtype=bool)[1:]
                    for audit in audits
                ]
            ).mean()
        ),
        "pose_frame_fraction": float(
            np.concatenate(
                [np.asarray(audit["series"]["exceedance_pose"], dtype=bool)[1:] for audit in audits]
            ).mean()
        ),
        "longest_consecutive_run_max": max(
            int(audit["exceedance"]["longest_consecutive_run"]) for audit in audits
        ),
        "longest_consecutive_run_median": float(
            np.median([int(audit["exceedance"]["longest_consecutive_run"]) for audit in audits])
        ),
        "number_of_segments_total": sum(
            int(audit["exceedance"]["number_of_segments"]) for audit in audits
        ),
        "transient_segment_count_total": sum(
            int(audit["exceedance"]["transient_segment_count"]) for audit in audits
        ),
        "persistent_segment_count_total": sum(
            int(audit["exceedance"]["persistent_segment_count"]) for audit in audits
        ),
    }
    phase_rows: list[dict[str, object]] = []
    for phase in ("APPROACH", "CONTACT", "GRASP", "LIFT", "LATE_MOTION"):
        trace_delta: list[np.ndarray] = []
        pose_delta: list[np.ndarray] = []
        mismatch: list[np.ndarray] = []
        relative: list[np.ndarray] = []
        for audit, trace in zip(audits, traces, strict=True):
            labels = np.asarray(trace["phase"]).astype("U24")
            mask = (
                np.isin(labels, ("MANIPULATION", "TERMINAL"))
                if phase == "LATE_MOTION"
                else labels == phase
            )
            trace_delta.append(np.asarray(audit["series"]["Delta_omega_trace"])[mask])
            pose_delta.append(np.asarray(audit["series"]["Delta_omega_pose"])[mask])
            mismatch.append(np.asarray(audit["series"]["trace_pose_mismatch"])[mask])
            relative.append(np.asarray(audit["series"]["relative_angular_twist_trace"])[mask])
        trace_values = np.concatenate(trace_delta)
        pose_values = np.concatenate(pose_delta)
        mismatch_values = np.concatenate(mismatch)
        relative_values = np.concatenate(relative)
        phase_rows.append(
            {
                "phase": phase,
                "frame_count": len(trace_values),
                "Delta_omega_trace_mean_radps": float(trace_values.mean()),
                "Delta_omega_trace_p95_radps": float(np.quantile(trace_values, 0.95)),
                "Delta_omega_trace_max_radps": float(trace_values.max()),
                "Delta_omega_pose_mean_radps": float(pose_values.mean()),
                "Delta_omega_pose_p95_radps": float(np.quantile(pose_values, 0.95)),
                "Delta_omega_pose_max_radps": float(pose_values.max()),
                "trace_pose_mismatch_mean_radps": float(mismatch_values.mean()),
                "trace_pose_mismatch_p95_radps": float(np.quantile(mismatch_values, 0.95)),
                "relative_angular_twist_mean_radps": float(relative_values.mean()),
                "relative_angular_twist_p95_radps": float(np.quantile(relative_values, 0.95)),
            }
        )
    result["worst_phase_by_trace_mean"] = max(
        phase_rows, key=lambda row: float(row["Delta_omega_trace_mean_radps"])
    )["phase"]
    result["terminal"] = {
        "trace_pass_episodes": sum(
            bool(audit["terminal"]["trace_pass_under_v1"]) for audit in audits
        ),
        "pose_pass_episodes": sum(
            bool(audit["terminal"]["pose_pass_under_v1"]) for audit in audits
        ),
        "trace_exceedance_count": sum(
            int(audit["terminal"]["trace_exceedance_count"]) for audit in audits
        ),
        "pose_exceedance_count": sum(
            int(audit["terminal"]["pose_exceedance_count"]) for audit in audits
        ),
        "endpoint_reference_estimator_mismatch_mean_radps": float(
            np.mean(
                [
                    float(audit["terminal"]["endpoint_reference_estimator_mismatch_radps"])
                    for audit in audits
                ]
            )
        ),
    }
    return result, phase_rows


def _episode_angular_row(episode: int, audit: Mapping[str, object]) -> dict[str, object]:
    return {
        "episode": episode,
        "trace_pose_mismatch_mean_radps": audit["measurement_consistency"]["mean"],
        "trace_pose_mismatch_median_radps": audit["measurement_consistency"]["median"],
        "trace_pose_mismatch_p95_radps": audit["measurement_consistency"]["p95"],
        "trace_pose_mismatch_max_radps": audit["measurement_consistency"]["max"],
        "reference_estimator_mismatch_mean_radps": audit["reference_estimator_consistency"]["mean"],
        "reference_estimator_mismatch_p95_radps": audit["reference_estimator_consistency"]["p95"],
        "Delta_omega_trace_mean_radps": audit["Delta_omega_trace"]["mean"],
        "Delta_omega_trace_median_radps": audit["Delta_omega_trace"]["median"],
        "Delta_omega_trace_p90_radps": audit["Delta_omega_trace"]["p90"],
        "Delta_omega_trace_p95_radps": audit["Delta_omega_trace"]["p95"],
        "Delta_omega_trace_p99_radps": audit["Delta_omega_trace"]["p99"],
        "Delta_omega_trace_max_radps": audit["Delta_omega_trace"]["max"],
        "Delta_omega_pose_mean_radps": audit["Delta_omega_pose"]["mean"],
        "Delta_omega_pose_p95_radps": audit["Delta_omega_pose"]["p95"],
        "exceedance_fraction": audit["exceedance"]["frame_fraction"],
        "longest_exceedance_run": audit["exceedance"]["longest_consecutive_run"],
        "exceedance_segments": audit["exceedance"]["number_of_segments"],
        "transient_segments": audit["exceedance"]["transient_segment_count"],
        "persistent_segments": audit["exceedance"]["persistent_segment_count"],
        "relative_angular_twist_mean_radps": audit["relative_angular_twist_trace"]["mean"],
        "relative_angular_twist_p95_radps": audit["relative_angular_twist_trace"]["p95"],
        "terminal_trace_pass_under_v1": audit["terminal"]["trace_pass_under_v1"],
        "terminal_pose_pass_under_v1": audit["terminal"]["pose_pass_under_v1"],
    }


def _write_angular_series(
    path: Path,
    *,
    episode: int,
    timestamps: np.ndarray,
    phase: np.ndarray,
    contact: np.ndarray,
    rotation_error_deg: np.ndarray,
    audit: Mapping[str, object],
) -> None:
    series = audit["series"]
    rows: list[dict[str, object]] = []
    for frame in range(len(timestamps)):
        rows.append(
            {
                "episode": episode,
                "frame": frame,
                "time_s": float(timestamps[frame]),
                "phase": str(phase[frame]),
                "hand_object_contact": bool(contact[frame]),
                "omega_ref_x_radps": float(series["omega_ref"][frame, 0]),
                "omega_ref_y_radps": float(series["omega_ref"][frame, 1]),
                "omega_ref_z_radps": float(series["omega_ref"][frame, 2]),
                "omega_ref_norm_radps": float(np.linalg.norm(series["omega_ref"][frame])),
                "omega_actual_trace_x_radps": float(series["omega_actual_trace"][frame, 0]),
                "omega_actual_trace_y_radps": float(series["omega_actual_trace"][frame, 1]),
                "omega_actual_trace_z_radps": float(series["omega_actual_trace"][frame, 2]),
                "omega_actual_trace_norm_radps": float(
                    np.linalg.norm(series["omega_actual_trace"][frame])
                ),
                "omega_actual_pose_x_radps": float(series["omega_actual_pose"][frame, 0]),
                "omega_actual_pose_y_radps": float(series["omega_actual_pose"][frame, 1]),
                "omega_actual_pose_z_radps": float(series["omega_actual_pose"][frame, 2]),
                "omega_actual_pose_norm_radps": float(
                    np.linalg.norm(series["omega_actual_pose"][frame])
                ),
                "Delta_omega_trace_radps": float(series["Delta_omega_trace"][frame]),
                "Delta_omega_pose_radps": float(series["Delta_omega_pose"][frame]),
                "trace_pose_mismatch_radps": float(series["trace_pose_mismatch"][frame]),
                "reference_estimator_mismatch_radps": float(
                    series["reference_estimator_mismatch"][frame]
                ),
                "relative_angular_twist_radps": float(
                    series["relative_angular_twist_trace"][frame]
                ),
                "rotation_pose_error_deg": float(rotation_error_deg[frame]),
                "incremental_rotation_angle_rad": float(
                    series["incremental_rotation_angle"][frame]
                ),
                "angular_acceleration_proxy_radps2": float(
                    series["angular_acceleration_proxy"][frame]
                ),
                "angular_limit_radps": float(series["angular_limit"][frame]),
                "threshold_exceeded": bool(series["exceedance_trace"][frame]),
            }
        )
    _write_csv(path, rows)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("STAGE16_PF_DF_MEAN_EMPTY")
    return float(array.mean())


def _pf_df_summary(clip: str, rows: list[Mapping[str, object]]) -> dict[str, object]:
    count = len(rows)

    def rate(name: str) -> dict[str, object]:
        passed = sum(bool(row[name]) for row in rows)
        return {"pass_count": passed, "total": count, "rate": passed / count}

    return {
        "clip": clip,
        "episodes": count,
        "PF": rate("pf"),
        "DF_pose": rate("df_pose"),
        "DF_linear_under_V1": rate("df_linear"),
        "DF_angular_trace_under_V1": rate("df_angular"),
        "DF_angular_pose_derived_under_V1": rate("df_angular_pose_derived"),
        "metrics": {
            name: _mean(float(row[name]) for row in rows)
            for name in (
                "E_r_mean_deg",
                "E_t_mean_cm",
                "E_j_mean_cm",
                "E_ft_mean_cm",
                "Delta_v_mean_mps",
                "Delta_v_p95_mps",
                "Delta_omega_trace_mean_radps",
                "Delta_omega_trace_p95_radps",
                "Delta_omega_pose_mean_radps",
                "Delta_omega_pose_p95_radps",
                "trace_pose_omega_mismatch_mean_radps",
                "lift_dz_m",
                "table_contact_before_lift_fraction",
                "table_contact_during_lift_fraction",
                "relative_angular_twist_lift_mean_radps",
            )
        },
        "PF_STATUS": (
            "PASS"
            if all(bool(row["pf"]) for row in rows)
            else "FAIL"
            if not any(bool(row["pf"]) for row in rows)
            else "PARTIAL"
        ),
        "DF_POSE_STATUS": (
            "PASS"
            if all(bool(row["df_pose"]) for row in rows)
            else "FAIL"
            if not any(bool(row["df_pose"]) for row in rows)
            else "PARTIAL"
        ),
        "DF_LINEAR_STATUS": (
            "PASS_UNDER_V1"
            if all(bool(row["df_linear"]) for row in rows)
            else "FAIL_UNDER_V1"
            if not any(bool(row["df_linear"]) for row in rows)
            else "PARTIAL_UNDER_V1"
        ),
        "DF_ANGULAR_STATUS": (
            "PASS_UNDER_V1"
            if all(bool(row["df_angular"]) for row in rows)
            else "FAIL_UNDER_V1"
            if not any(bool(row["df_angular"]) for row in rows)
            else "PARTIAL_UNDER_V1"
        ),
        "DF_OVERALL_PROFILE": "POSE_AND_LINEAR_AND_ANGULAR_REPORTED_SEPARATELY",
    }


def _metric_definitions() -> str:
    return """# Stage16 PF and DF Metric Definitions

## Physical Functionality

| PF Metric | Definition | Window | Hard gate? | Threshold authority |
| --- | --- | --- | --- | --- |
| prelift grasp readiness | at least two named PhysX fingertips persistent before/on LIFT | through LIFT onset | yes | existing 3-step persistence, >=2 fingers |
| persistent intended contact | source-required identity recall | semantic LIFT | report | frozen Strict V4 source mask |
| lift | actual object vertical displacement | terminal | yes | frozen >=5 cm |
| support transfer | table/object contact before and during LIFT | pre-LIFT and LIFT | report | no frozen fraction threshold |
| relative hand-object coupling | object minus wrist relative twist | LIFT | report | no frozen threshold |
| penetration | frozen absolute/inter-finger geometry | full | yes | Evaluation Suite V2 |
| causality | causal recorded PhysX execution | full | yes | Evaluation Suite V2/runtime receipts |
| action bounds | max absolute normalized action <=1 | full | yes | frozen action contract |
| hidden-control checks | no guidance/object write/wrist-root teleport | full | yes | frozen runtime provenance |

## Demonstration Fidelity

| DF Metric | Definition | Window | Status/gate | Threshold provenance |
| --- | --- | --- | --- | --- |
| Er | mean object rotation error | full | <30 deg | frozen V2 |
| Et | mean object translation error | full | <3 cm | frozen V2 |
| Ej | mean tracked joint error | full | <8 cm | frozen V2 |
| Eft | mean fingertip error | full | <6 cm | frozen V2 |
| Delta_v mean/p95 | norm(actual-reference) world linear velocity | full | report | Reference Kinematics V2 |
| terminal Delta_v | reference-relative linear error | last 20 | V1 inherited | legacy inherited, not newly validated |
| Delta_omega trace mean/p95 | PhysX trace minus reference world omega | full | report | Reference Kinematics V2 |
| Delta_omega pose mean/p95 | pose-derived actual minus reference world omega | full | report | same SO(3) estimator family |
| terminal Delta_omega | reference-relative angular error | last 20 | V1 inherited | legacy inherited, not newly validated |
| trace-vs-pose mismatch | trace omega minus pose-derived omega | full | attribution only | no new fidelity gate |

PF answers whether the robot physically performs the interaction. DF answers whether the resulting motion remains faithful to the demonstrated/reference motion. No unvalidated DF overall boolean is defined.
"""


def _replay_commands(selected: Mapping[str, int]) -> str:
    base = (
        "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python "
        "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop"
    )
    trace_650 = ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
    trace_105 = ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4"
    lines = [
        "# Visualization Commands",
        "",
        "All commands display recorded PhysX actual plus raw HOCap MANO/object with the retarget reference hidden. `M` toggles raw MOCAP and `R` toggles the retarget layer.",
        "",
        "## V4/170650",
        "",
        "```bash",
        f"{base} --trace {trace_650}/episode_000.npz --object hocap_170650 --no-reference-ghost",
        f"{base} --start-frame 92 --end-frame 230 --trace {trace_650}/episode_000.npz --object hocap_170650 --no-reference-ghost",
        f"{base} --start-frame 92 --end-frame 230 --trace {trace_650}/episode_000.npz --object hocap_170650 --no-reference-ghost --mocap-object-low-poly",
    ]
    for label in ("lowest", "median", "highest"):
        episode = int(selected[label])
        lines.append(
            f"{base} --trace {trace_650}/episode_{episode:03d}.npz --object hocap_170650 --no-reference-ghost"
        )
    lines.extend(
        [
            "```",
            "",
            "## V4/170105",
            "",
            "```bash",
            f"{base} --trace {trace_105}/episode_00.npz --object hocap_170105 --no-reference-ghost",
            f"{base} --start-frame 92 --end-frame 230 --trace {trace_105}/episode_00.npz --object hocap_170105 --no-reference-ghost",
            f"{base} --start-frame 92 --end-frame 230 --trace {trace_105}/episode_00.npz --object hocap_170105 --no-reference-ghost --mocap-object-low-poly",
            "```",
            "",
            "Authoritative markers are reported in `contact_timing/aggregate.csv`; the shared phase window is CONTACT frame 92 through the end of LIFT at frame 229 (`[92,230)`).",
        ]
    )
    return "\n".join(lines) + "\n"


def _final_markdown(
    *,
    timing: Mapping[str, Mapping[str, object]],
    attribution: Mapping[str, object],
    angular: Mapping[str, object],
    angular_decision: Mapping[str, object],
    phase_rows: list[Mapping[str, object]],
    finger_rows: list[Mapping[str, object]],
    pf_df: Mapping[str, Mapping[str, object]],
) -> str:
    a105 = timing["hocap_170105"]
    a650 = timing["hocap_170650"]
    s105 = pf_df["hocap_170105"]
    s650 = pf_df["hocap_170650"]

    def rate(summary: Mapping[str, object], name: str) -> str:
        value = summary[name]
        return f"{value['pass_count']}/{value['total']}"

    contact_rows = "\n".join(
        f"| {clip} | {row['raw_ready_median']} | {row['retarget_ready_median']} | {row['actual_ready_median']} ({row['actual_ready_range']}) | {row['lift_onset']} | {row['raw_margin_frames_median']} / {float(row['raw_margin_frames_median']) * 0.05:.2f} s | {row['retarget_margin_frames_median']} / {float(row['retarget_margin_frames_median']) * 0.05:.2f} s | {row['actual_margin_frames_median']} / {float(row['actual_margin_frames_median']) * 0.05:.2f} s | {row['raw_to_retarget_delay_frames_median']} | {row['retarget_to_actual_delay_frames_median']} ({row['retarget_to_actual_delay_frames_range']}) |"
        for clip, row in (("170105", a105), ("170650", a650))
    )
    fingers = "\n".join(
        f"| {str(row['clip']).removeprefix('hocap_')} | {row['finger']} | {row['raw_onset']} | {row['retarget_onset']} | {row['actual_first_median']} ({row['actual_first_range']}) | {row['actual_persistent_median']} ({row['actual_persistent_range']}) | {row['lift']} | {row['raw_to_retarget_frames']} | {row['retarget_to_actual_persistent_frames_median']} |"
        for row in finger_rows
    )
    phases = "\n".join(
        f"| {row['phase']} | {float(row['Delta_omega_trace_mean_radps']):.6f} | {float(row['Delta_omega_pose_mean_radps']):.6f} | {float(row['Delta_omega_trace_p95_radps']):.6f} | {float(row['relative_angular_twist_mean_radps']):.6f} |"
        for row in phase_rows
    )
    return f"""# Stage16 Contact Timing + Angular Twist + PF/DF Handoff

## Git and authority

`BRANCH=feature/ppo-physical`, `START_HEAD={START_HEAD}`. The exact 10 frozen V4/170105 C4 Eval10 traces and 20 frozen V4/170650 C4 Formal20 traces were analyzed offline. Raw contact authority is the immutable Strict V4 HOCap source mask; retarget authority is ReferenceContactContractV2 strong geometric contact; phase/LIFT comes from `reference_index`; angular pose differentiation is the Reference Kinematics V2 SO(3)-log world estimator.

## Contact timing layers

Positive margin means ready before LIFT; negative means after LIFT. Frames are 0.05 s.

LIFT is runtime frame 184 / 9.20 s and raw-source time 1.15 s / source frame 34.5 for both aligned clips.

| Clip | Raw ready | Retarget ready | Actual ready median (range) | LIFT | Raw margin frame/s | Retarget margin frame/s | Actual margin frame/s | Raw→Retarget | Retarget→Actual median (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{contact_rows}

`CONTACT_TIMING_LAYER_ROOT_CAUSE={attribution["CONTACT_TIMING_LAYER_ROOT_CAUSE"]}`

`CONFIDENCE={attribution["CONFIDENCE"]}`

### Per-finger timing

| Clip | Finger | Raw onset | Retarget onset | Actual first median (range) | Actual persistent median (range) | LIFT | Raw→Retarget | Retarget→Actual persistent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{fingers}

## Angular-twist audit — V4/170650 Formal20

`DOES_TRACE_OMEGA_MATCH_POSE_DERIVED_OMEGA={angular_decision["DOES_TRACE_OMEGA_MATCH_POSE_DERIVED_OMEGA"]}`

`IS_LARGE_DELTA_OMEGA={angular_decision["IS_LARGE_DELTA_OMEGA"]}`

`ANGULAR_TWIST_ROOT_CAUSE={angular_decision["ANGULAR_TWIST_ROOT_CAUSE"]}`

`WORST_ANGULAR_ERROR_PHASE={angular["worst_phase_by_trace_mean"]}`

| Metric | 170650 aggregate |
| --- | ---: |
| trace-vs-pose omega mismatch mean | {float(angular["measurement_consistency"]["mean"]):.6f} rad/s |
| trace-vs-pose mismatch p95 | {float(angular["measurement_consistency"]["p95"]):.6f} rad/s |
| trace-vs-pose mismatch max | {float(angular["measurement_consistency"]["max"]):.6f} rad/s |
| Δω trace mean | {float(angular["Delta_omega_trace"]["mean"]):.6f} rad/s |
| Δω pose mean | {float(angular["Delta_omega_pose"]["mean"]):.6f} rad/s |
| exceedance fraction | {float(angular["exceedance"]["frame_fraction"]):.6f} |
| longest exceedance run | {angular["exceedance"]["longest_consecutive_run_max"]} frames |
| spike/segment count | {angular["exceedance"]["number_of_segments_total"]} |
| relative angular twist | {float(angular["relative_angular_twist_trace"]["mean"]):.6f} rad/s |

### Phase-wise angular fidelity

| Phase | Δω trace mean | Δω pose mean | trace p95 | Relative angular twist |
| --- | ---: | ---: | ---: | ---: |
{phases}

The trace-versus-pose mismatch is evaluated before any wobble interpretation. Relative object/wrist rotation remains a proxy and is not promoted to friction, slip, or grasp-wrench evidence.

## PF and DF

PF is physical task completion and never consumes `Er/Et/Ej/Eft/Delta_v/Delta_omega`. DF is demonstration/reference fidelity and does not become true merely because a grasp succeeds. The V1 linear/angular statuses retain inherited thresholds whose scientific calibration is unresolved.

| Metric | V4/170105 | V4/170650 | Interpretation |
| --- | ---: | ---: | --- |
| PF | {rate(s105, "PF")} | {rate(s650, "PF")} | causal grasp-before-LIFT plus >=5 cm lift and safety |
| DF_pose | {rate(s105, "DF_pose")} | {rate(s650, "DF_pose")} | frozen V2 pose thresholds |
| DF_linear under V1 | {rate(s105, "DF_linear_under_V1")} | {rate(s650, "DF_linear_under_V1")} | inherited reference-relative terminal threshold |
| DF_angular trace under V1 | {rate(s105, "DF_angular_trace_under_V1")} | {rate(s650, "DF_angular_trace_under_V1")} | inherited reference-relative terminal threshold |
| DF_angular pose-derived under V1 | {rate(s105, "DF_angular_pose_derived_under_V1")} | {rate(s650, "DF_angular_pose_derived_under_V1")} | same estimator family, diagnostic comparison |
| Legacy SRqualified | 0/10 | 2/20 | immutable |
| SR_dynamic V1 | 0/10 | 2/20 | immutable |

V4/170650: `PHYSICAL_FUNCTIONALITY={s650["PF_STATUS"]}`, `POSE_FIDELITY={s650["DF_POSE_STATUS"]}`, `LINEAR_DYNAMIC_FIDELITY={s650["DF_LINEAR_STATUS"]}`, `ANGULAR_DYNAMIC_FIDELITY={s650["DF_ANGULAR_STATUS"]}`.

V4/170105: `PHYSICAL_FUNCTIONALITY={s105["PF_STATUS"]}`, `POSE_FIDELITY={s105["DF_POSE_STATUS"]}`, `LINEAR_DYNAMIC_FIDELITY={s105["DF_LINEAR_STATUS"]}`, `ANGULAR_DYNAMIC_FIDELITY={s105["DF_ANGULAR_STATUS"]}`.

`PF_DF_SEPARATION_CHANGES_INTERPRETATION=YES`: a physically completed lift can coexist with degraded angular demonstration fidelity, while close pose tracking cannot substitute for missing physical lift.

## Next actions

`NEXT_170105={_next_contact(str(attribution["CONTACT_TIMING_LAYER_ROOT_CAUSE"]))}`

`NEXT_170650={_next_angular(str(angular_decision["ANGULAR_TWIST_ROOT_CAUSE"]))}`

## Replay and manual acceptance

See `replay/visualization_commands.md`. For 170105 inspect when raw MANO, retarget geometry, and actual named contact become grasp-ready relative to LIFT. For 170650 compare the lowest/median/highest Δω episodes and distinguish visible object-in-hand rotation from common hand-object motion or short estimator spikes.

## Safety flags

`BRANCH=feature/ppo-physical`, `NEW_BRANCH_CREATED=NO`, `NEW_WORKTREE_CREATED=NO`, `GUIDANCE_WORKTREE_MODIFIED=NO`, `PPO_TRAINING_RUN=NO`, `PPO_OPTIMIZER_STEP=0`, `REWARD_CHANGED=NO`, `FRICTION_CHANGED=NO`, `MASS_CHANGED=NO`, `REFERENCE_CHANGED=NO`, `RETIMING_CHANGED=NO`, `SR_HOLD_IMPLEMENTED=NO`, `ENGINEERED_TERMINAL_HOLD_ADDED=NO`, `LEGACY_SRPHYSICS_MODIFIED=NO`, `SR_DYNAMIC_V1_MODIFIED=NO`, `ANGULAR_THRESHOLD_TUNED=NO`, `CONTROLLER_CHANGED=NO`, `ACTION_CHANGED=NO`, `GUIDANCE_ADDED=NO`, `OBJECT_STATE_WRITE_ADDED=NO`, `WRIST_ROOT_WRITE_ADDED=NO`, `RAW_MOCAP_REPLAY_REGRESSED=NO`, `HISTORICAL_ARTIFACTS_MODIFIED=NO`, `PUSHED=NO`, `PR_CREATED=NO`, `.local_TRACKED=NO`.
"""


def _next_contact(root: str) -> str:
    return {
        "RAW_TO_RETARGET_TIMING_LOSS_PRIMARY": "NEXT_CONTACT_AWARE_GEOMETRIC_RETARGET_REFINEMENT",
        "RETARGET_TO_PHYSICS_CONTACT_ACQUISITION_LAG_PRIMARY": "NEXT_OBJECT_AGNOSTIC_CONTACT_TIMING_PHYSICAL_REFINEMENT",
        "MULTI_STAGE_TIMING_LOSS_PRIMARY": "NEXT_JOINT_RETARGET_AND_PHYSICAL_CONTACT_TIMING_REFINEMENT",
    }.get(root, "NEXT_CONTACT_TIMING_AUTHORITY_REVIEW")


def _next_angular(root: str) -> str:
    if root == "ANGULAR_VELOCITY_MEASUREMENT_SEMANTICS_MISMATCH_PRIMARY":
        return "NEXT_ALIGN_ACTUAL_AND_REFERENCE_TWIST_MEASUREMENT_SEMANTICS"
    if root == "TRANSIENT_ANGULAR_SPIKES_PRIMARY":
        return "NEXT_REVIEW_DYNAMIC_FIDELITY_AGGREGATION"
    if root in {
        "PERSISTENT_ROTATIONAL_WOBBLE_PRIMARY",
        "HAND_OBJECT_RELATIVE_ROTATION_PRIMARY",
    }:
        return "NEXT_OBJECT_AGNOSTIC_ROTATIONAL_GRASP_STABILITY_DIAGNOSTIC"
    if root == "REFERENCE_ANGULAR_ESTIMATION_ARTIFACT_PRIMARY":
        return "NEXT_ALIGN_ACTUAL_AND_REFERENCE_TWIST_MEASUREMENT_SEMANTICS"
    return "NEXT_ANGULAR_TWIST_FACTOR_ISOLATION"


def _refresh(root: Path, args: argparse.Namespace) -> int:
    if not root.is_dir():
        raise FileNotFoundError(f"STAGE16_PF_DF_REPORT_MISSING:{root}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    commits = subprocess.run(
        ["git", "log", "--format=%H %s", f"{START_HEAD}..{head}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    _write_json(
        root / "git_commits.json",
        {
            "branch": "feature/ppo-physical",
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": head,
            "commits": commits,
        },
    )
    if args.validation_status:
        _write_json(
            root / "tests.json",
            {
                "status": args.validation_status,
                "legacy_evaluation_suite_v2_receipts_unchanged": args.legacy_parity,
                "stage16_dynamic_v1_receipts_unchanged": args.legacy_parity,
                "raw_mocap_replay_regression": args.replay_regression,
            },
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--validation-status", default="")
    parser.add_argument("--legacy-parity", default="NOT_RUN")
    parser.add_argument("--replay-regression", default="NOT_RUN")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.report_root.resolve()
    if args.refresh:
        return _refresh(root, args)
    if root.exists():
        raise FileExistsError(f"STAGE16_PF_DF_OUTPUT_ALREADY_EXISTS:{root}")
    root.mkdir(parents=True)
    gates = _gate_by_clip()
    contact_contract = ContactTimingContract()
    angular_contract = AngularAuditContract()
    pf_contract = PhysicalFunctionalityContract()
    df_contract = DemonstrationFidelityContract()
    _write_json(root / "contracts/contact_timing_contract.json", contact_contract.as_dict())
    _write_json(
        root / "contracts/angular_audit_contract.json",
        {
            **angular_contract.as_dict(),
            "legacy_angular_gate_by_clip": {clip: gate.as_dict() for clip, gate in gates.items()},
        },
    )
    _write_json(root / "contracts/pf_contract.json", pf_contract.as_dict())
    _write_json(
        root / "contracts/df_contract.json",
        {
            **df_contract.as_dict(),
            "legacy_terminal_gate_by_clip": {clip: gate.as_dict() for clip, gate in gates.items()},
        },
    )
    manifest = _input_manifest()
    _write_json(root / "input_manifest.json", manifest)

    overlays: dict[str, Any] = {}
    runtime_times_by_clip = {clip: _runtime_timestamps(clip) for clip in CLIPS}
    traces_by_clip: dict[str, list[dict[str, np.ndarray]]] = {clip: [] for clip in CLIPS}
    timing_by_clip: dict[str, list[dict[str, object]]] = {clip: [] for clip in CLIPS}
    all_finger_rows: list[dict[str, object]] = []
    for clip in CLIPS:
        paths = _trace_paths(clip)
        overlay = resolve_raw_mocap_overlay(
            trace_path=paths[0],
            frame_count=321,
            clip=clip,
            reference_path=WORLD_WRIST_ROOT / f"{clip}.world_wrist.stage16.npz",
        )
        if (
            overlay.coordinate_alignment["status"] != "PASS"
            or overlay.time_alignment["status"] != "PASS"
        ):
            raise ValueError(f"STAGE16_PF_DF_RAW_ALIGNMENT_NOT_PASS:{clip}")
        overlays[clip] = overlay
        raw, _ = _raw_mask(clip)
        retarget, _, _ = _retarget_mask(clip)
        for episode, path in enumerate(paths):
            trace = _load_trace(path)
            if not np.array_equal(trace["source_contact_mask"], raw):
                raise ValueError(f"STAGE16_PF_DF_TRACE_RAW_MASK_DRIFT:{clip}:{episode}")
            lift = np.flatnonzero(np.asarray(trace["phase"]) == "LIFT")
            if not len(lift):
                raise ValueError(f"STAGE16_PF_DF_LIFT_PHASE_MISSING:{clip}:{episode}")
            timing = contact_timing_metrics(
                raw_contact=raw,
                retarget_contact=retarget,
                actual_contact=np.asarray(trace["tip_pair_presence"], dtype=bool),
                actual_valid=np.asarray(trace["hand_object_pair_force_valid"], dtype=bool),
                lift_onset=int(lift[0]),
                timestamps_s=runtime_times_by_clip[clip],
                raw_timestamps_s=np.asarray(overlay.runtime_timestamps_s, dtype=np.float64),
                raw_frame_float=np.asarray(overlay.raw_frame_float, dtype=np.float64),
                contract=contact_contract,
            )
            timing_by_clip[clip].append(timing)
            traces_by_clip[clip].append(trace)
        episode_rows, finger_rows = _timing_rows(clip=clip, episodes=timing_by_clip[clip])
        _write_csv(
            root / f"contact_timing/v4_{clip.removeprefix('hocap_')}/episode_timing.csv",
            episode_rows,
        )
        all_finger_rows.extend(finger_rows)
    timing_summary = {clip: _timing_summary(clip, timing_by_clip[clip]) for clip in CLIPS}
    for clip in CLIPS:
        _write_json(
            root / f"contact_timing/v4_{clip.removeprefix('hocap_')}/summary.json",
            timing_summary[clip],
        )
    timing_aggregate_rows = [
        {
            "clip": clip,
            "raw_ready_median": summary["raw_ready_median"],
            "retarget_ready_median": summary["retarget_ready_median"],
            "actual_ready_median": summary["actual_ready_median"],
            "actual_ready_range": summary["actual_ready_range"],
            "lift": summary["lift_onset"],
            "lift_runtime_time_s": summary["lift_runtime_time_s"],
            "lift_raw_time_s": summary["lift_raw_time_s"],
            "lift_raw_frame_float": summary["lift_raw_frame_float"],
            "raw_margin_frames_median": summary["raw_margin_frames_median"],
            "retarget_margin_frames_median": summary["retarget_margin_frames_median"],
            "actual_margin_frames_median": summary["actual_margin_frames_median"],
            "raw_to_retarget_delay_frames_median": summary["raw_to_retarget_delay_frames_median"],
            "retarget_to_actual_delay_frames_median": summary[
                "retarget_to_actual_delay_frames_median"
            ],
            "prelift_ready_episodes": summary["prelift_ready_episodes"],
            "episodes": summary["episodes"],
        }
        for clip, summary in timing_summary.items()
    ]
    _write_csv(root / "contact_timing/aggregate.csv", timing_aggregate_rows)
    aggregate_fingers = _aggregate_finger_rows(all_finger_rows)
    _write_csv(root / "contact_timing/per_finger.csv", aggregate_fingers)
    attribution = timing_attribution(
        timing_summary["hocap_170105"], timing_summary["hocap_170650"], contract=contact_contract
    )
    _write_json(root / "contact_timing/attribution.json", attribution)

    audits_by_clip: dict[str, list[dict[str, object]]] = {clip: [] for clip in CLIPS}
    angular_rows_650: list[dict[str, object]] = []
    for clip in CLIPS:
        overlay = overlays[clip]
        gate = gates[clip]
        for episode, trace in enumerate(traces_by_clip[clip]):
            valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
            contact = (
                np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1) & valid
            )
            audit = angular_episode_audit(
                actual_object_pose_wxyz=np.asarray(trace["object_pose"]),
                actual_object_twist_world=np.asarray(trace["object_twist"]),
                reference_object_pose_wxyz=np.asarray(trace["object_reference"]),
                reference_object_twist_world=np.asarray(trace["object_twist_reference"]),
                wrist_pose_wxyz=np.asarray(trace["wrist_pose"]),
                wrist_twist_world=np.asarray(trace["wrist_twist_world"]),
                timestamps_s=runtime_times_by_clip[clip],
                phase=np.asarray(trace["phase"]),
                hand_object_contact=contact,
                valid=valid,
                contact_angular_limit_radps=gate.terminal_angular_speed_radps,
                free_angular_limit_radps=gate.terminal_free_object_angular_speed_radps,
                terminal_window_control_steps=gate.terminal_window_control_steps,
                contract=angular_contract,
            )
            audits_by_clip[clip].append(audit)
            if clip == "hocap_170650":
                angular_rows_650.append(_episode_angular_row(episode, audit))
                rotation = object_metric_series(
                    np.asarray(trace["object_pose"], dtype=np.float64),
                    np.asarray(trace["object_reference"], dtype=np.float64),
                )["e_r_deg"]
                _write_angular_series(
                    root / f"angular_twist/episodes/angular_twist_episode_{episode:03d}.csv",
                    episode=episode,
                    timestamps=runtime_times_by_clip[clip],
                    phase=np.asarray(trace["phase"]),
                    contact=contact,
                    rotation_error_deg=np.asarray(rotation, dtype=np.float64),
                    audit=audit,
                )
    angular_aggregate, phase_rows = _aggregate_angular(
        audits_by_clip["hocap_170650"], traces_by_clip["hocap_170650"]
    )
    aggregate_row = {
        "episode": "AGGREGATE",
        "trace_pose_mismatch_mean_radps": angular_aggregate["measurement_consistency"]["mean"],
        "trace_pose_mismatch_median_radps": angular_aggregate["measurement_consistency"]["median"],
        "trace_pose_mismatch_p95_radps": angular_aggregate["measurement_consistency"]["p95"],
        "trace_pose_mismatch_max_radps": angular_aggregate["measurement_consistency"]["max"],
        "reference_estimator_mismatch_mean_radps": angular_aggregate[
            "reference_estimator_consistency"
        ]["mean"],
        "reference_estimator_mismatch_p95_radps": angular_aggregate[
            "reference_estimator_consistency"
        ]["p95"],
        "Delta_omega_trace_mean_radps": angular_aggregate["Delta_omega_trace"]["mean"],
        "Delta_omega_trace_median_radps": angular_aggregate["Delta_omega_trace"]["median"],
        "Delta_omega_trace_p90_radps": angular_aggregate["Delta_omega_trace"]["p90"],
        "Delta_omega_trace_p95_radps": angular_aggregate["Delta_omega_trace"]["p95"],
        "Delta_omega_trace_p99_radps": angular_aggregate["Delta_omega_trace"]["p99"],
        "Delta_omega_trace_max_radps": angular_aggregate["Delta_omega_trace"]["max"],
        "Delta_omega_pose_mean_radps": angular_aggregate["Delta_omega_pose"]["mean"],
        "Delta_omega_pose_p95_radps": angular_aggregate["Delta_omega_pose"]["p95"],
        "exceedance_fraction": angular_aggregate["exceedance"]["frame_fraction"],
        "longest_exceedance_run": angular_aggregate["exceedance"]["longest_consecutive_run_max"],
        "exceedance_segments": angular_aggregate["exceedance"]["number_of_segments_total"],
        "transient_segments": angular_aggregate["exceedance"]["transient_segment_count_total"],
        "persistent_segments": angular_aggregate["exceedance"]["persistent_segment_count_total"],
        "relative_angular_twist_mean_radps": angular_aggregate["relative_angular_twist_trace"][
            "mean"
        ],
        "relative_angular_twist_p95_radps": angular_aggregate["relative_angular_twist_trace"][
            "p95"
        ],
        "terminal_trace_pass_under_v1": angular_aggregate["terminal"]["trace_pass_episodes"],
        "terminal_pose_pass_under_v1": angular_aggregate["terminal"]["pose_pass_episodes"],
    }
    _write_csv(root / "angular_twist/aggregate.csv", [*angular_rows_650, aggregate_row])
    _write_csv(root / "angular_twist/phase_summary.csv", phase_rows)
    _write_csv(
        root / "angular_twist/estimator_consistency.csv",
        [
            {
                "episode": row["episode"],
                "mean_radps": row["trace_pose_mismatch_mean_radps"],
                "median_radps": row["trace_pose_mismatch_median_radps"],
                "p95_radps": row["trace_pose_mismatch_p95_radps"],
                "max_radps": row["trace_pose_mismatch_max_radps"],
            }
            for row in [*angular_rows_650, aggregate_row]
        ],
    )
    angular_decision = angular_root_cause(angular_aggregate, contract=angular_contract)
    angular_decision["worst_phase"] = angular_aggregate["worst_phase_by_trace_mean"]
    angular_decision["reference_terminal_sanity"] = angular_aggregate["terminal"]
    _write_json(root / "angular_twist/root_cause.json", angular_decision)

    pf_df_rows: dict[str, list[dict[str, object]]] = {clip: [] for clip in CLIPS}
    for clip in CLIPS:
        gate = gates[clip]
        for episode, (trace, timing, dynamic, angular) in enumerate(
            zip(
                traces_by_clip[clip],
                timing_by_clip[clip],
                _dynamic_rows(clip),
                audits_by_clip[clip],
                strict=True,
            )
        ):
            valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
            contact = (
                np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1) & valid
            )
            delta_v = np.linalg.norm(
                np.asarray(trace["object_twist"], dtype=np.float64)[:, :3]
                - np.asarray(trace["object_twist_reference"], dtype=np.float64)[:, :3],
                axis=-1,
            )
            linear_pass = terminal_threshold_pass(
                delta_v,
                contact=contact,
                valid=valid,
                contact_limit=gate.terminal_linear_speed_mps,
                free_limit=gate.terminal_free_object_linear_speed_mps,
                terminal_steps=gate.terminal_window_control_steps,
            )
            pf = evaluate_physical_functionality(
                causal_execution=_bool(dynamic["causal_execution_pass"]),
                geometry_safe=_bool(dynamic["absolute_geometry_pass"]),
                action_bounds_safe=_bool(dynamic["action_bounds_pass"]),
                prelift_multifinger_grasp_ready=bool(timing["prelift_multifinger_grasp_ready"]),
                lift_dz_m=float(dynamic["lift_dz_m"]),
                no_hidden_control=_bool(dynamic["causal_execution_pass"]),
                contract=pf_contract,
            )
            df = evaluate_demonstration_fidelity(
                e_r_mean_deg=float(dynamic["E_r_mean_deg"]),
                e_t_mean_cm=float(dynamic["E_t_mean_cm"]),
                e_j_mean_cm=float(dynamic["E_j_mean_cm"]),
                e_ft_mean_cm=float(dynamic["E_ft_mean_cm"]),
                linear_pass_under_v1=linear_pass,
                angular_trace_pass_under_v1=bool(angular["terminal"]["trace_pass_under_v1"]),
                angular_pose_pass_under_v1=bool(angular["terminal"]["pose_pass_under_v1"]),
                contract=df_contract,
            )
            phase = np.asarray(trace["phase"]).astype("U24")
            lift_mask = (phase == "LIFT") & valid
            prelift_mask = (np.arange(len(phase)) < int(timing["lift_onset"])) & valid
            table = np.asarray(trace["table_object_contact"], dtype=bool)
            rel_angular = np.asarray(
                angular["series"]["relative_angular_twist_trace"], dtype=np.float64
            )
            row = {
                "clip": clip,
                "episode": episode,
                "pf": pf["pf"],
                "pf_failure_reasons": ";".join(pf["pf_failure_reasons"]),
                "df_pose": df["df_pose"],
                "df_linear": df["df_linear"],
                "df_angular": df["df_angular"],
                "df_angular_pose_derived": df["df_angular_pose_derived"],
                "E_r_mean_deg": float(dynamic["E_r_mean_deg"]),
                "E_t_mean_cm": float(dynamic["E_t_mean_cm"]),
                "E_j_mean_cm": float(dynamic["E_j_mean_cm"]),
                "E_ft_mean_cm": float(dynamic["E_ft_mean_cm"]),
                "Delta_v_mean_mps": float(dynamic["Delta_v_mean_mps"]),
                "Delta_v_p95_mps": float(dynamic["Delta_v_p95_mps"]),
                "Delta_omega_trace_mean_radps": angular["Delta_omega_trace"]["mean"],
                "Delta_omega_trace_p95_radps": angular["Delta_omega_trace"]["p95"],
                "Delta_omega_pose_mean_radps": angular["Delta_omega_pose"]["mean"],
                "Delta_omega_pose_p95_radps": angular["Delta_omega_pose"]["p95"],
                "trace_pose_omega_mismatch_mean_radps": angular["measurement_consistency"]["mean"],
                "prelift_grasp_ready": timing["prelift_multifinger_grasp_ready"],
                "named_source_contact_match_at_lift": timing["named_source_contact_match_at_lift"],
                "named_source_contact_recall_at_lift": timing[
                    "named_source_contact_recall_at_lift"
                ],
                "lift": pf["lift_success"],
                "lift_dz_m": float(dynamic["lift_dz_m"]),
                "table_support_release": bool(table[lift_mask].sum() == 0),
                "table_contact_before_lift_fraction": float(table[prelift_mask].mean()),
                "table_contact_during_lift_fraction": float(table[lift_mask].mean()),
                "relative_angular_twist_lift_mean_radps": float(rel_angular[lift_mask].mean()),
                "causality": pf["causal_execution"],
                "penetration_safe": pf["geometry_safe"],
                "action_bounds_safe": pf["action_bounds_safe"],
                "no_hidden_control": pf["no_hidden_control"],
                "legacy_SRqualified": _bool(dynamic["legacy_SRqualified"]),
                "SR_dynamic_V1": _bool(dynamic["SR_dynamic"]),
                "DF_THRESHOLD_PROVENANCE": df["THRESHOLD_PROVENANCE"],
            }
            pf_df_rows[clip].append(row)
        short = clip.removeprefix("hocap_")
        _write_csv(root / f"pf_df/v4_{short}_episode_receipts.csv", pf_df_rows[clip])
    pf_df_summary = {clip: _pf_df_summary(clip, pf_df_rows[clip]) for clip in CLIPS}
    for clip, summary in pf_df_summary.items():
        _write_json(root / f"pf_df/v4_{clip.removeprefix('hocap_')}_summary.json", summary)
    comparison = []
    for metric, key in (
        ("PF", "PF"),
        ("DF_pose", "DF_pose"),
        ("DF_linear_under_V1", "DF_linear_under_V1"),
        ("DF_angular_trace_under_V1", "DF_angular_trace_under_V1"),
        ("DF_angular_pose_derived_under_V1", "DF_angular_pose_derived_under_V1"),
    ):
        comparison.append(
            {
                "metric": metric,
                "V4_170105": f"{pf_df_summary['hocap_170105'][key]['pass_count']}/10",
                "V4_170650": f"{pf_df_summary['hocap_170650'][key]['pass_count']}/20",
                "interpretation": "PF physical outcome"
                if metric == "PF"
                else "DF fidelity dimension",
            }
        )
    _write_csv(root / "pf_df/comparison.csv", comparison)
    (root / "pf_df/metric_definitions.md").write_text(_metric_definitions(), encoding="utf-8")

    angular_order = sorted(
        range(20),
        key=lambda episode: float(angular_rows_650[episode]["Delta_omega_trace_mean_radps"]),
    )
    selected = {
        "lowest": angular_order[0],
        "median": angular_order[len(angular_order) // 2],
        "highest": angular_order[-1],
    }
    replay_root = root / "replay"
    replay_root.mkdir(parents=True, exist_ok=True)
    (replay_root / "visualization_commands.md").write_text(
        _replay_commands(selected), encoding="utf-8"
    )
    (replay_root / "manual_acceptance.md").write_text(
        "# Manual Acceptance\n\nFor 170105 compare raw MANO readiness, retarget named-finger geometry, actual persistent named contact, and LIFT onset. For 170650 compare the selected lowest/median/highest trace-Delta-omega episodes and decide whether visible motion is object-in-hand rotation, common hand-object motion, or a brief measurement spike. Manual review is not silently converted into a machine gate.\n",
        encoding="utf-8",
    )

    safety = {
        "BRANCH": "feature/ppo-physical",
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "PPO_TRAINING_RUN": "NO",
        "PPO_OPTIMIZER_STEP": 0,
        "REWARD_CHANGED": "NO",
        "FRICTION_CHANGED": "NO",
        "MASS_CHANGED": "NO",
        "REFERENCE_CHANGED": "NO",
        "RETIMING_CHANGED": "NO",
        "SR_HOLD_IMPLEMENTED": "NO",
        "ENGINEERED_TERMINAL_HOLD_ADDED": "NO",
        "LEGACY_SRPHYSICS_MODIFIED": "NO",
        "SR_DYNAMIC_V1_MODIFIED": "NO",
        "ANGULAR_THRESHOLD_TUNED": "NO",
        "CONTROLLER_CHANGED": "NO",
        "ACTION_CHANGED": "NO",
        "GUIDANCE_ADDED": "NO",
        "OBJECT_STATE_WRITE_ADDED": "NO",
        "WRIST_ROOT_WRITE_ADDED": "NO",
        "RAW_MOCAP_REPLAY_REGRESSED": "NO",
        "HISTORICAL_ARTIFACTS_MODIFIED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
    }
    final = {
        "schema_version": "Stage16ContactTimingAngularTwistPfDfHandoffV1",
        "git": {"branch": "feature/ppo-physical", "START_HEAD": START_HEAD},
        "contact_timing": timing_summary,
        "contact_timing_attribution": attribution,
        "angular_twist_170650": angular_aggregate,
        "angular_root_cause": angular_decision,
        "PF_DF": pf_df_summary,
        "PF_DF_SEPARATION_CHANGES_INTERPRETATION": "YES",
        "NEXT_170105": _next_contact(str(attribution["CONTACT_TIMING_LAYER_ROOT_CAUSE"])),
        "NEXT_170650": _next_angular(str(angular_decision["ANGULAR_TWIST_ROOT_CAUSE"])),
        "safety": safety,
    }
    _write_json(root / "final_summary.json", final)
    markdown = _final_markdown(
        timing=timing_summary,
        attribution=attribution,
        angular=angular_aggregate,
        angular_decision=angular_decision,
        phase_rows=phase_rows,
        finger_rows=aggregate_fingers,
        pf_df=pf_df_summary,
    )
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    _write_json(root / "tests.json", {"status": "NOT_RUN"})
    _write_json(
        root / "git_commits.json",
        {
            "branch": "feature/ppo-physical",
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip(),
            "commits": [],
        },
    )
    print(
        json.dumps(
            {
                "status": "STAGE16_CONTACT_TIMING_ANGULAR_PF_DF_COMPLETE",
                "report_root": str(root),
                "contact_root_cause": attribution["CONTACT_TIMING_LAYER_ROOT_CAUSE"],
                "angular_root_cause": angular_decision["ANGULAR_TWIST_ROOT_CAUSE"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
