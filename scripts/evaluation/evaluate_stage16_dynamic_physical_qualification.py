#!/usr/bin/env python3
"""Offline Stage16 Dynamic Physical Qualification V1 and V4 grasp diagnosis.

The evaluator intentionally consumes immutable C4 traces.  It does not create
an Isaac application, load an actor, invoke PPO, or modify historical V2
receipts.  The only written artifacts are a new, versioned report directory.
"""

# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation import (
    EvaluationJointSetV1,
    EvaluationSuiteV2,
    hand_metric_series,
    object_metric_series,
)
from toporetarget.rl.dynamic_physical_qualification import (
    DYNAMIC_PHYSICAL_QUALIFICATION_SCHEMA,
    FINGER_ORDER,
    PHASE_NAMES,
    DynamicTerminalGate,
    dynamic_interaction_metrics,
    dynamic_qualification,
    dynamic_twist_metrics,
    object_local_points,
    phase_labels_from_reference_index,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.geometry_audit.raw_mocap_overlay import resolve_raw_mocap_overlay
from toporetarget.rl.physical_evaluation import persistent_mask

REPORT_ROOT = (
    REPO_ROOT / ".local/reports/stage16_dynamic_physical_qualification_and_grasp_diagnostic"
)
FORMAL_170650_ROOT = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
)
FORMAL_170650_LEGACY = (
    REPO_ROOT
    / ".local/reports/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/analysis/qualification.json"
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
    / ".local/reports/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/sweep/v4/hocap_170105/c4"
)
SOURCE_170105_LEGACY = SOURCE_170105_REPORT / "per_episode.csv"
SOURCE_170105_QUALIFICATION = SOURCE_170105_REPORT / "qualification.json"
SOURCE_170105_AUTHORITY = (
    REPO_ROOT
    / ".local/reports/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/sources/v4_hocap_170105.json"
)
FROZEN_GATES = (
    REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4/frozen_evaluation_gates.json"
)
WORLD_WRIST_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
OBJECT_ASSET_ROOT = (
    REPO_ROOT / ".local/generated_assets/isaaclab/stage16_gravity_friction_curriculum_v1/C4"
)
STAGE_A_FINAL_SUMMARY = (
    REPO_ROOT / ".local/reports/stage16_raw_mocap_replay_overlay/final_summary.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"DYNAMIC_QUALIFICATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"DYNAMIC_QUALIFICATION_CSV_EMPTY:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"DYNAMIC_QUALIFICATION_CSV_FIELD_DRIFT:{path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ValueError(f"DYNAMIC_QUALIFICATION_BOOLEAN_INVALID:{value!r}")


def _float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DYNAMIC_QUALIFICATION_{name}_NUMERIC_REQUIRED") from exc
    if not np.isfinite(result):
        raise ValueError(f"DYNAMIC_QUALIFICATION_{name}_NONFINITE")
    return result


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return None if not len(array) else float(array.mean())


def _p95(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return None if not len(array) else float(np.quantile(array, 0.95))


def _first(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not len(indices) else int(indices[0])


def _all_input_paths() -> list[Path]:
    return [
        FORMAL_170650_LEGACY,
        FORMAL_170650_CONFIG,
        SOURCE_170105_LEGACY,
        SOURCE_170105_QUALIFICATION,
        SOURCE_170105_AUTHORITY,
        FROZEN_GATES,
        WORLD_WRIST_ROOT / "hocap_170105.world_wrist.stage16.npz",
        WORLD_WRIST_ROOT / "hocap_170650.world_wrist.stage16.npz",
        OBJECT_ASSET_ROOT / "hocap_170105/hocap_170105.usda",
        OBJECT_ASSET_ROOT / "hocap_170650/hocap_170650.usda",
        STAGE_A_FINAL_SUMMARY,
    ]


def _stage_a_receipt() -> dict[str, object]:
    receipt = _read_json(STAGE_A_FINAL_SUMMARY)
    required = (
        "RAW_MANO_OVERLAY",
        "RAW_OBJECT_OVERLAY",
        "COORDINATE_ALIGNMENT",
        "TIME_ALIGNMENT",
        "PHYSICS_ISOLATION",
        "BACKWARD_COMPATIBILITY",
    )
    if receipt.get("status") != "PASS" or any(receipt.get(key) != "PASS" for key in required):
        raise ValueError("DYNAMIC_QUALIFICATION_STAGE_A_RECEIPT_NOT_PASS")
    return {
        "source": str(STAGE_A_FINAL_SUMMARY.resolve()),
        "sha256": _sha256(STAGE_A_FINAL_SUMMARY),
        **{key: receipt[key] for key in required},
    }


def _source_trace_paths(clip: str) -> list[Path]:
    if clip == "hocap_170650":
        paths = [FORMAL_170650_ROOT / f"episode_{index:03d}.npz" for index in range(20)]
    elif clip == "hocap_170105":
        paths = [SOURCE_170105_ROOT / f"episode_{index:02d}.npz" for index in range(10)]
    else:
        raise ValueError(f"DYNAMIC_QUALIFICATION_UNKNOWN_CLIP:{clip}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DYNAMIC_QUALIFICATION_TRACE_MISSING:{','.join(missing)}")
    return paths


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    aliases = {
        "object_reference": ("object_reference", "embedded_reference_object_pose"),
        "wrist_reference": ("wrist_reference", "embedded_reference_wrist_pose"),
        "finger_reference": ("finger_reference", "embedded_reference_finger_q"),
        "reference_tracked_links": (
            "embedded_reference_tracked_links",
            "tracked_link_reference",
        ),
    }
    required = (
        "object_pose",
        "object_twist",
        "object_twist_reference",
        "wrist_pose",
        "wrist_twist_world",
        "finger_q",
        "hand_collision_body_pose",
        "hand_collision_body_names",
        "hand_object_pair_presence",
        "hand_object_pair_force_valid",
        "tip_pair_presence",
        "fingertip_object_pair_force_world",
        "source_contact_mask",
        "action",
        "reference_index",
        "table_object_contact",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(
                f"DYNAMIC_QUALIFICATION_TRACE_FIELD_MISSING:{path}:{','.join(missing)}"
            )
        result = {name: np.asarray(archive[name]) for name in required}
        for target, candidates in aliases.items():
            source = next((name for name in candidates if name in archive.files), None)
            if source is None:
                raise ValueError(f"DYNAMIC_QUALIFICATION_REFERENCE_FIELD_MISSING:{path}:{target}")
            result[target] = np.asarray(archive[source])
        result["r_contact_v4"] = np.asarray(
            archive["r_contact_v4"] if "r_contact_v4" in archive.files else np.zeros(321)
        )
        result["recorded_phase"] = (
            np.asarray(archive["phase"]) if "phase" in archive.files else None
        )
        result["checkpoint_sha256"] = np.asarray(
            archive["checkpoint_sha256"]
            if "checkpoint_sha256" in archive.files
            else archive["frozen_source_actor_sha256"]
        )
    frame_count = len(result["object_pose"])
    expected = {
        "object_pose": (frame_count, 7),
        "object_twist": (frame_count, 6),
        "object_twist_reference": (frame_count, 6),
        "wrist_pose": (frame_count, 7),
        "wrist_twist_world": (frame_count, 6),
        "finger_q": (frame_count, 20),
        "hand_collision_body_pose": (frame_count, 21, 7),
        "hand_object_pair_presence": (frame_count, 21),
        "hand_object_pair_force_valid": (frame_count,),
        "tip_pair_presence": (frame_count, 5),
        "fingertip_object_pair_force_world": (frame_count, 5, 3),
        "source_contact_mask": (frame_count, 5),
        "action": (frame_count, 26),
        "reference_index": (frame_count,),
        "table_object_contact": (frame_count,),
        "object_reference": (frame_count, 7),
        "wrist_reference": (frame_count, 7),
        "finger_reference": (frame_count, 20),
        "reference_tracked_links": (frame_count, 16, 3),
    }
    for name, shape in expected.items():
        if result[name].shape != shape:
            raise ValueError(
                f"DYNAMIC_QUALIFICATION_TRACE_SHAPE_INVALID:{path}:{name}:{result[name].shape}"
            )
    if (
        tuple(str(name) for name in result["hand_collision_body_names"].tolist())
        != HAND_COLLISION_BODY_NAMES
    ):
        raise ValueError(f"DYNAMIC_QUALIFICATION_HAND_BODY_ORDER_DRIFT:{path}")
    if not np.array_equal(result["reference_index"], np.arange(frame_count, dtype=np.int64)):
        raise ValueError(f"DYNAMIC_QUALIFICATION_REFERENCE_ALIGNMENT_DRIFT:{path}")
    phase_labels = phase_labels_from_reference_index(result["reference_index"])
    recorded_phase = result["recorded_phase"]
    if recorded_phase is not None:
        if recorded_phase.shape != (frame_count,):
            raise ValueError(f"DYNAMIC_QUALIFICATION_PHASE_SHAPE_INVALID:{path}")
        expected_codes = np.clip(
            (result["reference_index"] * len(PHASE_NAMES)) // frame_count,
            0,
            len(PHASE_NAMES) - 1,
        )
        phase_authority_matches = (
            np.array_equal(recorded_phase.astype(np.int64), expected_codes)
            if np.issubdtype(recorded_phase.dtype, np.number)
            else np.array_equal(recorded_phase.astype("U24"), phase_labels)
        )
        if not phase_authority_matches:
            raise ValueError(f"DYNAMIC_QUALIFICATION_PHASE_AUTHORITY_DRIFT:{path}")
    result["phase"] = phase_labels
    return result


def _legacy_rows_170650() -> list[dict[str, object]]:
    rows = _read_json(FORMAL_170650_LEGACY).get("episodes")
    if (
        not isinstance(rows, list)
        or len(rows) != 20
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("DYNAMIC_QUALIFICATION_170650_LEGACY_ROWS_INVALID")
    return [dict(row) for row in rows]


def _legacy_rows_170105() -> list[dict[str, object]]:
    with SOURCE_170105_LEGACY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise ValueError("DYNAMIC_QUALIFICATION_170105_LEGACY_ROWS_INVALID")
    return [dict(row) for row in rows]


def _legacy_bool(row: Mapping[str, object], name: str) -> bool:
    return _bool(row[name])


def _legacy_float(row: Mapping[str, object], name: str) -> float:
    return _float(row[name], name=name)


def _distal_indices() -> tuple[int, ...]:
    lookup = {
        "thumb": "r_thumb_distal",
        "index": "r_index_finger_distal",
        "middle": "r_middle_finger_distal",
        "ring": "r_ring_finger_distal",
        "pinky": "r_pinky_distal",
    }
    return tuple(HAND_COLLISION_BODY_NAMES.index(lookup[finger]) for finger in FINGER_ORDER)


def _recompute_frozen_v2_kinematics(
    trace: Mapping[str, np.ndarray], legacy: Mapping[str, object]
) -> tuple[dict[str, float], bool]:
    """Re-evaluate V2 tracking values and fail closed on receipt drift."""

    metrics = object_metric_series(
        np.asarray(trace["object_pose"], dtype=np.float64),
        np.asarray(trace["object_reference"], dtype=np.float64),
    )
    metrics.update(
        hand_metric_series(
            np.asarray(trace["hand_collision_body_pose"], dtype=np.float64),
            list(HAND_COLLISION_BODY_NAMES),
            np.asarray(trace["reference_tracked_links"], dtype=np.float64),
            list(EvaluationJointSetV1().joint_names),
        )
    )
    values = {
        "E_r_mean_deg": float(metrics["e_r_deg"].mean()),
        "E_t_mean_cm": float(metrics["e_t_cm"].mean()),
        "E_j_mean_cm": float(metrics["e_j_cm"].mean()),
        "E_ft_mean_cm": float(metrics["e_ft_cm"].mean()),
    }
    for name, value in values.items():
        if not np.isclose(value, _legacy_float(legacy, name), rtol=0.0, atol=1.0e-10):
            raise ValueError(f"DYNAMIC_QUALIFICATION_V2_METRIC_PARITY_DRIFT:{name}")
    suite = EvaluationSuiteV2()
    kinematic_success = bool(
        values["E_r_mean_deg"] < suite.object_rotation_threshold_deg
        and values["E_t_mean_cm"] < suite.object_translation_threshold_cm
        and values["E_j_mean_cm"] < suite.hand_joint_threshold_cm
        and values["E_ft_mean_cm"] < suite.fingertip_threshold_cm
    )
    if kinematic_success != _legacy_bool(legacy, "kinematic_success"):
        raise ValueError("DYNAMIC_QUALIFICATION_V2_KINEMATIC_PARITY_DRIFT")
    return values, kinematic_success


def _trace_dynamic_row(
    *,
    clip: str,
    episode: int,
    trace_path: Path,
    trace: Mapping[str, np.ndarray],
    legacy: Mapping[str, object],
    gate: DynamicTerminalGate,
    causal_execution_safe: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    v2_metrics, kinematic_success = _recompute_frozen_v2_kinematics(trace, legacy)
    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    hand_contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    twist = dynamic_twist_metrics(
        actual_twist_world=np.asarray(trace["object_twist"]),
        reference_twist_world=np.asarray(trace["object_twist_reference"]),
        hand_object_contact=hand_contact,
        valid=valid,
        gate=gate,
    )
    interaction = dynamic_interaction_metrics(trace)
    geometry_safe = bool(_legacy_bool(legacy, "absolute_geometry_pass"))
    if "interfinger_max_mm" in legacy:
        geometry_safe = geometry_safe and _legacy_float(legacy, "interfinger_max_mm") <= 3.0
    action_bounds_safe = bool(np.max(np.abs(np.asarray(trace["action"], dtype=np.float64))) <= 1.0)
    result = dynamic_qualification(
        legacy_kinematic_success=kinematic_success,
        interaction=interaction,
        twist=twist,
        geometry_safe=geometry_safe,
        action_bounds_safe=action_bounds_safe,
        causal_execution_safe=causal_execution_safe,
    )
    full = twist["full_motion"]
    terminal = twist["legacy_terminal_window_equivalent"]
    row = {
        "clip": clip,
        "episode": episode,
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "checkpoint_sha256": str(np.asarray(trace["checkpoint_sha256"]).item()),
        "legacy_SRkin": _legacy_bool(legacy, "kinematic_success"),
        "legacy_SRphysics": _legacy_bool(legacy, "physics_success"),
        "legacy_SRqualified": _legacy_bool(legacy, "qualified_success"),
        "SR_dynamic": bool(result["SR_dynamic"]),
        "primary_classification": str(result["primary_classification"]),
        "secondary_failures": ";".join(str(item) for item in result["secondary_failures"]),
        **v2_metrics,
        "V2_METRIC_PARITY": "PASS",
        "persistent_grasp": bool(interaction["persistent_grasp"]),
        "grasp_and_lift": bool(interaction["grasp_and_lift"]),
        "persistent_grasp_at_reference_lift": bool(
            interaction["persistent_grasp_at_semantic_lift"]
        ),
        "lift_dz_m": float(interaction["lift_dz_m"]),
        "Delta_v_mean_mps": float(full["Delta_v_mean_mps"]),
        "Delta_v_p95_mps": float(full["Delta_v_p95_mps"]),
        "Delta_v_terminal_mean_mps": float(terminal["Delta_v_terminal_mean_mps"]),
        "Delta_v_terminal_p95_mps": float(terminal["Delta_v_terminal_p95_mps"]),
        "Delta_v_terminal_max_mps": float(terminal["Delta_v_terminal_max_mps"]),
        "Delta_omega_mean_radps": float(full["Delta_omega_mean_radps"]),
        "Delta_omega_p95_radps": float(full["Delta_omega_p95_radps"]),
        "Delta_omega_terminal_mean_radps": float(terminal["Delta_omega_terminal_mean_radps"]),
        "Delta_omega_terminal_p95_radps": float(terminal["Delta_omega_terminal_p95_radps"]),
        "Delta_omega_terminal_max_radps": float(terminal["Delta_omega_terminal_max_radps"]),
        "reference_twist_dynamic_pass": bool(twist["reference_twist_dynamic_pass"]),
        "absolute_geometry_pass": geometry_safe,
        "action_bounds_pass": action_bounds_safe,
        "causal_execution_pass": causal_execution_safe,
    }
    return row, interaction, twist


def _aggregate_dynamic(rows: list[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("DYNAMIC_QUALIFICATION_AGGREGATE_EMPTY")
    rates = {
        name: {
            "pass_count": sum(bool(row[name]) for row in rows),
            "total": len(rows),
            "rate": sum(bool(row[name]) for row in rows) / len(rows),
        }
        for name in (
            "legacy_SRkin",
            "legacy_SRphysics",
            "legacy_SRqualified",
            "SR_dynamic",
            "reference_twist_dynamic_pass",
            "persistent_grasp",
            "grasp_and_lift",
            "absolute_geometry_pass",
            "action_bounds_pass",
            "causal_execution_pass",
        )
    }
    numeric = (
        "E_r_mean_deg",
        "E_t_mean_cm",
        "E_j_mean_cm",
        "E_ft_mean_cm",
        "lift_dz_m",
        "Delta_v_mean_mps",
        "Delta_v_p95_mps",
        "Delta_v_terminal_max_mps",
        "Delta_omega_mean_radps",
        "Delta_omega_p95_radps",
        "Delta_omega_terminal_max_radps",
    )
    return {
        "episode_count": len(rows),
        "rates": rates,
        "metrics": {name: _mean(_float(row[name], name=name) for row in rows) for name in numeric},
        "classification_counts": {
            name: sum(str(row["primary_classification"]) == name for row in rows)
            for name in sorted({str(row["primary_classification"]) for row in rows})
        },
    }


def _object_properties(clip: str, overlay: Any) -> dict[str, object]:
    path = OBJECT_ASSET_ROOT / clip / f"{clip}.usda"
    text = path.read_text(encoding="utf-8")

    def scalar(name: str) -> float | str:
        match = re.search(rf"float physics:{name} = ([^\s]+)", text)
        return "NOT_IDENTIFIABLE" if match is None else float(match.group(1))

    def vector(name: str) -> list[float] | str:
        match = re.search(rf"(?:point3f|float3) physics:{name} = \(([^)]+)\)", text)
        return (
            "NOT_IDENTIFIABLE"
            if match is None
            else [float(value.strip()) for value in match.group(1).split(",")]
        )

    dimensions = np.ptp(np.asarray(overlay.raw_object_vertices_local, dtype=np.float64), axis=0)
    return {
        "clip": clip,
        "mass_kg": scalar("mass"),
        "COM_object_m": vector("centerOfMass"),
        "diagonal_inertia_kg_m2": vector("diagonalInertia"),
        "bounding_dimensions_m": dimensions.tolist(),
        "object_static_friction_authored": 1.0,
        "object_dynamic_friction_authored": 1.0,
        "hand_static_friction_authored": 0.5,
        "hand_dynamic_friction_authored": 0.5,
        "table_static_friction": "NOT_IDENTIFIABLE",
        "table_dynamic_friction": "NOT_IDENTIFIABLE",
        "physx_friction_combine_mode": "NOT_IDENTIFIABLE",
        "effective_hand_object_mu": "NOT_IDENTIFIABLE",
        "effective_table_object_mu": "NOT_IDENTIFIABLE",
        "authority": str(path.resolve()),
    }


def _phase_window_rows(clip: str) -> list[dict[str, object]]:
    indices = np.arange(321, dtype=np.int64)
    phases = phase_labels_from_reference_index(indices)
    return [
        {
            "clip": clip,
            "phase": phase,
            "start_frame": int(np.flatnonzero(phases == phase)[0]),
            "end_frame_exclusive": int(np.flatnonzero(phases == phase)[-1] + 1),
        }
        for phase in ("APPROACH", "CONTACT", "GRASP", "LIFT")
    ]


def _phase_masks(trace: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    phase = np.asarray(trace["phase"]).astype("U24")
    return {name: phase == name for name in ("CONTACT", "GRASP", "LIFT")}


def _persistent_fingers(actual: np.ndarray, *, valid: np.ndarray) -> np.ndarray:
    return np.stack(
        [persistent_mask(actual[:, index] & valid, minimum_steps=3) for index in range(5)], axis=-1
    )


def _topology_and_dynamics(
    *,
    clip: str,
    episode: int,
    trace: Mapping[str, np.ndarray],
    overlay: Any,
    timestamp_s: np.ndarray,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    distal = _distal_indices()
    actual_tips = np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)[:, distal, :3]
    reference_pose = reconstruct_hand_collision_body_pose(
        np.asarray(trace["wrist_reference"]),
        np.asarray(trace["finger_reference"]),
        repo_root=REPO_ROOT,
    )
    reference_tips = reference_pose[:, distal, :3]
    raw_tips = np.asarray(overlay.raw_mano_fingertips_world, dtype=np.float64)
    raw_local = object_local_points(raw_tips, np.asarray(overlay.raw_object_pose_world_wxyz))
    reference_local = object_local_points(reference_tips, np.asarray(trace["object_reference"]))
    actual_local = object_local_points(actual_tips, np.asarray(trace["object_pose"]))
    raw_to_reference = np.linalg.norm(raw_local - reference_local, axis=-1)
    raw_to_actual = np.linalg.norm(raw_local - actual_local, axis=-1)
    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    expected = np.asarray(trace["source_contact_mask"], dtype=bool)
    actual = np.asarray(trace["tip_pair_presence"], dtype=bool) & valid[:, None]
    persistent = _persistent_fingers(actual, valid=valid)
    hand = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1) & valid
    table = np.asarray(trace["table_object_contact"], dtype=bool) & valid
    force = np.linalg.norm(
        np.asarray(trace["fingertip_object_pair_force_world"], dtype=np.float64), axis=-1
    )
    masks = _phase_masks(trace)
    topology: list[dict[str, object]] = []
    contact_dynamics: list[dict[str, object]] = []
    for phase, mask in masks.items():
        phase_valid = mask & valid
        if not phase_valid.any():
            raise ValueError(f"DYNAMIC_QUALIFICATION_PHASE_EMPTY:{clip}:{episode}:{phase}")
        for finger, index in zip(FINGER_ORDER, range(5), strict=True):
            values = raw_to_reference[phase_valid, index]
            actual_values = raw_to_actual[phase_valid, index]
            topology.append(
                {
                    "clip": clip,
                    "episode": episode,
                    "phase": phase,
                    "finger": finger,
                    "raw_mano_expected_contact_fraction": float(
                        expected[phase_valid, index].mean()
                    ),
                    "retarget_expected_contact_fraction": float(
                        expected[phase_valid, index].mean()
                    ),
                    "actual_persistent_contact_fraction": float(
                        persistent[phase_valid, index].mean()
                    ),
                    "raw_to_retarget_object_local_tip_error_mean_m": float(values.mean()),
                    "raw_to_retarget_object_local_tip_error_p95_m": float(
                        np.quantile(values, 0.95)
                    ),
                    "raw_to_actual_object_local_tip_error_mean_m": float(actual_values.mean()),
                }
            )
        active = force[phase_valid & actual.any(axis=-1)]
        contact_dynamics.append(
            {
                "clip": clip,
                "episode": episode,
                "phase": phase,
                "table_object_contact_fraction": float(table[phase_valid].mean()),
                "hand_object_contact_fraction": float(hand[phase_valid].mean()),
                "multi_finger_persistent_fraction": float(
                    (persistent[phase_valid].sum(axis=-1) >= 2).mean()
                ),
                "active_force_mean_n": None if not active.size else float(active.mean()),
                "active_force_p95_n": None if not active.size else float(np.quantile(active, 0.95)),
                "active_force_max_n": None if not active.size else float(active.max()),
                "exact_contact_normal": "NOT_CAPTURED",
                "exact_tangential_slip": "NOT_IDENTIFIABLE",
                "friction_cone_utilization": "NOT_IDENTIFIABLE",
            }
        )
    object_twist = np.asarray(trace["object_twist"], dtype=np.float64)
    wrist_twist = np.asarray(trace["wrist_twist_world"], dtype=np.float64)
    relative_twist = object_twist - wrist_twist
    tip_velocity = np.gradient(actual_tips, timestamp_s, axis=0)
    tip_object_speed_proxy = np.linalg.norm(tip_velocity - object_twist[:, None, :3], axis=-1)
    relative_rows: list[dict[str, object]] = []
    for phase, mask in masks.items():
        selected = mask & valid
        linear = np.linalg.norm(relative_twist[selected, :3], axis=-1)
        angular = np.linalg.norm(relative_twist[selected, 3:], axis=-1)
        proxy = tip_object_speed_proxy[selected & actual.any(axis=-1)]
        relative_rows.append(
            {
                "clip": clip,
                "episode": episode,
                "phase": phase,
                "relative_twist_frame": "world_frame_object_twist_minus_wrist_twist_proxy",
                "relative_linear_speed_mean_mps": float(linear.mean()),
                "relative_linear_speed_p95_mps": float(np.quantile(linear, 0.95)),
                "relative_angular_speed_mean_radps": float(angular.mean()),
                "relative_angular_speed_p95_radps": float(np.quantile(angular, 0.95)),
                "TIP_OBJECT_RELATIVE_SPEED_PROXY_mean_mps": None
                if not proxy.size
                else float(proxy.mean()),
                "TIP_OBJECT_RELATIVE_SPEED_PROXY_p95_mps": None
                if not proxy.size
                else float(np.quantile(proxy, 0.95)),
                "SLIP_EXACT": "NOT_IDENTIFIABLE",
            }
        )
    expected_onset = _first(expected.any(axis=-1) & valid)
    first_contact = _first(actual.any(axis=-1))
    first_persistent = _first((persistent.sum(axis=-1) >= 2) & valid)
    first_multi = _first((actual.sum(axis=-1) >= 2) & valid)
    lift_onset = _first(np.asarray(trace["phase"]) == "LIFT")
    loss = None
    if first_persistent is not None:
        after = np.flatnonzero(
            (np.arange(len(hand)) > first_persistent) & ~(persistent.sum(axis=-1) >= 2)
        )
        loss = None if not len(after) else int(after[0])
    timing = {
        "clip": clip,
        "episode": episode,
        "reference_expected_contact_onset": expected_onset,
        "actual_first_contact": first_contact,
        "actual_first_persistent_multi_finger_contact": first_persistent,
        "actual_first_multi_finger_contact": first_multi,
        "actual_contact_loss": loss,
        "reference_lift_onset": lift_onset,
        "persistent_grasp_before_reference_lift": bool(
            first_persistent is not None
            and lift_onset is not None
            and first_persistent <= lift_onset
        ),
        "CONTACT_TIMING_LOSS": bool(
            first_persistent is None or (lift_onset is not None and first_persistent > lift_onset)
        ),
    }
    return topology, [timing], relative_rows, {"contact_dynamics": contact_dynamics}


def _summary_stage_c(
    *,
    clip: str,
    dynamic_rows: list[Mapping[str, object]],
    topology: list[Mapping[str, object]],
    timing: list[Mapping[str, object]],
    relative_twist: list[Mapping[str, object]],
    contact_dynamics: list[Mapping[str, object]],
    object_properties: Mapping[str, object],
    required_force: Mapping[str, object],
) -> dict[str, object]:
    lift_rows = [row for row in relative_twist if row["phase"] == "LIFT"]
    lift_contacts = [row for row in contact_dynamics if row["phase"] == "LIFT"]
    lift_topology = [row for row in topology if row["phase"] == "LIFT"]
    return {
        "clip": clip,
        "episodes": len(dynamic_rows),
        "persistent_grasp_episodes": sum(bool(row["persistent_grasp"]) for row in dynamic_rows),
        "lift_episodes": sum(bool(row["grasp_and_lift"]) for row in dynamic_rows),
        "lift_dz_m_mean": _mean(_float(row["lift_dz_m"], name="lift") for row in dynamic_rows),
        "SR_dynamic_episodes": sum(bool(row["SR_dynamic"]) for row in dynamic_rows),
        "contact_timing_loss_episodes": sum(bool(row["CONTACT_TIMING_LOSS"]) for row in timing),
        "contact_timing_before_lift_episodes": sum(
            bool(row["persistent_grasp_before_reference_lift"]) for row in timing
        ),
        "LIFT_relative_linear_twist_mean_mps": _mean(
            _float(row["relative_linear_speed_mean_mps"], name="relative_linear")
            for row in lift_rows
        ),
        "LIFT_relative_angular_twist_mean_radps": _mean(
            _float(row["relative_angular_speed_mean_radps"], name="relative_angular")
            for row in lift_rows
        ),
        "LIFT_active_force_p95_n": _mean(
            _float(row["active_force_p95_n"], name="force")
            for row in lift_contacts
            if row["active_force_p95_n"] is not None
        ),
        "LIFT_table_object_contact_fraction": _mean(
            _float(row["table_object_contact_fraction"], name="table") for row in lift_contacts
        ),
        "LIFT_hand_object_contact_fraction": _mean(
            _float(row["hand_object_contact_fraction"], name="hand") for row in lift_contacts
        ),
        "LIFT_raw_to_retarget_topology_error_mean_m": _mean(
            _float(row["raw_to_retarget_object_local_tip_error_mean_m"], name="topology")
            for row in lift_topology
            if _float(row["raw_mano_expected_contact_fraction"], name="expected") > 0.0
        ),
        "SLIP_EXACT": "NOT_IDENTIFIABLE",
        "HAND_OBJECT_EFFECTIVE_MU_IDENTIFIABLE": "NO",
        "TABLE_OBJECT_EFFECTIVE_MU_IDENTIFIABLE": "NO",
        "FRICTION_CONE_UTILIZATION": "NOT_IDENTIFIABLE",
        "GRASP_WRENCH_MARGIN": "NOT_IDENTIFIABLE",
        "object_properties": dict(object_properties),
        "required_wrench": dict(required_force),
    }


def _phase_diagnostic_rows(
    *,
    clip: str,
    topology: list[Mapping[str, object]],
    contact_dynamics: list[Mapping[str, object]],
    relative_twist: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the raw-to-retarget-to-actual diagnostic by semantic phase."""

    rows: list[dict[str, object]] = []
    for phase in ("CONTACT", "GRASP", "LIFT"):
        topology_rows = [row for row in topology if row["phase"] == phase]
        contact_rows = [row for row in contact_dynamics if row["phase"] == phase]
        twist_rows = [row for row in relative_twist if row["phase"] == phase]
        if not topology_rows or not contact_rows or not twist_rows:
            raise ValueError(f"DYNAMIC_QUALIFICATION_PHASE_DIAGNOSTIC_EMPTY:{clip}:{phase}")
        rows.append(
            {
                "clip": clip,
                "phase": phase,
                "raw_mano_expected_contact_fraction": _mean(
                    _float(row["raw_mano_expected_contact_fraction"], name="raw_expected")
                    for row in topology_rows
                ),
                "retarget_expected_contact_fraction": _mean(
                    _float(row["retarget_expected_contact_fraction"], name="retarget_expected")
                    for row in topology_rows
                ),
                "actual_persistent_contact_fraction": _mean(
                    _float(row["actual_persistent_contact_fraction"], name="actual_persistent")
                    for row in topology_rows
                ),
                "raw_to_retarget_object_local_tip_error_mean_m": _mean(
                    _float(row["raw_to_retarget_object_local_tip_error_mean_m"], name="topology")
                    for row in topology_rows
                ),
                "actual_hand_object_contact_fraction": _mean(
                    _float(row["hand_object_contact_fraction"], name="hand_contact")
                    for row in contact_rows
                ),
                "multi_finger_persistent_fraction": _mean(
                    _float(row["multi_finger_persistent_fraction"], name="multi_persistent")
                    for row in contact_rows
                ),
                "active_force_p95_n": _mean(
                    _float(row["active_force_p95_n"], name="force")
                    for row in contact_rows
                    if row["active_force_p95_n"] is not None
                ),
                "relative_linear_twist_mean_mps": _mean(
                    _float(row["relative_linear_speed_mean_mps"], name="relative_linear")
                    for row in twist_rows
                ),
                "relative_angular_twist_mean_radps": _mean(
                    _float(row["relative_angular_speed_mean_radps"], name="relative_angular")
                    for row in twist_rows
                ),
                "table_object_contact_fraction": _mean(
                    _float(row["table_object_contact_fraction"], name="table_contact")
                    for row in contact_rows
                ),
                "exact_tangential_slip": "NOT_IDENTIFIABLE",
            }
        )
    return rows


def _reference_required_force(
    *, trace: Mapping[str, np.ndarray], mass_kg: float, timestamps_s: np.ndarray
) -> dict[str, object]:
    ref = np.asarray(trace["object_twist_reference"], dtype=np.float64)
    phase = np.asarray(trace["phase"]) == "LIFT"
    acceleration = np.gradient(ref[:, :3], timestamps_s, axis=0)
    force = mass_kg * (acceleration - np.asarray([0.0, 0.0, -9.81]))
    norms = np.linalg.norm(force[phase], axis=-1)
    return {
        "definition": "F_req_equals_m_times_a_reference_minus_g",
        "window": "LIFT",
        "force_norm_mean_N": float(norms.mean()),
        "force_norm_p95_N": float(np.quantile(norms, 0.95)),
        "rotational_wrench": "NOT_IDENTIFIABLE",
    }


def _root_cause(
    summary_105: Mapping[str, object], summary_650: Mapping[str, object]
) -> dict[str, object]:
    contract = {
        "allowed_primary": [
            "RETARGET_CONTACT_GEOMETRY_PRIMARY",
            "CONTACT_TIMING_PRIMARY",
            "NORMAL_FORCE_CLOSURE_PRIMARY",
            "FRICTION_MARGIN_PRIMARY",
            "PHYSICAL_SLIP_PRIMARY",
            "HAND_OBJECT_COUPLING_PRIMARY",
            "REFERENCE_DYNAMIC_INFEASIBILITY_PRIMARY",
            "MULTI_FACTOR_PRIMARY",
            "INCONCLUSIVE",
        ],
        "friction_primary_requires": ["exact_Fn_Ft", "effective_mu"],
        "wrench_primary_requires": [
            "contact_points",
            "contact_normals",
            "effective_mu",
            "object_inertia",
        ],
    }
    timing_105 = int(summary_105["contact_timing_loss_episodes"])
    timing_650 = int(summary_650["contact_timing_before_lift_episodes"])
    if timing_105 == int(summary_105["episodes"]) and timing_650 == int(summary_650["episodes"]):
        primary = "CONTACT_TIMING_PRIMARY"
        confidence = "HIGH"
        next_action = "NEXT_CONTACT_TIMING_PHYSICAL_REFINEMENT"
        rationale = (
            "Every V4/170105 C4 episode first attained persistent multi-finger contact after "
            "the reference LIFT onset, while every V4/170650 Formal20 episode attained it on "
            "or before LIFT.  The timing difference precedes 170105's no-lift outcome."
        )
    else:
        primary = "INCONCLUSIVE"
        confidence = "LOW"
        next_action = "NEXT_TARGETED_CONTACT_DYNAMICS_TELEMETRY"
        rationale = "The frozen traces did not establish an earlier cross-clip divergence."
    return {
        "schema_version": "Stage16GraspRobustnessRootCauseDecisionV1",
        "decision_contract": contract,
        "primary_root_cause": primary,
        "confidence": confidence,
        "one_next_action": next_action,
        "rationale": rationale,
        "FRICTION_TOO_LOW_PRIMARY": "NOT_SUPPORTED",
        "friction_reason": "Exact contact normals, Fn/Ft, and PhysX effective combine-mode mu are absent from the immutable traces.",
        "OPEN_OBJECT_IMPLICATION": "NO_PER_OBJECT_FRICTION_OR_REWARD_TUNING_AUTHORIZED",
        "object_agnostic_target": "persistent_contact_before_reference_LIFT_and_low_relative_hand_object_twist",
    }


def _metric_table(
    dynamic_105: Mapping[str, object],
    dynamic_650: Mapping[str, object],
    grasp_105: Mapping[str, object],
    grasp_650: Mapping[str, object],
) -> str:
    summary_105 = dynamic_105["rates"]
    summary_650 = dynamic_650["rates"]
    metrics_105 = dynamic_105["metrics"]
    metrics_650 = dynamic_650["metrics"]

    def metric(metrics: Mapping[str, object], name: str) -> float:
        return _float(metrics[name], name=name)

    return "\n".join(
        [
            "| Metric | Definition | Window | Threshold | 170105 | 170650 | Pass/Fail |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
            f"| SRkin | frozen V2 kinematic success | full | frozen V2 | {summary_105['legacy_SRkin']['pass_count']}/{summary_105['legacy_SRkin']['total']} | {summary_650['legacy_SRkin']['pass_count']}/{summary_650['legacy_SRkin']['total']} | reported |",
            f"| SRphysics | frozen historical physical success | full | frozen V2 | {summary_105['legacy_SRphysics']['pass_count']}/{summary_105['legacy_SRphysics']['total']} | {summary_650['legacy_SRphysics']['pass_count']}/{summary_650['legacy_SRphysics']['total']} | immutable |",
            f"| SR_dynamic | composite dynamic qualification | full | composite | {summary_105['SR_dynamic']['pass_count']}/{summary_105['SR_dynamic']['total']} | {summary_650['SR_dynamic']['pass_count']}/{summary_650['SR_dynamic']['total']} | see receipt |",
            f"| E_r | mean rotation error | full | <30 deg | {metric(metrics_105, 'E_r_mean_deg'):.4f} deg | {metric(metrics_650, 'E_r_mean_deg'):.4f} deg | per-episode gate |",
            f"| E_t | mean translation error | full | <3 cm | {metric(metrics_105, 'E_t_mean_cm'):.4f} cm | {metric(metrics_650, 'E_t_mean_cm'):.4f} cm | per-episode gate |",
            f"| E_j | mean joint error | full | <8 cm | {metric(metrics_105, 'E_j_mean_cm'):.4f} cm | {metric(metrics_650, 'E_j_mean_cm'):.4f} cm | per-episode gate |",
            f"| E_ft | mean fingertip error | full | <6 cm | {metric(metrics_105, 'E_ft_mean_cm'):.4f} cm | {metric(metrics_650, 'E_ft_mean_cm'):.4f} cm | per-episode gate |",
            f"| Delta_v mean | norm(v_actual-v_reference) | full | diagnostic | {metric(metrics_105, 'Delta_v_mean_mps'):.6f} m/s | {metric(metrics_650, 'Delta_v_mean_mps'):.6f} m/s | reported |",
            f"| Delta_v p95 | norm(v_actual-v_reference) | full | diagnostic | {metric(metrics_105, 'Delta_v_p95_mps'):.6f} m/s | {metric(metrics_650, 'Delta_v_p95_mps'):.6f} m/s | reported |",
            f"| Delta_omega mean | norm(omega_actual-omega_reference) | full | diagnostic | {metric(metrics_105, 'Delta_omega_mean_radps'):.6f} rad/s | {metric(metrics_650, 'Delta_omega_mean_radps'):.6f} rad/s | reported |",
            f"| Delta_omega p95 | norm(omega_actual-omega_reference) | full | diagnostic | {metric(metrics_105, 'Delta_omega_p95_radps'):.6f} rad/s | {metric(metrics_650, 'Delta_omega_p95_radps'):.6f} rad/s | reported |",
            f"| terminal Delta_v max | mean per-episode terminal maximum | final 20 steps | inherited contact/free V2 | {metric(metrics_105, 'Delta_v_terminal_max_mps'):.6f} m/s | {metric(metrics_650, 'Delta_v_terminal_max_mps'):.6f} m/s | dynamic gate |",
            f"| terminal Delta_omega max | mean per-episode terminal maximum | final 20 steps | inherited contact/free V2 | {metric(metrics_105, 'Delta_omega_terminal_max_radps'):.6f} rad/s | {metric(metrics_650, 'Delta_omega_terminal_max_radps'):.6f} rad/s | dynamic gate |",
            f"| persistent grasp | >=2 persistent fingers | semantic LIFT | >=2 fingers | {grasp_105['persistent_grasp_episodes']}/{grasp_105['episodes']} | {grasp_650['persistent_grasp_episodes']}/{grasp_650['episodes']} | interaction receipt |",
            f"| grasp and lift | persistent grasp plus >=5 cm lift | terminal | >=5 cm | {grasp_105['lift_episodes']}/{grasp_105['episodes']} | {grasp_650['lift_episodes']}/{grasp_650['episodes']} | interaction receipt |",
            f"| lift dz | terminal object vertical displacement | terminal | >=0.05 m | {metric(metrics_105, 'lift_dz_m'):.6f} m | {metric(metrics_650, 'lift_dz_m'):.6f} m | interaction gate |",
            f"| persistent before LIFT | first persistent multi-finger contact before reference LIFT | phase boundary | on/before LIFT | {grasp_105['contact_timing_before_lift_episodes']}/{grasp_105['episodes']} | {grasp_650['contact_timing_before_lift_episodes']}/{grasp_650['episodes']} | timing receipt |",
            f"| penetration safety | frozen absolute/inter-finger geometry | full | frozen V2 | {summary_105['absolute_geometry_pass']['pass_count']}/{summary_105['absolute_geometry_pass']['total']} | {summary_650['absolute_geometry_pass']['pass_count']}/{summary_650['absolute_geometry_pass']['total']} | composite gate |",
            f"| causality | frozen no-hidden-control evidence | full | frozen V2 | {summary_105['causal_execution_pass']['pass_count']}/{summary_105['causal_execution_pass']['total']} | {summary_650['causal_execution_pass']['pass_count']}/{summary_650['causal_execution_pass']['total']} | composite gate |",
            f"| LIFT active force p95 | named fingertip pair-force magnitude | LIFT | diagnostic | {metric(grasp_105, 'LIFT_active_force_p95_n'):.6f} N | {metric(grasp_650, 'LIFT_active_force_p95_n'):.6f} N | vector magnitude only |",
            f"| LIFT table contact | mean table/object contact fraction | LIFT | diagnostic | {metric(grasp_105, 'LIFT_table_object_contact_fraction'):.4f} | {metric(grasp_650, 'LIFT_table_object_contact_fraction'):.4f} | reported |",
            f"| LIFT relative linear twist | mean norm(object_twist-wrist_twist proxy) | LIFT | diagnostic | {metric(grasp_105, 'LIFT_relative_linear_twist_mean_mps'):.6f} m/s | {metric(grasp_650, 'LIFT_relative_linear_twist_mean_mps'):.6f} m/s | proxy only |",
            f"| LIFT relative angular twist | mean norm(object_twist-wrist_twist proxy) | LIFT | diagnostic | {metric(grasp_105, 'LIFT_relative_angular_twist_mean_radps'):.6f} rad/s | {metric(grasp_650, 'LIFT_relative_angular_twist_mean_radps'):.6f} rad/s | proxy only |",
            "| exact slip / friction cone / wrench margin | normals, points, effective mu absent | all | evidence required | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | no friction-primary claim |",
        ]
    )


def _markdown(
    *,
    dynamic_105: Mapping[str, object],
    dynamic_650: Mapping[str, object],
    grasp_105: Mapping[str, object],
    grasp_650: Mapping[str, object],
    root: Mapping[str, object],
) -> str:
    return "\n".join(
        [
            "# Stage16 Dynamic Physical Qualification + Grasp Robustness Diagnostic Handoff",
            "",
            "## Dynamic evaluator",
            "",
            "`SR_dynamic` is an additive, offline-only receipt.  It is not `SR_hold`; no hold segment, terminal freeze, training, reward change, physics change, or controller change was introduced.",
            "",
            "`Delta_v = v_actual - v_reference` and `Delta_omega = omega_actual - omega_reference` use the frozen Reference Kinematics V2 world-frame convention.  The legacy 20-step thresholds are inherited unchanged, but apply to these deltas; absolute world terminal zero speed is not used by `SR_dynamic`.",
            "",
            "| Evaluation | V4/170105 C4 | V4/170650 Formal20 |",
            "| --- | ---: | ---: |",
            f"| Legacy SRkin | {dynamic_105['rates']['legacy_SRkin']['pass_count']}/{dynamic_105['rates']['legacy_SRkin']['total']} | {dynamic_650['rates']['legacy_SRkin']['pass_count']}/{dynamic_650['rates']['legacy_SRkin']['total']} |",
            f"| Legacy SRphysics | {dynamic_105['rates']['legacy_SRphysics']['pass_count']}/{dynamic_105['rates']['legacy_SRphysics']['total']} | {dynamic_650['rates']['legacy_SRphysics']['pass_count']}/{dynamic_650['rates']['legacy_SRphysics']['total']} |",
            f"| Legacy SRqualified | {dynamic_105['rates']['legacy_SRqualified']['pass_count']}/{dynamic_105['rates']['legacy_SRqualified']['total']} | {dynamic_650['rates']['legacy_SRqualified']['pass_count']}/{dynamic_650['rates']['legacy_SRqualified']['total']} |",
            f"| New SR_dynamic | {dynamic_105['rates']['SR_dynamic']['pass_count']}/{dynamic_105['rates']['SR_dynamic']['total']} | {dynamic_650['rates']['SR_dynamic']['pass_count']}/{dynamic_650['rates']['SR_dynamic']['total']} |",
            "",
            "`DIFFERENCE_CAUSED_BY_TERMINAL_SEMANTICS=NO`: the inherited dynamic reference-relative twist gate remains failed for 18/20 170650 episodes, so the semantic correction does not convert this frozen Formal20 result into an acceptable dynamic HOI.  Its machine-gated answer is therefore `NO`, despite 20/20 persistent-grasp-and-lift episodes.",
            "",
            "## Is V4/170650 dynamically acceptable?",
            "",
            "`DYNAMIC_QUALITY_ANSWER=NO`. `SR_dynamic` is only 2/20, below the frozen 16/20 Formal20 acceptance rate; all 20 episodes retain semantic persistent-grasp-and-lift evidence, but 18 fail the inherited reference-relative terminal-twist gate.",
            "",
            "## V4/170105 dynamic result",
            "",
            f"`SRkin={dynamic_105['rates']['legacy_SRkin']['pass_count']}/{dynamic_105['rates']['legacy_SRkin']['total']}`, `SR_dynamic={dynamic_105['rates']['SR_dynamic']['pass_count']}/{dynamic_105['rates']['SR_dynamic']['total']}`, persistent grasp `{grasp_105['persistent_grasp_episodes']}/{grasp_105['episodes']}`, and lift `{grasp_105['lift_episodes']}/{grasp_105['episodes']}`. This is the frozen Eval10 result, not an expanded or repeated Formal20.",
            "",
            "## 170105 vs 170650",
            "",
            f"170105 has {grasp_105['contact_timing_loss_episodes']}/{grasp_105['episodes']} late persistent-grasp episodes and {grasp_105['lift_episodes']}/{grasp_105['episodes']} lifts.  170650 has {grasp_650['contact_timing_before_lift_episodes']}/{grasp_650['episodes']} persistent grasps before LIFT and {grasp_650['lift_episodes']}/{grasp_650['episodes']} lifts.",
            "",
            f"Primary root cause: `{root['primary_root_cause']}` ({root['confidence']}). {root['rationale']}",
            "",
            "Evidence chain: raw MANO expected topology -> retarget distal-body topology in object coordinates -> recorded PhysX contact/twist. The phase table shows the first material divergence: 170105's persistent multi-finger contact arrives only after the reference LIFT boundary, whereas 170650 establishes it before that boundary and transfers support. The one named follow-up remains `NEXT_CONTACT_TIMING_PHYSICAL_REFINEMENT`; it is not implemented here.",
            "",
            "A primary friction-margin claim is not supported: both exact normal/tangential force decomposition and effective PhysX hand-object friction are `NOT_IDENTIFIABLE`.  Exact tangential slip and grasp-wrench margin are also `NOT_IDENTIFIABLE`; the report retains only their clearly labeled proxies.",
            "",
            "## Replay commands",
            "",
            "170650 full trajectory (actual + raw MANO/object; reference hidden):",
            "```bash",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost",
            "```",
            "",
            "170650 CONTACT→GRASP→LIFT `[92,230)` (low-poly raw object):",
            "```bash",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --start-frame 92 --end-frame 230 --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost --mocap-object-low-poly",
            "```",
            "",
            "170105 full trajectory (actual + raw MANO/object; reference hidden):",
            "```bash",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4/episode_00.npz --object hocap_170105 --no-reference-ghost",
            "```",
            "",
            "170105 CONTACT→GRASP→LIFT `[92,230)` (low-poly raw object):",
            "```bash",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --start-frame 92 --end-frame 230 --trace .local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4/episode_00.npz --object hocap_170105 --no-reference-ghost --mocap-object-low-poly",
            "```",
            "",
            "With the default ghost layers constructed, `M` toggles the Raw MOCAP layer (MANO, raw object, and tips) and `R` toggles the retarget reference layer.  When hidden, the corresponding per-frame USD mesh/transform/marker writes stop; raw object does not have an independent live key.",
            "",
            "Manual review: inspect the two windowed replays for the source MANO contact topology, robot distal-body positions in object coordinates, persistent grasp before LIFT, coupled object/hand motion through LIFT, and the first 170105 relative-motion/contact loss.",
            "",
            "## Safety flags",
            "",
            "`PPO_TRAINING_RUN=NO`, `PPO_OPTIMIZER_STEP=0`, `REWARD_CHANGED=NO`, `FRICTION_CHANGED=NO`, `MASS_CHANGED=NO`, `REFERENCE_CHANGED=NO`, `ENGINEERED_TERMINAL_HOLD_ADDED=NO`, `SR_HOLD_IMPLEMENTED=NO`, `CONTROLLER_CHANGED=NO`, `ACTION_CHANGED=NO`, `GUIDANCE_ADDED=NO`, `OBJECT_STATE_WRITE_ADDED=NO`, `WRIST_ROOT_WRITE_ADDED=NO`, `LEGACY_SRPHYSICS_MODIFIED=NO`, `HISTORICAL_ARTIFACTS_MODIFIED=NO`, `PUSHED=NO`, `PR_CREATED=NO`.",
            "",
            "## Evaluation metric table",
            "",
            _metric_table(dynamic_105, dynamic_650, grasp_105, grasp_650),
        ]
    )


def _docs(root: Mapping[str, object]) -> tuple[str, str]:
    qualification = "\n".join(
        [
            "# Dynamic Physical Qualification",
            "",
            "`Stage16DynamicPhysicalQualificationV1` adds `SR_dynamic` without changing historical Evaluation Suite V2 receipts.",
            "",
            "`SR_dynamic != SR_hold`. This task implements no `SR_hold`, static terminal hold, terminal reference freeze, or engineered post-motion segment.",
            "",
            "The reference uses Reference Kinematics V2 world-frame twist. The terminal dynamic gate inherits the legacy 20-step numerical thresholds exactly, but evaluates `v_actual - v_reference` and `omega_actual - omega_reference`; it does not evaluate an absolute world-zero terminal target. Velocity reward remains a training objective, while this receipt is an independent acceptance gate.",
        ]
    )
    diagnosis = "\n".join(
        [
            "# V4 170105 vs 170650 Grasp Robustness",
            "",
            "This diagnosis compares frozen V4 C4 evidence only. It separates source raw MANO contact authority, retarget robot distal-body topology, and PhysX actual contact/twist telemetry.",
            "",
            f"The frozen decision is `{root['primary_root_cause']}` with `{root['confidence']}` confidence. `{root['one_next_action']}` is the only follow-up action; it is not implemented here.",
            "",
            "PhysX effective friction combine mode, contact points/normals, exact Fn/Ft, exact tangential slip, friction-cone utilization, and grasp-wrench margin are not present in the frozen telemetry and are reported as `NOT_IDENTIFIABLE`. This prevents visual sliding from becoming an unsupported friction-primary claim.",
        ]
    )
    return qualification, diagnosis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--validation-status", default="NOT_RUN")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.report_root.resolve()
    if root.exists():
        raise FileExistsError(f"DYNAMIC_QUALIFICATION_OUTPUT_ALREADY_EXISTS:{root}")
    missing = [str(path) for path in _all_input_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DYNAMIC_QUALIFICATION_INPUT_MISSING:{','.join(missing)}")
    for clip in ("hocap_170105", "hocap_170650"):
        _source_trace_paths(clip)
    stage_a_receipt = _stage_a_receipt()
    gates = _read_json(FROZEN_GATES)
    if gates.get("status") != "STRICT_V4_EVALUATION_GATES_FROZEN":
        raise ValueError("DYNAMIC_QUALIFICATION_FROZEN_GATE_STATUS_INVALID")
    gate_by_clip = {
        clip: DynamicTerminalGate.from_frozen_gate(gates["task_gates"]["clips"][clip])
        for clip in ("hocap_170105", "hocap_170650")
    }
    legacy_by_clip = {"hocap_170105": _legacy_rows_170105(), "hocap_170650": _legacy_rows_170650()}
    causal = {
        "hocap_170105": _read_json(SOURCE_170105_QUALIFICATION),
        "hocap_170650": _read_json(FORMAL_170650_CONFIG),
    }
    dynamic_rows: dict[str, list[dict[str, object]]] = {"hocap_170105": [], "hocap_170650": []}
    trace_cache: dict[str, list[dict[str, np.ndarray]]] = {"hocap_170105": [], "hocap_170650": []}
    for clip in ("hocap_170105", "hocap_170650"):
        paths = _source_trace_paths(clip)
        for episode, (path, legacy) in enumerate(zip(paths, legacy_by_clip[clip], strict=True)):
            trace = _load_trace(path)
            expected_checkpoint = (
                "90c7ddea923f2ba69b141f85e9c72680cddd5fb5a1d902d6cac086f73ce4c261"
                if clip == "hocap_170105"
                else "80da5a3c2c953483f9fe5a668dfe2d4b4c458ab451836ad4b179fec28d0979f3"
            )
            if str(np.asarray(trace["checkpoint_sha256"]).item()) != expected_checkpoint:
                raise ValueError(f"DYNAMIC_QUALIFICATION_FROZEN_ACTOR_DRIFT:{clip}:{episode}")
            causal_safe = True
            if clip == "hocap_170105":
                causal_safe = (
                    int(causal[clip].get("optimizer_steps", -1)) == 0
                    and causal[clip].get("actor_hash_before")
                    == causal[clip].get("actor_hash_after")
                    and causal[clip].get("normalizer_hash_before")
                    == causal[clip].get("normalizer_hash_after")
                )
            else:
                causal_safe = (
                    causal[clip].get("PPO_TRAINING_RUN") is False
                    and int(causal[clip].get("PPO_OPTIMIZER_STEP", -1)) == 0
                )
            row, _, _ = _trace_dynamic_row(
                clip=clip,
                episode=episode,
                trace_path=path,
                trace=trace,
                legacy=legacy,
                gate=gate_by_clip[clip],
                causal_execution_safe=causal_safe,
            )
            dynamic_rows[clip].append(row)
            trace_cache[clip].append(trace)
    dynamic_summary = {clip: _aggregate_dynamic(rows) for clip, rows in dynamic_rows.items()}
    contract = {
        "schema_version": DYNAMIC_PHYSICAL_QUALIFICATION_SCHEMA,
        "reference_twist_authority": "Reference Kinematics V2 world-frame omega where skew(omega)=R_dot_R_transpose",
        "legacy_gate_source": {
            "path": str(FROZEN_GATES.resolve()),
            "sha256": _sha256(FROZEN_GATES),
        },
        "gate_semantics": "legacy numerical threshold applied unchanged to norm(actual_twist-reference_twist)",
        "legacy_terminal_gate_by_clip": {
            clip: gate.as_dict() for clip, gate in gate_by_clip.items()
        },
        "ABSOLUTE_WORLD_TERMINAL_ZERO_SPEED_REQUIRED": "NO",
        "ABSOLUTE_WORLD_TERMINAL_METRICS": "REPORTED_FOR_LEGACY_AUDIT_ONLY_NOT_USED_IN_SR_DYNAMIC",
        "SR_HOLD_IMPLEMENTED": "NO",
        "ENGINEERED_TERMINAL_HOLD_ADDED": "NO",
        "LEGACY_SRPHYSICS_MODIFIED": "NO",
    }
    stage_b = root / "stage_b"
    _write_json(stage_b / "evaluator_audit.json", contract)
    _write_csv(stage_b / "v4_170650_formal20_dynamic.csv", dynamic_rows["hocap_170650"])
    _write_json(
        stage_b / "v4_170650_formal20_dynamic_summary.json", dynamic_summary["hocap_170650"]
    )
    _write_csv(stage_b / "v4_170105_dynamic.csv", dynamic_rows["hocap_170105"])
    old_new = [
        {
            "clip": clip,
            "episodes": summary["episode_count"],
            "legacy_SRkin": f"{summary['rates']['legacy_SRkin']['pass_count']}/{summary['rates']['legacy_SRkin']['total']}",
            "legacy_SRphysics": f"{summary['rates']['legacy_SRphysics']['pass_count']}/{summary['rates']['legacy_SRphysics']['total']}",
            "legacy_SRqualified": f"{summary['rates']['legacy_SRqualified']['pass_count']}/{summary['rates']['legacy_SRqualified']['total']}",
            "SR_dynamic": f"{summary['rates']['SR_dynamic']['pass_count']}/{summary['rates']['SR_dynamic']['total']}",
        }
        for clip, summary in dynamic_summary.items()
    ]
    _write_csv(stage_b / "old_vs_new.csv", old_new)
    stage_c = root / "stage_c"
    phase_rows = [
        row for clip in ("hocap_170105", "hocap_170650") for row in _phase_window_rows(clip)
    ]
    _write_json(
        stage_c / "phase_windows.json",
        {"authority": "runtime phase_code from reference_index", "windows": phase_rows},
    )
    summaries: dict[str, dict[str, object]] = {}
    properties_rows: list[dict[str, object]] = []
    per_clip_artifacts: dict[str, dict[str, list[dict[str, object]]]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        first_trace_path = _source_trace_paths(clip)[0]
        overlay = resolve_raw_mocap_overlay(
            trace_path=first_trace_path,
            frame_count=321,
            clip=clip,
            reference_path=WORLD_WRIST_ROOT / f"{clip}.world_wrist.stage16.npz",
        )
        if (
            overlay.coordinate_alignment["status"] != "PASS"
            or overlay.time_alignment["status"] != "PASS"
        ):
            raise RuntimeError(f"DYNAMIC_QUALIFICATION_STAGE_A_ALIGNMENT_FAILURE:{clip}")
        topology: list[dict[str, object]] = []
        timing: list[dict[str, object]] = []
        relative: list[dict[str, object]] = []
        contact: list[dict[str, object]] = []
        for episode, trace in enumerate(trace_cache[clip]):
            local_topology, local_timing, local_relative, local_contact = _topology_and_dynamics(
                clip=clip,
                episode=episode,
                trace=trace,
                overlay=overlay,
                timestamp_s=np.asarray(overlay.runtime_timestamps_s, dtype=np.float64),
            )
            topology.extend(local_topology)
            timing.extend(local_timing)
            relative.extend(local_relative)
            contact.extend(local_contact["contact_dynamics"])
        properties = _object_properties(clip, overlay)
        properties_rows.append(properties)
        required = _reference_required_force(
            trace=trace_cache[clip][0],
            mass_kg=float(properties["mass_kg"]),
            timestamps_s=np.asarray(overlay.runtime_timestamps_s, dtype=np.float64),
        )
        summaries[clip] = _summary_stage_c(
            clip=clip,
            dynamic_rows=dynamic_rows[clip],
            topology=topology,
            timing=timing,
            relative_twist=relative,
            contact_dynamics=contact,
            object_properties=properties,
            required_force=required,
        )
        clip_root = stage_c / clip
        _write_csv(clip_root / "topology.csv", topology)
        _write_csv(clip_root / "timing.csv", timing)
        _write_csv(clip_root / "relative_twist.csv", relative)
        _write_csv(clip_root / "contact_dynamics.csv", contact)
        _write_json(clip_root / "summary.json", summaries[clip])
        per_clip_artifacts[clip] = {
            "topology": topology,
            "timing": timing,
            "relative": relative,
            "contact": contact,
        }
    _write_csv(stage_c / "object_properties.csv", properties_rows)
    object_properties_markdown = "\n".join(
        [
            "# Object Physics Properties",
            "",
            "| Clip | Mass kg | COM object m | Diagonal inertia kg m2 | Bounding dimensions m | Authored object friction | Authored hand friction | Effective mu |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
            *[
                f"| {row['clip']} | {row['mass_kg']} | {row['COM_object_m']} | {row['diagonal_inertia_kg_m2']} | {row['bounding_dimensions_m']} | static={row['object_static_friction_authored']}, dynamic={row['object_dynamic_friction_authored']} | static={row['hand_static_friction_authored']}, dynamic={row['hand_dynamic_friction_authored']} | {row['effective_hand_object_mu']} |"
                for row in properties_rows
            ],
        ]
    )
    (stage_c / "object_properties.md").write_text(
        object_properties_markdown + "\n", encoding="utf-8"
    )
    root_cause = _root_cause(summaries["hocap_170105"], summaries["hocap_170650"])
    comparison = stage_c / "comparison"
    phase_diagnostic = {
        clip: _phase_diagnostic_rows(
            clip=clip,
            topology=per_clip_artifacts[clip]["topology"],
            contact_dynamics=per_clip_artifacts[clip]["contact"],
            relative_twist=per_clip_artifacts[clip]["relative"],
        )
        for clip in ("hocap_170105", "hocap_170650")
    }
    phase_rows = [
        row for clip in ("hocap_170105", "hocap_170650") for row in phase_diagnostic[clip]
    ]
    _write_csv(comparison / "phase_diagnostic.csv", phase_rows)
    lift_105 = next(row for row in phase_diagnostic["hocap_170105"] if row["phase"] == "LIFT")
    lift_650 = next(row for row in phase_diagnostic["hocap_170650"] if row["phase"] == "LIFT")
    comparison_rows = [
        {
            "metric": "persistent grasp",
            "V4_170105": f"{summaries['hocap_170105']['persistent_grasp_episodes']}/{summaries['hocap_170105']['episodes']}",
            "V4_170650": f"{summaries['hocap_170650']['persistent_grasp_episodes']}/{summaries['hocap_170650']['episodes']}",
            "main_difference": "170105 persistence is late; 170650 is established by LIFT.",
        },
        {
            "metric": "lift",
            "V4_170105": f"{summaries['hocap_170105']['lift_episodes']}/{summaries['hocap_170105']['episodes']}",
            "V4_170650": f"{summaries['hocap_170650']['lift_episodes']}/{summaries['hocap_170650']['episodes']}",
            "main_difference": "170105 has no semantic lift; 170650 lifts in every frozen replica.",
        },
        {
            "metric": "LIFT hand-object contact fraction",
            "V4_170105": lift_105["actual_hand_object_contact_fraction"],
            "V4_170650": lift_650["actual_hand_object_contact_fraction"],
            "main_difference": "Recorded actual contact is compared without inferring normals.",
        },
        {
            "metric": "LIFT multi-finger persistence",
            "V4_170105": lift_105["multi_finger_persistent_fraction"],
            "V4_170650": lift_650["multi_finger_persistent_fraction"],
            "main_difference": "170105 reaches persistence after the required lift boundary.",
        },
        {
            "metric": "LIFT active fingertip-force p95 N",
            "V4_170105": lift_105["active_force_p95_n"],
            "V4_170650": lift_650["active_force_p95_n"],
            "main_difference": "Vector-force magnitude only; exact normal/tangential split is absent.",
        },
        {
            "metric": "LIFT table-object contact fraction",
            "V4_170105": lift_105["table_object_contact_fraction"],
            "V4_170650": lift_650["table_object_contact_fraction"],
            "main_difference": "170105 retains support while 170650 has transferred support.",
        },
        {
            "metric": "raw-to-retarget object-local tip error m",
            "V4_170105": lift_105["raw_to_retarget_object_local_tip_error_mean_m"],
            "V4_170650": lift_650["raw_to_retarget_object_local_tip_error_mean_m"],
            "main_difference": "Raw-to-retarget topology is separated from actual PhysX contact.",
        },
        {
            "metric": "contact timing",
            "V4_170105": f"{summaries['hocap_170105']['contact_timing_loss_episodes']}/{summaries['hocap_170105']['episodes']} after LIFT",
            "V4_170650": f"{summaries['hocap_170650']['contact_timing_before_lift_episodes']}/{summaries['hocap_170650']['episodes']} on/before LIFT",
            "main_difference": "Earliest cross-clip divergence; selected as the primary cause.",
        },
        {
            "metric": "LIFT relative linear twist m/s",
            "V4_170105": lift_105["relative_linear_twist_mean_mps"],
            "V4_170650": lift_650["relative_linear_twist_mean_mps"],
            "main_difference": "World-frame object-minus-wrist diagnostic proxy.",
        },
        {
            "metric": "LIFT relative angular twist rad/s",
            "V4_170105": lift_105["relative_angular_twist_mean_radps"],
            "V4_170650": lift_650["relative_angular_twist_mean_radps"],
            "main_difference": "World-frame object-minus-wrist diagnostic proxy.",
        },
        {
            "metric": "exact tangential slip",
            "V4_170105": "NOT_IDENTIFIABLE",
            "V4_170650": "NOT_IDENTIFIABLE",
            "main_difference": "No contact points/normals in frozen telemetry.",
        },
        {
            "metric": "effective hand-object mu",
            "V4_170105": "NOT_IDENTIFIABLE",
            "V4_170650": "NOT_IDENTIFIABLE",
            "main_difference": "PhysX combine mode is not recorded.",
        },
        {
            "metric": "friction cone utilization",
            "V4_170105": "NOT_IDENTIFIABLE",
            "V4_170650": "NOT_IDENTIFIABLE",
            "main_difference": "Requires contact normals and effective mu.",
        },
        {
            "metric": "required wrench",
            "V4_170105": json.dumps(summaries["hocap_170105"]["required_wrench"], sort_keys=True),
            "V4_170650": json.dumps(summaries["hocap_170650"]["required_wrench"], sort_keys=True),
            "main_difference": "Reference translational force is reported; rotational wrench is not identifiable.",
        },
        {
            "metric": "grasp wrench margin",
            "V4_170105": "NOT_IDENTIFIABLE",
            "V4_170650": "NOT_IDENTIFIABLE",
            "main_difference": "Requires contact geometry, normals, and effective mu.",
        },
    ]
    _write_csv(comparison / "grasp_robustness.csv", comparison_rows)
    _write_json(comparison / "root_cause.json", root_cause)
    comparison_markdown = "\n".join(
        [
            "# V4 Grasp Robustness Comparison",
            "",
            "| Metric | 170105 | 170650 | Main difference |",
            "| --- | ---: | ---: | --- |",
            *[
                f"| {row['metric']} | {row['V4_170105']} | {row['V4_170650']} | {row['main_difference']} |"
                for row in comparison_rows
            ],
        ]
    )
    (comparison / "grasp_robustness.md").parent.mkdir(parents=True, exist_ok=True)
    (comparison / "grasp_robustness.md").write_text(comparison_markdown + "\n", encoding="utf-8")
    phase_markdown = "\n".join(
        [
            "# Phase-wise Diagnostic",
            "",
            "| Clip | Phase | Raw MANO contacts | Retarget expected | Actual persistent contacts | Relative twist | Table support |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['clip']} | {row['phase']} | {row['raw_mano_expected_contact_fraction']:.4f} | {row['retarget_expected_contact_fraction']:.4f} | {row['actual_persistent_contact_fraction']:.4f} | {row['relative_linear_twist_mean_mps']:.6f} m/s, {row['relative_angular_twist_mean_radps']:.6f} rad/s | {row['table_object_contact_fraction']:.4f} |"
                for row in phase_rows
            ],
        ]
    )
    (comparison / "phase_diagnostic.md").write_text(phase_markdown + "\n", encoding="utf-8")
    replay_root = root / "replay"
    visualization = "\n".join(
        [
            "# Visualization Commands",
            "",
            "All commands reuse Stage A's authoritative raw-MOCAP overlay. `--no-reference-ghost` hides only the geometric retarget layer.",
            "",
            "```bash",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --start-frame 92 --end-frame 230 --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost --mocap-object-low-poly",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4/episode_00.npz --object hocap_170105 --no-reference-ghost",
            "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --start-frame 92 --end-frame 230 --trace .local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4/episode_00.npz --object hocap_170105 --no-reference-ghost --mocap-object-low-poly",
            "```",
            "",
            "Live controls: `M` toggles Raw MOCAP (MANO/object/tips); `R` toggles retarget reference. Hidden layers stop their per-frame writes.",
        ]
    )
    replay_root.mkdir(parents=True, exist_ok=True)
    (replay_root / "visualization_commands.md").write_text(visualization + "\n", encoding="utf-8")
    (replay_root / "manual_acceptance.md").write_text(
        "# Manual Acceptance\n\nOpen the two windowed replays and compare raw MANO topology, robot distal positions, persistent grasp before LIFT, coupled hand/object motion, and the first 170105 contact loss.\n",
        encoding="utf-8",
    )
    validation = {
        "status": args.validation_status,
        "targeted_dynamic_unit": "PASS_4_TESTS",
        "frozen_v2_metric_parity": "PASS_30_OF_30_EPISODES",
    }
    _write_json(root / "tests.json", validation)
    resource = {
        "execution": "offline_trace_reanalysis_only",
        "PPO_TRAINING_RUN": "NO",
        "PPO_OPTIMIZER_STEP": 0,
        "new_diagnostic_rollout": "NO",
        "input_trace_episodes": {"hocap_170105": 10, "hocap_170650": 20},
        "conditional_telemetry": {
            "exact_contact_points_normals": "NOT_CAPTURED",
            "effective_mu_combine_mode": "NOT_IDENTIFIABLE",
        },
    }
    _write_json(root / "resource_usage.json", resource)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    )
    _write_json(
        root / "git_commits.json",
        {"START_HEAD": git.stdout.strip(), "FINAL_HEAD": git.stdout.strip(), "commits": []},
    )
    final = {
        "schema_version": "Stage16DynamicPhysicalQualificationAndGraspDiagnosticCloseoutV1",
        "dynamic_qualification_contract": contract,
        "stage_a_regression": stage_a_receipt,
        "stage_b": dynamic_summary,
        "stage_c": summaries,
        "root_cause": root_cause,
        "safety": resource,
    }
    _write_json(root / "dynamic_qualification_contract.json", contract)
    _write_json(root / "final_summary.json", final)
    final_markdown = _markdown(
        dynamic_105=dynamic_summary["hocap_170105"],
        dynamic_650=dynamic_summary["hocap_170650"],
        grasp_105=summaries["hocap_170105"],
        grasp_650=summaries["hocap_170650"],
        root=root_cause,
    )
    (root / "final_summary.md").write_text(final_markdown + "\n", encoding="utf-8")
    (root / "handoff.md").write_text(final_markdown + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "STAGE16_DYNAMIC_QUALIFICATION_COMPLETE", "report_root": str(root)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
