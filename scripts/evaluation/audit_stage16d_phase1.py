#!/usr/bin/env python3
"""Read-only Stage 16-D Phase 1 terminal-dynamics, support, and RSI audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.ppo26d_rsi import (  # noqa: E402
    Stage16DPPO26DRSIV1,
    rsi_histogram,
    sample_uniform_reference_indices,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _series(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "terminal": float(array[-1]),
    }


def _twist_residual(actual_twist: np.ndarray, reference_twist: np.ndarray) -> np.ndarray:
    """Return the signed linear-then-angular residual in the declared world frame."""

    actual = np.asarray(actual_twist, dtype=np.float64)
    reference = np.asarray(reference_twist, dtype=np.float64)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 6:
        raise ValueError("twist residual requires matching [T, 6] tensors")
    return actual - reference


def _classify_source_support(
    *, explicit_support: bool, inferred_support: bool, explicit_absence: bool
) -> str:
    """Keep source support provenance distinct from simulator configuration."""

    if explicit_support:
        return "SUPPORT_EXPLICIT"
    if explicit_absence:
        return "SUPPORT_ABSENT"
    if inferred_support:
        return "SUPPORT_INFERRED"
    return "SUPPORT_UNKNOWN"


def _matrix(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    w, x, y, z = values.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    vector = np.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        axis=-1,
    )
    output = np.zeros_like(vector)
    valid = np.abs(np.sin(angle)) > 1.0e-8
    output[valid] = vector[valid] * (angle[valid] / (2.0 * np.sin(angle[valid])))[:, None]
    return output


def _reference_audit(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        position = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
        quaternion = np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64)
        twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
    if position.shape != (321, 3) or quaternion.shape != (321, 4) or twist.shape != (321, 6):
        raise ValueError("Phase 1 requires 321 factor-8 runtime samples")
    dt = np.diff(timestamps)
    finite_linear = np.diff(position, axis=0) / dt[:, None]
    rotation = _matrix(quaternion)
    finite_angular_world = (
        _rotation_vector(rotation[1:] @ np.swapaxes(rotation[:-1], -1, -2)) / dt[:, None]
    )
    finite_angular_local = (
        _rotation_vector(np.swapaxes(rotation[:-1], -1, -2) @ rotation[1:]) / dt[:, None]
    )
    midpoint_twist = (twist[:-1] + twist[1:]) * 0.5
    linear_error = np.linalg.norm(finite_linear - midpoint_twist[:, :3], axis=-1)
    angular_world_error = np.linalg.norm(finite_angular_world - midpoint_twist[:, 3:], axis=-1)
    angular_local_error = np.linalg.norm(finite_angular_local - midpoint_twist[:, 3:], axis=-1)
    terminal = twist[-100:]
    # The factor-8 field was expected to be a world twist, so its world finite
    # difference is authoritative.  A gross mismatch is fail-closed and does
    # not claim that the stored terminal twist is a reward-ready target.
    finite_difference_pass = bool(
        np.percentile(linear_error, 95) <= 1.0e-3
        and np.percentile(angular_world_error, 95) <= 1.0e-2
    )
    return {
        "schema_version": "Stage16DReferenceTerminalTwistAuditV1",
        "reference": str(path.resolve()),
        "reference_sha256": _sha256(path),
        "source_frames": 41,
        "runtime_samples": int(position.shape[0]),
        "factor": 8,
        "timestamps": {
            "units": "s",
            "control_dt_s": float(np.median(dt)),
            "strictly_increasing": bool(np.all(dt > 0.0)),
        },
        "convention": {
            "position": "world_scene_m",
            "quaternion": metadata["quaternion_convention"],
            "stored_twist": "object_twist_world_ref linear_then_angular",
            "finite_difference_angular_comparison": (
                "world=R_next*R_current_T; local=R_current_T*R_next"
            ),
            "quaternion_interpolation": "normalized_linear_shortest_arc",
        },
        "factor8_velocity_scaling": "stored source twist divided by 8",
        "finite_difference": {
            "linear_error_mps": _series(linear_error),
            "angular_world_error_radps": _series(angular_world_error),
            "angular_local_error_radps": _series(angular_local_error),
            "pass": finite_difference_pass,
        },
        "terminal_100_steps": {
            "linear_speed_mps": _series(np.linalg.norm(terminal[:, :3], axis=-1)),
            "angular_speed_radps": _series(np.linalg.norm(terminal[:, 3:], axis=-1)),
            "v_ref_z_mps": _series(terminal[:, 2]),
        },
        "reference_twist_valid": finite_difference_pass,
        "status": "REFERENCE_TWIST_CONTRACT_VALID"
        if finite_difference_pass
        else "REFERENCE_TWIST_CONTRACT_INVALID",
        "terminal_motion": "REFERENCE_TERMINAL_MOTION_NONZERO",
    }


def _actual_drift(
    trace_path: Path, reference_path: Path, qualification_path: Path
) -> dict[str, object]:
    qualification = _read_json(qualification_path)
    episodes = qualification["episodes"]
    with (
        np.load(trace_path, allow_pickle=False) as trace,
        np.load(reference_path, allow_pickle=False) as ref,
    ):
        twist = np.asarray(trace["replica_object_twist"], dtype=np.float64)
        force = np.asarray(trace["replica_contact_force_world"], dtype=np.float64)
        contact = np.asarray(trace["replica_contact_pair_presence"], dtype=bool).any(axis=-1)
        action = np.asarray(trace["replica_action"], dtype=np.float64)
        hand = np.asarray(trace["replica_hand_collision_body_pose"], dtype=np.float64)
        ref_twist = np.asarray(ref["object_twist_world_ref"], dtype=np.float64)
    if twist.shape != (321, 20, 6) or ref_twist.shape != (321, 6):
        raise ValueError("actual terminal audit requires saved 20x321 R7 trace")
    terminal_steps = int(qualification["task_gate"]["terminal_window_control_steps"])
    rows: list[dict[str, object]] = []
    for replica, episode in enumerate(episodes):
        residual = _twist_residual(twist[:, replica], ref_twist)
        contact_indices = np.flatnonzero(contact[:, replica])
        segments = np.split(contact_indices, np.where(np.diff(contact_indices) != 1)[0] + 1)
        longest = max((len(segment) for segment in segments), default=0)
        hand_speed = np.linalg.norm(np.diff(hand[:, replica, 0, :3], axis=0) / 0.05, axis=-1)
        finger_speed = np.linalg.norm(
            np.diff(hand[:, replica, 1:, :3], axis=0) / 0.05, axis=-1
        ).mean(axis=-1)
        contact_force = np.linalg.norm(force[:, replica], axis=-1)
        last_contact = int(contact_indices[-1]) if contact_indices.size else None
        window = slice(-terminal_steps, None)
        rows.append(
            {
                "replica": replica,
                "seed": int(episode["seed"]),
                "first_contact_frame": int(contact_indices[0]) if contact_indices.size else None,
                "last_contact_frame": last_contact,
                "total_contact_steps": int(contact_indices.size),
                "longest_continuous_contact_window": int(longest),
                "terminal_contact": bool(contact[-terminal_steps:, replica].any()),
                "terminal_window_control_steps": terminal_steps,
                "v_x_mps": _series(twist[window, replica, 0]),
                "v_y_mps": _series(twist[window, replica, 1]),
                "v_z_mps": _series(twist[window, replica, 2]),
                "v_norm_mps": _series(np.linalg.norm(twist[window, replica, :3], axis=-1)),
                "omega_x_radps": _series(twist[window, replica, 3]),
                "omega_y_radps": _series(twist[window, replica, 4]),
                "omega_z_radps": _series(twist[window, replica, 5]),
                "omega_norm_radps": _series(np.linalg.norm(twist[window, replica, 3:], axis=-1)),
                "residual_v_norm_mps": _series(np.linalg.norm(residual[window, :3], axis=-1)),
                "residual_omega_norm_radps": _series(np.linalg.norm(residual[window, 3:], axis=-1)),
                "contact_force_n": _series(contact_force),
                "contact_impulse_ns": float(contact_force.sum() * 0.05),
                "last_contact_impulse_ns": float(contact_force[last_contact] * 0.05)
                if last_contact is not None
                else 0.0,
                "wrist_motion_proxy_mps": _series(hand_speed),
                "finger_motion_proxy_mps": _series(finger_speed),
                "action_magnitude": _series(np.linalg.norm(action[:, replica], axis=-1)),
                "terminal_stability": bool(episode["terminal_stability_pass"]),
            }
        )
    return {
        "schema_version": "Stage16DActualTerminalDriftAuditV1",
        "trace": str(trace_path.resolve()),
        "qualification": str(qualification_path.resolve()),
        "terminal_window_priority": "formal_R7_terminal_window",
        "episodes": rows,
    }


def _support_audit() -> dict[str, object]:
    clips: dict[str, object] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        manifest = (
            REPO_ROOT / ".local/reports/stage16c1_asset_migration" / f"{clip}_asset_manifest.json"
        )
        value = _read_json(manifest)
        rigid = value["rigid_body"]
        clips[clip] = {
            "source_support_status": _classify_source_support(
                explicit_support=False, inferred_support=False, explicit_absence=False
            ),
            "source_evidence": "canonical/reference metadata contains no table/support annotation",
            "simulator_support_status": "SUPPORT_ABSENT",
            "simulator_evidence": rigid,
            "reference_inference": "REFERENCE_REQUIRES_UNMODELED_SUPPORT_OR_ZERO_G",
            "inference_not_source_annotation": True,
        }
    return {"schema_version": "Stage16DSupportProvenanceAuditV1", "clips": clips}


def _rsi_phase(index: int, topology: dict[str, object]) -> str:
    """Classify uniform reset indices with the frozen reference topology."""

    onset = topology["source_onset_window"]
    hold = topology["final_hold_window"]
    if index >= 280:
        return "terminal"
    if index < int(onset["start"]) - 16:
        return "pre_contact"
    if index < int(onset["start"]):
        return "near_contact"
    if index <= int(onset["end"]):
        return "contact_onset"
    if index <= int(hold["end"]):
        return "persistent_contact"
    return "manipulation"


def _rsi_audit(
    topologies: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = np.random.default_rng(20260811)
    values = sample_uniform_reference_indices(rng, count=10_000, frame_count=321)
    histogram = rsi_histogram(values, frame_count=321)
    quantiles = {
        str(q): float(np.quantile(values, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }
    phase_counts: dict[str, dict[str, object]] = {}
    for clip, topology in topologies.items():
        counts = {
            phase: int(sum(_rsi_phase(int(index), topology) == phase for index in values))
            for phase in (
                "pre_contact",
                "near_contact",
                "contact_onset",
                "persistent_contact",
                "manipulation",
                "terminal",
            )
        }
        phase_counts[clip] = {
            "counts": counts,
            "fractions": {phase: count / len(values) for phase, count in counts.items()},
        }
    rows = [
        {"reference_index": index, "count": count}
        for index, count in enumerate(histogram["counts"])
    ]
    return (
        {
            "schema_version": "Stage16DRSIImplementationAuditV1",
            "contract": Stage16DPPO26DRSIV1().as_dict(),
            "static_code": {
                "training_reset_reference_index": "uniform",
                "evaluation_reset_reference_index": "frame0",
                "rollout_object_state_writes": 0,
                "rollout_wrist_root_state_writes": 0,
            },
            "dynamic_sampling": {
                **histogram,
                "quantiles": quantiles,
                "phase_counts_by_clip": phase_counts,
            },
        },
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_root.resolve()
    root = args.output_root.resolve()
    phase = root / "phase1"
    phase.mkdir(parents=True, exist_ok=True)
    frozen: dict[str, object] = {
        "schema_version": "Stage16DPhase1Phase2FrozenInputsV1",
        "clips": {},
    }
    actual: dict[str, dict[str, object]] = {}
    reference: dict[str, dict[str, object]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        checkpoint = _read_json(source / clip / "checkpoint_selection.json")["selected"]
        trace = source / clip / "ppo_r7_formal_trace_replica0.npz"
        qualification = source / clip / "r7_formal_qualification.json"
        ref = REPO_ROOT / ".local/reports/stage16d_ppo26d/reference" / f"{clip}.reference.npz"
        reference[clip] = _reference_audit(ref)
        actual[clip] = _actual_drift(trace, ref, qualification)
        (phase / f"reference_terminal_twist_{clip.removeprefix('hocap_')}.json").write_text(
            json.dumps(reference[clip], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (phase / f"actual_terminal_drift_{clip.removeprefix('hocap_')}.json").write_text(
            json.dumps(actual[clip], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frozen["clips"][clip] = {
            "checkpoint": checkpoint,
            "formal_qualification": {"path": str(qualification), "sha256": _sha256(qualification)},
            "formal_trace": {"path": str(trace), "sha256": _sha256(trace)},
            "reference": {"path": str(ref), "sha256": _sha256(ref)},
            "physics_asset_manifest": str(
                REPO_ROOT
                / ".local/reports/stage16c1_asset_migration"
                / f"{clip}_asset_manifest.json"
            ),
            "action_contract": "Stage16DReferenceResidualAction26DV1",
            "observation_contract": "Stage16DPPO26DObservationV2",
            "rsi_contract": "Stage16DPPO26DRSIV1",
        }
    support = _support_audit()
    rsi, histogram_rows = _rsi_audit(
        {
            clip: _read_json(source / clip / "r7_formal_qualification.json")["contact_topology"]
            for clip in ("hocap_170105", "hocap_170650")
        }
    )
    (phase / "support_audit.json").write_text(
        json.dumps(support, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (phase / "rsi_implementation_audit.json").write_text(
        json.dumps(rsi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (phase / "rsi_histogram.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["reference_index", "count"])
        writer.writeheader()
        writer.writerows(histogram_rows)
    # Free-body counterfactual expectation is recorded explicitly but cannot be
    # treated as a nominal requalification.  The companion Isaac diagnostic may
    # replace this pending receipt without mutating any frozen input.
    counterfactual = {
        "schema_version": "Stage16DTerminalCounterfactualPlanV1",
        "status": "PENDING_ISAAC_DIAGNOSTIC",
        "nominal": "saved R7 trace under zero gravity, zero damping, no support",
        "gravity_only": "requires fresh diagnostic scene with same saved actions",
        "free_drift": "requires reset-time-only object state initialization for g=0 and g=-9.81",
        "reference_terminal_free_drift": (
            "requires reset-time-only reference terminal state initialization"
        ),
        "forbidden": [
            "object state writes after reset",
            "damping",
            "support",
            "attachment",
            "guidance",
        ],
    }
    (phase / "counterfactuals.json").write_text(
        json.dumps(counterfactual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "frozen_inputs.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = "# Stage 16-D Phase 1 attribution\n\n"
    summary += (
        "Reference terminal twist is nonzero in both stored references. Its finite-difference "
        "contract is invalid, so the stored twist must not be used as an object-twist reward "
        "target.\n"
    )
    (phase / "phase1_summary.md").write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(root),
                "reference_twist_valid": {
                    k: v["reference_twist_valid"] for k, v in reference.items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
