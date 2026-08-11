#!/usr/bin/env python3
"""Recompute bounded Stage 16-D terminal attribution against Reference V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"
DEFAULT_V1_CONTINUATION = REPO_ROOT / ".local/reports/stage16d_ppo26d_continuation"
DEFAULT_FROZEN_PHASE1_ROOT = REPO_ROOT / ".local/reports/stage16d_phase1_phase2/phase1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _series(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "final": float(values[-1]),
    }


def _longest_run(values: np.ndarray) -> int:
    current = 0
    longest = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _terminal_rows(
    *, trace_path: Path, reference_path: Path, evaluation_path: Path, gate: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = _json(evaluation_path)
    episodes = evaluation.get("frame_zero")
    if (
        not isinstance(episodes, list)
        or len(episodes) != 20
        or any(int(row["start_reference_index"]) != 0 for row in episodes)
    ):
        raise ValueError("PHASE1_RERUN_REQUIRES_20_FRESH_FRAME_ZERO_EPISODES")
    with (
        np.load(trace_path, allow_pickle=False) as trace,
        np.load(reference_path, allow_pickle=False) as reference,
    ):
        actual = np.asarray(trace["replica_object_twist"], dtype=np.float64)
        contact = np.asarray(trace["replica_contact_pair_presence"], dtype=bool).any(axis=-1)
        action = np.asarray(trace["replica_action"], dtype=np.float64)
        reference_twist = np.asarray(reference["object_twist_world_ref"], dtype=np.float64)
    if actual.shape != (321, 20, 6) or reference_twist.shape != (321, 6):
        raise ValueError("PHASE1_RERUN_TRACE_OR_REFERENCE_SHAPE_INVALID")
    if contact.shape != (321, 20) or action.shape != (321, 20, 26):
        raise ValueError("PHASE1_RERUN_CONTACT_OR_ACTION_SHAPE_INVALID")
    terminal_steps = int(gate["terminal_window_control_steps"])
    terminal = slice(-terminal_steps, None)
    residual = actual - reference_twist[:, None, :]
    rows: list[dict[str, Any]] = []
    for replica, episode in enumerate(episodes):
        terminal_contact_mask = contact[terminal, replica]
        linear = np.linalg.norm(actual[terminal, replica, :3], axis=-1)
        angular = np.linalg.norm(actual[terminal, replica, 3:], axis=-1)
        linear_limit = np.where(
            terminal_contact_mask,
            float(gate["terminal_linear_speed_mps"]),
            float(gate["terminal_free_object_linear_speed_mps"]),
        )
        angular_limit = np.where(
            terminal_contact_mask,
            float(gate["terminal_angular_speed_radps"]),
            float(gate["terminal_free_object_angular_speed_radps"]),
        )
        contact_indices = np.flatnonzero(contact[:, replica])
        required_contact_steps = int(
            np.ceil(float(gate["terminal_required_contact_fraction"]) * terminal_steps)
        )
        terminal_contact = int(terminal_contact_mask.sum()) >= required_contact_steps
        terminal_kinematic = bool(np.all((linear <= linear_limit) & (angular <= angular_limit)))
        residual_linear = np.linalg.norm(residual[terminal, replica, :3], axis=-1)
        residual_angular = np.linalg.norm(residual[terminal, replica, 3:], axis=-1)
        rows.append(
            {
                "replica": replica,
                "seed": int(episode["seed"]),
                "last_contact_frame": int(contact_indices[-1]) if contact_indices.size else None,
                "total_contact_steps": int(contact_indices.size),
                "longest_contact_window": _longest_run(contact[:, replica]),
                "terminal_contact_steps": int(terminal_contact_mask.sum()),
                "terminal_contact": terminal_contact,
                "terminal_kinematic": terminal_kinematic,
                "terminal_stability": bool(terminal_contact and terminal_kinematic),
                "terminal_stability_pass": bool(terminal_contact and terminal_kinematic),
                "actual_v_norm_mps": _series(linear),
                "actual_omega_norm_radps": _series(angular),
                "delta_v_norm_mps": _series(residual_linear),
                "delta_omega_norm_radps": _series(residual_angular),
                "action_norm": _series(np.linalg.norm(action[:, replica], axis=-1)),
            }
        )
    reference_terminal = reference_twist[-100:]
    diagnostic = {
        "reference_terminal_v_norm_mps": _series(
            np.linalg.norm(reference_terminal[:, :3], axis=-1)
        ),
        "reference_terminal_omega_norm_radps": _series(
            np.linalg.norm(reference_terminal[:, 3:], axis=-1)
        ),
        "terminal_contact_rate": sum(bool(row["terminal_contact"]) for row in rows) / len(rows),
        "terminal_stability_rate": sum(bool(row["terminal_stability"]) for row in rows) / len(rows),
        "terminal_window_control_steps": terminal_steps,
    }
    return rows, diagnostic


def _attribution(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    failures = [row for row in rows if not bool(row["terminal_stability"])]
    angular_scale = float(gate["terminal_free_object_angular_speed_radps"])
    linear_scale = float(gate["terminal_free_object_linear_speed_mps"])
    meaningful = [
        row
        for row in failures
        if float(row["delta_v_norm_mps"]["mean"]) >= linear_scale
        or float(row["delta_omega_norm_radps"]["mean"]) >= angular_scale
    ]
    contact_failures = [row for row in failures if not bool(row["terminal_contact"])]
    meaningful_fraction = len(meaningful) / max(len(failures), 1)
    if meaningful_fraction >= 0.5:
        primary = "POLICY_TRACKING_ERROR"
        secondary = "CONTACT_IMPULSE_RESIDUAL"
        confidence = "medium"
    elif contact_failures:
        primary = "CONTACT_LOSS"
        secondary = "POLICY_TRACKING_ERROR"
        confidence = "medium"
    else:
        primary = "UNKNOWN"
        secondary = "ZERO_GRAVITY_VELOCITY_PERSISTENCE"
        confidence = "low"
    return {
        "primary": primary,
        "secondary": secondary,
        "confidence": confidence,
        "failure_episode_count": len(failures),
        "meaningful_residual_definition": {
            "linear_mps": linear_scale,
            "angular_radps": angular_scale,
            "source": "existing free-object terminal-stability velocity scales",
        },
        "meaningful_residual_count": len(meaningful),
        "meaningful_residual_fraction": meaningful_fraction,
        "terminal_contact_failure_count": len(contact_failures),
        "labels": [
            primary,
            secondary,
            "ZERO_GRAVITY_VELOCITY_PERSISTENCE",
            "SUPPORT_MISSING",
        ],
        "reference_twist_contract": "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED",
    }


def _counterfactual_summary(path: Path, *, clip: str, reference_sha256: str) -> dict[str, Any]:
    """Validate one executed V2 counterfactual and retain only its receipt summary."""

    value = _json(path)
    if value.get("clip") != clip:
        raise ValueError(f"counterfactual clip mismatch: {path}")
    frozen = value.get("frozen_inputs")
    if not isinstance(frozen, dict) or frozen.get("reference_sha256") != reference_sha256:
        raise ValueError(f"counterfactual is not bound to the V2 reference: {path}")
    trajectories = value.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 2:
        raise ValueError(f"counterfactual must retain two representative trajectories: {path}")
    compact = []
    for trajectory in trajectories:
        summary = trajectory.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"counterfactual trajectory summary missing: {path}")
        compact.append(
            {
                "label": trajectory["label"],
                "replica": int(trajectory["replica"]),
                "formal_terminal_stability": bool(trajectory["formal_terminal_stability"]),
                "object_z_final_m": float(summary["object_z_m"]["final"]),
                "linear_speed_final_mps": float(summary["linear_speed_mps"]["final"]),
                "angular_speed_final_radps": float(summary["angular_speed_radps"]["final"]),
                "vertical_displacement_m": (
                    None
                    if "vertical_displacement_m" not in summary
                    else float(summary["vertical_displacement_m"])
                ),
                "contact_steps": summary.get("contact_steps"),
                "ended_early": summary.get("ended_early"),
                "final_termination_reason": summary.get("final_termination_reason"),
            }
        )
    return {
        "path": str(path),
        "sha256": _hash(path),
        "case": value["case"],
        "gravity": value["gravity"],
        "initial_state": value.get("initial_state"),
        "same_saved_26d_actions": value.get("same_saved_26d_actions"),
        "physics_contract": value.get("physics_contract"),
        "reset_time_object_state_initialization_calls": value.get(
            "reset_time_object_state_initialization_calls"
        ),
        "post_initialization_object_state_write_calls": value.get(
            "post_initialization_object_state_write_calls"
        ),
        "trajectories": compact,
    }


def _counterfactuals(*, phase: Path, references: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Aggregate the bounded, already-executed V2 CF1/CF2/CF3 receipts."""

    output: dict[str, Any] = {}
    expected = {
        "CF1_same_action_gravity_only": "cf1_gravity_{short}.json",
        "CF2_last_contact_free_drift_g0": "cf2_last_contact_g0_{short}.json",
        "CF2_last_contact_free_drift_gravity": "cf2_last_contact_gravity_{short}.json",
        "CF3_reference_terminal_free_drift_g0": "cf3_reference_terminal_g0_{short}.json",
        "CF3_reference_terminal_free_drift_gravity": "cf3_reference_terminal_gravity_{short}.json",
    }
    for clip in CLIPS:
        short = clip.removeprefix("hocap_")
        receipts = {
            label: _counterfactual_summary(
                phase / template.format(short=short),
                clip=clip,
                reference_sha256=references[clip]["sha256"],
            )
            for label, template in expected.items()
        }
        output[clip] = {
            "representatives": [
                {
                    "label": row["label"],
                    "replica": row["replica"],
                    "formal_terminal_stability": row["formal_terminal_stability"],
                }
                for row in receipts["CF1_same_action_gravity_only"]["trajectories"]
            ],
            "receipts": receipts,
            "interpretation": {
                "gravity_replay": (
                    "same saved actions lose contact and terminate under gravity; this is a "
                    "physics-sensitivity diagnostic, not a nominal requalification"
                ),
                "zero_g_free_drift": (
                    "velocity persists without contact because damping remains nominal zero; "
                    "the result does not establish a hidden controller"
                ),
                "reference_terminal": (
                    "V2 terminal state has nonzero but small twist and is classified "
                    "TERMINAL_POSE_STILL_MOVING"
                ),
            },
        }
    return {
        "schema_version": "Stage16DPhase1RReferenceV2CounterfactualsV1",
        "status": "COMPLETE_BOUNDED_V2_COUNTERFACTUALS",
        "reference_kinematics_version": 2,
        "maximum_representatives_per_clip": 2,
        "clips": output,
    }


def _frozen_phase1_context(frozen_root: Path) -> dict[str, Any]:
    """Carry forward only unchanged support/RSI semantics with immutable provenance."""

    paths = {
        "support_audit": frozen_root / "support_audit.json",
        "rsi_implementation_audit": frozen_root / "rsi_implementation_audit.json",
    }
    result: dict[str, Any] = {}
    for label, path in paths.items():
        value = _json(path)
        result[label] = {"path": str(path), "sha256": _hash(path), "value": value}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--v1-continuation-root", type=Path, default=DEFAULT_V1_CONTINUATION)
    parser.add_argument("--frozen-phase1-root", type=Path, default=DEFAULT_FROZEN_PHASE1_ROOT)
    parser.add_argument(
        "--finalize-counterfactuals",
        action="store_true",
        help="Require and aggregate the five bounded V2 counterfactual receipts per clip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    phase = root / "phase1_rerun"
    diagnostics = phase / "v1_policy_on_v2_reference"
    continuation = args.v1_continuation_root.resolve()
    frozen_phase1 = args.frozen_phase1_root.resolve()
    summaries: dict[str, Any] = {}
    for clip in CLIPS:
        short = clip.removeprefix("hocap_")
        evaluation = diagnostics / clip / "phase1r_v2_evaluation.json"
        trace = diagnostics / clip / "ppo_phase1r_v2_trace_replica0.npz"
        reference = root / "references" / f"{clip}.reference_kinematics_v2.npz"
        prior = continuation / clip / "r7_formal_qualification.json"
        gate = _json(prior)["task_gate"]
        rows, terminal = _terminal_rows(
            trace_path=trace, reference_path=reference, evaluation_path=evaluation, gate=gate
        )
        attribution = _attribution(rows, gate)
        twist_report = {
            "schema_version": "Stage16DPhase1RReferenceV2TerminalTwistV1",
            "clip": clip,
            "reference_kinematics_version": 2,
            "policy_reference_version": 1,
            "policy_label": "V1_POLICY_ON_V2_REFERENCE_DIAGNOSTIC",
            "fresh_frame_zero_episodes": 20,
            "reference": {"path": str(reference), "sha256": _hash(reference)},
            "trace": {"path": str(trace), "sha256": _hash(trace)},
            "terminal_reference": terminal,
            "episodes": rows,
        }
        _write(phase / f"terminal_twist_{short}.json", twist_report)
        _write(phase / f"actual_terminal_drift_{short}.json", twist_report)
        _write(phase / f"attribution_{short}.json", attribution)
        _write(
            diagnostics / clip / "phase1r_v2_qualification.json",
            {
                "schema_version": "Stage16DPhase1RDiagnosticQualificationV1",
                "status": "PHASE1R_V2_DIAGNOSTIC_COMPLETE",
                "clip": clip,
                "episodes": rows,
                "task_gate": gate,
                "reference_kinematics_version": 2,
            },
        )
        summaries[clip] = {
            "reference": {"path": str(reference), "sha256": _hash(reference)},
            "terminal": terminal,
            "attribution": attribution,
        }
    counterfactual = None
    if args.finalize_counterfactuals:
        counterfactual = _counterfactuals(
            phase=phase,
            references={clip: summaries[clip]["reference"] for clip in CLIPS},
        )
        _write(phase / "counterfactuals.json", counterfactual)
    summary = {
        "schema_version": "Stage16DPhase1RReferenceV2SummaryV1",
        "status": "COMPLETE_FRESH_V1_POLICY_ON_V2_REFERENCE_DIAGNOSTIC",
        "reference_kinematics_version": 2,
        "policy_reference_version": 1,
        "counterfactual_status": (
            "PENDING_BOUNDED_V2_COUNTERFACTUALS"
            if counterfactual is None
            else counterfactual["status"]
        ),
        "frozen_support_and_rsi_context": _frozen_phase1_context(frozen_phase1),
        "clips": summaries,
    }
    _write(phase / "summary.json", summary)
    lines = ["# Stage 16-D Phase 1-R", ""]
    lines.append("Fresh 20-episode V1-policy-on-V2-reference diagnostics were completed.")
    lines.append(f"Counterfactuals: {summary['counterfactual_status']}.")
    lines.append(
        "Support provenance remains source-unknown and simulator-support-absent; RSI remains "
        "a reset-only distributional limitation."
    )
    for clip, result in summaries.items():
        terminal = result["terminal"]
        attribution = result["attribution"]
        lines.extend(
            [
                "",
                f"## {clip}",
                "",
                f"- terminal stability: {terminal['terminal_stability_rate']:.2%}",
                f"- terminal contact: {terminal['terminal_contact_rate']:.2%}",
                f"- primary attribution: {attribution['primary']}",
            ]
        )
    (phase / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(phase)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
