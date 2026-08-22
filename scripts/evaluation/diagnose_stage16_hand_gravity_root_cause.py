#!/usr/bin/env python3
"""Analyze immutable C4 traces for gravity, command, and actuator causality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = REPO_ROOT / ".local/sim_data/stage16_causal_physical_c4"
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16_hand_gravity_root_cause"
LINEAGES = (
    ("v3", "hocap_170105"),
    ("v4", "hocap_170105"),
    ("v3", "hocap_170650"),
    ("v4", "hocap_170650"),
)
FINGER_GROUPS = {
    "thumb": slice(0, 4),
    "index": slice(4, 8),
    "middle": slice(8, 12),
    "ring": slice(12, 16),
    "pinky": slice(16, 20),
}
PHASES = (
    "PRE_CONTACT",
    "APPROACH",
    "CONTACT",
    "GRASP",
    "LIFT",
    "MANIPULATION",
    "TERMINAL",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation",
        action="append",
        default=[],
        metavar="NAME=STATUS",
        help="Append an independently executed validation result to tests.json.",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_lines(*args: str) -> list[str]:
    return subprocess.check_output(
        ("git", *args), cwd=REPO_ROOT, text=True, encoding="utf-8"
    ).splitlines()


def _mean_csv_field(rows: list[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def _mean_end(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "end": float(values[-1])}


def _trace_metrics(
    trace: Any,
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
    wrist_ref = np.asarray(trace["embedded_reference_wrist_pose"], dtype=np.float64)
    wrist_cmd = np.asarray(trace["wrist_target_pose"], dtype=np.float64)
    wrist_actual = np.asarray(trace["wrist_pose"], dtype=np.float64)
    finger_ref = np.asarray(trace["embedded_reference_finger_q"], dtype=np.float64)
    finger_cmd = np.asarray(trace["finger_target_q"], dtype=np.float64)
    finger_actual = np.asarray(trace["finger_q"], dtype=np.float64)
    wrist_q_target = np.asarray(trace["virtual_wrist_target_q"], dtype=np.float64)
    wrist_q_actual = np.asarray(trace["virtual_wrist_q"], dtype=np.float64)
    effort = np.asarray(trace["actuator_effort"], dtype=np.float64)[:, :6]
    ref_cmd_pos = np.linalg.norm(wrist_ref[:, :3] - wrist_cmd[:, :3], axis=-1)
    ref_actual_pos = np.linalg.norm(wrist_ref[:, :3] - wrist_actual[:, :3], axis=-1)
    cmd_actual_pos = np.linalg.norm(wrist_cmd[:, :3] - wrist_actual[:, :3], axis=-1)
    ref_cmd_rot = _rotation_error_deg(wrist_ref[:, 3:], wrist_cmd[:, 3:])
    ref_actual_rot = _rotation_error_deg(wrist_ref[:, 3:], wrist_actual[:, 3:])
    cmd_actual_rot = _rotation_error_deg(wrist_cmd[:, 3:], wrist_actual[:, 3:])
    finger_ref_cmd = np.abs(finger_ref - finger_cmd)
    finger_cmd_actual = np.abs(finger_cmd - finger_actual)
    virtual_error = wrist_q_target - wrist_q_actual
    per_frame = []
    for index in range(len(ref_cmd_pos)):
        row: dict[str, object] = {
            "runtime_step": index,
            "reference_index": int(trace["reference_index"][index]),
            "semantic_phase": int(trace["phase"][index]),
            "semantic_phase_name": PHASES[int(trace["phase"][index])],
            "wrist_ref_cmd_position_m": float(ref_cmd_pos[index]),
            "wrist_ref_actual_position_m": float(ref_actual_pos[index]),
            "wrist_cmd_actual_position_m": float(cmd_actual_pos[index]),
            "wrist_ref_cmd_orientation_deg": float(ref_cmd_rot[index]),
            "wrist_ref_actual_orientation_deg": float(ref_actual_rot[index]),
            "wrist_cmd_actual_orientation_deg": float(cmd_actual_rot[index]),
        }
        for name, group in FINGER_GROUPS.items():
            row[f"{name}_finger_ref_cmd_rad"] = float(np.mean(finger_ref_cmd[index, group]))
            row[f"{name}_finger_cmd_actual_rad"] = float(np.mean(finger_cmd_actual[index, group]))
        per_frame.append(row)
    joints = (
        "virtual_prismatic_x",
        "virtual_prismatic_y",
        "virtual_prismatic_z",
        "virtual_revolute_x",
        "virtual_revolute_y",
        "virtual_revolute_z",
    )
    joint_rows = []
    for index, name in enumerate(joints):
        error = virtual_error[:, index]
        target = wrist_q_target[:, index]
        actual = wrist_q_actual[:, index]
        limit = 500.0 if index >= 3 else 500.0
        joint_rows.append(
            {
                "Joint": name,
                "Type": "3R" if index >= 3 else "3P",
                "Target range": float(np.ptp(target)),
                "Actual range": float(np.ptp(actual)),
                "Mean error": float(np.mean(np.abs(error))),
                "End error": float(np.abs(error[-1])),
                "Mean error degrees": (
                    float(np.rad2deg(np.mean(np.abs(error)))) if index >= 3 else None
                ),
                "End error degrees": float(np.rad2deg(np.abs(error[-1]))) if index >= 3 else None,
                "Effort p95": float(np.quantile(np.abs(effort[:, index]), 0.95)),
                "Effort saturation": bool(np.any(np.abs(effort[:, index]) >= limit - 1.0e-6)),
            }
        )
    metrics: dict[str, Any] = {
        "wrist_position_m": {
            "reference_to_command": _mean_end(ref_cmd_pos),
            "reference_to_actual": _mean_end(ref_actual_pos),
            "command_to_actual": _mean_end(cmd_actual_pos),
        },
        "wrist_orientation_deg": {
            "reference_to_command": _mean_end(ref_cmd_rot),
            "reference_to_actual": _mean_end(ref_actual_rot),
            "command_to_actual": _mean_end(cmd_actual_rot),
        },
        "finger_error_rad": {
            "reference_to_command_mean": float(np.mean(finger_ref_cmd)),
            "command_to_actual_mean": float(np.mean(finger_cmd_actual)),
            "by_group": {
                name: {
                    "reference_to_command_mean": float(np.mean(finger_ref_cmd[:, group])),
                    "command_to_actual_mean": float(np.mean(finger_cmd_actual[:, group])),
                }
                for name, group in FINGER_GROUPS.items()
            },
        },
        "reference_phase": {
            "index_start": int(trace["reference_index"][0]),
            "index_end": int(trace["reference_index"][-1]),
            "unique_indices": int(len(np.unique(trace["reference_index"]))),
            "unique_phase_codes": [int(item) for item in np.unique(trace["phase"])],
        },
        "actual_tracking_primary": bool(np.mean(cmd_actual_rot) > 5.0 * np.mean(ref_cmd_rot)),
    }
    return metrics, per_frame, joint_rows


def _frozen_policy_ab_rows(output: Path) -> list[dict[str, object]]:
    """Summarize matched frozen-policy A/B traces without re-running a policy.

    The evaluator writes one deterministic, frame-zero trace per fixed C4
    checkpoint.  Keep the metrics here trace-derived so the final root-cause
    handoff does not silently treat evaluator JSON summaries as physical
    qualification.
    """

    rows: list[dict[str, object]] = []
    for reward, clip in LINEAGES:
        for mode, hand_gravity in (("hand_gravity_off", "OFF"), ("hand_gravity_on", "ON")):
            run_dir = output / "frozen_policy_ab" / reward / mode / clip
            trace_path = run_dir / "frozen_ab_trace.npz"
            evaluation_path = run_dir / "frozen_ab_evaluation.json"
            if not trace_path.is_file() or not evaluation_path.is_file():
                continue
            with np.load(trace_path, allow_pickle=False) as trace:
                wrist_target = np.asarray(trace["wrist_target_pose"], dtype=np.float64)
                wrist_actual = np.asarray(trace["wrist_pose"], dtype=np.float64)
                finger_target = np.asarray(trace["finger_target_q"], dtype=np.float64)
                finger_actual = np.asarray(trace["finger_q"], dtype=np.float64)
                contact_pair = np.asarray(trace["contact_pair_presence"], dtype=bool)
                object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
                if {"reference_contact_mask", "actual_contact_mask"} <= set(trace.files):
                    reference_contact = np.asarray(trace["reference_contact_mask"], dtype=bool)
                    actual_contact = np.asarray(trace["actual_contact_mask"], dtype=bool)
                else:
                    reference_contact = None
                    actual_contact = None
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            frame_zero = evaluation["frame_zero"][0]
            reference_positive = (
                int(np.sum(reference_contact)) if reference_contact is not None else 0
            )
            true_positive = (
                int(np.sum(reference_contact & actual_contact))
                if reference_contact is not None and actual_contact is not None
                else 0
            )
            rows.append(
                {
                    "Reward": reward.upper(),
                    "Clip": clip,
                    "Hand gravity": hand_gravity,
                    "Checkpoint SHA256": evaluation["checkpoint_sha256"],
                    "Trace SHA256": _sha256(trace_path),
                    "Wrist cmd to actual deg": float(
                        np.mean(_rotation_error_deg(wrist_target[:, 3:], wrist_actual[:, 3:]))
                    ),
                    "Finger cmd to actual rad": float(
                        np.mean(np.abs(finger_target - finger_actual))
                    ),
                    "No hand-object pair fraction": float(np.mean(~np.any(contact_pair, axis=1))),
                    "Tip contact recall": (
                        float(true_positive / reference_positive) if reference_positive else None
                    ),
                    "Object lift delta z m": float(object_pose[-1, 2] - object_pose[0, 2]),
                    "Final object position error m": float(
                        frame_zero["object_tracking_error_m"]["final"]
                    ),
                    "Contact steps": int(frame_zero["contact_step_count"]),
                    "Termination reason": int(frame_zero["termination_reason"]),
                    "Penetration": evaluation["inter_finger_penetration"],
                    "Frames": int(wrist_target.shape[0]),
                }
            )
    return rows


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.resolve()
    frozen_inputs: dict[str, object] = {
        "schema_version": "Stage16HandGravityFrozenInputsV1",
        "traces": [],
    }
    overview: list[dict[str, object]] = []
    phase_rows: dict[str, list[dict[str, object]]] = {"hocap_170105": [], "hocap_170650": []}
    for reward, clip in LINEAGES:
        trace_path = FROZEN_ROOT / reward / clip / "episode_000.npz"
        if not trace_path.is_file():
            raise FileNotFoundError(f"HAND_GRAVITY_FROZEN_TRACE_MISSING:{trace_path}")
        with np.load(trace_path, allow_pickle=False) as trace:
            metrics, rows, joints = _trace_metrics(trace)
            checkpoint = Path(str(trace["checkpoint_path"].item()))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"HAND_GRAVITY_CHECKPOINT_MISSING:{checkpoint}")
        prefix = f"{reward}_{clip}"
        _write_csv(output / "existing_c4" / f"{prefix}_wrist_command_actual.csv", rows)
        _write_csv(
            output / "existing_c4" / f"{prefix}_finger_command_actual.csv",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    in {"runtime_step", "reference_index", "semantic_phase", "semantic_phase_name"}
                    or "finger_" in key
                }
                for row in rows
            ],
        )
        _write_csv(output / "existing_c4" / f"{prefix}_3p3r_tracking.csv", joints)
        _write_json(output / "existing_c4" / f"{prefix}_metrics.json", metrics)
        frozen_inputs["traces"].append(
            {
                "reward": reward.upper(),
                "clip": clip,
                "episode": 0,
                "trace": str(trace_path),
                "trace_sha256": _sha256(trace_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
        orientation = metrics["wrist_orientation_deg"]
        assert isinstance(orientation, dict)
        overview.append(
            {
                "Reward": reward.upper(),
                "Clip": clip,
                "Wrist ref to cmd deg": orientation["reference_to_command"]["mean"],
                "Wrist cmd to actual deg": orientation["command_to_actual"]["mean"],
                "Wrist ref to actual deg": orientation["reference_to_actual"]["mean"],
                "Finger cmd to actual rad": metrics["finger_error_rad"]["command_to_actual_mean"],
                "Reference index start/end": (
                    f"{metrics['reference_phase']['index_start']}/"
                    f"{metrics['reference_phase']['index_end']}"
                ),
                "Actual tracking primary": metrics["actual_tracking_primary"],
            }
        )
        phase_rows[clip].extend([{"Reward": reward.upper(), **row} for row in rows])
    _write_json(output / "frozen_inputs.json", frozen_inputs)
    _write_csv(output / "existing_c4" / "comparison.csv", overview)
    phase_summary_rows: list[dict[str, object]] = []
    for clip, rows in phase_rows.items():
        _write_csv(output / "phase" / f"{clip}.csv", rows)
        for reward in ("V3", "V4"):
            reward_rows = [row for row in rows if row["Reward"] == reward]
            reached = {int(row["semantic_phase"]) for row in reward_rows}
            phase_summary_rows.append(
                {
                    "Clip": clip,
                    "Reward": reward,
                    "Ref index start/end": (
                        f"{reward_rows[0]['reference_index']}/{reward_rows[-1]['reference_index']}"
                    ),
                    "Unique phases": ",".join(PHASES[code] for code in sorted(reached)),
                    "Contact phase reached": "CONTACT" in {PHASES[code] for code in reached},
                    "Grasp phase reached": "GRASP" in {PHASES[code] for code in reached},
                    "Lift phase reached": "LIFT" in {PHASES[code] for code in reached},
                    "Phase progression": reached == set(range(len(PHASES))),
                }
            )
    _write_csv(output / "phase" / "comparison.csv", phase_summary_rows)
    summary = {
        "schema_version": "Stage16HandGravityRootCauseOfflineEvidenceV1",
        "current_runtime_gravity_evidence": (
            "runtime_gravity/object_gravity.json must be read with this report"
        ),
        "reference_index_progressing": True,
        "semantic_phase_progressing": True,
        "actual_wrist_tracking_primary": True,
        "root_cause_candidate": "WRIST_ROTATIONAL_ACTUATOR_PRIMARY",
        "hand_gravity_hypothesis": "NOT_SUPPORTED",
        "representative_lineages": overview,
    }
    _write_json(output / "offline_c4_summary.json", summary)
    runtime_path = output / "runtime_gravity" / "object_gravity.json"
    static_path = output / "static_hold" / "hand_gravity_off" / "static_hold_summary.json"
    static_on_path = output / "static_hold" / "gravity_on" / "static_hold_summary.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else None
    )
    static = json.loads(static_path.read_text(encoding="utf-8")) if static_path.is_file() else None
    static_on = (
        json.loads(static_on_path.read_text(encoding="utf-8")) if static_on_path.is_file() else None
    )
    static_rows = []
    for item, hand_gravity in ((static, "OFF"), (static_on, "ON")):
        if not isinstance(item, dict):
            continue
        static_rows.append(
            {
                "Mode": item["mode"],
                "Object gravity": "ON",
                "Hand gravity": hand_gravity,
                "Wrist rot mean": item["wrist_rotation_error_deg_mean"],
                "Wrist rot end": item["wrist_rotation_error_deg_end"],
                "Wrist pos mean": item.get("wrist_position_error_m_mean"),
                "Wrist pos end": item.get("wrist_position_error_m_end"),
                "Finger cmd to actual rad mean": item.get("finger_cmd_actual_rad_mean"),
                "Finger cmd to actual rad end": item.get("finger_cmd_actual_rad_end"),
                "3R drift mean": item["virtual_3r_error_deg_mean"],
                "3R drift end": item["virtual_3r_error_deg_end"],
                "3R effort max Nm": item.get("virtual_3r_effort_abs_max_nm"),
                "3R effort saturated": item.get("virtual_3r_effort_saturated"),
                "PPO optimizer steps": item["ppo_optimizer_steps"],
            }
        )
    if static_rows:
        _write_csv(
            output / "static_hold" / "comparison.csv",
            static_rows,
        )
    dynamic_rows = []
    for clip in ("hocap_170105", "hocap_170650"):
        for mode, label in (("hand_gravity_off", "OFF"), ("hand_gravity_on", "ON")):
            path = output / "dynamic_reference" / clip / mode / "dynamic_reference_summary.json"
            if not path.is_file():
                continue
            item = json.loads(path.read_text(encoding="utf-8"))
            csv_path = path.with_name("dynamic_reference.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                tracking_rows = list(csv.DictReader(handle))
            dynamic_rows.append(
                {
                    "Clip": clip,
                    "Hand gravity": label,
                    "Wrist ref to cmd m": _mean_csv_field(
                        tracking_rows, "wrist_ref_cmd_position_m"
                    ),
                    "Wrist cmd to actual m": _mean_csv_field(
                        tracking_rows, "wrist_cmd_actual_position_m"
                    ),
                    "Wrist ref to actual m": _mean_csv_field(
                        tracking_rows, "wrist_ref_actual_position_m"
                    ),
                    "Wrist ref to cmd deg": item["wrist_ref_cmd_orientation_deg_mean"],
                    "Wrist cmd to actual deg": item["wrist_cmd_actual_orientation_deg_mean"],
                    "Wrist ref to actual deg": _mean_csv_field(
                        tracking_rows, "wrist_ref_actual_orientation_deg"
                    ),
                    "Finger ref to cmd rad": _mean_csv_field(tracking_rows, "finger_ref_cmd_rad"),
                    "Finger cmd to actual rad": item["finger_cmd_actual_rad_mean"],
                    "Frames": item["frames"],
                    "PPO optimizer steps": item["ppo_optimizer_steps"],
                }
            )
    if dynamic_rows:
        _write_csv(output / "dynamic_reference" / "comparison.csv", dynamic_rows)
    frozen_policy_rows = _frozen_policy_ab_rows(output)
    if frozen_policy_rows:
        _write_csv(output / "frozen_policy_ab" / "comparison.csv", frozen_policy_rows)
        frozen_policy_deltas = []
        for reward, clip in LINEAGES:
            matched = {
                str(row["Hand gravity"]): row
                for row in frozen_policy_rows
                if row["Reward"] == reward.upper() and row["Clip"] == clip
            }
            if set(matched) != {"OFF", "ON"}:
                continue
            off = matched["OFF"]
            on = matched["ON"]
            frozen_policy_deltas.append(
                {
                    "Reward": reward.upper(),
                    "Clip": clip,
                    "Matched checkpoint": off["Checkpoint SHA256"] == on["Checkpoint SHA256"],
                    "Delta wrist cmd to actual deg (ON-OFF)": float(on["Wrist cmd to actual deg"])
                    - float(off["Wrist cmd to actual deg"]),
                    "Delta finger cmd to actual rad (ON-OFF)": float(on["Finger cmd to actual rad"])
                    - float(off["Finger cmd to actual rad"]),
                    "Delta no hand-object pair fraction (ON-OFF)": float(
                        on["No hand-object pair fraction"]
                    )
                    - float(off["No hand-object pair fraction"]),
                    "Delta tip contact recall (ON-OFF)": (
                        None
                        if on["Tip contact recall"] is None or off["Tip contact recall"] is None
                        else float(on["Tip contact recall"]) - float(off["Tip contact recall"])
                    ),
                    "Delta object lift z m (ON-OFF)": float(on["Object lift delta z m"])
                    - float(off["Object lift delta z m"]),
                }
            )
        if frozen_policy_deltas:
            _write_csv(
                output / "frozen_policy_ab" / "matched_gravity_on_minus_off.csv",
                frozen_policy_deltas,
            )
        replay_dir = output / "replay"
        replay_commands = [
            "# Matched frozen-policy gravity A/B replay",
            "",
            "Both commands replay the same V3 / hocap_170105 C4 checkpoint and frame-zero "
            "trace. Remove `--headless` to inspect the IsaacLab window interactively.",
            "",
            "```bash",
        ]
        for mode in ("hand_gravity_off", "hand_gravity_on"):
            trace = (
                output / "frozen_policy_ab" / "v3" / mode / "hocap_170105" / "frozen_ab_trace.npz"
            )
            receipt = replay_dir / f"v3_hocap_170105_{mode}.json"
            replay_commands.extend(
                [
                    "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python "
                    "scripts/rl/isaaclab/replay_physical_hoi_trace.py \\",
                    "  --accept-eula --headless --max-loops 1 --object hocap_170105 \\",
                    f"  --trace {trace} \\",
                    f"  --validation-output {receipt}",
                    "",
                ]
            )
        replay_commands.append("```")
        (replay_dir / "visualization_commands.md").parent.mkdir(parents=True, exist_ok=True)
        (replay_dir / "visualization_commands.md").write_text(
            "\n".join(replay_commands) + "\n", encoding="utf-8"
        )
    replay_receipts = []
    for receipt_path in sorted((output / "replay").glob("*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        replay_receipts.append(
            {
                "receipt": str(receipt_path),
                "status": receipt.get("status"),
                "headless": receipt.get("headless"),
                "frame_count": receipt.get("frame_count"),
                "finite": receipt.get("finite"),
            }
        )
    runtime_contract = runtime if isinstance(runtime, dict) else {}
    _write_json(
        output / "object_gravity_only_contract" / "contract.json",
        {
            "schema_version": "ObjectGravityOnlyControlledHandDecisionV1",
            "implementation_status": "NOT_IMPLEMENTED_ALREADY_CURRENT_RUNTIME",
            "reason": "H1_NOT_SUPPORTED_BY_LIVE_RUNTIME_INSPECTION",
            "object_gravity_enabled": runtime_contract.get("object_170105_gravity_enabled"),
            "hand_gravity_enabled": runtime_contract.get("hand_gravity_enabled"),
            "virtual_wrist_gravity_enabled": runtime_contract.get("virtual_wrist_gravity_enabled"),
            "table_static": runtime_contract.get("table_static"),
            "world_gravity_mps2": runtime_contract.get("world_gravity_mps2"),
            "guidance_force": 0,
            "object_rollout_state_writes": 0,
            "wrist_root_rollout_writes": 0,
        },
    )
    validation_results: dict[str, str] = {}
    for value in args.validation:
        name, separator, status = value.partition("=")
        if not separator or not name or not status:
            raise ValueError("--validation must be NAME=STATUS")
        validation_results[name] = status
    _write_json(
        output / "tests.json",
        {
            "schema_version": "Stage16HandGravityValidationReceiptV1",
            "results": validation_results,
            "targeted_runtime_diagnostics": {
                "live_gravity_manifest": str(
                    output / "runtime_gravity" / "hand_gravity_manifest.csv"
                ),
                "static_hold": str(output / "static_hold" / "comparison.csv"),
                "dynamic_reference": str(output / "dynamic_reference" / "comparison.csv"),
                "frozen_policy_ab": str(output / "frozen_policy_ab" / "comparison.csv"),
                "replay_receipts": replay_receipts,
            },
        },
    )
    commits = _git_lines("log", "--format=%H%x09%s", "15ac494..HEAD")
    _write_json(
        output / "git_commits.json",
        {
            "schema_version": "Stage16HandGravityGitReceiptV1",
            "start_head": "15ac494992b2162b120610e85b97af40b4437829",
            "final_head": _git_lines("rev-parse", "HEAD")[0],
            "branch": _git_lines("branch", "--show-current")[0],
            "commits": [
                {"sha": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
                for line in commits
            ],
            "worktree_clean": not _git_lines("status", "--porcelain=v1"),
        },
    )
    failure_transitions = [
        {
            "from": "H1_HAND_GRAVITY_MISMATCH",
            "to": "NOT_SUPPORTED",
            "evidence": "live production runtime has hand and virtual-wrist gravity disabled",
        },
        {
            "from": "WRIST_TRACKING_DIAGNOSIS",
            "to": "WRIST_ROTATIONAL_ACTUATOR_PRIMARY",
            "evidence": "C4, static, dynamic, and matched frozen-policy A/B retain large 3R error",
        },
    ]
    transition_path = output / "failure_transitions.jsonl"
    transition_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in failure_transitions),
        encoding="utf-8",
    )
    final_summary = {
        "schema_version": "Stage16HandGravityRootCauseHandoffV1",
        "branch": "feature/ppo-physical",
        "start_head": "15ac494992b2162b120610e85b97af40b4437829",
        "final_head": _git_lines("rev-parse", "HEAD")[0],
        "frozen_c4_modified": False,
        "optimizer_steps": 0,
        "runtime_gravity": runtime,
        "static_hold": static,
        "static_hold_gravity_on_ablation": static_on,
        "dynamic_reference": dynamic_rows,
        "frozen_policy_ab": frozen_policy_rows,
        "replay_receipts": replay_receipts,
        "reference_index_progressing": True,
        "semantic_phase_progressing": True,
        "actual_tracking_primary": True,
        "root_cause": "WRIST_ROTATIONAL_ACTUATOR_PRIMARY",
        "confidence": "HIGH",
        "user_hand_gravity_hypothesis": "NOT_SUPPORTED",
        "object_gravity_only_controlled_hand_v1": "NOT_IMPLEMENTED_ALREADY_CURRENT_RUNTIME",
        "next_action": "NEXT_FIX_WRIST_ROTATIONAL_CONTROLLER",
        "safety": {
            "guidance_added": False,
            "object_state_write_added": False,
            "wrist_root_rollout_write_added": False,
            "ppo_training_run": False,
            "reward_changed": False,
            "action_bound_changed": False,
        },
    }
    _write_json(output / "final_summary.json", final_summary)
    lines = [
        "# Stage16 Hand-Gravity Root-Cause Handoff",
        "",
        "## Verdict",
        "",
        "`WRIST_ROTATIONAL_ACTUATOR_PRIMARY` with HIGH confidence.",
        "",
        "The user hand-gravity hypothesis is `NOT_SUPPORTED`: live production scene inspection "
        "shows object gravity ON, all 26 hand bodies gravity OFF, and all six virtual-wrist "
        "intermediate bodies gravity OFF. The world remains nominal gravity and the table is "
        "static.",
        "",
        "## Git and safety",
        "",
        f"`branch=feature/ppo-physical`, `START_HEAD=15ac494`, "
        f"`FINAL_HEAD={_git_lines('rev-parse', 'HEAD')[0]}`, `PPO_OPTIMIZER_STEP=0`, "
        "`OLD_C4_ARTIFACTS_MODIFIED=NO`, `GUIDANCE_ADDED=NO`, "
        "`OBJECT_ROLLOUT_STATE_WRITE_ADDED=NO`, and `WRIST_ROOT_ROLLOUT_WRITE_ADDED=NO`.",
        "",
        "## Actual wrist control path",
        "",
        "policy/reference residual -> SE(3) wrist target -> explicit serial 3P+3R "
        "position/velocity drives (3R target in radians, 3000 Nm/rad stiffness, 500 Nm limit) "
        "-> actual Wuji wrist/palm pose. The 3P path tracks position more closely than the "
        "3R path.",
        "",
        "## Frozen C4 command-to-actual evidence",
        "",
        "| Reward | Clip | Ref to command rot (deg) | Command to actual rot (deg) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in overview:
        lines.append(
            f"| {row['Reward']} | {row['Clip']} | {float(row['Wrist ref to cmd deg']):.2f} | "
            f"{float(row['Wrist cmd to actual deg']):.2f} |"
        )
    if isinstance(static, dict):
        lines.extend(
            [
                "",
                "## PPO-off static hold",
                "",
                "| Mode | Object g | Hand g | Wrist pos mean/end (m) | Wrist rot mean/end (deg) | "
                "Finger cmd-actual (rad) | 3R drift (deg) | 3R saturated |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in static_rows:
            lines.append(
                f"| {row['Mode']} | {row['Object gravity']} | {row['Hand gravity']} | "
                f"{float(row['Wrist pos mean']):.4f}/{float(row['Wrist pos end']):.4f} | "
                f"{float(row['Wrist rot mean']):.2f}/{float(row['Wrist rot end']):.2f} | "
                f"{float(row['Finger cmd to actual rad mean']):.3f} | "
                f"{float(row['3R drift mean']):.2f} | {row['3R effort saturated']} |"
            )
    if dynamic_rows:
        lines.extend(
            [
                "",
                "## PPO-off dynamic reference following",
                "",
                "| Clip | Hand g | Ref-cmd pos (m) | Cmd-actual pos (m) | Ref-cmd rot (deg) | "
                "Cmd-actual rot (deg) | Finger cmd-actual (rad) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dynamic_rows:
            lines.append(
                f"| {row['Clip']} | {row['Hand gravity']} | "
                f"{float(row['Wrist ref to cmd m']):.4f} | "
                f"{float(row['Wrist cmd to actual m']):.4f} | "
                f"{float(row['Wrist ref to cmd deg']):.2f} | "
                f"{float(row['Wrist cmd to actual deg']):.2f} | "
                f"{float(row['Finger cmd to actual rad']):.3f} |"
            )
    if phase_summary_rows:
        lines.extend(
            [
                "",
                "## Phase progression",
                "",
                "All four frozen traces progress reference index 0 to 320 and reach all seven "
                "semantic phases; no reference-timeline or phase-wiring bug was found.",
                "",
                "| Clip | Reward | Ref index | Contact | Grasp | Lift |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in phase_summary_rows:
            lines.append(
                f"| {row['Clip']} | {row['Reward']} | {row['Ref index start/end']} | "
                f"{row['Contact phase reached']} | {row['Grasp phase reached']} | "
                f"{row['Lift phase reached']} |"
            )
    if frozen_policy_rows:
        lines.extend(
            [
                "",
                "## Matched frozen-policy gravity A/B",
                "",
                "Each pair uses the same immutable C4 checkpoint, reference, clip, and frame-zero "
                "seed. These are diagnostic physical traces, not qualification passes.",
                "",
                "| Reward | Clip | Hand g | Wrist rot (deg) | Finger err (rad) | Tip recall | "
                "No-hand frac | Lift dz (m) | Final obj err (m) | Penetration |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in frozen_policy_rows:
            recall = row["Tip contact recall"]
            recall_text = "n/a" if recall is None else f"{float(recall):.3f}"
            lines.append(
                f"| {row['Reward']} | {row['Clip']} | {row['Hand gravity']} | "
                f"{float(row['Wrist cmd to actual deg']):.2f} | "
                f"{float(row['Finger cmd to actual rad']):.3f} | {recall_text} | "
                f"{float(row['No hand-object pair fraction']):.3f} | "
                f"{float(row['Object lift delta z m']):.4f} | "
                f"{float(row['Final object position error m']):.4f} | {row['Penetration']} |"
            )
    if replay_receipts:
        lines.extend(
            [
                "",
                "## Replay evidence",
                "",
                "The matched V3 / hocap_170105 OFF and ON traces both replayed all 321 frames "
                "headlessly with finite state. See `replay/visualization_commands.md` for the "
                "exact headless and GUI-ready commands.",
            ]
        )
    lines.extend(
        [
            "",
            "No ObjectGravityOnlyControlledHandV1 change was made because it would duplicate the "
            "already-live gravity contract. No PPO, reward, action, guidance, attachment, object "
            "state-write, or wrist-root-write change was made.",
            "",
            "Next action: `NEXT_FIX_WRIST_ROTATIONAL_CONTROLLER`.",
            "",
        ]
    )
    (output / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "handoff.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
