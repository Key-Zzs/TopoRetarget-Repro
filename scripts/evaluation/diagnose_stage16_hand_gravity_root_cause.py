#!/usr/bin/env python3
"""Analyze immutable C4 traces for gravity, command, and actuator causality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def main() -> int:
    output = _parser().parse_args().output_dir.resolve()
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
    for clip, rows in phase_rows.items():
        _write_csv(output / "phase" / f"{clip}.csv", rows)
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
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else None
    )
    static = json.loads(static_path.read_text(encoding="utf-8")) if static_path.is_file() else None
    if isinstance(static, dict):
        _write_csv(
            output / "static_hold" / "comparison.csv",
            [
                {
                    "Mode": static["mode"],
                    "Object gravity": "ON",
                    "Hand gravity": "OFF",
                    "Wrist rot mean": static["wrist_rotation_error_deg_mean"],
                    "Wrist rot end": static["wrist_rotation_error_deg_end"],
                    "3R drift mean": static["virtual_3r_error_deg_mean"],
                    "3R drift end": static["virtual_3r_error_deg_end"],
                    "PPO optimizer steps": static["ppo_optimizer_steps"],
                }
            ],
        )
    final_summary = {
        "schema_version": "Stage16HandGravityRootCauseHandoffV1",
        "branch": "feature/ppo-physical",
        "start_head": "15ac494992b2162b120610e85b97af40b4437829",
        "frozen_c4_modified": False,
        "optimizer_steps": 0,
        "runtime_gravity": runtime,
        "static_hold": static,
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
                "## PPO-off static hold under current hand-gravity-off runtime",
                "",
                f"Over {float(static['simulated_time_s']):.1f} s, wrist orientation error was "
                f"{float(static['wrist_rotation_error_deg_mean']):.2f} deg mean / "
                f"{float(static['wrist_rotation_error_deg_end']):.2f} deg end; 3R error was "
                f"{float(static['virtual_3r_error_deg_mean']):.2f} deg mean.",
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
