#!/usr/bin/env python3
"""Classify Stage 16-D replica failures before any terminal-only refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OLD_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"


def _angular_speed(quaternion: np.ndarray, frequency_hz: float = 20.0) -> np.ndarray:
    normalized = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    dot = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0)) * frequency_hz


def _terminal_hold_duration(stable: np.ndarray) -> int:
    duration = 0
    for value in stable[::-1]:
        if not bool(value):
            break
        duration += 1
    return duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--terminal-window", type=int, default=20)
    parser.add_argument("--linear-speed-limit", type=float, default=0.10)
    parser.add_argument("--angular-speed-limit", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suffix = args.clip.removeprefix("hocap_")
    qualification_path = OLD_REPORT_ROOT / f"trajectory_qualification_{suffix}_v3.json"
    trace_path = OLD_REPORT_ROOT / f"trajectory_trace_{suffix}_v3.npz"
    geometry_path = REPORT_ROOT / f"corrected_runtime_penetration_pairs_{suffix}.npz"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    episodes = qualification["episodes"]
    with np.load(trace_path, allow_pickle=False) as trace:
        object_pose = np.asarray(trace["replica_object_pose"], dtype=np.float64)
        saturation = np.asarray(trace["saturation"], dtype=bool)
    with np.load(geometry_path, allow_pickle=False) as geometry:
        signed = np.asarray(geometry["signed_separation_m"], dtype=np.float64)
        depth = np.asarray(geometry["penetration_depth_m"], dtype=np.float64)
    if object_pose.shape != (321, len(episodes), 7):
        raise RuntimeError("STAGE16D_TERMINAL_ANALYSIS_TRACE_SHAPE_FAILURE")
    linear_speed = np.linalg.norm(np.diff(object_pose[..., :3], axis=0) * 20.0, axis=-1)
    angular_speed = _angular_speed(object_pose[..., 3:])
    terminal_start = 321 - args.terminal_window
    successful = [int(row["replica"]) for row in episodes if row["success"]]
    failed = [int(row["replica"]) for row in episodes if not row["success"]]
    success_final_position = np.median(object_pose[-1, successful, :3], axis=0)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        replica = int(episode["replica"])
        stable = (linear_speed[:, replica] <= args.linear_speed_limit) & (
            angular_speed[:, replica] <= args.angular_speed_limit
        )
        terminal_stable = stable[terminal_start - 1 :]
        offline_contact = signed[:, replica].min(axis=1) <= 0.002
        terminal_contact = offline_contact[terminal_start:]
        row = {
            "replica": replica,
            "qualification_success": bool(episode["success"]),
            "termination_reason_code": int(episode["termination_reason_code"]),
            "failure_runtime_index": 320 if not episode["success"] else None,
            "first_unstable_terminal_index": (
                int(terminal_start + np.flatnonzero(~terminal_stable)[0])
                if bool((~terminal_stable).any())
                else None
            ),
            "final_linear_speed_mps": float(linear_speed[-1, replica]),
            "final_angular_speed_radps": float(angular_speed[-1, replica]),
            "terminal_linear_speed_max_mps": float(
                linear_speed[terminal_start - 1 :, replica].max()
            ),
            "terminal_angular_speed_max_radps": float(
                angular_speed[terminal_start - 1 :, replica].max()
            ),
            "terminal_hold_duration_control_steps": _terminal_hold_duration(terminal_stable),
            "terminal_offline_contact_frame_rate": float(terminal_contact.mean()),
            "terminal_offline_overlap_frame_rate": float(
                (signed[terminal_start:, replica].min(axis=1) < 0.0).mean()
            ),
            "terminal_max_runtime_proxy_penetration_m": float(
                depth[terminal_start:, replica].max(initial=0.0)
            ),
            "final_object_position_world_m": object_pose[-1, replica, :3].tolist(),
            "final_position_deviation_from_success_median_m": float(
                np.linalg.norm(object_pose[-1, replica, :3] - success_final_position)
            ),
            "semantic_progress": float(episode["semantic_progress"]),
            "contact_recall": float(episode["contact_recall"]),
            "contact_causality_pass": bool(episode["contact_causality_pass"]),
            "action_saturation_terminal_any": bool(saturation[terminal_start:].any()),
        }
        if not row["qualification_success"]:
            if (
                row["final_angular_speed_radps"] > args.angular_speed_limit
                and row["semantic_progress"] >= 0.80
                and row["contact_recall"] >= 0.80
            ):
                row["failure_class"] = "TERMINAL_OBJECT_TWIST"
            elif row["terminal_offline_contact_frame_rate"] < 0.50:
                row["failure_class"] = "TERMINAL_CONTACT_LOSS"
            elif row["terminal_hold_duration_control_steps"] < args.terminal_window:
                row["failure_class"] = "TERMINAL_HOLD_TOO_SHORT"
            else:
                row["failure_class"] = "MIXED_FAILURE"
        else:
            row["failure_class"] = "SUCCESS"
        rows.append(row)
    failed_rows = [row for row in rows if row["replica"] in failed]
    terminal_classes = {
        "TERMINAL_CONTACT_LOSS",
        "TERMINAL_OBJECT_TWIST",
        "TERMINAL_HOLD_TOO_SHORT",
        "TERMINAL_WRIST_OR_FINGER_DRIFT",
    }
    terminal_local = bool(failed_rows) and all(
        row["failure_class"] in terminal_classes for row in failed_rows
    )
    payload = {
        "schema_version": "Stage16DTerminalFailureAnalysisV1",
        "clip": args.clip,
        "qualification_source": str(qualification_path.relative_to(REPO_ROOT)),
        "trace_source": str(trace_path.relative_to(REPO_ROOT)),
        "formal_geometry_source": str(geometry_path.relative_to(REPO_ROOT)),
        "terminal_window": {
            "start_frame": terminal_start,
            "end_frame": 320,
            "control_steps": args.terminal_window,
            "linear_speed_limit_mps": args.linear_speed_limit,
            "angular_speed_limit_radps": args.angular_speed_limit,
        },
        "successful_replicas": successful,
        "failed_replicas": failed,
        "failure_count": len(failed),
        "failure_classes": {
            name: sum(row["failure_class"] == name for row in failed_rows)
            for name in sorted({row["failure_class"] for row in failed_rows})
        },
        "all_preterminal_semantic_contact_causality_pass": all(
            row["semantic_progress"] >= 0.80
            and row["contact_recall"] >= 0.80
            and row["contact_causality_pass"]
            for row in failed_rows
        ),
        "terminal_only_refinement_authorized": terminal_local,
        "recommended_path": (
            "TERMINAL_TAIL_REFINEMENT_V1" if terminal_local else "GLOBAL_BOUNDED_OPTIMIZER_FALLBACK"
        ),
        "replicas": rows,
    }
    output = args.output or REPORT_ROOT / f"terminal_failure_analysis_{suffix}.json"
    csv_output = args.csv_output or REPORT_ROOT / f"successful_vs_failed_{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": "TERMINAL_ANALYSIS_COMPLETE",
                "failed": failed,
                "terminal_only_refinement_authorized": terminal_local,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
