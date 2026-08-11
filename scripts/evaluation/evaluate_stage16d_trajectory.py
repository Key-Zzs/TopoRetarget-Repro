#!/usr/bin/env python3
"""Evaluate one saved 20-replica Stage 16-D R7 trace with Evaluation Suite V2.

The evaluator is Isaac-free by design: it consumes frozen post-physics traces,
their embedded reference, and the formal R7 qualification.  It therefore never
changes an environment, policy, reward, physics configuration, or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation import (  # noqa: E402
    EvaluationFingertipSetV1,
    EvaluationJointSetV1,
    EvaluationSuiteV2,
    PhysicsEpisodeEvidence,
    aggregate_rollouts,
    hand_metric_series,
    object_metric_series,
    timeline_rows,
    trajectory_success,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _series_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("metric series must be finite non-empty [T]")
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "terminal": float(array[-1]),
    }


def _absolute_penetration_pass(qualification: dict[str, Any]) -> bool:
    gates = qualification.get("geometry", {}).get("absolute_gates", {})
    if not isinstance(gates, dict) or not gates:
        raise ValueError("R7 qualification lacks absolute hand-object penetration gates")
    return all(bool(value) for value in gates.values())


def _terminal_stability_series(
    object_twist: np.ndarray, contact: np.ndarray, qualification: dict[str, Any]
) -> np.ndarray:
    gate = qualification.get("task_gate")
    if not isinstance(gate, dict):
        raise ValueError("R7 qualification lacks task gate")
    values = np.zeros(object_twist.shape[0], dtype=bool)
    terminal_steps = int(gate["terminal_window_control_steps"])
    if terminal_steps <= 0 or terminal_steps > object_twist.shape[0]:
        raise ValueError("invalid R7 terminal window")
    terminal = slice(-terminal_steps, None)
    linear = np.linalg.norm(object_twist[terminal, :3], axis=-1)
    angular = np.linalg.norm(object_twist[terminal, 3:], axis=-1)
    terminal_contact = contact[terminal]
    linear_limit = np.where(
        terminal_contact,
        float(gate["terminal_linear_speed_mps"]),
        float(gate["terminal_free_object_linear_speed_mps"]),
    )
    angular_limit = np.where(
        terminal_contact,
        float(gate["terminal_angular_speed_radps"]),
        float(gate["terminal_free_object_angular_speed_radps"]),
    )
    values[terminal] = (linear <= linear_limit) & (angular <= angular_limit)
    return values


def _episode_row(
    *,
    replica: int,
    actual_object_pose: np.ndarray,
    actual_object_twist: np.ndarray,
    actual_hand_pose: np.ndarray,
    contact: np.ndarray,
    reference_object_pose: np.ndarray,
    reference_link_positions: np.ndarray,
    collision_names: list[str],
    reference_names: list[str],
    qualification_row: dict[str, Any],
    qualification: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metrics = object_metric_series(actual_object_pose, reference_object_pose)
    metrics.update(
        hand_metric_series(
            actual_hand_pose,
            collision_names,
            reference_link_positions,
            reference_names,
        )
    )
    physics = PhysicsEpisodeEvidence(
        terminal_contact_pass=bool(qualification_row["terminal_contact_pass"]),
        terminal_stability_pass=bool(qualification_row["terminal_stability_pass"]),
        contact_causality_pass=bool(qualification_row["contact_causality_pass"]),
        inter_finger_penetration_pass=bool(qualification_row["inter_finger_penetration_pass"]),
        absolute_hand_object_penetration_pass=_absolute_penetration_pass(qualification),
        action_bounds_pass=bool(qualification_row["action_bounds_pass"]),
        no_hidden_force=bool(qualification_row["no_hidden_control"]),
        no_object_rollout_state_write=int(qualification_row["formal_object_state_writes"]) == 0,
        no_wrist_root_teleport=int(qualification_row["formal_wrist_state_writes"]) == 0,
    )
    success = trajectory_success(
        metrics,
        complete=bool(qualification_row["complete_trajectory"]),
        physics=physics,
    )
    series = {name: _series_summary(values) for name, values in metrics.items()}
    terminal_stability = _terminal_stability_series(actual_object_twist, contact, qualification)
    row: dict[str, object] = {
        "replica": replica,
        "seed": int(qualification_row["seed"]),
        "E_r_mean_deg": success["E_r_mean_deg"],
        "E_r_p95_deg": series["e_r_deg"]["p95"],
        "E_r_max_deg": series["e_r_deg"]["max"],
        "E_r_terminal_deg": series["e_r_deg"]["terminal"],
        "E_t_mean_cm": success["E_t_mean_cm"],
        "E_t_p95_cm": series["e_t_cm"]["p95"],
        "E_t_max_cm": series["e_t_cm"]["max"],
        "E_t_terminal_cm": series["e_t_cm"]["terminal"],
        "E_j_mean_cm": success["E_j_mean_cm"],
        "E_j_p95_cm": series["e_j_cm"]["p95"],
        "E_j_max_cm": series["e_j_cm"]["max"],
        "E_j_terminal_cm": series["e_j_cm"]["terminal"],
        "E_ft_mean_cm": success["E_ft_mean_cm"],
        "E_ft_p95_cm": series["e_ft_cm"]["p95"],
        "E_ft_max_cm": series["e_ft_cm"]["max"],
        "E_ft_terminal_cm": series["e_ft_cm"]["terminal"],
        "kinematic_success": success["kinematic_success"],
        "physics_success": success["physics_success"],
        "qualified_success": success["qualified_success"],
        "old_ppo_task_success": bool(qualification_row["terminal_stability_pass"]),
        "reference_completion": bool(qualification_row["reached_reference_end"]),
        "terminal_contact": bool(qualification_row["terminal_contact_pass"]),
        "terminal_stability": bool(qualification_row["terminal_stability_pass"]),
        "contact_causality": bool(qualification_row["contact_causality_pass"]),
        "inter_finger_penetration": bool(qualification_row["inter_finger_penetration_pass"]),
        "absolute_hand_object_penetration": physics.absolute_hand_object_penetration_pass,
        "action_bounds": bool(qualification_row["action_bounds_pass"]),
        "no_hidden_force": bool(qualification_row["no_hidden_control"]),
        "no_object_rollout_state_write": physics.no_object_rollout_state_write,
        "no_wrist_root_teleport": physics.no_wrist_root_teleport,
        "source_relative_geometry_fidelity": bool(qualification.get("geometry_formal_pass")),
    }
    timeline = timeline_rows(metrics, contact=contact, terminal_stability=terminal_stability)
    for point in timeline:
        point.update({"replica": replica, "seed": int(qualification_row["seed"])})
    return row, timeline


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty per-episode CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--per-episode", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    args = parser.parse_args()

    qualification = _read_json(args.qualification.resolve())
    formal = qualification.get("episodes")
    if not isinstance(formal, list) or len(formal) != 20:
        raise ValueError("Evaluation Suite V2 requires 20 frozen formal R7 episodes")
    with np.load(args.reference.resolve(), allow_pickle=False) as reference_archive:
        reference_object_pose = np.concatenate(
            (
                np.asarray(
                    reference_archive["object_pose_translation_world_ref"], dtype=np.float64
                ),
                np.asarray(
                    reference_archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64
                ),
            ),
            axis=-1,
        )
        reference_links = np.asarray(reference_archive["tracked_link_positions_world_ref"])
        metadata = json.loads(str(reference_archive["metadata"].item()))
        reference_names = [str(name) for name in metadata["tracked_link_names"]]
    with np.load(args.trace.resolve(), allow_pickle=False) as trace:
        required = {
            "replica_object_pose",
            "replica_object_twist",
            "replica_hand_collision_body_pose",
            "replica_contact_pair_presence",
            "hand_collision_body_names",
        }
        missing = sorted(required.difference(trace.files))
        if missing:
            raise ValueError(f"all-replica R7 trace lacks: {missing}")
        actual_object = np.asarray(trace["replica_object_pose"], dtype=np.float64)
        actual_twist = np.asarray(trace["replica_object_twist"], dtype=np.float64)
        actual_hand = np.asarray(trace["replica_hand_collision_body_pose"], dtype=np.float64)
        contacts = np.asarray(trace["replica_contact_pair_presence"], dtype=bool).any(axis=-1)
        collision_names = [str(name) for name in trace["hand_collision_body_names"].tolist()]
    if actual_object.shape != (321, 20, 7) or actual_twist.shape != (321, 20, 6):
        raise ValueError("R7 trace must be exactly [321, 20, 7/6]")
    if actual_hand.shape[:2] != (321, 20) or contacts.shape != (321, 20):
        raise ValueError("R7 hand/contact series has invalid dimensions")
    if reference_object_pose.shape != (321, 7) or reference_links.shape[0] != 321:
        raise ValueError("reference must be factor-8 321 samples")

    rows: list[dict[str, object]] = []
    timelines: list[dict[str, object]] = []
    for replica, formal_row in enumerate(formal):
        if int(formal_row["replica"]) != replica:
            raise ValueError("formal R7 replica ordering is not stable")
        row, timeline = _episode_row(
            replica=replica,
            actual_object_pose=actual_object[:, replica],
            actual_object_twist=actual_twist[:, replica],
            actual_hand_pose=actual_hand[:, replica],
            contact=contacts[:, replica],
            reference_object_pose=reference_object_pose,
            reference_link_positions=reference_links,
            collision_names=collision_names,
            reference_names=reference_names,
            qualification_row=formal_row,
            qualification=qualification,
        )
        rows.append(row)
        timelines.extend(timeline)
    summary = {
        "schema_version": "TopoRetargetEvaluationSuiteV2ResultV1",
        "clip": qualification["clip"],
        "contract": EvaluationSuiteV2().as_dict(),
        "joint_set": EvaluationJointSetV1().as_dict(),
        "fingertip_set": EvaluationFingertipSetV1().as_dict(),
        "formal_r7_qualification": str(args.qualification.resolve()),
        "trace": str(args.trace.resolve()),
        "reference": str(args.reference.resolve()),
        "aggregate": aggregate_rollouts(rows),
        "legacy": {
            "old_ppo_task_success": {
                "pass_count": sum(bool(row["old_ppo_task_success"]) for row in rows),
                "total": len(rows),
            },
            "source_relative_geometry_fidelity": bool(qualification["geometry_formal_pass"]),
            "absolute_hand_object_penetration": _absolute_penetration_pass(qualification),
        },
    }
    for path in (args.per_episode, args.summary, args.timeline):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.per_episode, rows)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.timeline.open("w", encoding="utf-8") as stream:
        for row in timelines:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"clip": qualification["clip"], "summary": str(args.summary.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
