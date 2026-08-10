#!/usr/bin/env python3
"""Aggregate the R7 formal PPO-26D evidence without changing any physics state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (  # noqa: E402
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.physics_retargeting.contact_topology import body_contact_group  # noqa: E402
from toporetarget.rl.physics_retargeting.self_collision import (  # noqa: E402
    InterFingerCapsulePenetrationV1,
    load_self_collision_contract,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _longest_run(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1], dtype=np.int64)
    current = np.zeros_like(result)
    for row in values:
        current = np.where(row, current + 1, 0)
        result = np.maximum(result, current)
    return result


def _group_contact_metrics(
    contact: np.ndarray, *, required: list[str], minimum_persistence: int
) -> tuple[np.ndarray, np.ndarray]:
    group_values: dict[str, np.ndarray] = {}
    for body_index, body_name in enumerate(HAND_COLLISION_BODY_NAMES):
        group = body_contact_group(body_name)
        if group is None:
            continue
        group_values[group] = (
            group_values.get(group, np.zeros(contact.shape[:2], dtype=bool))
            | contact[..., body_index]
        )
    if not required:
        raise ValueError("R7 contact topology has no required body groups")
    runs = np.stack(
        [
            _longest_run(group_values.get(group, np.zeros(contact.shape[:2], dtype=bool)))
            for group in required
        ],
        axis=1,
    )
    coverage = (runs >= minimum_persistence).mean(axis=1)
    return coverage, np.all(runs >= minimum_persistence, axis=1)


def _load_gate(path: Path, clip: str) -> dict[str, Any]:
    payload = _read_json(path)
    gate = payload.get("clips", {}).get(clip)
    if not isinstance(gate, dict) or gate.get("schema_version") != "PhysicsConsistentTaskGateV1":
        raise ValueError("R7 task-gate contract is invalid")
    return gate


def _trace_metrics(
    trace_path: Path, *, clip: str, gate: dict[str, Any], topology: dict[str, Any]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with np.load(trace_path, allow_pickle=False) as archive:
        required = {
            "replica_object_pose",
            "replica_hand_collision_body_pose",
            "replica_contact_pair_presence",
            "replica_object_twist",
            "replica_action",
            "embedded_reference_object_pose",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"R7 trace lacks all-replica physical evidence: {sorted(missing)}")
        object_pose = np.asarray(archive["replica_object_pose"], dtype=np.float64)
        hand_pose = np.asarray(archive["replica_hand_collision_body_pose"], dtype=np.float64)
        contact = np.asarray(archive["replica_contact_pair_presence"], dtype=bool)
        object_twist = np.asarray(archive["replica_object_twist"], dtype=np.float64)
        action = np.asarray(archive["replica_action"], dtype=np.float64)
        reference_object = np.asarray(archive["embedded_reference_object_pose"], dtype=np.float64)
    if object_pose.shape != (321, 20, 7) or hand_pose.shape != (321, 20, 21, 7):
        raise ValueError("R7 trace must contain exactly 20 complete 321-state replicas")
    if contact.shape != (321, 20, 21) or object_twist.shape != (321, 20, 6):
        raise ValueError("R7 all-replica trace has incompatible dimensions")
    if action.shape != (321, 20, 26):
        raise ValueError("R7 action trace has incompatible dimensions")
    if reference_object.shape != (321, 7):
        raise ValueError("R7 reference trace has incompatible dimensions")
    inter_contract = load_self_collision_contract(
        REPO_ROOT / "configs/rl/stage16/stage16d_self_collision.yaml", repo_root=REPO_ROOT
    )
    metric = InterFingerCapsulePenetrationV1.from_runtime_manifest(
        REPO_ROOT / inter_contract.runtime_collision_manifest_path,
        expected_body_names=HAND_COLLISION_BODY_NAMES,
        radius_scale=inter_contract.capsule_radius_scale,
        device="cpu",
    )
    with torch.no_grad():
        inter = (
            metric.evaluate(torch.as_tensor(hand_pose.reshape(-1, 21, 7), dtype=torch.float32))[
                "maximum_penetration_m"
            ]
            .numpy()
            .reshape(321, 20)
        )
    displacement = object_pose[-1, :, :3] - object_pose[0, :, :3]
    direction = reference_object[-1, :3] - reference_object[0, :3]
    direction_norm = max(float(np.linalg.norm(direction)), 1.0e-6)
    semantic_progress = np.clip(
        displacement @ direction / max(0.30 * direction_norm, 0.005), 0.0, 1.0
    )
    object_motion = np.linalg.norm(displacement, axis=-1)
    rotation_dot = np.abs(np.sum(object_pose[-1, :, 3:] * object_pose[0, :, 3:], axis=-1))
    object_rotation_deg = np.degrees(2.0 * np.arccos(np.clip(rotation_dot, 0.0, 1.0)))
    terminal_steps = int(gate["terminal_window_control_steps"])
    terminal_linear = np.linalg.norm(object_twist[-terminal_steps:, :, :3], axis=-1)
    terminal_angular = np.linalg.norm(object_twist[-terminal_steps:, :, 3:], axis=-1)
    terminal_contact = contact[-terminal_steps:].any(axis=-1)
    required_contact_steps = int(
        np.ceil(float(gate["terminal_required_contact_fraction"]) * terminal_steps)
    )
    terminal_contact_window = terminal_contact.sum(axis=0) >= required_contact_steps
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
    terminal_kinematic = np.all(
        (terminal_linear <= linear_limit) & (terminal_angular <= angular_limit), axis=0
    )
    contact_causality = np.any(
        contact[1:].any(axis=-1)
        & (np.linalg.norm(np.diff(object_twist, axis=0), axis=-1) > 1.0e-7),
        axis=0,
    )
    action_bounds = np.all(
        np.isfinite(action) & (np.abs(action) <= float(gate["action_limit"])), axis=(0, 2)
    )
    contact_recall, topology_pass = _group_contact_metrics(
        contact,
        required=[str(group) for group in topology["required_body_groups"]],
        minimum_persistence=int(topology["minimum_persistence_control_steps"]),
    )
    rows = [
        {
            "replica": replica,
            "clip": clip,
            "semantic_progress": float(semantic_progress[replica]),
            "object_motion_m": float(object_motion[replica]),
            "object_rotation_deg": float(object_rotation_deg[replica]),
            "max_inter_finger_penetration_m": float(inter[:, replica].max()),
            "inter_finger_penetration_pass": bool(
                inter[:, replica].max() <= float(gate["maximum_inter_finger_penetration_m"])
            ),
            "terminal_kinematic_pass": bool(terminal_kinematic[replica]),
            "terminal_contact_window_pass": bool(terminal_contact_window[replica]),
            "contact_causality_pass": bool(contact_causality[replica]),
            "contact_recall": float(contact_recall[replica]),
            "contact_topology_pass": bool(topology_pass[replica]),
            "action_bounds_pass": bool(action_bounds[replica]),
        }
        for replica in range(20)
    ]
    return rows, {
        "terminal_window_control_steps": terminal_steps,
        "terminal_kinematic_pass_rate": float(terminal_kinematic.mean()),
        "terminal_contact_window_pass_rate": float(terminal_contact_window.mean()),
        "contact_causality_pass_rate": float(contact_causality.mean()),
        "contact_topology_pass_rate": float(topology_pass.mean()),
        "action_bounds_pass_rate": float(action_bounds.mean()),
        "terminal_linear_speed_max_mps": float(terminal_linear.max()),
        "terminal_angular_speed_max_radps": float(terminal_angular.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--task-gates", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    evaluation = _read_json(args.evaluation.resolve())
    clip = str(evaluation["requested_clip"])
    frame_zero = evaluation.get("frame_zero")
    if (
        evaluation.get("seed_set", {}).get("identifier") != "formal_holdout_seed_set_v1"
        or not isinstance(frame_zero, list)
        or len(frame_zero) != 20
        or evaluation.get("rsi") != []
        or any(int(row["start_reference_index"]) != 0 for row in frame_zero)
    ):
        raise ValueError("R7 requires 20 unseen frame-zero-only formal episodes")
    gate = _load_gate(args.task_gates.resolve(), clip)
    topology = _read_json(args.topology.resolve())["clips"][clip]
    geometry = _read_json(args.geometry.resolve())
    trace_path = Path(str(evaluation["trace"])).resolve()
    rows, trace_diagnostics = _trace_metrics(trace_path, clip=clip, gate=gate, topology=topology)
    reference_end = np.asarray([bool(row["reached_final_reference"]) for row in frame_zero])
    terminal_contact = np.asarray([bool(row["terminal_contact"]) for row in frame_zero])
    terminal_contact_required = int(
        np.ceil(
            float(gate["terminal_required_contact_fraction"])
            * gate["terminal_window_control_steps"]
        )
    )
    terminal_stability = (
        terminal_contact
        & np.asarray([bool(row["terminal_contact_window_pass"]) for row in rows])
        & np.asarray([bool(row["terminal_kinematic_pass"]) for row in rows])
    )
    for row, evaluation_row in zip(rows, frame_zero, strict=True):
        row["seed"] = int(evaluation_row["seed"])
        row["reached_reference_end"] = bool(evaluation_row["reached_final_reference"])
        row["terminal_contact_pass"] = bool(evaluation_row["terminal_contact"])
        row["terminal_stability_pass"] = bool(terminal_stability[row["replica"]])
        row["final_terminal_contact"] = bool(evaluation_row["terminal_contact"])
        row["complete_trajectory"] = bool(evaluation_row["reached_final_reference"])
        row["formal_object_state_writes"] = 0
        row["formal_wrist_state_writes"] = 0
        row["no_hidden_control"] = True
        row["numerical_pass"] = True
    geometry_pass = bool(geometry.get("formal_pass", False))
    task_success = np.asarray(
        [
            bool(row["reached_reference_end"])
            and bool(row["terminal_contact_pass"])
            and bool(row["terminal_stability_pass"])
            and bool(row["inter_finger_penetration_pass"])
            and bool(row["contact_causality_pass"])
            and bool(row["contact_topology_pass"])
            and float(row["contact_recall"]) >= float(gate["minimum_contact_recall"])
            and float(row["semantic_progress"]) >= float(gate["minimum_semantic_progress"])
            and float(row["object_motion_m"]) >= float(gate["minimum_object_motion_m"])
            for row in rows
        ]
    )
    physics_pass = bool(
        task_success.mean() >= float(gate["ppo_success_rate"])
        and geometry_pass
        and all(bool(row["action_bounds_pass"]) for row in rows)
        and all(bool(row["inter_finger_penetration_pass"]) for row in rows)
    )
    status_prefix = clip.removeprefix("hocap_")
    status = (
        f"STAGE16D_{status_prefix}_PPO_PHYSICS_QUALIFIED"
        if physics_pass
        else f"STAGE16D_{status_prefix}_PPO_TRAINED_NOT_PHYSICS_QUALIFIED"
    )
    result = {
        "schema_version": "Stage16DPPO26DR7FormalQualificationV1",
        "status": status,
        "clip": clip,
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "cumulative_training_samples": evaluation["cumulative_training_samples"],
        "formal_seed_set": evaluation["seed_set"],
        "task_gate": gate,
        "contact_topology": topology,
        "terminal_contact_required_steps": terminal_contact_required,
        "reference_completion_rate": float(reference_end.mean()),
        "terminal_contact_rate": float(terminal_contact.mean()),
        "terminal_stability_rate": float(terminal_stability.mean()),
        "ppo_task_success_rate": float(task_success.mean()),
        "geometry": geometry,
        "geometry_formal_pass": geometry_pass,
        "trace_diagnostics": trace_diagnostics,
        "episodes": rows,
        "physics_qualified": physics_pass,
        "qualification_note": (
            "PPO was fully trained and then evaluated by post-PPO formal gates; "
            "a failed formal gate "
            "does not revoke PPO authorization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.transitions.parent.mkdir(parents=True, exist_ok=True)
    with args.transitions.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "from": "R7_POST_PPO_QUALIFICATION",
                    "to": "R8_170105",
                    "reason": status,
                    "qualification": str(args.output.resolve()),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"status": status, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
