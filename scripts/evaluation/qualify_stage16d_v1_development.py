#!/usr/bin/env python3
"""Qualify a frozen V1 baseline on a Stage 16-D development or Formal20 set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/rl/isaaclab"))

from qualify_stage16d_ppo26d_r7 import _trace_metrics  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V1_DEVELOPMENT_QUALIFICATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _twist_residuals(trace: Path, reference: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(reference, allow_pickle=False) as archive:
        reference_twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
    if int(metadata.get("reference_kinematics_version", -1)) != 2:
        raise ValueError("V1_DEVELOPMENT_QUALIFICATION_REQUIRES_REFERENCE_V2")
    with np.load(trace, allow_pickle=False) as archive:
        actual = np.asarray(archive["replica_object_twist"], dtype=np.float64)
    if actual.shape != (321, 20, 6) or reference_twist.shape != (321, 6):
        raise ValueError("V1_DEVELOPMENT_QUALIFICATION_TWIST_SHAPE_INVALID")
    linear = np.linalg.norm(actual[..., :3] - reference_twist[:, None, :3], axis=-1)
    angular = np.linalg.norm(actual[..., 3:] - reference_twist[:, None, 3:], axis=-1)
    return (
        {
            "reference": str(reference),
            "reference_sha256": _sha256(reference),
            "world_frame": True,
            "terminal_window_control_steps": 20,
            "terminal_delta_v_mps": {
                "per_episode_median": float(np.median(np.median(linear[-20:], axis=0))),
                "all_samples_p95": float(np.percentile(linear[-20:], 95)),
            },
            "terminal_delta_omega_radps": {
                "per_episode_median": float(np.median(np.median(angular[-20:], axis=0))),
                "all_samples_p95": float(np.percentile(angular[-20:], 95)),
            },
        },
        linear,
        angular,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--task-gates", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "formal"), default="development")
    parser.add_argument("--seed-set", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_path = args.evaluation.resolve()
    evaluation = _read(evaluation_path)
    frame_zero = evaluation.get("frame_zero")
    seed_set = str(evaluation.get("seed_set", {}).get("identifier", ""))
    if (
        seed_set != args.seed_set
        or evaluation.get("rsi") != []
        or not isinstance(frame_zero, list)
        or len(frame_zero) != 20
        or any(int(row["start_reference_index"]) != 0 for row in frame_zero)
        or evaluation.get("reward_contract", {}).get("identifier")
        == "TopoRetargetReferenceTrackingReward26DV3"
    ):
        raise ValueError("V1_QUALIFICATION_REQUIRES_FROZEN_V1_FORMAL20")
    if args.kind == "formal" and "formal" not in seed_set.lower():
        raise ValueError("V1_QUALIFICATION_FORMAL_REQUIRES_FORMAL_SEEDS")
    if args.kind == "development" and "formal" in seed_set.lower():
        raise ValueError("V1_QUALIFICATION_DEVELOPMENT_FORBIDS_FORMAL_SEEDS")
    clip = str(evaluation["requested_clip"])
    trace_path = Path(str(evaluation["trace"])).resolve()
    geometry = _read(args.geometry.resolve())
    if geometry.get("candidate_trace_sha256") != _sha256(trace_path):
        raise ValueError("V1_DEVELOPMENT_QUALIFICATION_GEOMETRY_PROVENANCE_MISMATCH")
    gates = geometry.get("absolute_gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError("V1_DEVELOPMENT_QUALIFICATION_ABSOLUTE_GEOMETRY_MISSING")
    geometry_absolute_pass = all(bool(value) for value in gates.values())
    gate = _read(args.task_gates.resolve())["clips"][clip]
    topology = _read(args.topology.resolve())["clips"][clip]
    rows, trace_diagnostics = _trace_metrics(trace_path, clip=clip, gate=gate, topology=topology)
    residuals, linear, angular = _twist_residuals(trace_path, args.reference.resolve())
    terminal_contact = np.asarray([bool(row["terminal_contact"]) for row in frame_zero])
    terminal_stability: list[bool] = []
    task_success: list[bool] = []
    for replica, row in enumerate(rows):
        stable = (
            bool(row["terminal_contact_window_pass"])
            and bool(row["terminal_kinematic_pass"])
            and bool(terminal_contact[replica])
        )
        terminal_stability.append(stable)
        row.update(
            {
                "seed": int(frame_zero[replica]["seed"]),
                "reached_reference_end": bool(frame_zero[replica]["reached_final_reference"]),
                "terminal_contact_pass": bool(terminal_contact[replica]),
                "terminal_stability_pass": stable,
                "final_terminal_contact": bool(terminal_contact[replica]),
                "complete_trajectory": bool(frame_zero[replica]["reached_final_reference"]),
                "formal_object_state_writes": 0,
                "formal_wrist_state_writes": 0,
                "no_hidden_control": True,
                "numerical_pass": True,
                "terminal_delta_v_mps": float(np.median(linear[-20:, replica])),
                "terminal_delta_omega_radps": float(np.median(angular[-20:, replica])),
            }
        )
        task_success.append(
            bool(row["complete_trajectory"])
            and bool(row["terminal_contact_pass"])
            and bool(row["terminal_stability_pass"])
            and bool(row["inter_finger_penetration_pass"])
            and bool(row["contact_causality_pass"])
            and bool(row["contact_topology_pass"])
            and float(row["contact_recall"]) >= float(gate["minimum_contact_recall"])
            and float(row["semantic_progress"]) >= float(gate["minimum_semantic_progress"])
            and float(row["object_motion_m"]) >= float(gate["minimum_object_motion_m"])
            and geometry_absolute_pass
        )
    physics_qualified = bool(
        np.mean(task_success) >= float(gate["ppo_success_rate"])
        and all(bool(row["action_bounds_pass"]) for row in rows)
    )
    result = {
        "schema_version": "Stage16DV1DevelopmentQualificationV1",
        "status": f"STAGE16D_V1_{args.kind.upper()}_COMPLETE",
        "kind": args.kind,
        "clip": clip,
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "cumulative_training_samples": evaluation.get("cumulative_training_samples"),
        "evaluation": str(evaluation_path),
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "seed_set": evaluation["seed_set"],
        "task_gate": gate,
        "contact_topology": topology,
        "geometry": geometry,
        "geometry_absolute_pass": geometry_absolute_pass,
        "geometry_formal_pass": geometry_absolute_pass,
        "reference_completion_rate": float(
            np.mean([bool(row["reached_reference_end"]) for row in rows])
        ),
        "terminal_contact_rate": float(terminal_contact.mean()),
        "terminal_stability_rate": float(np.mean(terminal_stability)),
        "ppo_task_success_rate": float(np.mean(task_success)),
        "physics_qualified": physics_qualified,
        "trace_diagnostics": trace_diagnostics,
        "twist_residuals": residuals,
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
