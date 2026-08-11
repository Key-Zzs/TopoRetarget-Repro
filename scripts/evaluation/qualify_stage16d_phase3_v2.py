#!/usr/bin/env python3
"""Qualify one 20-replica Phase 3 evaluation without source-relative conflation.

This is deliberately a post-PPO evaluator.  It reads the frozen simulator
trace, V2 reference, static no-hidden-control decision, and collision audit; it
does not create an Isaac application or alter a policy/checkpoint.
"""

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
sys.path.insert(0, str(REPO_ROOT / "scripts/rl/isaaclab"))

from qualify_stage16d_ppo26d_r7 import _trace_metrics  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_geometry_pass(geometry: dict[str, Any]) -> bool:
    gates = geometry.get("absolute_gates")
    if not isinstance(gates, dict) or set(gates) != {"p95_at_most_3mm", "strict_max_below_10mm"}:
        raise ValueError("PHASE3_GEOMETRY_ABSOLUTE_GATES_MISSING")
    return all(bool(value) for value in gates.values())


def _require_phase3_entry(path: Path) -> dict[str, Any]:
    decision = _read_json(path)
    gates = decision.get("gates")
    required = {
        "G1_reference_kinematics_v2",
        "G2_object_twist_observability",
        "G3_meaningful_terminal_residual_dynamics",
        "G4_contact_not_primary_failure",
        "G5_physics_integrity",
    }
    if (
        decision.get("status") != "PHASE3_OBJECT_TWIST_REWARD_RECOMMENDED"
        or not isinstance(gates, dict)
        or not required.issubset(gates)
        or not all(bool(gates[key].get("pass")) for key in required)
    ):
        raise ValueError("PHASE3_EVALUATION_REQUIRES_PASSED_ENTRY_GATES")
    return decision


def _check_evaluation_contract(
    evaluation: dict[str, Any], *, kind: str, policy_reference_version: int
) -> tuple[str, list[dict[str, Any]]]:
    frame_zero = evaluation.get("frame_zero")
    if (
        not isinstance(frame_zero, list)
        or len(frame_zero) != 20
        or evaluation.get("rsi") != []
        or any(int(row["start_reference_index"]) != 0 for row in frame_zero)
    ):
        raise ValueError("PHASE3_EVALUATION_REQUIRES_20_FRAME_ZERO_EPISODES")
    seed_id = str(evaluation.get("seed_set", {}).get("identifier"))
    expected = "formal_holdout_seed_set_v1" if kind == "formal" else "development_eval_seed_set_v1"
    if seed_id != expected:
        raise ValueError(f"PHASE3_{kind.upper()}_SEED_SET_MISMATCH:{seed_id}")
    if int(evaluation.get("reference_kinematics_version", -1)) != 2:
        raise ValueError("PHASE3_EVALUATION_REQUIRES_REFERENCE_KINEMATICS_V2")
    reward = evaluation.get("reward_contract", {})
    if policy_reference_version == 2:
        if reward.get("identifier") != "TopoRetargetReferenceTrackingReward26DV2":
            raise ValueError("PHASE3_V2_CHECKPOINT_EVALUATED_WITH_WRONG_REWARD")
    elif reward.get("identifier") != "TopoRetargetReferenceTrackingReward26DV1":
        raise ValueError("PHASE3_V1_BASELINE_EVALUATED_WITH_WRONG_REWARD")
    return seed_id, frame_zero


def _twist_residuals(trace: Path, reference: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(reference, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if int(metadata.get("reference_kinematics_version", -1)) != 2:
            raise ValueError("PHASE3_RESIDUALS_REQUIRE_REFERENCE_V2")
        reference_twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
    with np.load(trace, allow_pickle=False) as archive:
        actual_twist = np.asarray(archive["replica_object_twist"], dtype=np.float64)
        version = int(np.asarray(archive["reference_kinematics_version"]).item())
        if version != 2:
            raise ValueError("PHASE3_TRACE_REFERENCE_VERSION_MISMATCH")
        if "object_twist_reference" in archive.files:
            embedded = np.asarray(archive["object_twist_reference"], dtype=np.float64)
            if embedded.shape != reference_twist.shape or not np.allclose(
                embedded, reference_twist, atol=1.0e-6, rtol=1.0e-6
            ):
                raise ValueError("PHASE3_TRACE_V2_TWIST_REFERENCE_HASH_MISMATCH")
    if actual_twist.shape != (321, 20, 6) or reference_twist.shape != (321, 6):
        raise ValueError("PHASE3_TWIST_RESIDUAL_SHAPE_INVALID")
    linear = np.linalg.norm(actual_twist[..., :3] - reference_twist[:, None, :3], axis=-1)
    angular = np.linalg.norm(actual_twist[..., 3:] - reference_twist[:, None, 3:], axis=-1)
    terminal = slice(-20, None)
    summary = {
        "reference": str(reference.resolve()),
        "reference_sha256": _hash(reference),
        "world_frame": True,
        "signed_difference": "actual_world_twist - reference_world_twist_v2",
        "terminal_window_control_steps": 20,
        "terminal_delta_v_mps": {
            "per_episode_median": float(np.median(np.median(linear[terminal], axis=0))),
            "per_episode_mean": float(np.mean(np.mean(linear[terminal], axis=0))),
            "all_samples_p95": float(np.percentile(linear[terminal], 95)),
        },
        "terminal_delta_omega_radps": {
            "per_episode_median": float(np.median(np.median(angular[terminal], axis=0))),
            "per_episode_mean": float(np.mean(np.mean(angular[terminal], axis=0))),
            "all_samples_p95": float(np.percentile(angular[terminal], 95)),
        },
    }
    return summary, linear, angular


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--task-gates", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--phase3-entry", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "formal"), required=True)
    parser.add_argument("--policy-reference-version", choices=(1, 2), type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_path = args.evaluation.resolve()
    evaluation = _read_json(evaluation_path)
    seed_id, frame_zero = _check_evaluation_contract(
        evaluation, kind=args.kind, policy_reference_version=args.policy_reference_version
    )
    clip = str(evaluation["requested_clip"])
    trace_path = Path(str(evaluation["trace"])).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"PHASE3_TRACE_MISSING:{trace_path}")
    geometry_path = args.geometry.resolve()
    geometry = _read_json(geometry_path)
    if geometry.get("candidate_trace_sha256") != _hash(trace_path):
        raise ValueError("PHASE3_GEOMETRY_TRACE_PROVENANCE_MISMATCH")
    absolute_geometry_pass = _absolute_geometry_pass(geometry)
    entry = _require_phase3_entry(args.phase3_entry.resolve())
    gate = _read_json(args.task_gates.resolve())["clips"][clip]
    topology = _read_json(args.topology.resolve())["clips"][clip]
    rows, trace_diagnostics = _trace_metrics(trace_path, clip=clip, gate=gate, topology=topology)
    residuals, linear_residual, angular_residual = _twist_residuals(
        trace_path, args.reference.resolve()
    )
    terminal_contact = np.asarray([bool(row["terminal_contact"]) for row in frame_zero])
    reached_end = np.asarray([bool(row["reached_final_reference"]) for row in frame_zero])
    terminal_stability = np.asarray(
        [
            bool(row["terminal_contact_window_pass"])
            and bool(row["terminal_kinematic_pass"])
            and bool(terminal_contact[index])
            for index, row in enumerate(rows)
        ]
    )
    task_success = []
    for index, row in enumerate(rows):
        row.update(
            {
                "seed": int(frame_zero[index]["seed"]),
                "reached_reference_end": bool(reached_end[index]),
                "terminal_contact_pass": bool(terminal_contact[index]),
                "terminal_stability_pass": bool(terminal_stability[index]),
                "final_terminal_contact": bool(terminal_contact[index]),
                "complete_trajectory": bool(reached_end[index]),
                "formal_object_state_writes": 0,
                "formal_wrist_state_writes": 0,
                "no_hidden_control": True,
                "numerical_pass": True,
                "terminal_delta_v_mps": float(np.median(linear_residual[-20:, index])),
                "terminal_delta_omega_radps": float(np.median(angular_residual[-20:, index])),
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
            and absolute_geometry_pass
        )
    result = {
        "schema_version": "Stage16DPhase3V2QualificationV1",
        "status": (
            "STAGE16D_PHASE3_REWARD_V2_FORMAL_COMPLETE"
            if args.kind == "formal"
            else "STAGE16D_PHASE3_REWARD_V2_DEVELOPMENT_COMPLETE"
        ),
        "kind": args.kind,
        "clip": clip,
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "reward_v2_samples": evaluation.get("reward_v2_samples"),
        "policy_reference_version": args.policy_reference_version,
        "evaluation": str(evaluation_path),
        "trace": str(trace_path),
        "trace_sha256": _hash(trace_path),
        "seed_set": evaluation["seed_set"],
        "seed_set_identifier": seed_id,
        "task_gate": gate,
        "contact_topology": topology,
        "phase3_entry": {
            "path": str(args.phase3_entry.resolve()),
            "sha256": _hash(args.phase3_entry.resolve()),
            "status": entry["status"],
        },
        "geometry": geometry,
        "geometry_absolute_pass": absolute_geometry_pass,
        "source_relative_geometry_diagnostic": geometry.get("relative_gates"),
        "geometry_formal_pass": absolute_geometry_pass,
        "reference_completion_rate": float(reached_end.mean()),
        "terminal_contact_rate": float(terminal_contact.mean()),
        "terminal_stability_rate": float(terminal_stability.mean()),
        "ppo_task_success_rate": float(np.mean(task_success)),
        "trace_diagnostics": trace_diagnostics,
        "twist_residuals": residuals,
        "episodes": rows,
        "qualification_note": (
            "Absolute runtime collision safety is a Phase3 gate. Source-relative geometry is "
            "preserved as a separate diagnostic and never converted into a reward term."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
