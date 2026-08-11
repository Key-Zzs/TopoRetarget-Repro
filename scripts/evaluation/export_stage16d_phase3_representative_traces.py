#!/usr/bin/env python3
"""Export provenance-bearing best and representative-failure Phase 3 traces.

The PPO evaluator stores one canonical all-replica physical trace.  This helper
does not replay, simulate, or alter it: it selects the requested formal
replicas from the frozen evaluation/qualification receipts and materializes
two self-contained copies whose metadata tells the replay tool which replica
to render.  The selected actual/reference world twists and the exact Reward V2
twist terms are recorded explicitly for each role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PHASE3_REPRESENTATIVE_TRACE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_success(row: dict[str, Any]) -> bool:
    required = (
        "complete_trajectory",
        "terminal_contact_pass",
        "terminal_stability_pass",
        "inter_finger_penetration_pass",
        "contact_causality_pass",
        "contact_topology_pass",
    )
    if any(name not in row for name in required):
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_QUALIFICATION_FIELDS_MISSING")
    return all(bool(row[name]) for name in required)


def _final_error(evaluation_row: dict[str, Any]) -> float:
    tracking = evaluation_row.get("object_tracking_error_m")
    if not isinstance(tracking, dict) or not isinstance(tracking.get("final"), (int, float)):
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_FINAL_ERROR_MISSING")
    return float(tracking["final"])


def _select_replicas(
    *, evaluation: dict[str, Any], qualification: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_rows = evaluation.get("frame_zero")
    qualification_rows = qualification.get("episodes")
    if not isinstance(evaluation_rows, list) or not isinstance(qualification_rows, list):
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_EPISODE_LIST_MISSING")
    if len(evaluation_rows) != 20 or len(qualification_rows) != 20:
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_REQUIRES_20_FORMAL_EPISODES")

    joined: list[dict[str, Any]] = []
    for replica, (evaluation_row, qualification_row) in enumerate(
        zip(evaluation_rows, qualification_rows, strict=True)
    ):
        if not isinstance(evaluation_row, dict) or not isinstance(qualification_row, dict):
            raise ValueError("PHASE3_REPRESENTATIVE_TRACE_EPISODE_OBJECT_REQUIRED")
        seed = evaluation_row.get("seed")
        if not isinstance(seed, int) or qualification_row.get("seed") != seed:
            raise ValueError("PHASE3_REPRESENTATIVE_TRACE_SEED_PROVENANCE_MISMATCH")
        joined.append(
            {
                "replica": replica,
                "seed": seed,
                "success": _episode_success(qualification_row),
                "final_error_m": _final_error(evaluation_row),
                "semantic_progress": float(qualification_row.get("semantic_progress", 0.0)),
            }
        )

    best = max(
        joined,
        key=lambda row: (
            bool(row["success"]),
            float(row["semantic_progress"]),
            -float(row["final_error_m"]),
            -int(row["replica"]),
        ),
    )
    failures = sorted(
        (row for row in joined if not bool(row["success"])),
        key=lambda row: (float(row["final_error_m"]), int(row["replica"])),
    )
    if not failures:
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_NO_FORMAL_FAILURE")
    failure = failures[(len(failures) - 1) // 2]
    return {"best_progress": best, "representative_failure": failure}, joined, failures


def _role_arrays(
    arrays: dict[str, np.ndarray], *, selected: dict[str, Any], role: str, source_sha256: str
) -> dict[str, np.ndarray]:
    required = {
        "reference_kinematics_version",
        "object_twist_reference",
        "replica_object_twist",
        "replica_object_pose",
        "replica_hand_collision_body_pose",
        "replica_contact_pair_presence",
        "replica_action",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"PHASE3_REPRESENTATIVE_TRACE_FIELDS_MISSING:{missing}")
    version = int(np.asarray(arrays["reference_kinematics_version"]).item())
    if version != 2:
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_REQUIRES_REFERENCE_V2")
    actual = np.asarray(arrays["replica_object_twist"], dtype=np.float64)
    reference = np.asarray(arrays["object_twist_reference"], dtype=np.float64)
    if actual.shape != (321, 20, 6) or reference.shape != (321, 6):
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_TWIST_SHAPE_INVALID")
    replica = int(selected["replica"])
    if not 0 <= replica < actual.shape[1]:
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_REPLICA_OUT_OF_RANGE")
    selected_actual = actual[:, replica]
    delta = selected_actual - reference
    linear_error = np.linalg.norm(delta[:, :3], axis=-1)
    angular_error = np.linalg.norm(delta[:, 3:], axis=-1)
    result = dict(arrays)
    result.update(
        {
            "representative_trace_schema_version": np.asarray(
                "Stage16DPhase3RepresentativeTraceV1"
            ),
            "representative_trace_role": np.asarray(role),
            "selected_replica": np.asarray(replica, dtype=np.int64),
            "selected_seed": np.asarray(int(selected["seed"]), dtype=np.int64),
            "selection_reason": np.asarray(
                "max_success_then_semantic_progress_then_lower_final_error"
                if role == "best_progress"
                else "median_final_error_among_formal_qualification_failures"
            ),
            "source_formal_trace_sha256": np.asarray(source_sha256),
            "selected_object_linear_velocity_world": selected_actual[:, :3].astype(np.float32),
            "selected_object_angular_velocity_world": selected_actual[:, 3:].astype(np.float32),
            "selected_object_linear_velocity_reference_world": reference[:, :3].astype(np.float32),
            "selected_object_angular_velocity_reference_world": reference[:, 3:].astype(np.float32),
            "selected_delta_object_linear_velocity_world": delta[:, :3].astype(np.float32),
            "selected_delta_object_angular_velocity_world": delta[:, 3:].astype(np.float32),
            "selected_error_obj_vel": linear_error.astype(np.float32),
            "selected_error_obj_ang_vel": angular_error.astype(np.float32),
            "selected_reward_obj_vel": np.exp(-np.square(linear_error / 0.075)).astype(np.float32),
            "selected_reward_obj_ang_vel": np.exp(-np.square(angular_error / 0.125)).astype(
                np.float32
            ),
        }
    )
    return result


def export_representative_traces(
    *,
    evaluation_path: Path,
    qualification_path: Path,
    source_trace_path: Path,
    best_output: Path,
    failure_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    evaluation = _read_json(evaluation_path.resolve())
    qualification = _read_json(qualification_path.resolve())
    if evaluation.get("seed_set", {}).get("identifier") != "formal_holdout_seed_set_v1":
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_REQUIRES_FORMAL_HOLDOUT")
    if qualification.get("kind") != "formal" or qualification.get("trace") != str(
        source_trace_path.resolve()
    ):
        raise ValueError("PHASE3_REPRESENTATIVE_TRACE_PROVENANCE_MISMATCH")
    selected, joined, failures = _select_replicas(
        evaluation=evaluation, qualification=qualification
    )
    with np.load(source_trace_path.resolve(), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    source_sha256 = _sha256(source_trace_path.resolve())
    outputs = {
        "best_progress": best_output.resolve(),
        "representative_failure": failure_output.resolve(),
    }
    for role, output in outputs.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            **_role_arrays(arrays, selected=selected[role], role=role, source_sha256=source_sha256),
        )
    result = {
        "schema_version": "Stage16DPhase3RepresentativeTraceExportV1",
        "status": "STAGE16D_PHASE3_REPRESENTATIVE_TRACES_EXPORTED",
        "formal_evaluation": str(evaluation_path.resolve()),
        "formal_qualification": str(qualification_path.resolve()),
        "source_trace": str(source_trace_path.resolve()),
        "source_trace_sha256": source_sha256,
        "selected": selected,
        "failure_count": len(failures),
        "formal_episode_count": len(joined),
        "outputs": {
            role: {"path": str(path), "sha256": _sha256(path)} for role, path in outputs.items()
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--best-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_representative_traces(
        evaluation_path=args.evaluation,
        qualification_path=args.qualification,
        source_trace_path=args.source_trace,
        best_output=args.best_output,
        failure_output=args.failure_output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps({"status": result["status"], "outputs": result["outputs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
