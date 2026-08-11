#!/usr/bin/env python3
"""Select and materialize replay-ready representative Reward V3 Formal20 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V3_REPLAY_TRACE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_qualified(row: dict[str, Any]) -> bool:
    required = (
        "complete_trajectory",
        "terminal_contact_pass",
        "terminal_stability_pass",
        "inter_finger_penetration_pass",
        "contact_causality_pass",
        "contact_topology_pass",
        "action_bounds_pass",
        "no_hidden_control",
    )
    if any(name not in row for name in required):
        raise ValueError("V3_REPLAY_TRACE_QUALIFICATION_FIELDS_MISSING")
    return all(bool(row[name]) for name in required)


def _joined_rows(evaluation: dict[str, Any], qualification: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation_rows = evaluation.get("frame_zero")
    qualification_rows = qualification.get("episodes")
    if (
        not isinstance(evaluation_rows, list)
        or not isinstance(qualification_rows, list)
        or len(evaluation_rows) != 20
        or len(qualification_rows) != 20
    ):
        raise ValueError("V3_REPLAY_TRACE_REQUIRES_FORMAL20")
    joined: list[dict[str, Any]] = []
    for replica, (evaluation_row, qualification_row) in enumerate(
        zip(evaluation_rows, qualification_rows, strict=True)
    ):
        if (
            not isinstance(evaluation_row, dict)
            or not isinstance(qualification_row, dict)
            or int(qualification_row.get("replica", -1)) != replica
            or evaluation_row.get("seed") != qualification_row.get("seed")
        ):
            raise ValueError("V3_REPLAY_TRACE_REPLICA_OR_SEED_PROVENANCE_MISMATCH")
        final_error = evaluation_row.get("object_tracking_error_m", {}).get("final")
        if not isinstance(final_error, (int, float)):
            raise ValueError("V3_REPLAY_TRACE_FINAL_ERROR_MISSING")
        joined.append(
            {
                "replica": replica,
                "seed": int(evaluation_row["seed"]),
                "qualified": _episode_qualified(qualification_row),
                "semantic_progress": float(qualification_row["semantic_progress"]),
                "final_error_m": float(final_error),
            }
        )
    return joined


def _selected_trace(
    arrays: dict[str, np.ndarray], *, selected: dict[str, Any], role: str, source_sha256: str
) -> dict[str, np.ndarray]:
    required = {
        "replica_object_twist",
        "replica_reference_contact_mask",
        "replica_actual_contact_mask",
        "replica_fingertip_object_pair_force_world",
        "replica_fingertip_object_pair_force_valid",
        "replica_contact_reward",
        "fingertip_link_names",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"V3_REPLAY_TRACE_FIELDS_MISSING:{missing}")
    replica = int(selected["replica"])
    twist = np.asarray(arrays["replica_object_twist"], dtype=np.float64)
    reference_twist = np.asarray(arrays["object_twist_reference"], dtype=np.float64)
    if twist.shape != (321, 20, 6) or reference_twist.shape != (321, 6):
        raise ValueError("V3_REPLAY_TRACE_TWIST_SHAPE_INVALID")
    if not 0 <= replica < 20:
        raise ValueError("V3_REPLAY_TRACE_REPLICA_INVALID")
    delta = twist[:, replica] - reference_twist
    pair_force = np.asarray(arrays["replica_fingertip_object_pair_force_world"], dtype=np.float64)[
        :, replica
    ]
    expected = np.asarray(arrays["replica_reference_contact_mask"], dtype=bool)[:, replica]
    s_contact = (np.linalg.norm(pair_force, axis=-1) * expected).sum(axis=-1)
    return {
        **arrays,
        "representative_trace_schema_version": np.asarray("Stage16DRewardV3RepresentativeTraceV1"),
        "representative_trace_role": np.asarray(role),
        "selected_replica": np.asarray(replica, dtype=np.int64),
        "selected_seed": np.asarray(int(selected["seed"]), dtype=np.int64),
        "selection_reason": np.asarray(
            {
                "best_qualified": "qualified_then_higher_semantic_progress_then_lower_final_error",
                "best_progress": "higher_semantic_progress_then_lower_final_error",
                "representative_failure": "median_final_error_among_unqualified_formal_episodes",
            }[role]
        ),
        "source_formal_trace_sha256": np.asarray(source_sha256),
        "selected_object_linear_velocity_world": twist[:, replica, :3].astype(np.float32),
        "selected_object_angular_velocity_world": twist[:, replica, 3:].astype(np.float32),
        "selected_delta_object_linear_velocity_world": delta[:, :3].astype(np.float32),
        "selected_delta_object_angular_velocity_world": delta[:, 3:].astype(np.float32),
        "selected_reference_contact_mask": expected,
        "selected_actual_contact_mask": np.asarray(
            arrays["replica_actual_contact_mask"], dtype=bool
        )[:, replica],
        "selected_fingertip_object_pair_force_world": pair_force.astype(np.float32),
        "selected_fingertip_object_pair_force_valid": np.asarray(
            arrays["replica_fingertip_object_pair_force_valid"], dtype=bool
        )[:, replica],
        "selected_S_contact_n": s_contact.astype(np.float32),
        "selected_r_contact": np.asarray(arrays["replica_contact_reward"], dtype=np.float32)[
            :, replica
        ],
    }


def _write(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--best-qualified-output", type=Path, required=True)
    parser.add_argument("--best-progress-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_path = args.evaluation.resolve()
    qualification_path = args.qualification.resolve()
    trace_path = args.source_trace.resolve()
    evaluation = _read(evaluation_path)
    qualification = _read(qualification_path)
    seed_set = str(evaluation.get("seed_set", {}).get("identifier", ""))
    if (
        qualification.get("kind") != "formal"
        or qualification.get("status") != "STAGE16D_REWARD_V3_FORMAL_COMPLETE"
        or "formal" not in seed_set.lower()
        or qualification.get("trace") != str(trace_path)
    ):
        raise ValueError("V3_REPLAY_TRACE_REQUIRES_FROZEN_FORMAL_HOLDOUT")
    rows = _joined_rows(evaluation, qualification)
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    source_sha256 = _sha256(trace_path)
    qualified = [row for row in rows if bool(row["qualified"])]
    failures = [row for row in rows if not bool(row["qualified"])]
    selected: dict[str, dict[str, Any]] = {
        "best_progress": max(
            rows,
            key=lambda row: (float(row["semantic_progress"]), -float(row["final_error_m"])),
        )
    }
    if qualified:
        selected["best_qualified"] = max(
            qualified,
            key=lambda row: (float(row["semantic_progress"]), -float(row["final_error_m"])),
        )
    if failures:
        ordered = sorted(
            failures, key=lambda row: (float(row["final_error_m"]), int(row["replica"]))
        )
        selected["representative_failure"] = ordered[(len(ordered) - 1) // 2]
    requested_outputs = {
        "best_qualified": args.best_qualified_output.resolve(),
        "best_progress": args.best_progress_output.resolve(),
        "representative_failure": args.failure_output.resolve(),
    }
    outputs: dict[str, dict[str, str] | None] = {}
    for role, path in requested_outputs.items():
        if role not in selected:
            outputs[role] = None
            continue
        outputs[role] = {
            "path": str(path),
            "sha256": _write(
                path,
                _selected_trace(
                    arrays, selected=selected[role], role=role, source_sha256=source_sha256
                ),
            ),
        }
    manifest = {
        "schema_version": "Stage16DRewardV3RepresentativeTraceExportV1",
        "status": "STAGE16D_REWARD_V3_REPRESENTATIVE_TRACES_EXPORTED",
        "formal_evaluation": str(evaluation_path),
        "formal_qualification": str(qualification_path),
        "source_trace": str(trace_path),
        "source_trace_sha256": source_sha256,
        "formal_episode_count": len(rows),
        "qualified_episode_count": len(qualified),
        "failed_episode_count": len(failures),
        "selected": selected,
        "outputs": outputs,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "outputs": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
