#!/usr/bin/env python3
"""Export real selected-episode V4 trace slices from one Formal20 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_REPRESENTATIVE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _source_pass(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("source_tip_recall") or 0.0) >= 0.50
        and float(row.get("persistent_source_tip_recall") or 0.0) >= 0.50
    )


def _select(
    *, qualification: dict[str, Any], suite: dict[str, Any], audit: dict[str, Any]
) -> dict[str, dict[str, Any] | None]:
    q_rows = qualification.get("episodes")
    suite_rows = suite.get("episodes")
    contact_rows = audit.get("per_replica")
    if (
        not isinstance(q_rows, list)
        or not isinstance(suite_rows, list)
        or not isinstance(contact_rows, list)
        or len(q_rows) != 20
        or len(suite_rows) != 20
        or len(contact_rows) != 20
    ):
        raise ValueError("STRICT_V4_REPRESENTATIVE_FORMAL20_ROWS_REQUIRED")
    rows: list[dict[str, Any]] = []
    for replica, (q, suite_row, contact) in enumerate(
        zip(q_rows, suite_rows, contact_rows, strict=True)
    ):
        if int(q["replica"]) != replica or int(suite_row["replica"]) != replica:
            raise ValueError("STRICT_V4_REPRESENTATIVE_REPLICA_ORDER_INVALID")
        rows.append(
            {
                "replica": replica,
                "seed": int(q["seed"]),
                "qualified": bool(suite_row["qualified_success"]),
                "physics": bool(suite_row["physics_success"]),
                "interaction": bool(suite_row["qualified_success"]) and _source_pass(contact),
                "source_tip_recall": float(contact["source_tip_recall"] or 0.0),
                "persistent_source_tip_recall": float(
                    contact["persistent_source_tip_recall"] or 0.0
                ),
                "semantic_progress": float(q["semantic_progress"]),
                "final_error_m": float(suite_row["E_t_terminal_cm"]) / 100.0,
            }
        )
    selection: dict[str, dict[str, Any] | None] = {
        "best_interaction_qualified": None,
        "best_physics_qualified": None,
        "representative_source_contact_failure": min(
            rows,
            key=lambda row: (
                row["persistent_source_tip_recall"],
                row["source_tip_recall"],
                -row["final_error_m"],
                row["replica"],
            ),
        ),
    }
    interaction = [row for row in rows if row["interaction"]]
    physics = [row for row in rows if row["physics"]]
    if interaction:
        selection["best_interaction_qualified"] = max(
            interaction,
            key=lambda row: (
                row["persistent_source_tip_recall"],
                row["source_tip_recall"],
                row["semantic_progress"],
                -row["final_error_m"],
            ),
        )
    if physics:
        selection["best_physics_qualified"] = max(
            physics,
            key=lambda row: (row["semantic_progress"], -row["final_error_m"]),
        )
    events = audit.get("no_tip_no_hand_flight_events")
    if not isinstance(events, list):
        raise ValueError("STRICT_V4_REPRESENTATIVE_FLIGHT_EVENTS_MISSING")
    no_hand = [row for row in events if row.get("event_type") == "NO_HAND_OBJECT_CONTACT_FLIGHT"]
    selection["representative_no_hand_flight_recontact"] = (
        max(
            no_hand,
            key=lambda row: (
                int(row["duration_control_steps"]),
                -int(row["replica"]),
            ),
        )
        if no_hand
        else None
    )
    return selection


def _slice(
    trace_path: Path, *, replica: int, role: str, selected: dict[str, Any]
) -> dict[str, np.ndarray]:
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays: dict[str, np.ndarray] = {}
        for name in archive.files:
            value = np.asarray(archive[name])
            if name.startswith("replica_") and value.shape[:2] == (321, 20):
                arrays[name.removeprefix("replica_")] = value[:, replica]
            elif name in {
                "embedded_reference_object_pose",
                "embedded_reference_wrist_pose",
                "embedded_reference_finger_q",
                "embedded_reference_tracked_links",
                "object_twist_reference",
                "fingertip_link_names",
                "hand_body_names",
                "fingertip_force_sensor_indices",
            }:
                arrays[name] = value
        arrays.update(
            {
                "representative_trace_schema_version": np.asarray(
                    "Stage16DStrictPerFingerV4RepresentativeTraceV1"
                ),
                "representative_trace_role": np.asarray(role),
                "selected_replica": np.asarray(replica, dtype=np.int64),
                "selected_seed": np.asarray(int(selected["seed"]), dtype=np.int64),
                "source_formal_trace_sha256": np.asarray(_sha256(trace_path)),
            }
        )
    return arrays


def _write(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--evaluation-suite", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = _read(args.evaluation.resolve())
    qualification = _read(args.qualification.resolve())
    suite = _read(args.evaluation_suite.resolve())
    audit = _read(args.source_audit.resolve())
    trace = args.source_trace.resolve()
    seed_set = str(evaluation.get("seed_set", {}).get("identifier", ""))
    if (
        qualification.get("status") != "STAGE16D_STRICT_V4_FORMAL_COMPLETE"
        or qualification.get("trace") != str(trace)
        or audit.get("trace", {}).get("sha256") != _sha256(trace)
        or suite.get("evaluation_kind") != "formal"
        or "formal" not in seed_set.lower()
    ):
        raise ValueError("STRICT_V4_REPRESENTATIVE_REQUIRES_FROZEN_FORMAL_HOLDOUT")
    selected = _select(qualification=qualification, suite=suite, audit=audit)
    output = args.output_dir.resolve()
    outputs: dict[str, dict[str, str] | str] = {}
    for role, row in selected.items():
        if row is None:
            outputs[role] = "NO_SUCH_EPISODE"
            continue
        replica = int(row["replica"])
        path = output / f"{role}.npz"
        outputs[role] = {
            "path": str(path),
            "sha256": _write(path, _slice(trace, replica=replica, role=role, selected=row)),
        }
    manifest = {
        "schema_version": "Stage16DStrictPerFingerV4RepresentativeTraceExportV1",
        "status": "STAGE16D_STRICT_V4_REPRESENTATIVE_TRACES_EXPORTED",
        "formal_evaluation": str(args.evaluation.resolve()),
        "formal_qualification": str(args.qualification.resolve()),
        "evaluation_suite": str(args.evaluation_suite.resolve()),
        "source_audit": str(args.source_audit.resolve()),
        "source_trace": str(trace),
        "source_trace_sha256": _sha256(trace),
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
