#!/usr/bin/env python3
"""Freeze the shared empirical stable dynamic-contact reference before optimization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.contracts import (  # noqa: E402
    GEOMETRY_QUERY_CONTRACT,
)
from toporetarget.rl.geometry_audit.dynamic_contact_reference import (  # noqa: E402
    SelectedStableCalibrationV1,
    freeze_empirical_dynamic_contact_reference,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"
SOURCE_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_rows(qualification: dict[str, Any]) -> tuple[SelectedStableCalibrationV1, ...]:
    rows = []
    for result in qualification["results"]:
        suffix = result["object_id"].removeprefix("hocap_")
        source = _load(SOURCE_ROOT / f"source_runtime_penetration_{suffix}.json")
        epsilon = GEOMETRY_QUERY_CONTRACT.metric_epsilon_m
        rows.append(
            SelectedStableCalibrationV1(
                object_id=result["object_id"],
                family_id=result["family_id"],
                candidate_id=result["candidate_id"],
                qualification_sha256=result["report_sha256"],
                replica_max_penetration_m=tuple(result["replica_max_penetration_m"]),
                replica_active_p95_penetration_m=tuple(result["replica_active_p95_penetration_m"]),
                v1_max_limit_m=float(source["max_penetration_m"]) * 1.10 + epsilon,
                v1_active_p95_limit_m=float(source["p95_penetration_m"]) * 1.10 + epsilon,
                stable_gate_passed=bool(result["qualification"]["passed"]),
            )
        )
    return tuple(rows)


def main() -> int:
    qualification = _load(REPORT_ROOT / "stable_grasp_qualification.json")
    rows = _selected_rows(qualification)
    required = tuple((row.object_id, row.family_id) for row in rows)
    serialized = {
        "schema_version": "SelectedStableCalibrationSetV1",
        "rows": [row.as_dict() for row in rows],
        "required_object_family_pairs": [list(pair) for pair in required],
    }
    _write(REPORT_ROOT / "selected_stable_calibrations.json", serialized)
    _write(
        REPORT_ROOT / "calibration_provenance.json",
        {
            "schema_version": "StableGraspCalibrationProvenanceV1",
            "selected": [
                {
                    "object_id": row.object_id,
                    "family_id": row.family_id,
                    "candidate_id": row.candidate_id,
                    "qualification_sha256": row.qualification_sha256,
                }
                for row in rows
            ],
            "corrected_trajectory_used": False,
            "optimizer_result_observed": False,
            "ppo_result_observed": False,
            "shared_rule": True,
        },
    )
    if not rows or any(not row.stable_gate_passed for row in rows):
        reference = {
            "schema_version": "EmpiricalStableDynamicContactReferenceV1",
            "status": "STAGE16D_STABLE_GRASP_CALIBRATION_BLOCKED",
            "created": False,
            "reason": "stable 20-replica coverage is incomplete",
        }
    elif all(row.v1_passed() for row in rows):
        reference = {
            "schema_version": "EmpiricalStableDynamicContactReferenceV1",
            "status": "NOT_CREATED_V1_ATTAINABLE",
            "created": False,
            "reason": "every selected stable calibration passes V1 in 20/20 replicas",
        }
    else:
        reference = {
            "status": "STAGE16D_EMPIRICAL_DYNAMIC_CONTACT_REFERENCE_FROZEN",
            "created": True,
            **freeze_empirical_dynamic_contact_reference(
                rows, required_object_family_pairs=required
            ),
        }
    _write(REPORT_ROOT / "empirical_dynamic_contact_reference.json", reference)
    print(json.dumps({"status": reference["status"], "output": str(REPORT_ROOT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
