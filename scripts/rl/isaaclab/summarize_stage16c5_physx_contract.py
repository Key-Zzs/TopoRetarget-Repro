#!/usr/bin/env python3
"""Fail-close the frozen R2 PhysX matrix and emit a selection ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GPU_IDS = ("G0", "G1", "G2", "G3", "G4", "G5")
CPU_IDS = ("C0",)
GPU_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
EARLY_STAGES = ("S0", "S1", "S2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R2_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage_payload(candidate_dir: Path, stage: str) -> tuple[Path | None, dict[str, Any] | None]:
    names = {
        "S0": "S0_smoke.json",
        "S1": "S1_pre_contact.json",
        "S2": "S2_contact_onset.json",
        "S3": "S3_sustained_contact.json",
        "S4": "S4_post_contact.json",
        "S5": "S5_full_replication.json",
    }
    path = candidate_dir / names[stage]
    return (path, _read(path)) if path.exists() else (None, None)


def _stage_summary(path: Path | None, payload: dict[str, Any] | None) -> dict[str, object]:
    if path is None or payload is None:
        return {"status": "NOT_RUN"}
    tolerance = payload.get("stage_result", {}).get("global_tolerances", {})
    if not isinstance(tolerance, dict):
        tolerance = {}
    return {
        "status": str(payload.get("result")),
        "path": str(path),
        "pid": payload.get("process", {}).get("pid"),
        "pgid": payload.get("process", {}).get("pgid"),
        "runtime_values_match_contract": payload.get("runtime_config", {}).get(
            "runtime_values_match_contract"
        ),
        "tolerance_status": tolerance.get("status"),
    }


def main() -> int:
    args = parse_args()
    matrix = _read(args.matrix)
    if matrix.get("matrix_frozen") is not True:
        raise ValueError("STAGE16C5A_R2_CANDIDATE_MATRIX_NOT_FROZEN")
    candidates = matrix.get("candidates")
    if (
        not isinstance(candidates, dict)
        or tuple(matrix.get("candidate_order", ())) != (*GPU_IDS, *CPU_IDS)
        or set(candidates) != {*GPU_IDS, *CPU_IDS}
    ):
        raise ValueError("STAGE16C5A_R2_CANDIDATE_MATRIX_CONTENT_DRIFT")

    rows: dict[str, dict[str, object]] = {}
    for candidate_id in GPU_IDS:
        candidate_dir = args.reports_dir / "candidates" / candidate_id
        reports = {stage: _stage_payload(candidate_dir, stage) for stage in GPU_STAGES}
        for stage in EARLY_STAGES:
            if reports[stage][1] is None:
                raise ValueError(f"STAGE16C5A_R2_REQUIRED_STAGE_MISSING:{candidate_id}:{stage}")
        early_pass = all(reports[stage][1]["result"] == "PASS" for stage in EARLY_STAGES)
        full_pass = early_pass and all(
            reports[stage][1] is not None and reports[stage][1]["result"] == "PASS"
            for stage in GPU_STAGES[3:]
        )
        rows[candidate_id] = {
            "identifier": candidates[candidate_id]["identifier"],
            "config_sha256": candidates[candidate_id]["config_sha256"],
            "early_pass": early_pass,
            "full_pass": full_pass,
            "stages": {stage: _stage_summary(*reports[stage]) for stage in GPU_STAGES},
        }

    cpu_rows: dict[str, dict[str, object]] = {}
    for candidate_id in CPU_IDS:
        candidate_dir = args.reports_dir / "candidates" / candidate_id
        reports = {stage: _stage_payload(candidate_dir, stage) for stage in GPU_STAGES}
        cpu_rows[candidate_id] = {
            "identifier": candidates[candidate_id]["identifier"],
            "config_sha256": candidates[candidate_id]["config_sha256"],
            "stages": {stage: _stage_summary(*reports[stage]) for stage in GPU_STAGES},
        }

    eligible = [candidate_id for candidate_id, row in rows.items() if row["full_pass"]]
    if eligible:
        decision = "SELECTED_PENDING_C3P_C4P"
        selected_candidate = sorted(eligible)[0]
        permitted = ["C3P_SEMANTIC_REGRESSION"]
        blocked = []
    else:
        decision = "NO_GPU_PHYSX_CONTRACT_SELECTED"
        selected_candidate = None
        permitted = ["CPU_DIAGNOSTIC_ONLY"]
        blocked = ["C3P", "C4P", "O1", "C5B", "C5C", "PPO"]
    selection = {
        "schema_version": "stage16c5a_r2_physx_contract_selection_v1",
        "matrix_path": str(args.matrix),
        "matrix_sha256": matrix.get("matrix_sha256"),
        "decision": decision,
        "selected_candidate_id": selected_candidate,
        "gpu_candidates": rows,
        "cpu_diagnostic_candidates": cpu_rows,
        "permitted_next": permitted,
        "blocked_next": blocked,
        "selection_rule": "all S0-S5 must pass; no tolerance softening or fallback allowed",
    }
    transitions = []
    for candidate_id, row in rows.items():
        first_failed = next(
            (stage for stage, outcome in row["stages"].items() if outcome["status"] == "FAIL"),
            None,
        )
        transitions.append(
            {
                "candidate_id": candidate_id,
                "first_failed_stage": first_failed,
                "status": "REJECTED" if first_failed else "ELIGIBLE",
            }
        )
    summary = {
        "schema_version": "stage16c5a_r2_physx_contract_summary_v1",
        "result": decision,
        "selection": selection,
        "failure_transitions": transitions,
    }
    _write_once(args.output_dir / "physics_contract_selection.json", selection)
    _write_once(args.output_dir / "stage16c5a_r2_summary.json", summary)
    transition_path = args.output_dir / "stage16c5a_r2_failure_transitions.jsonl"
    if transition_path.exists():
        raise FileExistsError(f"STAGE16C5A_R2_REFUSES_OVERWRITE: {transition_path}")
    transition_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions), encoding="utf-8"
    )
    print(json.dumps({"result": decision, "selected_candidate_id": selected_candidate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
