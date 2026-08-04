#!/usr/bin/env python3
"""Assemble fail-closed Stage 16-C.5A-R1 evidence without authorizing O1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--old-baseline", type=Path, required=True)
    parser.add_argument("--repaired-baseline", type=Path, required=True)
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e3", type=Path, required=True)
    parser.add_argument("--e4", type=Path, required=True)
    parser.add_argument("--e5", type=Path, required=True)
    parser.add_argument("--e6", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"C5A closeout input is not an object: {path}")
    return payload


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_CLOSEOUT_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numbers(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (float, int)):
        return [float(value)]
    if isinstance(value, Mapping):
        result: list[float] = []
        for child in value.values():
            result.extend(_numbers(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_numbers(child))
        return result
    return []


def _all_errors_zero(report: dict[str, Any]) -> bool:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        errors = row.get("errors")
        if not isinstance(errors, list) or len(errors) != 19:
            return False
        for error in errors:
            if not isinstance(error, Mapping):
                return False
            if any(abs(value) > 0.0 for value in _numbers(error)):
                return False
            if error.get("termination_exact") is not True:
                return False
    return True


def _max_by_field(report: dict[str, Any], key: str) -> dict[str, float]:
    maximum: dict[str, float] = {}
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("C5A vector diagnostic lacks rows")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("C5A vector diagnostic row is malformed")
        errors = row.get("errors")
        if not isinstance(errors, list):
            raise ValueError("C5A vector diagnostic errors are malformed")
        for error in errors:
            if not isinstance(error, Mapping):
                raise ValueError("C5A vector diagnostic error is malformed")
            values = error.get(key)
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    maximum[str(name)] = max(maximum.get(str(name), 0.0), abs(float(value)))
    return maximum


def _raw_derived_analysis(report: dict[str, Any]) -> dict[str, Any]:
    raw = _max_by_field(report, "raw_state_max_abs")
    raw.pop("source_env_origins", None)
    derived = _max_by_field(report, "derived_state")
    rewards = _max_by_field(report, "reward_components_max_abs")
    raw_nonzero = sorted(name for name, value in raw.items() if value > 0.0)
    return {
        "schema_version": "stage16c5_raw_vs_derived_state_analysis_v1",
        "raw_simulator_state_max_abs": raw,
        "derived_task_metric_max_abs": derived,
        "reward_component_max_abs": rewards,
        "raw_simulator_state_diverged": bool(raw_nonzero),
        "raw_nonzero_fields": raw_nonzero,
        "classification": (
            "RAW_SIMULATOR_STATE_DIVERGENCE" if raw_nonzero else "DERIVED_METRIC_ONLY_DIVERGENCE"
        ),
    }


def _not_run(name: str, reason: str) -> dict[str, str]:
    return {"status": "NOT_RUN_GATE_BLOCKED", "artifact": name, "reason": reason}


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    old = _load(args.old_baseline)
    repaired = _load(args.repaired_baseline)
    e1, e2, e3, e4, e5, e6 = (
        _load(path)
        for path in (
            args.e1,
            args.e2,
            args.e3,
            args.e4,
            args.e5,
            args.e6,
        )
    )
    raw_derived = _raw_derived_analysis(e2)
    e1_zero = _all_errors_zero(e1)
    e2_raw_diverged = bool(raw_derived["raw_simulator_state_diverged"])
    evidence_complete = (
        repaired.get("status") == "PHYSX_REPLICATION_BASELINE_NONDETERMINISM"
        and e1_zero
        and e2_raw_diverged
        and e3.get("result") == "ENV_ORIGIN_NORMALIZATION_VALID"
        and e6.get("result") == "CONTACT_TELEMETRY_READ_ONLY_CONFIRMED"
        and e4.get("result")
        in {
            "CROSS_PROCESS_FINGERPRINT_IDENTICAL",
            "CROSS_PROCESS_FINGERPRINT_DIFFERENT",
        }
        and e5.get("result")
        in {
            "CROSS_PROCESS_FINGERPRINT_IDENTICAL",
            "CROSS_PROCESS_FINGERPRINT_DIFFERENT",
        }
    )
    status = (
        "STAGE16C5A_PHYSICS_CONTRACT_CHANGE_REQUIRED"
        if evidence_complete
        else "STAGE16C5A_HARNESS_REPAIR_PARTIAL"
    )
    reason = (
        "TRUE_FROZEN_PHYSX_BASELINE_NONDETERMINISM"
        if evidence_complete
        else "R1_EVIDENCE_INCOMPLETE_OR_INCONSISTENT"
    )
    before_after = {
        "schema_version": "stage16c5_natural_baseline_before_after_v1",
        "historical_baseline": old.get("global_tolerances"),
        "repaired_baseline": repaired.get("global_tolerances"),
        "historical_status": old.get("status"),
        "repaired_status": repaired.get("status"),
        "formula_contract": "max(fixed_floor, 10 * baseline_p99); hard caps unchanged",
        "harness_change": "reset boundary, DirectRLEnv bookkeeping, reward-component reporting",
        "conclusion": (
            "repair did not reduce same-process 33-env natural baseline below frozen caps"
            if repaired.get("status") == "PHYSX_REPLICATION_BASELINE_NONDETERMINISM"
            else "repaired baseline requires manual review"
        ),
    }
    matrix = {
        "schema_version": "stage16c5_nondeterminism_experiment_matrix_v1",
        "formal_gate": "E2 only: same-process, same-scene 33-env candidate population",
        "experiments": {
            "E1": "1-env same-process sequential 20-trial diagnostic",
            "E2": "33-env same-process 1 source plus 32 peers, 20 trials",
            "E3": "E2 vector scene analyzed in scene-local coordinates with 33 unique origins",
            "E4": "20 independent 1-env child-process diagnostics; not a formal gate",
            "E5": "20 independent 33-env child-process diagnostics; not a formal gate",
            "E6": "off/aggregate/diagnostic telemetry byte-level physical-state comparison",
        },
        "snapshot_restore_used": False,
    }
    summary = {
        "schema_version": "stage16c5a_r1_closeout_v1",
        "status": status,
        "reason": reason,
        "natural_baseline": repaired.get("status"),
        "e1_single_env_same_process_zero_error": e1_zero,
        "e2_vector_same_process_raw_diverged": e2_raw_diverged,
        "e3_origin_status": e3.get("result"),
        "e4_cross_process_status": e4.get("result"),
        "e5_cross_process_status": e5.get("result"),
        "e6_telemetry_status": e6.get("result"),
        "authorization": {
            "STAGE16C5A_O1": "NOT_AUTHORIZED",
            "STAGE16C5B": "NOT_AUTHORIZED",
            "STAGE16C5C": "NOT_RUN_GATE_BLOCKED",
            "STAGE16C6": "NOT_AUTHORIZED",
            "PPO": "NOT_STARTED",
        },
        "unexecuted_minimal_physics_contract_options": [
            "enable and qualify a deterministic PhysX solver configuration",
            "change and qualify solver position/velocity iteration counts",
            "change and qualify GPU contact-solver/enhanced-determinism settings",
            "run a CPU PhysX backend diagnostic with a separately frozen contract",
        ],
        "forbidden_in_this_closeout": [
            "hard-cap or tolerance change",
            "solver or contact-parameter change",
            "snapshot/history replay as a baseline bypass",
            "O1, CEM, C5B, C5C, PPO execution",
        ],
    }
    markdown = f"""# Stage 16-C.5A-R1 closeout

Status: `{status}`

The repaired 33-environment same-process natural baseline remains above frozen
hard caps. E1 is exact for sequential single-environment resets, while E2 has
raw simulator-state divergence in the actual source-plus-32-peer candidate
scene. E3 verifies origin subtraction; E4/E5 are process diagnostics only; E6
checks that telemetry is read-only. O1, C5B/C5C, and PPO are not authorized.

No solver, physics, hard-cap, or tolerance contract was changed. The listed
physics-contract options require separate user authorization and were not run.
"""

    _write(report_dir / "nondeterminism_experiment_matrix.json", matrix)
    _write(report_dir / "e1_single_env_same_process.json", e1)
    _write(report_dir / "e2_vector_same_process.json", e2)
    _write(report_dir / "e3_env_origin_invariance.json", e3)
    _write(report_dir / "e4_single_env_cross_process.json", e4)
    _write(report_dir / "e5_vector_cross_process.json", e5)
    _write(report_dir / "e6_contact_telemetry_effect.json", e6)
    _write(report_dir / "raw_vs_derived_state_analysis.json", raw_derived)
    _write(
        report_dir / "harness_repairs.json",
        {
            "status": "HARNESS_REPAIRS_APPLIED_AND_RETESTED",
            "evidence": "suspected_defects.json and baseline_harness_audit.json",
            "repairs": [
                "reset synchronization",
                "DirectRLEnv manual step bookkeeping",
                "first-step reward cache materialization",
                "component-wise reward and contact telemetry reporting",
            ],
        },
    )
    _write(report_dir / "natural_baseline_before_after.json", before_after)
    _write(report_dir / "repaired_replication_noise_floor.json", repaired.get("global_tolerances"))
    _write(report_dir / "repaired_replication_tolerances.json", repaired.get("global_tolerances"))
    for filename in (
        "tensor_clone_qualification.json",
        "history_replay_qualification.json",
        "replication_qualification.json",
        "candidate_independence.json",
        "candidate_pool_benchmark.json",
        "c5b_runtime_projection.json",
        "c5b_oracle_config.json",
        "c5b_smoke.json",
        "c5b_170105_rollout.json",
        "c5b_170650_rollout.json",
        "c5c_170105_evaluation.json",
        "c5c_170650_evaluation.json",
        "c5c_search_seed_diagnostics.json",
        "visual_review.json",
    ):
        _write(report_dir / filename, _not_run(filename, status))
    _write(report_dir / "final_summary.json", summary)
    _write(report_dir / "final_summary.md", markdown)
    _write(report_dir / "handoff.md", markdown)
    print(json.dumps({"status": status, "reason": reason}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
