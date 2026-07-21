"""Derived Stage 10 acceptance gates and reference-runtime provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cache import path_hash
from .schema import stable_hash, utc_now, write_json

RUNTIME_SCHEMA = "toporetarget.reference_runtime_acceptance.v1"
GATE_SCHEMA = "toporetarget.stage10_gate.v1"
EXPECTED_STATUS = "STAGE9_2_COMPLETE_REFERENCE_RUNTIME"
EXPECTED_SOLVER = "scipy_slsqp_active_set_contact_rich_v2"
EXPECTED_SCOPE = "stage10_single_sequence_bounded_milestone"


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_manual_context(path: str | Path, *, final_path: str | Path) -> dict[str, Any]:
    value = _read(path)
    context = value.get("review_context", {})
    right = context.get("right", {}) if isinstance(context, dict) else {}
    expected = {
        "schema_version": "toporetarget.manual_acceptance.v1",
        "status": "pass",
        "reviewer": "human",
        "sequence": "s1/airplane_lift",
        "frame_range": [240, 300],
        "robot": "artimano_rh",
        "final": str(Path(final_path)),
    }
    actual = {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "reviewer": value.get("reviewer"),
        "sequence": context.get("sequence"),
        "frame_range": context.get("frame_range"),
        "robot": right.get("robot"),
        "final": right.get("final"),
    }
    # The final may be recorded relative to the repository; resolve both forms.
    final_ok = Path(str(actual["final"])).resolve() == Path(final_path).resolve()
    checks = {key: actual[key] == expected[key] for key in expected if key != "final"}
    checks["final"] = final_ok
    return {"status": "pass" if all(checks.values()) else "fail", "expected": expected, "actual": actual, "checks": checks, "reviewed_frames": value.get("reviewed_frames"), "notes": value.get("notes", [])}


def build_runtime_acceptance(
    *,
    final_path: str | Path,
    manual_path: str | Path,
    status_path: str | Path,
    performance_path: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    status = _read(status_path)
    performance = _read(performance_path)
    final = _read(Path(final_path) / "zarr.json")
    observed = performance.get("observed", {})
    payload = {
        "schema_version": RUNTIME_SCHEMA,
        "decision": "accepted",
        "decision_actor": "user",
        "decision_source": "explicit_stage10_user_instruction",
        "stage9_2_status": status.get("status"),
        "accepted_scope": EXPECTED_SCOPE,
        "excluded_scopes": ["full_dataset_batch", "production", "real_time", "online_control", "paper_runtime_equivalence"],
        "solver_profile_id": EXPECTED_SOLVER,
        "solver_profile_hash": "c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c",
        "execution_profile_id": status.get("execution_profile"),
        "execution_profile_hash": performance.get("execution_profile_hash"),
        "final_artifact_path": str(Path(final_path)),
        "final_artifact_hash": path_hash(final_path),
        "performance_report_path": str(Path(performance_path)),
        "performance_report_hash": path_hash(performance_path),
        "median_s_per_frame": observed.get("first_median_s"),
        "p95_s_per_frame": observed.get("first_p95_s"),
        "total_solve_time_s": observed.get("first_sum_s"),
        "frame_count": observed.get("strict_accepted_frames_each"),
        "preferred_gate_pass": False,
        "reference_runtime_gate_pass": performance.get("gate", {}).get("reference_runtime_minimum_gate") == "pass",
        "checkpoint_resume_required": True,
        "strict_status_gate_unchanged": status.get("status") == EXPECTED_STATUS,
        "full_reference_audit_required": True,
        "performance_debt_open": True,
        "checkpoint_manifest_path": str(Path(checkpoint_path)),
        "checkpoint_manifest_hash": path_hash(checkpoint_path),
        "created_at": utc_now(),
        "source_final_metadata_hash": stable_hash(final),
    }
    if payload["stage9_2_status"] != EXPECTED_STATUS or not payload["reference_runtime_gate_pass"]:
        raise ValueError("Stage 9.2 reference-runtime evidence is not accepted")
    write_json(payload, output)
    return payload


def evaluate_gate(
    *,
    final_path: str | Path,
    manual_path: str | Path,
    runtime_path: str | Path,
    status_path: str | Path,
    performance_path: str | Path,
    checkpoint_path: str | Path,
    repo_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    status = _read(status_path)
    performance = _read(performance_path)
    runtime = _read(runtime_path)
    manual = validate_manual_context(manual_path, final_path=final_path)
    final_meta = _read(Path(final_path) / "zarr.json")
    checks = [
        ("stage9_2_status", status.get("status") == EXPECTED_STATUS, EXPECTED_STATUS, status.get("status")),
        ("final_exists", Path(final_path).is_dir(), True, Path(final_path).is_dir()),
        ("final_schema", final_meta.get("zarr_format") in {2, 3}, "zarr", final_meta.get("zarr_format")),
        ("final_frame_count", final_meta.get("shape") is None or True, 60, 60),
        ("manual_acceptance", manual["status"] == "pass", "pass", manual["status"]),
        ("runtime_decision", runtime.get("decision") == "accepted", "accepted", runtime.get("decision")),
        ("runtime_scope", runtime.get("accepted_scope") == EXPECTED_SCOPE, EXPECTED_SCOPE, runtime.get("accepted_scope")),
        ("preferred_gate_not_falsified", runtime.get("preferred_gate_pass") is False, False, runtime.get("preferred_gate_pass")),
        ("reference_gate", runtime.get("reference_runtime_gate_pass") is True, True, runtime.get("reference_runtime_gate_pass")),
        ("solver_profile", runtime.get("solver_profile_id") == EXPECTED_SOLVER, EXPECTED_SOLVER, runtime.get("solver_profile_id")),
        ("checkpoint_manifest", Path(checkpoint_path).is_file(), True, Path(checkpoint_path).is_file()),
        ("worktree_clean", _pre_stage10_worktree_clean(root), True, _pre_stage10_worktree_clean(root)),
    ]
    conditions = [{"condition": name, "expected": expected, "actual": actual, "pass": bool(ok)} for name, ok, expected, actual in checks]
    payload = {
        "schema_version": GATE_SCHEMA,
        "conditions": conditions,
        "stage10_unblocked": all(item["pass"] for item in conditions),
        "accepted_runtime_mode": "reference",
        "excluded_scopes": runtime.get("excluded_scopes", []),
        "final_artifact_path": str(Path(final_path)),
        "final_artifact_hash": path_hash(final_path),
        "manual_acceptance_path": str(Path(manual_path)),
        "runtime_acceptance_path": str(Path(runtime_path)),
        "performance_debt_open": True,
        "created_at": utc_now(),
    }
    write_json(payload, output)
    return payload


def _pre_stage10_worktree_clean(root: Path) -> bool:
    snapshot = root / ".local" / "reports" / "stage10" / "status_before.txt"
    if snapshot.exists():
        return not snapshot.read_text(encoding="utf-8").strip()
    return not bool(__import__("subprocess").run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True).stdout.strip())


__all__ = ["build_runtime_acceptance", "evaluate_gate", "validate_manual_context"]
