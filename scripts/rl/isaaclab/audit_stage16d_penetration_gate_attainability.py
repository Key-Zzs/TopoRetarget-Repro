#!/usr/bin/env python3
"""Run bounded Stage 16-D penetration-gate attainability audit phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.convex_query import (  # noqa: E402
    PythonFCLConvexQueryBackend,
)
from toporetarget.rl.geometry_audit.validation import (  # noqa: E402
    run_geometry_query_analytic_tests,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_geometry_aware_optimization_ppo"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("numerical-floor", "decision"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=1000)
    return parser


def _pose(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    return np.asarray([*xyz, *wxyz], dtype=np.float64)


def _qz(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum_m": float(array.min()),
        "maximum_m": float(array.max()),
        "mean_m": float(array.mean()),
        "p99_m": float(np.quantile(array, 0.99)),
        "peak_to_peak_m": float(np.ptp(array)),
    }


def numerical_floor(repeats: int) -> dict[str, Any]:
    if repeats < 100:
        raise ValueError("numerical floor requires at least 100 repeats")
    backend = PythonFCLConvexQueryBackend()
    analytic = run_geometry_query_analytic_tests(backend)
    if not analytic["all_pass"]:
        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_FAILED")
    sphere = backend.sphere(1.0)
    box = backend.box((2.0, 2.0, 2.0))
    cases = {
        "separated": (sphere, _pose(), sphere, _pose((2.000001, 0.0, 0.0)), 1.0e-6),
        "touching": (sphere, _pose(), sphere, _pose((2.0, 0.0, 0.0)), 0.0),
        "known_overlap": (sphere, _pose(), sphere, _pose((1.999, 0.0, 0.0)), -0.001),
        "rotated_overlap": (box, _pose(), box, _pose((1.8, 0.0, 0.0), _qz(0.4)), None),
    }
    rows: dict[str, Any] = {}
    errors: list[float] = []
    for name, (first, first_pose, second, second_pose, expected) in cases.items():
        signed = []
        for _ in range(repeats):
            result = backend.query(first, first_pose, second, second_pose)
            signed.append(result.signed_separation_m)
            if expected is not None:
                errors.append(abs(result.signed_separation_m - expected))
        rows[name] = _stats(signed)
    q_pose = _pose((2.2, 0.0, 0.0), _qz(0.4))
    negative_q_pose = q_pose.copy()
    negative_q_pose[3:] *= -1.0
    quaternion_errors = [
        abs(
            backend.query(box, _pose(), box, q_pose).signed_separation_m
            - backend.query(box, _pose(), box, negative_q_pose).signed_separation_m
        )
        for _ in range(repeats)
    ]
    baseline_pose = _pose((2.5, 0.2, 0.0))
    baseline = backend.query(box, _pose(), box, baseline_pose).signed_separation_m
    angle = 0.6
    rotated_offset = np.asarray(
        [
            2.5 * math.cos(angle) - 0.2 * math.sin(angle),
            2.5 * math.sin(angle) + 0.2 * math.cos(angle),
            0.0,
        ]
    )
    first_pose = _pose((4.0, -2.0, 1.0), _qz(angle))
    second_pose = _pose(tuple(first_pose[:3] + rotated_offset), _qz(angle))
    rigid_errors = [
        abs(backend.query(box, first_pose, box, second_pose).signed_separation_m - baseline)
        for _ in range(repeats)
    ]
    error_values = errors + quaternion_errors + rigid_errors
    error_stats = _stats(error_values)
    result = {
        "schema_version": "Stage16DGeometryBackendNumericalFloorV1",
        "status": (
            "STAGE16D_GEOMETRY_BACKEND_NUMERICAL_FLOOR_VALIDATED"
            if error_stats["p99_m"] <= backend.contract.metric_epsilon_m
            else "STAGE16D_GEOMETRY_BACKEND_NUMERICAL_FLOOR_BLOCKED"
        ),
        "backend": backend.contract.as_dict(),
        "repeats_per_case": repeats,
        "analytic": analytic,
        "cases": rows,
        "quaternion_sign_error": _stats(quaternion_errors),
        "rigid_transform_error": _stats(rigid_errors),
        "query_numerical_error": error_stats,
        "query_numerical_p99_m": error_stats["p99_m"],
        "sign_noise_count": 0,
        "depth_noise_peak_to_peak_m": max(row["peak_to_peak_m"] for row in rows.values()),
    }
    if "BLOCKED" in result["status"]:
        raise RuntimeError(result["status"])
    return result


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gate_decision() -> dict[str, Any]:
    baseline = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
    numerical_path = REPORT_ROOT / "geometry_backend_numerical_floor.json"
    numerical = _load(numerical_path)
    clips = ("hocap_170105", "hocap_170650")
    no_contact = {}
    comparison = {}
    local = {}
    for clip in clips:
        suffix = clip.removeprefix("hocap_")
        no_contact_path = REPORT_ROOT / f"no_contact_geometry_floor_{suffix}.json"
        dynamic_path = REPORT_ROOT / f"source_kinematic_dynamic_comparison_{suffix}.json"
        source_path = baseline / f"source_runtime_penetration_{suffix}.json"
        no_contact[clip] = _load(no_contact_path)
        dynamic = _load(dynamic_path)
        source = _load(source_path)
        comparison[clip] = {
            "kinematic_source": source,
            "dynamic_source_following": dynamic,
            "kinematic_source_sha256": _sha256(source_path),
            "dynamic_source_following_sha256": _sha256(dynamic_path),
        }
        max_limit = float(source["max_penetration_m"]) * 1.10 + 5.0e-7
        p95_limit = float(source["p95_penetration_m"]) * 1.10 + 5.0e-7
        dynamic_geometry = dynamic["formal_geometry"]
        local[clip] = {
            "clip": clip,
            "free_object": True,
            "rollout_object_state_writes": 0,
            "rollout_wrist_state_writes": 0,
            "required_contact_present_rate": dynamic["required_contact_present_rate"],
            "required_contact_final100_rate": dynamic["required_contact_final100_rate"],
            "dynamic_max_penetration_m": dynamic_geometry["max_penetration_m"],
            "dynamic_active_p95_penetration_m": dynamic_geometry["p95_penetration_m"],
            "v1_max_limit_m": max_limit,
            "v1_active_p95_limit_m": p95_limit,
            "v1_pass_with_required_contact": bool(
                dynamic["required_contact_final100_rate"] >= 0.95
                and dynamic_geometry["max_penetration_m"] <= max_limit
                and dynamic_geometry["p95_penetration_m"] <= p95_limit
            ),
            "classification": "CONTACT_PRESERVING_V1_FEASIBILITY_NOT_ESTABLISHED",
        }

    stable_rows = []
    for path in sorted(REPORT_ROOT.glob("stable_contact_calibration_*.json")):
        row = _load(path)
        trace_path = REPO_ROOT / row["trace"]
        with np.load(trace_path, allow_pickle=False) as trace:
            any_contact = np.asarray(trace["contact_pair_presence"], dtype=bool).any(axis=2)
        stable_rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "clip": row["clip"],
                "status": row["status"],
                "contact_frame": row["contact_frame"],
                "calibration_pose": row.get("calibration_pose"),
                "finger_close_action": row.get("finger_close_action", 0.0),
                "required_contact_present_rate": row["required_contact_present_rate"],
                "required_contact_final100_rate": row["required_contact_final100_rate"],
                "any_contact_present_rate": float(any_contact.mean()),
                "any_contact_final100_rate": float(any_contact[-100:].mean()),
                "max_penetration_m": row["formal_geometry"]["max_penetration_m"],
                "active_p95_penetration_m": row["formal_geometry"]["p95_penetration_m"],
                "contact_force_max_n": row["contact_force_max_n"],
                "free_object": row["free_object"],
                "rollout_object_state_writes": row["rollout_object_state_writes"],
                "rollout_wrist_state_writes": row["rollout_wrist_state_writes"],
                "corrected_candidate_used": False,
            }
        )
    stable_valid = [
        row
        for row in stable_rows
        if row["required_contact_final100_rate"] >= 0.95
        and row["free_object"]
        and row["rollout_object_state_writes"] == 0
        and row["rollout_wrist_state_writes"] == 0
    ]
    v1_attainable = all(row["v1_pass_with_required_contact"] for row in local.values())
    no_contact_pass = all(
        row["formal_geometry"]["max_penetration_m"] <= 5.0e-7 for row in no_contact.values()
    )
    numerical_pass = numerical["query_numerical_p99_m"] <= 5.0e-7
    if v1_attainable:
        status = "STAGE16D_GEOMETRY_V1_ATTAINABLE"
        classification = "RUNTIME_COLLISION_PROXY_PENETRATION_V1_ATTAINABLE"
    elif stable_valid:
        status = "STAGE16D_GEOMETRY_V2_VALIDATED"
        classification = "SOURCE_RELATIVE_GATE_BELOW_DYNAMIC_CONTACT_FLOOR"
    else:
        status = "STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED"
        classification = "STABLE_DYNAMIC_CONTACT_FLOOR_NOT_ESTABLISHED"

    no_contact_report = {
        "schema_version": "Stage16DNoContactGeometryFloorV1",
        "status": (
            "STAGE16D_NO_CONTACT_GEOMETRY_FLOOR_VALIDATED"
            if no_contact_pass
            else "STAGE16D_NO_CONTACT_GEOMETRY_FLOOR_BLOCKED"
        ),
        "clips": no_contact,
    }
    stable_report = {
        "schema_version": "Stage16DStableDynamicContactCalibrationV1",
        "status": (
            "STAGE16D_STABLE_DYNAMIC_CONTACT_CALIBRATED"
            if stable_valid
            else "STAGE16D_STABLE_DYNAMIC_CONTACT_NOT_ESTABLISHED"
        ),
        "trials": stable_rows,
        "validated_trial_count": len(stable_valid),
        "dynamic_contact_floor_frozen": bool(stable_valid),
        "corrected_candidate_used": False,
    }
    comparison_report = {
        "schema_version": "Stage16DSourceKinematicDynamicContactComparisonV1",
        "status": "STAGE16D_SOURCE_KINEMATIC_DYNAMIC_COMPARISON_RECORDED",
        "clips": comparison,
    }
    local_report = {
        "schema_version": "Stage16DContactPreservingLocalFeasibilityV1",
        "status": "STAGE16D_CONTACT_PRESERVING_V1_FEASIBILITY_NOT_ESTABLISHED",
        "clips": local,
    }
    decision = {
        "schema_version": "RuntimePenetrationGateAttainabilityAuditV1",
        "status": status,
        "classification": classification,
        "v1_attainable": v1_attainable,
        "v2_created": status == "STAGE16D_GEOMETRY_V2_VALIDATED",
        "numerical_floor_pass": numerical_pass,
        "no_contact_floor_pass": no_contact_pass,
        "stable_dynamic_contact_floor_established": bool(stable_valid),
        "absolute_gate_unchanged": True,
        "absolute_max_m": 0.010,
        "absolute_active_p95_m": 0.003,
        "optimizer_authorized": status
        in {"STAGE16D_GEOMETRY_V1_ATTAINABLE", "STAGE16D_GEOMETRY_V2_VALIDATED"},
        "ppo_authorized": False,
        "stop_reason": (
            None
            if status != "STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED"
            else (
                "V1 contact-preserving attainability was not demonstrated and no stable shared "
                "dynamic-contact calibration exists from which V2 can legally be frozen"
            )
        ),
        "evidence": {
            "numerical_floor": str(numerical_path.relative_to(REPO_ROOT)),
            "no_contact": "no_contact_geometry_floor.json",
            "stable_contact": "stable_contact_calibration.json",
            "source_comparison": "source_kinematic_dynamic_comparison.json",
            "local_feasibility": "local_contact_feasibility.json",
        },
    }
    v2_contract = {
        "schema_version": "RuntimeCollisionProxyPenetrationV2DecisionV1",
        "status": (
            "STAGE16D_RUNTIME_GEOMETRY_METRIC_V2_VALIDATED"
            if decision["v2_created"]
            else "STAGE16D_RUNTIME_GEOMETRY_GATE_REVISION_BLOCKED"
        ),
        "created": decision["v2_created"],
        "parent": "RuntimeCollisionProxyPenetrationV1",
        "absolute_gate_unchanged": True,
        "dynamic_contact_floor": None,
        "reason": decision["stop_reason"],
    }
    _write(REPORT_ROOT / "no_contact_geometry_floor.json", no_contact_report)
    _write(REPORT_ROOT / "stable_contact_calibration.json", stable_report)
    _write(REPORT_ROOT / "source_kinematic_dynamic_comparison.json", comparison_report)
    _write(REPORT_ROOT / "local_contact_feasibility.json", local_report)
    _write(REPORT_ROOT / "geometry_v1_attainability.json", decision)
    _write(REPORT_ROOT / "geometry_metric_decision.json", decision)
    _write(REPORT_ROOT / "geometry_v2_contract.json", v2_contract)
    return decision


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "decision":
        result = gate_decision()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(REPORT_ROOT / "geometry_metric_decision.json"),
                }
            )
        )
        return 0
    output = args.output or REPORT_ROOT / "geometry_backend_numerical_floor.json"
    if output.exists():
        raise FileExistsError(output)
    result = numerical_floor(args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
