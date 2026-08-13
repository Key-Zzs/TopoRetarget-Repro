#!/usr/bin/env python3
"""Select development candidates or aggregate formal stable-grasp qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"
OBJECT_IDS = ("hocap_170105", "hocap_170650")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select-development", "formal"), required=True)
    parser.add_argument("--level", choices=("C1", "C2"), default="C1")
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    return parser


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
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["selection_metrics"]
    return (
        -float(metrics["topology_coverage"]),
        -float(metrics["terminal_hold_stability"]),
        -float(metrics["contact_persistence"]),
        float(metrics["terminal_linear_speed_p95_max_mps"]),
        float(metrics["terminal_angular_speed_p95_max_radps"]),
        float(metrics["active_p95_penetration_m"]),
        float(metrics["max_penetration_m"]),
        float(metrics["mean_absolute_effort"]),
        float(metrics["mean_action_variation"]),
        str(row["candidate"]["candidate_id"]),
    )


def _select_development(output: Path, level: str) -> dict[str, Any]:
    matrix = _load(output / f"stable_grasp_candidate_matrix_{level.lower()}.json")
    selections: list[dict[str, Any]] = []
    missing: list[str] = []
    for object_id in OBJECT_IDS:
        family_ids = sorted({row["family_id"] for row in matrix["objects"][object_id]})
        for family_id in family_ids:
            expected = {
                row["candidate_id"]
                for row in matrix["objects"][object_id]
                if row["family_id"] == family_id
            }
            reports = []
            for candidate_id in sorted(expected):
                path = output / f"calibration_dev_{object_id}_{candidate_id}.json"
                if not path.is_file():
                    missing.append(str(path.relative_to(REPO_ROOT)))
                    continue
                report = _load(path)
                if report["candidate"]["candidate_id"] != candidate_id:
                    raise RuntimeError("STAGE16D_CALIBRATION_RESULT_IDENTITY_FAILURE")
                reports.append({**report, "report_path": str(path.relative_to(REPO_ROOT))})
            eligible = [row for row in reports if row.get("development_pass") is True]
            if eligible:
                best = min(eligible, key=_rank_key)
                selections.append(
                    {
                        "object_id": object_id,
                        "family_id": family_id,
                        "candidate_id": best["candidate"]["candidate_id"],
                        "development_report": best["report_path"],
                        "development_report_sha256": _sha256(REPO_ROOT / best["report_path"]),
                        "selection_metrics": best["selection_metrics"],
                        "ranking": (
                            "hard gate, topology, hold, persistence, twist, geometry, "
                            "effort, action variation, candidate ID"
                        ),
                    }
                )
    expected_family_count = sum(
        len({row["family_id"] for row in matrix["objects"][object_id]}) for object_id in OBJECT_IDS
    )
    complete = not missing and len(selections) == expected_family_count
    return {
        "schema_version": "StableGraspDevelopmentSelectionV1",
        "status": (
            "STAGE16D_STABLE_GRASP_DEVELOPMENT_SELECTION_VALIDATED"
            if complete
            else "STAGE16D_STABLE_GRASP_DEVELOPMENT_SELECTION_INCOMPLETE"
        ),
        "level": level,
        "expected_family_count": expected_family_count,
        "missing_reports": missing,
        "selections": selections,
        "all_candidates_reported": not missing,
        "eligible_family_count": len(selections),
        "c2_authorized": not complete and level == "C1" and not missing,
    }


def _formal(output: Path, level: str) -> dict[str, Any]:
    selection = _load(output / f"stable_grasp_selection_{level.lower()}.json")
    if not selection["selections"]:
        raise RuntimeError("STAGE16D_STABLE_GRASP_FORMAL_WITHOUT_DEVELOPMENT_SELECTION")
    rows: list[dict[str, Any]] = []
    for selected in selection["selections"]:
        path = output / (
            f"calibration_formal_{selected['object_id']}_{selected['candidate_id']}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        report = _load(path)
        if (
            report["replicas"] != 20
            or report["candidate"]["candidate_id"] != selected["candidate_id"]
        ):
            raise RuntimeError("STAGE16D_CALIBRATION_FORMAL_IDENTITY_FAILURE")
        rows.append(
            {
                "object_id": selected["object_id"],
                "family_id": selected["family_id"],
                "candidate_id": selected["candidate_id"],
                "report": str(path.relative_to(REPO_ROOT)),
                "report_sha256": _sha256(path),
                "status": report["status"],
                "qualification": report["qualification"],
                "selection_metrics": report["selection_metrics"],
                "formal_geometry": report["formal_geometry"],
                "replica_max_penetration_m": report["replica_max_penetration_m"],
                "replica_active_p95_penetration_m": report["replica_active_p95_penetration_m"],
            }
        )
    for object_id in OBJECT_IDS:
        object_rows = [row for row in rows if row["object_id"] == object_id]
        _write(
            output / f"calibration_results_{object_id.removeprefix('hocap_')}.json",
            {
                "schema_version": "StableGraspCalibrationObjectResultsV1",
                "object_id": object_id,
                "results": object_rows,
                "all_required_families_passed": bool(object_rows)
                and all(row["qualification"]["passed"] for row in object_rows),
            },
        )
    passed = all(row["qualification"]["passed"] for row in rows)
    geometry = {
        "schema_version": "StableGraspExactGeometryAuditV1",
        "backend": "python-fcl==0.7.0.11",
        "metric": "RuntimeCollisionProxyPenetrationV1",
        "rows": [
            {
                key: row[key]
                for key in (
                    "object_id",
                    "family_id",
                    "candidate_id",
                    "report",
                    "report_sha256",
                    "formal_geometry",
                    "replica_max_penetration_m",
                    "replica_active_p95_penetration_m",
                )
            }
            for row in rows
        ],
    }
    _write(output / "stable_grasp_exact_geometry.json", geometry)
    return {
        "schema_version": "StableGraspCalibrationQualificationSetV1",
        "status": (
            "STAGE16D_STABLE_GRASP_CALIBRATION_VALIDATED"
            if passed
            else "STAGE16D_STABLE_GRASP_CALIBRATION_PARTIAL"
        ),
        "level": level,
        "all_required_object_family_pairs_passed": passed,
        "results": rows,
        "corrected_trajectory_used": False,
        "rollout_state_writes": 0,
        "hidden_support": False,
    }


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    if args.phase == "select-development":
        result = _select_development(output, args.level)
        destination = output / f"stable_grasp_selection_{args.level.lower()}.json"
    else:
        result = _formal(output, args.level)
        destination = output / "stable_grasp_qualification.json"
    _write(destination, result)
    print(json.dumps({"status": result["status"], "output": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
