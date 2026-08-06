#!/usr/bin/env python3
"""Freeze the D.4R4 RuntimeCollisionProxyPenetration V1/V2 decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.dynamic_contact_reference import (  # noqa: E402
    SelectedStableCalibrationV1,
    decide_geometry_v1_v2,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"


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


def main() -> int:
    selected = _load(REPORT_ROOT / "selected_stable_calibrations.json")
    rows = tuple(
        SelectedStableCalibrationV1(
            **{
                key: value
                for key, value in row.items()
                if key not in {"schema_version", "v1_passed_20_of_20"}
            }
        )
        for row in selected["rows"]
    )
    required = tuple(tuple(pair) for pair in selected["required_object_family_pairs"])
    decision = decide_geometry_v1_v2(rows, required_object_family_pairs=required)
    attainability = {
        "schema_version": "Stage16DStableGraspGeometryV1AttainabilityV1",
        "status": (
            "STAGE16D_GEOMETRY_V1_ATTAINABLE"
            if decision.get("v1_attainable")
            else decision["status"]
        ),
        "calibrations": [row.as_dict() for row in rows],
        "all_required_pairs_stable": bool(rows) and all(row.stable_gate_passed for row in rows),
        "v1_passed_all_pairs_20_of_20": bool(rows) and all(row.v1_passed() for row in rows),
    }
    _write(REPORT_ROOT / "geometry_v1_attainability.json", attainability)
    _write(REPORT_ROOT / "geometry_v1_v2_decision.json", decision)
    v2 = (
        decision["v2_contract"]
        if decision.get("v2_created")
        else {
            "schema_version": "RuntimeCollisionProxyPenetrationV2",
            "status": "NOT_CREATED",
            "reason": decision["status"],
            "absolute_gate_unchanged": True,
        }
    )
    _write(REPORT_ROOT / "geometry_v2_contract.json", v2)
    ledger = f"""# RuntimeCollisionProxyPenetration parent/child ledger

- Decision: `{decision["status"]}`
- Parent: `RuntimeCollisionProxyPenetrationV1`
- Child created: `{str(bool(decision.get("v2_created"))).lower()}`
- Absolute 10 mm / 3 mm gate changed: `false`
- Threshold split by clip or topology: `false`
- Corrected trajectory used for calibration: `false`
- Optimizer/PPO result used for calibration: `false`
- Empirical reference is not physical truth or a mathematical lower bound.
"""
    (REPORT_ROOT / "parent_child_contract_ledger.md").write_text(ledger, encoding="utf-8")
    print(json.dumps({"status": decision["status"], "output": str(REPORT_ROOT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
