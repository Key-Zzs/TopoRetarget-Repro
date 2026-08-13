#!/usr/bin/env python3
"""Freeze the bounded stable-grasp candidate matrix and action schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.stable_grasp_calibration import (  # noqa: E402
    GraspTopologyFamilyV1,
    StableGraspCalibrationActionScheduleV1,
    StableGraspCalibrationGateV1,
    freeze_candidate_matrix,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("C1", "C2"), required=True)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    return parser


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
    args = _parser().parse_args()
    output = args.output_root.resolve()
    family_contract = _load(output / "grasp_topology_family_contract.json")
    families = tuple(GraspTopologyFamilyV1(**row) for row in family_contract["families"])
    object_ids = sorted({clip for family in families for clip in family.applicable_clips})
    matrix = freeze_candidate_matrix(object_ids=object_ids, families=families, level=args.level)
    _write(output / f"stable_grasp_candidate_matrix_{args.level.lower()}.json", matrix)
    if args.level == "C1":
        _write(output / "stable_grasp_candidate_matrix.json", matrix)
        schedule = StableGraspCalibrationActionScheduleV1()
        _write(
            output / "calibration_action_schedule.json",
            {
                **schedule.as_dict(),
                "stability_gate": StableGraspCalibrationGateV1().as_dict(),
                "frozen_before_calibration": True,
            },
        )
    print(
        json.dumps(
            {
                "status": "STAGE16D_STABLE_GRASP_MATRIX_FROZEN",
                "level": args.level,
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
