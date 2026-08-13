#!/usr/bin/env python3
"""Fail-closed qualification for the materialized Stage 16-D reference V2."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.reference_kinematics import (  # noqa: E402
    qualify_reference_kinematics_v2,
)

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"
DEFAULT_SOURCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
DEFAULT_V1_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d/reference"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_transition(path: Path, state: str, *, reason: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "state": state,
                    "reason": reason,
                },
                sort_keys=True,
            )
            + "\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    source_root = args.source_root.resolve()
    v1_root = args.v1_root.resolve()
    transitions = output / "failure_transitions.jsonl"
    reports: dict[str, dict[str, Any]] = {}
    try:
        for clip in CLIPS:
            reports[clip] = qualify_reference_kinematics_v2(
                source_root / f"{clip}.world_wrist.stage16.npz",
                v1_root / f"{clip}.reference.npz",
                output / "references" / f"{clip}.reference_kinematics_v2.npz",
            )
        all_valid = all(
            report["status"] == "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED"
            for report in reports.values()
        )
        aggregate = {
            "schema_version": "ReferenceKinematicsQualificationV2Aggregate",
            "clips": reports,
            "status": "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED"
            if all_valid
            else "STAGE16D_REFERENCE_KINEMATICS_V2_BLOCKED",
        }
        _write_json(output / "reference_kinematics_qualification.json", aggregate)
        _write_json(
            output / "factor8_scaling_validation.json",
            {"clips": {clip: report["factor8_scaling"] for clip, report in reports.items()}},
        )
        _write_json(
            output / "integral_consistency.json",
            {"clips": {clip: report["integral_consistency"] for clip, report in reports.items()}},
        )
        _write_json(
            output / "terminal_reference_semantics.json",
            {
                "clips": {
                    clip: report["terminal_reference_semantics"] for clip, report in reports.items()
                }
            },
        )
        _write_json(
            output / "linear_velocity_contract.json",
            {
                "clips": {
                    clip: report["linear_velocity_contract"] for clip, report in reports.items()
                }
            },
        )
        _write_json(
            output / "angular_velocity_contract.json",
            {
                "clips": {
                    clip: report["angular_velocity_contract"] for clip, report in reports.items()
                }
            },
        )
        _append_transition(
            transitions,
            "KINEMATICS_QUALIFICATION" if all_valid else "BLOCKED",
            reason="all source-key, time, pose/twist, factor-8, and integral gates evaluated",
        )
        print(
            json.dumps({"status": aggregate["status"], "output_root": str(output)}, sort_keys=True)
        )
        return 0 if all_valid else 2
    except Exception as error:
        _append_transition(transitions, "BLOCKED", reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
