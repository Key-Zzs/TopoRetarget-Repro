#!/usr/bin/env python3
"""Freeze inputs and materialize the pose/time-primary Stage 16-D reference V2."""

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
    Stage16DReferenceTimeV2,
    inspect_v1_reference,
    materialize_reference_kinematics_v2,
    sha256_file,
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
    source_root = args.source_root.resolve()
    v1_root = args.v1_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite versioned V2 output: {output}")
    output.mkdir(parents=True)
    transitions = output / "failure_transitions.jsonl"
    _append_transition(
        transitions, "INPUT_FREEZE", reason="output root created without touching V1"
    )
    audits: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {
        "schema_version": "Stage16DReferenceKinematicsV2FrozenInputsV1",
        "clips": {},
    }
    try:
        for clip in CLIPS:
            source = source_root / f"{clip}.world_wrist.stage16.npz"
            v1 = v1_root / f"{clip}.reference.npz"
            audit = inspect_v1_reference(source, v1)
            audits[clip] = audit
            inputs["clips"][clip] = {
                "native_source": {"path": str(source), "sha256": sha256_file(source)},
                "factor8_v1": {"path": str(v1), "sha256": sha256_file(v1)},
                "v1_declared_source_sha256": audit["source"]["sha256"],
                "source_hash_matches_v1_metadata": True,
            }
        _write_json(output / "frozen_inputs.json", inputs)
        _write_json(output / "input_hashes.json", inputs)
        _append_transition(
            transitions, "V1_REFERENCE_AUDIT", reason="source hashes match V1 metadata"
        )
        _write_json(output / "reference_v1_audit.json", {"clips": audits})
        materialized: dict[str, Any] = {}
        for clip in CLIPS:
            materialized[clip] = materialize_reference_kinematics_v2(
                source_root / f"{clip}.world_wrist.stage16.npz",
                v1_root / f"{clip}.reference.npz",
                output / "references" / f"{clip}.reference_kinematics_v2.npz",
            )
        _append_transition(
            transitions, "POSE_AUDIT", reason="source keys audited before V2 materialization"
        )
        _append_transition(
            transitions, "TIMESTAMP_REPAIR", reason="runtime timestamps fixed to i * 0.05 s"
        )
        _append_transition(
            transitions,
            "LINEAR_VELOCITY_REPAIR",
            reason="V2 linear twists derived from final V2 positions and timestamps",
        )
        _append_transition(
            transitions,
            "ANGULAR_VELOCITY_REPAIR",
            reason="V2 angular twists derived in world convention with SO3 logs",
        )
        time_contract = Stage16DReferenceTimeV2().as_dict()
        _write_json(output / "reference_timestamp_contract.json", time_contract)
        _write_json(output / "timestamp_contract.json", time_contract)
        _write_json(
            output / "interpolation_contract.json",
            {
                "authority_order": ["timestamps", "pose trajectory", "pose-derived twist"],
                "translation": "pose-derived-tangent cubic Hermite",
                "rotation": "shortest-arc normalized linear quaternion interpolation",
                "old_independent_twist_interpolation": "forbidden",
                "materialized": materialized,
            },
        )
        print(
            json.dumps(
                {
                    "status": "STAGE16D_REFERENCE_KINEMATICS_V2_MATERIALIZED",
                    "output_root": str(output),
                    "references": materialized,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        _append_transition(transitions, "BLOCKED", reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
