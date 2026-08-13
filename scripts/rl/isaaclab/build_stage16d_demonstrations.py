#!/usr/bin/env python3
"""Build demonstrations only when the formal trajectory gate authorizes them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.physics_retargeting.entry_gate import trajectory_entry_decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, action="append", required=True)
    parser.add_argument("--geometry", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.qualification) != 2 or len(args.geometry) != 2:
        raise ValueError("Stage16D demonstration gate requires both clips")
    decisions = [
        trajectory_entry_decision(
            json.loads(qualification.read_text()), json.loads(geometry.read_text())
        )
        for qualification, geometry in zip(args.qualification, args.geometry, strict=True)
    ]
    authorized = [row for row in decisions if row["authorization"] != "PPO_NOT_AUTHORIZED_FOR_CLIP"]
    result = {
        "schema_version": "PhysicsCorrectionDemonstrationManifestV1",
        "status": "STAGE16D_DEMONSTRATION_EXPORT_NOT_AUTHORIZED"
        if not authorized
        else "STAGE16D_DEMONSTRATION_EXPORT_AUTHORIZED_NOT_EXECUTED",
        "trajectory_level_split": {"train": 0.8, "validation": 0.2},
        "frame_level_random_split": False,
        "future_state_in_observation": False,
        "hidden_simulator_state_in_observation": False,
        "decisions": decisions,
        "trajectories": [],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
