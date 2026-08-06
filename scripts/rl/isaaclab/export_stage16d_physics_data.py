#!/usr/bin/env python3
"""Inventory partial trajectories and guard PhysicsQualifiedIsaacTrajectoryV2 export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, action="append", required=True)
    parser.add_argument("--ppo-evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluations = [json.loads(path.read_text()) for path in args.ppo_evaluation]
    ppo_pass = all(row["status"] == "STAGE16D_SINGLE_CLIP_PPO_VALIDATED" for row in evaluations)
    inventory = []
    for root in args.trajectory_root:
        files = [path for path in root.rglob("*") if path.is_file()]
        inventory.append(
            {
                "root": str(root.resolve()),
                "schema": "PhysicsConsistentRetargetedTrajectoryV1",
                "files": [
                    {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in sorted(files)
                ],
            }
        )
    result = {
        "schema_version": "Stage16DPhysicsDataInventoryV1",
        "status": "STAGE16D_PHYSICS_DATA_EXPORTED"
        if ppo_pass
        else "STAGE16D_PHYSICS_DATA_PARTIAL_BLOCKED",
        "qualified_v2_exported": ppo_pass,
        "partial_v1_inventory": inventory,
        "ppo_evaluations": evaluations,
        "reason": None if ppo_pass else "PPO qualification was not authorized or executed",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
