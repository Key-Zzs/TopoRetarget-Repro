#!/usr/bin/env python3
"""Guarded Stage 16-D BC and single-clip PPO entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.physics_retargeting.entry_gate import trajectory_entry_decision
from toporetarget.rl.ppo.physics_correction_ppo import PhysicsCorrectionPPOV1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--bc-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = trajectory_entry_decision(
        json.loads(args.qualification.read_text()), json.loads(args.geometry.read_text())
    )
    result = {
        "schema_version": "Stage16DSingleClipPPOTrainingV1",
        "clip": args.clip,
        "status": f"STAGE16D_{args.clip.removeprefix('hocap_')}_PPO_NOT_RUN",
        "authorization": decision,
        "contract": PhysicsCorrectionPPOV1().as_dict(),
        "bc": "NOT_RUN_TRAJECTORY_GATE_BLOCKED",
        "samples": 0,
        "checkpoints": [],
        "workers_started": 0,
        "reason": "formal trajectory geometry gate did not authorize BC or PPO",
    }
    bc_result = {
        "schema_version": "Stage16DBehaviorCloningTrainingV1",
        "clip": args.clip,
        "status": "STAGE16D_BC_NOT_RUN",
        "actor_only": True,
        "critic_pseudo_labels": False,
        "epochs": 0,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "validation_action_loss": None,
        "reason": result["reason"],
    }
    args.bc_output.write_text(json.dumps(bc_result, indent=2, sort_keys=True) + "\n")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
