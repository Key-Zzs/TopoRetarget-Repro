#!/usr/bin/env python3
"""Guarded balanced two-clip Stage 16-D PPO entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.physics_retargeting.entry_gate import two_clip_ppo_authorized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.evaluation) != 2:
        raise ValueError("two-clip PPO gate requires two single-clip evaluations")
    evaluations = [json.loads(path.read_text()) for path in args.evaluation]
    authorized = two_clip_ppo_authorized(*evaluations)
    result = {
        "schema_version": "Stage16DTwoClipPPOV1",
        "status": "STAGE16D_TWO_CLIP_PPO_AUTHORIZED_NOT_EXECUTED"
        if authorized
        else "STAGE16D_TWO_CLIP_PPO_NOT_RUN",
        "balanced_clip_sampling": True,
        "shared_observation_normalization": True,
        "clip_identity_in_observation": False,
        "clip_specific_parameters": False,
        "samples": 0,
        "workers_started": 0,
        "single_clip_evaluations": evaluations,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
