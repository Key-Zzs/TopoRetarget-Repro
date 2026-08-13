#!/usr/bin/env python3
"""Qualify a Stage 16-D PPO run or preserve its guarded NOT_RUN state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training.read_text())
    clip = str(training["clip"])
    result = {
        "schema_version": "Stage16DPPOQualificationV1",
        "clip": clip,
        "status": f"STAGE16D_{clip.removeprefix('hocap_')}_PPO_NOT_RUN",
        "frame_zero_full_episode": True,
        "evaluation_episodes": 0,
        "checkpoint_reload": "NOT_RUN",
        "samples": int(training["samples"]),
        "reason": training["reason"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
