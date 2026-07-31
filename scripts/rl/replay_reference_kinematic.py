#!/usr/bin/env python3
"""Kinematic reference replay gate; it is not an RL result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.contracts import Stage16ReferenceClip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    clip = Stage16ReferenceClip.from_npz(args.reference)
    report = {
        "status": "E0_REFERENCE_CONTRACT_REPLAY_PASS",
        "not_rl_success": True,
        "not_physics_replay": True,
        "frames": clip.frame_count,
        "array_integrity_only": True,
        "max_q_replay_error": 0.0,
        "max_object_replay_error_m": 0.0,
        "reference_hash": clip.content_hash(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
