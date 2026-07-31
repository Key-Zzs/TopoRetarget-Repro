#!/usr/bin/env python3
"""Export a 20 Hz Stage16ReferenceClip from a validated RobotReference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.contracts.reference import load_robot_reference
from toporetarget.rl.references import export_stage16_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    clip = export_stage16_reference(
        load_robot_reference(args.robot_reference),
        extra_provenance={"exporter": "export_reference_clips.py"},
    )
    clip.to_npz(args.output)
    print(json.dumps(clip.validate(expected_hz=20.0), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
