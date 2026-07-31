#!/usr/bin/env python3
"""Materialize one accepted Stage-12 HOCap final for Stage-16 use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.contracts.reference import save_robot_reference
from toporetarget.rl.stage12_reference import (
    materialize_accepted_stage12_reference,
    write_obj_mesh,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-trajectory", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--robot-reference-output", required=True, type=Path)
    parser.add_argument("--object-mesh-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    reference, vertices, faces, mesh_metadata = materialize_accepted_stage12_reference(
        final_trajectory=args.final_trajectory,
        canonical=args.canonical,
        checkpoint_manifest=args.checkpoint_manifest,
    )
    save_robot_reference(reference, args.robot_reference_output)
    mesh_path = write_obj_mesh(args.object_mesh_output, vertices, faces)
    report = {
        "status": "ACCEPTED_STAGE12_HOCAP_REFERENCE_MATERIALIZED",
        "robot_reference": str(args.robot_reference_output.resolve()),
        "robot_reference_validation": reference.validate(),
        "object_mesh": str(mesh_path.resolve()),
        "object_mesh_metadata": mesh_metadata,
        "source_final_frames": reference.num_frames,
        "source_final_fps": reference.fps,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
