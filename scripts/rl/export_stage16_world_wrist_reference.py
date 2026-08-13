#!/usr/bin/env python3
"""Export one direct Stage-12 world-frame Stage-16B reference and object mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco

from toporetarget.rl.stage12_reference import materialize_accepted_stage12_reference, write_obj_mesh
from toporetarget.rl.world_wrist import export_world_wrist_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-trajectory", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object-mesh-output", required=True, type=Path)
    parser.add_argument("--wuji-mjcf", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    robot_reference, vertices, faces, mesh_metadata = materialize_accepted_stage12_reference(
        final_trajectory=args.final_trajectory,
        canonical=args.canonical,
        checkpoint_manifest=args.checkpoint_manifest,
    )
    model = mujoco.MjModel.from_xml_path(str(args.wuji_mjcf))
    lower = model.jnt_range[: model.njnt, 0]
    upper = model.jnt_range[: model.njnt, 1]
    source_hashes = {
        "final_zarr_metadata": _sha256(args.final_trajectory / "zarr.json"),
        "canonical_zarr_metadata": _sha256(args.canonical / "zarr.json"),
        "checkpoint_manifest": _sha256(args.checkpoint_manifest),
    }
    world_reference = export_world_wrist_reference(
        robot_reference,
        source_hashes=source_hashes,
        engineering_assumptions=["ENGINEERING_ASSUMPTION_ZERO_GRAVITY_NO_GROUND"],
    )
    validation = world_reference.validate(joint_lower=lower, joint_upper=upper)
    output = world_reference.to_npz(args.output)
    mesh = write_obj_mesh(args.object_mesh_output, vertices, faces)
    report = {
        "status": "STAGE16B_WORLD_REFERENCE_VALIDATED",
        "reference": str(output.resolve()),
        "object_mesh": str(mesh.resolve()),
        "object_mesh_sha256": _sha256(mesh),
        "object_mesh_metadata": mesh_metadata,
        "source_artifacts": {
            "final_trajectory": str(args.final_trajectory.resolve()),
            "canonical": str(args.canonical.resolve()),
            "checkpoint_manifest": str(args.checkpoint_manifest.resolve()),
            "hashes": source_hashes,
        },
        "validation": validation,
        "not_derived_from_legacy_base_relative_npz": True,
        "engineering_extension": "WORLD_WRIST_FINGER_TRACKING_PROTOCOL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "reference", "object_mesh")}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
