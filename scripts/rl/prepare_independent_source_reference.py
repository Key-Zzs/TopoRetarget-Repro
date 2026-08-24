#!/usr/bin/env python3
"""Prepare one accepted geometric trajectory for independent source-policy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import atomic_write_json  # noqa: E402
from toporetarget.rl.reference_tracking.ppo26d_reference import (  # noqa: E402
    export_factor8_reference,
    sha256_file,
)
from toporetarget.rl.reference_tracking.reference_kinematics import (  # noqa: E402
    materialize_reference_kinematics_v2,
    qualify_reference_kinematics_v2,
)
from toporetarget.rl.stage12_reference import (  # noqa: E402
    materialize_accepted_stage12_reference,
    write_obj_mesh,
)
from toporetarget.rl.world_wrist import export_world_wrist_reference  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--final-trajectory", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--wuji-mjcf", type=Path, required=True)
    parser.add_argument("--world-reference-output", type=Path, required=True)
    parser.add_argument("--object-mesh-output", type=Path, required=True)
    parser.add_argument("--reference-v1-output", type=Path, required=True)
    parser.add_argument("--reference-v2-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_SOURCE_REFERENCE_INPUT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _require_distinct_outputs(paths: tuple[Path, ...]) -> None:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("INDEPENDENT_SOURCE_REFERENCE_OUTPUT_ALIAS")
    existing = [str(path) for path in resolved if path.exists()]
    if existing:
        raise FileExistsError(f"INDEPENDENT_SOURCE_REFERENCE_REFUSES_OVERWRITE:{existing}")


def main() -> int:
    args = _parser().parse_args()
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("INDEPENDENT_SOURCE_REFERENCE_REQUIRES_MUJOCO_ENVIRONMENT") from exc
    final = args.final_trajectory.resolve()
    canonical = args.canonical.resolve()
    checkpoint_manifest = args.checkpoint_manifest.resolve()
    mjcf = args.wuji_mjcf.resolve()
    outputs = (
        args.world_reference_output,
        args.object_mesh_output,
        args.reference_v1_output,
        args.reference_v2_output,
        args.report,
    )
    _require_distinct_outputs(outputs)
    for path in (final / "zarr.json", canonical / "zarr.json", checkpoint_manifest, mjcf):
        if not path.is_file():
            raise FileNotFoundError(f"INDEPENDENT_SOURCE_REFERENCE_INPUT_MISSING:{path}")

    robot_reference, vertices, faces, mesh_metadata = materialize_accepted_stage12_reference(
        final_trajectory=final,
        canonical=canonical,
        checkpoint_manifest=checkpoint_manifest,
    )
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    joint_lower = model.jnt_range[: model.njnt, 0]
    joint_upper = model.jnt_range[: model.njnt, 1]
    source_hashes = {
        "final_zarr_metadata": sha256_file(final / "zarr.json"),
        "canonical_zarr_metadata": sha256_file(canonical / "zarr.json"),
        "checkpoint_manifest": sha256_file(checkpoint_manifest),
    }
    world = export_world_wrist_reference(
        robot_reference,
        source_hashes=source_hashes,
        engineering_assumptions=["ENGINEERING_ASSUMPTION_ZERO_GRAVITY_NO_GROUND"],
    )
    validation = world.validate(joint_lower=joint_lower, joint_upper=joint_upper)
    world_path = world.to_npz(args.world_reference_output.resolve())
    object_path = write_obj_mesh(args.object_mesh_output.resolve(), vertices, faces)
    v1 = export_factor8_reference(world_path, args.reference_v1_output.resolve())
    v2 = materialize_reference_kinematics_v2(
        world_path,
        args.reference_v1_output.resolve(),
        args.reference_v2_output.resolve(),
    )
    qualification = qualify_reference_kinematics_v2(
        world_path,
        args.reference_v1_output.resolve(),
        args.reference_v2_output.resolve(),
    )
    if qualification.get("status") != "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED":
        raise RuntimeError("INDEPENDENT_SOURCE_REFERENCE_V2_QUALIFICATION_FAILED")
    report: dict[str, Any] = {
        "schema_version": "IndependentSourcePolicyReferencePreparationV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "source_artifacts": {
            "final_trajectory_zarr": _artifact(final / "zarr.json"),
            "canonical_zarr": _artifact(canonical / "zarr.json"),
            "checkpoint_manifest": _artifact(checkpoint_manifest),
            "wuji_mjcf": _artifact(mjcf),
        },
        "world_reference": _artifact(world_path),
        "world_reference_validation": validation,
        "object_mesh": {
            **_artifact(object_path),
            "metadata": mesh_metadata,
        },
        "factor8_v1": v1,
        "factor8_v2": v2,
        "factor8_v2_qualification": qualification,
        "policy_outcomes_observed": False,
    }
    atomic_write_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
