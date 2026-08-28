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

from toporetarget.evaluation.retarget_semantic_validity import (  # noqa: E402
    require_semantic_admission,
)
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

_REFERENCE_EXECUTABILITY_CHECKS = (
    "source_key_preservation",
    "timestamps",
    "quaternion",
    "finite",
    "linear_fd_consistency",
    "angular_so3_consistency",
    "world_angular_convention",
)
_REFERENCE_FIDELITY_ONLY_CHECKS = (
    "factor8_scaling",
    "integral_consistency",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--final-trajectory", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--geometric-receipt", type=Path, required=True)
    parser.add_argument("--semantic-qualification", type=Path, required=True)
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


def require_numerical_solver_success(
    receipt_path: Path,
    *,
    clip_id: str,
    canonical: Path,
    final: Path,
    checkpoint_manifest: Path,
) -> dict[str, str]:
    """Validate the numerical half of production retarget admission."""

    resolved = receipt_path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("INDEPENDENT_SOURCE_REFERENCE_GEOMETRIC_RECEIPT_INVALID") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "PASS"
        or payload.get("clip_id", payload.get("episode_id")) != clip_id
    ):
        raise ValueError("INDEPENDENT_SOURCE_REFERENCE_NUMERICAL_SOLVER_NONPASS")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("INDEPENDENT_SOURCE_REFERENCE_GEOMETRIC_ARTIFACTS_MISSING")
    expected = {
        "canonical": canonical.resolve(),
        "final": final.resolve(),
        "checkpoint_manifest": checkpoint_manifest.resolve(),
    }
    for name, expected_path in expected.items():
        row = artifacts.get(name)
        reported = Path(str(row.get("path", ""))).resolve() if isinstance(row, dict) else Path()
        if reported != expected_path:
            raise ValueError(f"INDEPENDENT_SOURCE_REFERENCE_GEOMETRIC_BINDING_MISMATCH:{name}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "status": "PASS"}


def reference_executability_v2(
    *, qualification: dict[str, Any], world_validation: dict[str, Any]
) -> dict[str, Any]:
    """Separate readable reference authority from derivative-fidelity diagnostics."""

    checks = qualification.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    hard_checks = {name: checks.get(name) is True for name in _REFERENCE_EXECUTABILITY_CHECKS}
    hard_checks["world_reference_valid"] = world_validation.get("valid") is True
    diagnostic_checks = {name: checks.get(name) is True for name in _REFERENCE_FIDELITY_ONLY_CHECKS}
    passed = all(hard_checks.values())
    return {
        "schema_version": "IndependentSourceReferenceExecutabilityV2",
        "status": "PASS" if passed else "FAIL",
        "hard_checks": hard_checks,
        "failed_hard_checks": [name for name, value in hard_checks.items() if not value],
        "fidelity_only_diagnostics": diagnostic_checks,
        "full_reference_kinematics_v2_status": qualification.get("status"),
        "admission_rule": (
            "finite complete readable world/reference rows and internally consistent stored "
            "kinematics are hard; factor8 relative-derivative and integral reconstruction "
            "accuracy remain reported fidelity diagnostics"
        ),
        "policy_outcomes_observed": False,
    }


def main() -> int:
    args = _parser().parse_args()
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
    numerical_admission = require_numerical_solver_success(
        args.geometric_receipt,
        clip_id=args.clip_id,
        canonical=canonical,
        final=final,
        checkpoint_manifest=checkpoint_manifest,
    )
    semantic_admission = require_semantic_admission(
        args.semantic_qualification,
        identifier=args.clip_id,
        canonical=canonical,
        final=final,
    )
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("INDEPENDENT_SOURCE_REFERENCE_REQUIRES_MUJOCO_ENVIRONMENT") from exc

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
    executability = reference_executability_v2(
        qualification=qualification, world_validation=validation
    )
    report: dict[str, Any] = {
        "schema_version": "IndependentSourcePolicyReferencePreparationV2",
        "status": "PASS" if executability["status"] == "PASS" else "FAIL",
        "clip_id": args.clip_id,
        "source_artifacts": {
            "final_trajectory_zarr": _artifact(final / "zarr.json"),
            "canonical_zarr": _artifact(canonical / "zarr.json"),
            "checkpoint_manifest": _artifact(checkpoint_manifest),
            "wuji_mjcf": _artifact(mjcf),
            "numerical_solver_qualification": numerical_admission,
            "retarget_semantic_qualification": semantic_admission,
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
        "reference_executability_v2": executability,
        "policy_outcomes_observed": False,
    }
    atomic_write_json(args.report.resolve(), report)
    if executability["status"] != "PASS":
        raise RuntimeError("INDEPENDENT_SOURCE_REFERENCE_EXECUTABILITY_V2_FAILED")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
