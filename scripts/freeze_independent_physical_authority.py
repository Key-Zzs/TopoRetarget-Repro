#!/usr/bin/env python3
"""Freeze command-level production authority for one held-out five-clip manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_manifest,
    atomic_write_json,
    freeze_method_contract,
    validate_authority_manifest,
)

RETARGET_SOLVER_PROFILE_ID = "wuji_continuous_sequential_v1"
RETARGET_EXECUTION_PROFILE_ID = "wuji_continuous_sequential_fast_exact_v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-object-authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_frozen_manifest(manifest)
    primary = json.loads(args.primary_object_authority.read_text(encoding="utf-8"))
    if manifest.get("primary_object_authority_sha256") != primary.get("authority_sha256"):
        raise ValueError("INDEPENDENT_PRODUCTION_AUTHORITY_PRIMARY_HASH_MISMATCH")
    clips = [str(row["clip_id"]) for row in manifest["clips"]]
    execution = RefinementExecutionProfile.load(RETARGET_EXECUTION_PROFILE_ID, REPO_ROOT)
    if not (
        execution.math_equivalent
        and execution.paper_objective_unchanged
        and execution.paper_constraints_unchanged
        and execution.continuity_contract_unchanged
        and execution.final_full_surface_audit
    ):
        raise ValueError("INDEPENDENT_RETARGET_EXECUTION_PROFILE_NOT_QUALIFIED")
    common = {"supported_clips": clips, "max_concurrent_clips": 1}
    authority = {
        "schema_version": "IndependentPhysicalRefinementProductionAuthorityV2",
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "primary_object_authority_sha256": primary["authority_sha256"],
        "authorities": {
            "retarget": {
                **common,
                "method": {
                    "solver_profile_id": RETARGET_SOLVER_PROFILE_ID,
                    "execution_profile_id": execution.profile_id,
                    "execution_profile_sha256": execution.profile_hash,
                    "math_equivalent": execution.math_equivalent,
                    "paper_objective_unchanged": execution.paper_objective_unchanged,
                    "paper_constraints_unchanged": execution.paper_constraints_unchanged,
                    "continuity_contract_unchanged": execution.continuity_contract_unchanged,
                    "final_full_surface_audit": execution.final_full_surface_audit,
                },
                "command": [
                    sys.executable,
                    "scripts/run_hocap_geometric_retarget_v2.py",
                    "--manifest",
                    str(manifest_path),
                    "--primary-object-authority",
                    str(args.primary_object_authority.resolve()),
                    "--clip-id",
                    "{clip_id}",
                    "--data-root",
                    "/mnt/nas/storage/Ref2Dex_storage/HOCap",
                    "--run-root",
                    "{run_root}",
                    "--report-root",
                    "{report_root}",
                    "--execution-profile",
                    execution.profile_id,
                ],
                "receipt": "{clip_report_root}/geometric_retarget_receipt.json",
            },
            "source_policy": {
                **common,
                "command": [
                    sys.executable,
                    "scripts/rl/isaaclab/run_independent_source_policy.py",
                    "--manifest",
                    str(manifest_path),
                    "--primary-object-authority",
                    str(args.primary_object_authority.resolve()),
                    "--clip-id",
                    "{clip_id}",
                    "--geometric-receipt",
                    "{clip_report_root}/geometric_retarget_receipt.json",
                    "--run-root",
                    "{run_root}",
                    "--report-root",
                    "{report_root}",
                    "--wuji-mjcf",
                    "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml",
                    "--strict-v4-contract",
                    ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json",
                    "--num-envs",
                    "1024",
                    "--accept-eula",
                ],
                "receipt": "{clip_report_root}/source_policy/source_policy_receipt.json",
            },
            "support": {
                **common,
                "command": [
                    sys.executable,
                    "scripts/physics/run_independent_physical_support.py",
                    "--manifest",
                    str(manifest_path),
                    "--clip-id",
                    "{clip_id}",
                    "--source-policy-receipt",
                    "{clip_report_root}/source_policy/source_policy_receipt.json",
                    "--run-root",
                    "{run_root}",
                    "--report-root",
                    "{report_root}",
                    "--strict-v4-contract",
                    ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json",
                    "--base-runtime-geometry-manifest",
                    ".local/reports/stage16d_metric_qualification_and_ppo/runtime_collision_geometry_manifest.json",
                    "--accept-eula",
                ],
                "receipt": "{clip_report_root}/support/support_receipt.json",
            },
            "frozen_evaluation": {
                **common,
                "command": [
                    sys.executable,
                    "scripts/evaluation/run_independent_frozen_physical_evaluation.py",
                    "--manifest",
                    str(manifest_path),
                    "--clip-id",
                    "{clip_id}",
                    "--source-policy-receipt",
                    "{clip_report_root}/source_policy/source_policy_receipt.json",
                    "--support-receipt",
                    "{clip_report_root}/support/support_receipt.json",
                    "--strict-v4-contract",
                    ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json",
                    "--run-root",
                    "{run_root}",
                    "--report-root",
                    "{report_root}",
                    "--accept-eula",
                ],
                "receipt": (
                    "{clip_report_root}/physical_refinement/frozen_evaluation_receipt.json"
                ),
            },
            "physical_refinement": {
                **common,
                "command": [
                    sys.executable,
                    "scripts/rl/isaaclab/run_independent_physical_refinement.py",
                    "--clip-id",
                    "{clip_id}",
                    "--frozen-evaluation-receipt",
                    "{clip_report_root}/physical_refinement/frozen_evaluation_receipt.json",
                    "--accept-eula",
                ],
                "receipt": (
                    "{clip_report_root}/physical_refinement/physical_refinement_receipt.json"
                ),
            },
            "qualification": {
                **common,
                "command": [
                    sys.executable,
                    "scripts/evaluation/finalize_independent_physical_result.py",
                    "--clip-id",
                    "{clip_id}",
                    "--frozen-evaluation-receipt",
                    "{clip_report_root}/physical_refinement/frozen_evaluation_receipt.json",
                    "--output",
                    "{clip_report_root}/final_result.json",
                ],
                "receipt": "{clip_report_root}/final_result.json",
            },
            "trace_export": {
                **common,
                "command": [
                    sys.executable,
                    "scripts/evaluation/export_independent_physical_trace.py",
                    "--clip-id",
                    "{clip_id}",
                    "--final-result",
                    "{clip_report_root}/final_result.json",
                    "--source-policy-receipt",
                    "{clip_report_root}/source_policy/source_policy_receipt.json",
                    "--support-receipt",
                    "{clip_report_root}/support/support_receipt.json",
                    "--output-root",
                    "{clip_report_root}/replay",
                    "--accept-eula",
                ],
                "receipt": "{clip_report_root}/replay/trace_export_receipt.json",
            },
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"INDEPENDENT_PRODUCTION_AUTHORITY_REFUSES_OVERWRITE:{output}")
    _contract, method_hash = freeze_method_contract(args.contract_root.resolve())
    authority["method_contract_hash"] = method_hash
    check = validate_authority_manifest(authority, clips)
    if check.get("valid") is not True:
        raise RuntimeError(f"INDEPENDENT_PRODUCTION_AUTHORITY_INVALID:{check}")
    atomic_write_json(output, authority)
    print(json.dumps({"status": "PASS", "output": str(output), **check}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
