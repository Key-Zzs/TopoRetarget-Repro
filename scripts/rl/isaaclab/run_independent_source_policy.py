#!/usr/bin/env python3
"""Build one independent geometry-to-source-policy lineage.

Production stops at L0 and delegates all further optimization to the
finite-support, grouped-multiplicative, full-physics route. Standalone
Strict-V4 PPO is forbidden by this runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_episode_manifest,
    atomic_write_json,
)
from toporetarget.runtime.gpu_preflight import (  # noqa: E402
    validate_gpu_preflight_receipt,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

L0_SAMPLES = 1_024_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-object-authority", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--geometric-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--wuji-mjcf", type=Path, required=True)
    parser.add_argument("--interaction-contact-contract", type=Path, required=True)
    parser.add_argument(
        "--source-policy-profile",
        choices=("source_controller_auto_v2",),
        default="source_controller_auto_v2",
    )
    parser.add_argument(
        "--stop-after-cpu-authorities",
        action="store_true",
        help=(
            "Freeze reference and source-contact prerequisites without launching Isaac or PPO; "
            "a later invocation in the same roots resumes from their PASS receipts."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--gpu-preflight-receipt",
        type=Path,
        help="Required before Isaac import or L0; omitted only for CPU-authorities stop.",
    )
    parser.add_argument("--support-preflight-receipt", type=Path)
    parser.add_argument("--base-runtime-geometry-manifest", type=Path)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_SOURCE_POLICY_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_SOURCE_POLICY_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _receipt_artifact(row: object) -> Path:
    if not isinstance(row, dict):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_RECEIPT_ARTIFACT_REQUIRED")
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file() or row.get("sha256") != sha256_file(path):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_RECEIPT_ARTIFACT_DRIFT")
    return path


def _run_step(
    name: str,
    command: list[str],
    *,
    log_root: Path,
    expected_artifacts: tuple[Path, ...] = (),
) -> dict[str, Any]:
    receipt_path = log_root / f"{name}.receipt.json"
    if receipt_path.is_file():
        previous = _json(receipt_path)
        if (
            previous.get("status") == "PASS"
            and previous.get("command") == command
            and all(path.is_file() for path in expected_artifacts)
        ):
            return {**previous, "resumed_from_pass_receipt": True}
        raise FileExistsError(
            f"INDEPENDENT_SOURCE_POLICY_REFUSES_RECEIPT_OVERWRITE:{receipt_path.resolve()}"
        )
    log_root.mkdir(parents=True, exist_ok=True)
    started = _utc()
    tick = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_root / f"{name}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    missing_artifacts = [str(path.resolve()) for path in expected_artifacts if not path.is_file()]
    passed = result.returncode == 0 and not missing_artifacts
    receipt = {
        "stage": name,
        "status": "PASS" if passed else "FAIL",
        "command": command,
        "expected_artifacts": [str(path.resolve()) for path in expected_artifacts],
        "missing_artifacts": missing_artifacts,
        "started_utc": started,
        "ended_utc": _utc(),
        "wall_seconds": time.perf_counter() - tick,
        "returncode": result.returncode,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }
    atomic_write_json(receipt_path, receipt)
    if not passed:
        raise RuntimeError(f"INDEPENDENT_SOURCE_POLICY_STAGE_FAILED:{name}:{log_path}")
    return receipt


def _zero_network_rollout_parity(
    direct_root: Path, network_root: Path, output: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    overall = True
    for episode in range(10):
        direct_path = direct_root / "traces" / f"episode_{episode:02d}.npz"
        network_path = network_root / "traces" / f"episode_{episode:02d}.npz"
        with (
            np.load(direct_path, allow_pickle=False) as direct_archive,
            np.load(network_path, allow_pickle=False) as network_archive,
        ):
            direct_fields = set(direct_archive.files)
            network_fields = set(network_archive.files)
            fields_match = direct_fields == network_fields
            max_abs = 0.0
            values_match = fields_match
            if fields_match:
                for name in sorted(direct_fields):
                    direct = np.asarray(direct_archive[name])
                    network = np.asarray(network_archive[name])
                    if direct.shape != network.shape:
                        values_match = False
                        break
                    if np.issubdtype(direct.dtype, np.number):
                        delta = float(
                            np.max(np.abs(direct.astype(np.float64) - network.astype(np.float64)))
                        )
                        max_abs = max(max_abs, delta)
                        values_match = values_match and delta <= 1.0e-6
                    else:
                        values_match = values_match and bool(np.array_equal(direct, network))
            passed = fields_match and values_match
            overall = overall and passed
            rows.append(
                {
                    "episode": episode,
                    "fields_match": fields_match,
                    "maximum_absolute_difference": max_abs,
                    "tolerance": 1.0e-6,
                    "passed": passed,
                }
            )
    receipt = {
        "schema_version": "ZeroResidualNetworkRolloutParityV1",
        "status": "PASS" if overall else "FAIL",
        "deterministic_reference_follower": str(direct_root.resolve()),
        "zero_output_network": str(network_root.resolve()),
        "absolute_tolerance": 1.0e-6,
        "episodes": rows,
    }
    atomic_write_json(output, receipt)
    return receipt


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.num_envs != 1024:
        raise ValueError("INDEPENDENT_SOURCE_POLICY_REQUIRES_EULA_AND_1024_ENVS")
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_CLIP_ID_INVALID")
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    assert_frozen_episode_manifest(manifest)
    rows = [row for row in manifest["clips"] if row.get("clip_id") == args.clip_id]
    if len(rows) != 1 or args.clip_id in {"hocap_170105", "hocap_170650"}:
        raise ValueError("INDEPENDENT_SOURCE_POLICY_CLIP_NOT_HELD_OUT")
    lineage_seed = int(rows[0]["selection_key"][:8], 16) & 0x7FFFFFFF
    authority_path = args.primary_object_authority.resolve()
    authority = _json(authority_path)
    if manifest.get("primary_object_authority_sha256") != authority.get("authority_sha256"):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_PRIMARY_AUTHORITY_MISMATCH")
    geometry_path = args.geometric_receipt.resolve()
    geometry = _json(geometry_path)
    if (
        geometry.get("status") != "PASS"
        or geometry.get("clip_id") != args.clip_id
        or geometry.get("primary_object_id") != rows[0].get("primary_object_id")
        or geometry.get("selection_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("INDEPENDENT_SOURCE_POLICY_GEOMETRY_RECEIPT_INVALID")

    run_root = args.run_root.resolve() / args.clip_id / "source_policy"
    report_root = args.report_root.resolve() / "clips" / args.clip_id / "source_policy"
    log_root = report_root / "logs"
    reference_root = run_root / "references"
    object_root = run_root / "objects"
    usd_parent = run_root / "object_usd"
    usd_root = usd_parent / args.clip_id
    contact_root = run_root / "source_contact"
    world_reference = reference_root / f"{args.clip_id}.world_wrist.stage16.npz"
    object_mesh = object_root / f"{args.clip_id}.obj"
    reference_v1 = reference_root / f"{args.clip_id}.reference_v1.npz"
    reference_v2 = reference_root / f"{args.clip_id}.reference_kinematics_v2.npz"
    reference_report = report_root / "reference_preparation.json"
    object_usd_report = report_root / "object_usd.json"
    object_usd = usd_root / f"{args.clip_id}.usda"
    source_contact_receipt = contact_root / args.clip_id / "source_contact_authority.json"
    l0_root = run_root / "l0"
    l0_checkpoint = (
        l0_root / args.clip_id / f"stage16d_ppo26d_{args.clip_id.removeprefix('hocap_')}_l0.pt"
    )
    l0_result = l0_root / args.clip_id / "l0_training.json"
    source_controller_root = report_root / "source_controller_v2"
    zero_direct_root = source_controller_root / "zero_residual_deterministic"
    zero_network_root = source_controller_root / "zero_residual_network"
    l0_qualification_root = source_controller_root / "corrected_l0"
    zero_checkpoint = run_root / "zero_residual" / "zero_residual_source.pt"
    zero_result = run_root / "zero_residual" / "source_training.json"
    zero_parity = source_controller_root / "zero_network_rollout_parity.json"
    contracts_root = args.report_root.resolve() / "clips" / args.clip_id / "physical_contracts"
    physical_contract_receipt = contracts_root / "physical_contract_receipt.json"
    runtime_geometry = contracts_root / "runtime_collision_geometry_manifest.json"
    evaluation_gates = contracts_root / "frozen_evaluation_gates.json"
    seed_manifest = contracts_root / "evaluation_seed_manifest.json"
    final_receipt = report_root / "source_policy_receipt.v4.json"
    if final_receipt.exists():
        raise FileExistsError(f"INDEPENDENT_SOURCE_POLICY_REFUSES_OVERWRITE:{final_receipt}")

    final = Path(str(geometry["artifacts"]["final"]["path"])).resolve()
    canonical = Path(str(geometry["artifacts"]["canonical"]["path"])).resolve()
    checkpoint_manifest = final.parent / "continuous_checkpoints" / "manifest.json"
    mjcf = args.wuji_mjcf.resolve()
    strict_contract = args.interaction_contact_contract.resolve()
    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        steps.append(
            _run_step(
                "prepare_reference",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-rl",
                    "python",
                    "scripts/rl/prepare_independent_source_reference.py",
                    "--clip-id",
                    args.clip_id,
                    "--final-trajectory",
                    str(final),
                    "--canonical",
                    str(canonical),
                    "--checkpoint-manifest",
                    str(checkpoint_manifest),
                    "--wuji-mjcf",
                    str(mjcf),
                    "--world-reference-output",
                    str(world_reference),
                    "--object-mesh-output",
                    str(object_mesh),
                    "--reference-v1-output",
                    str(reference_v1),
                    "--reference-v2-output",
                    str(reference_v2),
                    "--report",
                    str(reference_report),
                ],
                log_root=log_root,
                expected_artifacts=(
                    world_reference,
                    object_mesh,
                    reference_v1,
                    reference_v2,
                    reference_report,
                ),
            )
        )
        steps.append(
            _run_step(
                "materialize_source_contact",
                [
                    sys.executable,
                    "scripts/evaluation/materialize_independent_hocap_source_contact.py",
                    "--manifest",
                    str(manifest_path),
                    "--primary-object-authority",
                    str(authority_path),
                    "--clip-id",
                    args.clip_id,
                    "--world-reference",
                    str(world_reference),
                    "--reference-v2",
                    str(reference_v2),
                    "--interaction-contact-contract",
                    str(strict_contract),
                    "--output-root",
                    str(contact_root),
                ],
                log_root=log_root,
                expected_artifacts=(source_contact_receipt,),
            )
        )
        if args.stop_after_cpu_authorities:
            prerequisite_receipt = report_root / "source_policy_prerequisites_receipt.json"
            if prerequisite_receipt.exists():
                raise FileExistsError(
                    "INDEPENDENT_SOURCE_POLICY_REFUSES_PREREQUISITE_OVERWRITE:"
                    f"{prerequisite_receipt}"
                )
            contact = _json(source_contact_receipt)
            if (
                contact.get("schema_version") != "IndependentHOCapSourceContactAuthorityV2"
                or contact.get("status") != "PASS"
                or contact.get("support_contact_authority", {}).get("scope")
                != "all_annotated_source_hands"
            ):
                raise RuntimeError("INDEPENDENT_SOURCE_POLICY_CONTACT_PREREQUISITE_INVALID")
            prerequisite = {
                "schema_version": "IndependentSourcePolicyPrerequisitesReceiptV2",
                "status": "PASS",
                "clip_id": args.clip_id,
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "primary_object_authority_sha256": authority["authority_sha256"],
                "source_policy_profile": "source_controller_auto_v2",
                "terminal_scope": "CPU_AUTHORITIES_ONLY",
                "isaac_object_import": "NOT_RUN",
                "l0_training": "NOT_RUN",
                "standalone_strict_v4_training": "FORBIDDEN_NOT_RUN",
                "ppo_optimizer_steps": 0,
                "artifacts": {
                    "world_reference": _artifact(world_reference),
                    "reference_v1": _artifact(reference_v1),
                    "reference_v2": _artifact(reference_v2),
                    "object_mesh": _artifact(object_mesh),
                    "source_contact": _artifact(source_contact_receipt),
                },
                "stages": steps,
                "productive_run_seconds": time.perf_counter() - started,
                "technical_retry_seconds": 0.0,
                "retry_count": 0,
                "cache_hit": False,
            }
            atomic_write_json(prerequisite_receipt, prerequisite)
            print(json.dumps(prerequisite, indent=2, sort_keys=True))
            return 0
        if (
            args.gpu_preflight_receipt is None
            or args.support_preflight_receipt is None
            or args.base_runtime_geometry_manifest is None
        ):
            raise ValueError(
                "GPU_SUPPORT_AND_GEOMETRY_RECEIPTS_REQUIRED_BEFORE_SOURCE_CONTROLLER_AUTO_V2"
            )
        gpu_preflight_path = args.gpu_preflight_receipt.resolve()
        validate_gpu_preflight_receipt(gpu_preflight_path)
        support_preflight_path = args.support_preflight_receipt.resolve()
        support_preflight = _json(support_preflight_path)
        if (
            support_preflight.get("schema_version")
            != "IndependentPhysicalSupportPreflightReceiptV1"
            or support_preflight.get("status") != "PASS"
            or support_preflight.get("clip_id") != args.clip_id
            or support_preflight.get("gpu_physx_authorized") is not True
        ):
            raise ValueError("INDEPENDENT_SOURCE_POLICY_SUPPORT_PREFLIGHT_INVALID")
        support_proxy = _receipt_artifact(
            support_preflight.get("artifacts", {}).get("support_proxy")
        )
        support_asset = _receipt_artifact(
            support_preflight.get("artifacts", {}).get("support_asset")
        )
        base_geometry = args.base_runtime_geometry_manifest.resolve()
        _artifact(base_geometry)
        steps.append(
            _run_step(
                "import_object_usd",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-isaaclab",
                    "python",
                    "scripts/rl/isaaclab/import_hocap_objects.py",
                    "--mesh",
                    str(object_mesh),
                    "--object-id",
                    args.clip_id,
                    "--output-dir",
                    str(usd_root),
                    "--report",
                    str(object_usd_report),
                    "--accept-eula",
                ],
                log_root=log_root,
                expected_artifacts=(object_usd, object_usd_report),
            )
        )
        contact = _json(source_contact_receipt)
        if (
            contact.get("schema_version") != "IndependentHOCapSourceContactAuthorityV2"
            or contact.get("status") != "PASS"
            or contact.get("support_contact_authority", {}).get("scope")
            != "all_annotated_source_hands"
        ):
            raise RuntimeError("INDEPENDENT_SOURCE_POLICY_CONTACT_CONTRACT_INVALID")
        strict_mask = _receipt_artifact(contact["artifacts"]["strict_mask"])
        reference_distance = _receipt_artifact(contact["artifacts"]["reference_distance"])
        steps.append(
            _run_step(
                "freeze_physical_contracts",
                [
                    sys.executable,
                    "scripts/rl/prepare_independent_physical_contracts.py",
                    "--selection-manifest",
                    str(manifest_path),
                    "--clip-id",
                    args.clip_id,
                    "--world-reference",
                    str(world_reference),
                    "--reference-v2",
                    str(reference_v2),
                    "--object-mesh",
                    str(object_mesh),
                    "--object-usd",
                    str(object_usd),
                    "--strict-source-mask",
                    str(strict_mask),
                    "--base-runtime-geometry-manifest",
                    str(base_geometry),
                    "--output-root",
                    str(contracts_root),
                ],
                log_root=log_root,
                expected_artifacts=(
                    physical_contract_receipt,
                    runtime_geometry,
                    evaluation_gates,
                    seed_manifest,
                ),
            )
        )
        qualifier_base = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "toporetarget-isaaclab",
            "python",
            "scripts/rl/isaaclab/qualify_zero_residual_source_controller.py",
            "--accept-eula",
            "--clip",
            args.clip_id,
            "--episodes",
            "10",
            "--reference",
            str(reference_v2),
            "--object-usd",
            str(object_usd),
            "--support-proxy",
            str(support_proxy),
            "--support-asset",
            str(support_asset),
            "--contact-contract",
            str(strict_contract),
            "--contact-mask-root",
            str(strict_mask.parent),
            "--reference-distance-root",
            str(reference_distance.parent),
            "--object-mesh-root",
            str(object_mesh.parent),
            "--runtime-geometry-manifest",
            str(runtime_geometry),
            "--frozen-evaluation-gates",
            str(evaluation_gates),
            "--seed-manifest",
            str(seed_manifest),
        ]
        steps.append(
            _run_step(
                "qualify_zero_residual_deterministic_v2",
                [
                    *qualifier_base,
                    "--controller-mode",
                    "ZERO_RESIDUAL_DETERMINISTIC",
                    "--output",
                    str(zero_direct_root),
                ],
                log_root=log_root,
                expected_artifacts=(zero_direct_root / "qualification.json",),
            )
        )
        zero_direct = _json(zero_direct_root / "qualification.json")
        if zero_direct.get("source_controller_executability_v2") == "PASS":
            steps.append(
                _run_step(
                    "materialize_zero_residual_network",
                    [
                        "conda",
                        "run",
                        "--no-capture-output",
                        "-n",
                        "toporetarget-isaaclab",
                        "python",
                        "scripts/rl/materialize_zero_residual_source.py",
                        "--qualification",
                        str(zero_direct_root / "qualification.json"),
                        "--checkpoint",
                        str(zero_checkpoint),
                        "--result",
                        str(zero_result),
                    ],
                    log_root=log_root,
                    expected_artifacts=(zero_checkpoint, zero_result),
                )
            )
            steps.append(
                _run_step(
                    "qualify_zero_residual_network_v2",
                    [
                        *qualifier_base,
                        "--controller-mode",
                        "ZERO_RESIDUAL_NETWORK",
                        "--checkpoint",
                        str(zero_checkpoint),
                        "--output",
                        str(zero_network_root),
                    ],
                    log_root=log_root,
                    expected_artifacts=(zero_network_root / "qualification.json",),
                )
            )
            zero_network_path = zero_network_root / "qualification.json"
            zero_network = _json(zero_network_path)
            parity = _zero_network_rollout_parity(zero_direct_root, zero_network_root, zero_parity)
            if (
                zero_network.get("source_controller_executability_v2") != "PASS"
                or parity["status"] != "PASS"
            ):
                raise RuntimeError("ZERO_RESIDUAL_NETWORK_PARITY_OR_EXECUTABILITY_FAILED")
            receipt = {
                "schema_version": "IndependentSourcePolicyReceiptV4",
                "status": "PASS",
                "clip_id": args.clip_id,
                "primary_object_id": rows[0]["primary_object_id"],
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "primary_object_authority_sha256": authority["authority_sha256"],
                "source_policy_profile": "source_controller_auto_v2",
                "selected_route": "ZERO_RESIDUAL",
                "source_controller_executability_v2": "PASS",
                "source_controller_fidelity_v2": zero_network["source_controller_fidelity_v2"],
                "l0_samples": 0,
                "standalone_strict_v4_samples": 0,
                "checkpoint": str(zero_checkpoint),
                "checkpoint_sha256": sha256_file(zero_checkpoint),
                "source_training_result": _artifact(zero_result),
                "source_qualification": _artifact(zero_network_path),
                "standalone_strict_v4_training": {
                    "status": "FORBIDDEN_NOT_RUN",
                    "samples": 0,
                    "ppo_optimizer_steps": 0,
                },
                "required_downstream_contract": {
                    "support": "finite_inferred_table_proxy_v1",
                    "gravity_scale": 1.0,
                    "friction_scale": 1.0,
                    "reward_aggregation": "grouped_multiplicative_v1",
                    "interaction_term": "u10_per_finger_pair_contact_primitive_v1",
                    "rse_enabled": True,
                    "standalone_strict_v4_ppo": False,
                    "evaluation_first": True,
                    "ppo_only_on_frozen_failure": True,
                },
                "lineage": {
                    "actor_root": str(zero_checkpoint.parent.resolve()),
                    "critic_root": str(zero_checkpoint.parent.resolve()),
                    "optimizer_root": str(zero_checkpoint.parent.resolve()),
                    "normalizer_root": str(zero_checkpoint.parent.resolve()),
                    "rng_seed": lineage_seed,
                },
                "artifacts": {
                    "gpu_preflight": _artifact(gpu_preflight_path),
                    "support_preflight": _artifact(support_preflight_path),
                    "world_reference": _artifact(world_reference),
                    "reference_v1": _artifact(reference_v1),
                    "reference_v2": _artifact(reference_v2),
                    "object_mesh": _artifact(object_mesh),
                    "object_usd": _artifact(object_usd),
                    "source_contact": _artifact(source_contact_receipt),
                    "physical_contracts": _artifact(physical_contract_receipt),
                    "zero_residual_qualification": _artifact(
                        zero_direct_root / "qualification.json"
                    ),
                    "zero_network_parity": _artifact(zero_parity),
                },
                "stages": steps,
                "productive_run_seconds": time.perf_counter() - started,
                "technical_retry_seconds": 0.0,
                "retry_count": 0,
                "cache_hit": False,
            }
            atomic_write_json(final_receipt, receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        steps.append(
            _run_step(
                "train_l0",
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "toporetarget-isaaclab",
                    "python",
                    "scripts/rl/isaaclab/train_stage16d_ppo26d.py",
                    "--clip",
                    args.clip_id,
                    "--reference",
                    str(reference_v2),
                    "--object-usd",
                    str(object_usd),
                    "--output-root",
                    str(l0_root),
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(lineage_seed),
                    "--continuous-virtual-wrist-angles",
                    "--accept-eula",
                ],
                log_root=log_root,
                expected_artifacts=(l0_checkpoint, l0_result),
            )
        )
        l0 = _json(l0_result)
        contact = _json(source_contact_receipt)
        if (
            l0.get("status") != "STAGE16D_PPO26D_L0_COMPLETE_NOT_YET_QUALIFIED"
            or int(l0.get("cumulative_samples", -1)) != L0_SAMPLES
            or int(l0.get("target_l0_samples", -1)) != L0_SAMPLES
            or int(l0.get("seed", -1)) != lineage_seed
            or contact.get("schema_version") != "IndependentHOCapSourceContactAuthorityV2"
            or contact.get("status") != "PASS"
            or contact.get("support_contact_authority", {}).get("scope")
            != "all_annotated_source_hands"
        ):
            raise RuntimeError("INDEPENDENT_SOURCE_POLICY_L0_FINAL_CONTRACT_INVALID")
        steps.append(
            _run_step(
                "qualify_corrected_bounded_l0_v2",
                [
                    *qualifier_base,
                    "--controller-mode",
                    "CORRECTED_L0",
                    "--checkpoint",
                    str(l0_checkpoint),
                    "--optimizer-steps",
                    "250",
                    "--training-samples",
                    str(L0_SAMPLES),
                    "--output",
                    str(l0_qualification_root),
                ],
                log_root=log_root,
                expected_artifacts=(l0_qualification_root / "qualification.json",),
            )
        )
        l0_qualification_path = l0_qualification_root / "qualification.json"
        l0_qualification = _json(l0_qualification_path)
        if l0_qualification.get("source_controller_executability_v2") != "PASS":
            raise RuntimeError("SOURCE_CONTROLLER_HARD_FAILURE")
        receipt = {
            "schema_version": "IndependentSourcePolicyReceiptV4",
            "status": "PASS",
            "clip_id": args.clip_id,
            "primary_object_id": rows[0]["primary_object_id"],
            "selection_manifest_sha256": manifest["manifest_sha256"],
            "primary_object_authority_sha256": authority["authority_sha256"],
            "source_policy_profile": "source_controller_auto_v2",
            "selected_route": "CORRECTED_L0",
            "source_controller_executability_v2": "PASS",
            "source_controller_fidelity_v2": l0_qualification["source_controller_fidelity_v2"],
            "l0_samples": L0_SAMPLES,
            "standalone_strict_v4_samples": 0,
            "checkpoint": str(l0_checkpoint),
            "checkpoint_sha256": sha256_file(l0_checkpoint),
            "source_training_result": _artifact(l0_result),
            "source_qualification": _artifact(l0_qualification_path),
            "standalone_strict_v4_training": {
                "status": "FORBIDDEN_NOT_RUN",
                "samples": 0,
                "ppo_optimizer_steps": 0,
            },
            "required_downstream_contract": {
                "support": "finite_inferred_table_proxy_v1",
                "gravity_scale": 1.0,
                "friction_scale": 1.0,
                "reward_aggregation": "grouped_multiplicative_v1",
                "interaction_term": "u10_per_finger_pair_contact_primitive_v1",
                "rse_enabled": True,
                "standalone_strict_v4_ppo": False,
                "evaluation_first": True,
                "ppo_only_on_frozen_failure": True,
            },
            "lineage": {
                "actor_root": str((l0_root / args.clip_id).resolve()),
                "critic_root": str((l0_root / args.clip_id).resolve()),
                "optimizer_root": str((l0_root / args.clip_id).resolve()),
                "normalizer_root": str((l0_root / args.clip_id).resolve()),
                "rng_seed": lineage_seed,
            },
            "artifacts": {
                "gpu_preflight": _artifact(gpu_preflight_path),
                "support_preflight": _artifact(support_preflight_path),
                "world_reference": _artifact(world_reference),
                "reference_v1": _artifact(reference_v1),
                "reference_v2": _artifact(reference_v2),
                "object_mesh": _artifact(object_mesh),
                "object_usd": _artifact(object_usd),
                "source_contact": _artifact(source_contact_receipt),
                "l0_result": _artifact(l0_result),
                "physical_contracts": _artifact(physical_contract_receipt),
                "zero_residual_qualification": _artifact(zero_direct_root / "qualification.json"),
            },
            "stages": steps,
            "productive_run_seconds": time.perf_counter() - started,
            "technical_retry_seconds": 0.0,
            "retry_count": 0,
            "cache_hit": False,
        }
        atomic_write_json(final_receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        atomic_write_json(
            report_root / "source_policy_failure.json",
            {
                "schema_version": "IndependentSourcePolicyFailureV1",
                "status": "FAIL",
                "clip_id": args.clip_id,
                "reason": f"{type(error).__name__}:{error}",
                "completed_stages": steps,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
