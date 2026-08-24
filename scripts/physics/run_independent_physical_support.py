#!/usr/bin/env python3
"""Prepare, counterfactually validate, and freeze one independent support contract."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_manifest,
    atomic_write_json,
)
from toporetarget.runtime.gpu_preflight import (  # noqa: E402
    validate_gpu_preflight_receipt,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--source-policy-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--base-runtime-geometry-manifest", type=Path, required=True)
    parser.add_argument(
        "--gpu-preflight-receipt",
        type=Path,
        help="Required before the PhysX support smoke; omitted only for --preflight-only.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run CPU support authority/geometry gates and never launch Isaac/PhysX.",
    )
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_SUPPORT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_SUPPORT_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _receipt_path(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file() or row.get("sha256") != sha256_file(path):
        raise ValueError("INDEPENDENT_SUPPORT_RECEIPT_ARTIFACT_DRIFT")
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
            f"INDEPENDENT_SUPPORT_REFUSES_RECEIPT_OVERWRITE:{receipt_path.resolve()}"
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
        raise RuntimeError(f"INDEPENDENT_SUPPORT_STAGE_FAILED:{name}:{log_path}")
    return receipt


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("INDEPENDENT_SUPPORT_REQUIRES_EULA")
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_SUPPORT_CLIP_ID_INVALID")
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    assert_frozen_manifest(manifest)
    rows = [row for row in manifest["clips"] if row.get("clip_id") == args.clip_id]
    if len(rows) != 1 or args.clip_id in {"hocap_170105", "hocap_170650"}:
        raise ValueError("INDEPENDENT_SUPPORT_CLIP_NOT_HELD_OUT")
    source_receipt_path = args.source_policy_receipt.resolve()
    source = _json(source_receipt_path)
    source_schema = source.get("schema_version")
    allowed_source_schemas = {"IndependentSourcePolicyReceiptV3"}
    if args.preflight_only:
        allowed_source_schemas.add("IndependentSourcePolicyPrerequisitesReceiptV2")
    if (
        source_schema not in allowed_source_schemas
        or source.get("status") != "PASS"
        or source.get("clip_id") != args.clip_id
        or source.get("selection_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("INDEPENDENT_SUPPORT_SOURCE_POLICY_RECEIPT_INVALID")
    if source_schema == "IndependentSourcePolicyReceiptV3" and (
        source.get("source_policy_profile") != "l0_then_physical_grouped_rse_v1"
        or int(source.get("l0_samples", -1)) != 1_024_000
        or int(source.get("standalone_strict_v4_samples", -1)) != 0
        or source.get("required_downstream_contract", {}).get("reward_aggregation")
        != "grouped_multiplicative_v1"
        or source.get("required_downstream_contract", {}).get("interaction_term")
        != "u10_per_finger_pair_contact_primitive_v1"
        or source.get("required_downstream_contract", {}).get("rse_enabled") is not True
        or source.get("required_downstream_contract", {}).get("standalone_strict_v4_ppo")
        is not False
    ):
        raise ValueError("INDEPENDENT_SUPPORT_L0_SOURCE_POLICY_CONTRACT_INVALID")
    if source_schema == "IndependentSourcePolicyPrerequisitesReceiptV2" and (
        source.get("source_policy_profile") != "l0_then_physical_grouped_rse_v1"
        or source.get("terminal_scope") != "CPU_AUTHORITIES_ONLY"
        or source.get("isaac_object_import") != "NOT_RUN"
        or source.get("l0_training") != "NOT_RUN"
        or source.get("standalone_strict_v4_training") != "FORBIDDEN_NOT_RUN"
        or int(source.get("ppo_optimizer_steps", -1)) != 0
    ):
        raise ValueError("INDEPENDENT_SUPPORT_CPU_PREREQUISITE_CONTRACT_INVALID")

    reference = _receipt_path(source["artifacts"]["world_reference"])
    reference_v2 = _receipt_path(source["artifacts"]["reference_v2"])
    object_mesh = _receipt_path(source["artifacts"]["object_mesh"])
    object_usd = (
        _receipt_path(source["artifacts"]["object_usd"])
        if source_schema == "IndependentSourcePolicyReceiptV3"
        else None
    )
    source_contact_path = _receipt_path(source["artifacts"]["source_contact"])
    source_contact = _json(source_contact_path)
    if (
        source_contact.get("schema_version") != "IndependentHOCapSourceContactAuthorityV2"
        or source_contact.get("status") != "PASS"
        or source_contact.get("clip_id") != args.clip_id
        or source_contact.get("selection_manifest_sha256") != manifest["manifest_sha256"]
        or source_contact.get("support_contact_authority", {}).get("scope")
        != "all_annotated_source_hands"
    ):
        raise ValueError("INDEPENDENT_SUPPORT_SOURCE_CONTACT_RECEIPT_INVALID")
    strict_mask = _receipt_path(source_contact["artifacts"]["strict_mask"])
    native_contact = _receipt_path(source_contact["artifacts"]["support_native"])
    reference_root = reference.parent
    object_root = object_mesh.parent
    object_usd_parent = object_usd.parent.parent if object_usd is not None else None
    support_report = args.report_root.resolve() / "clips" / args.clip_id / "support"
    support_asset_root = args.run_root.resolve() / args.clip_id / "support_assets"
    support_asset = support_asset_root / args.clip_id / "support_proxy.usda"
    support_proxy = support_report / "inference" / args.clip_id / "table_proxy.json"
    support_resolution = support_report / "inference" / args.clip_id / "support_resolution.json"
    support_geometry = support_report / "inference" / args.clip_id / "geometry_validation.json"
    support_plane_fit = support_report / "inference" / args.clip_id / "plane_fit.json"
    support_preflight = support_report / "support_preflight.json"
    physics_root = support_report / "physics" / args.clip_id
    contracts_root = args.report_root.resolve() / "clips" / args.clip_id / "physical_contracts"
    final_receipt = support_report / (
        "support_preflight_receipt.json" if args.preflight_only else "support_receipt.json"
    )
    if final_receipt.exists():
        raise FileExistsError(f"INDEPENDENT_SUPPORT_REFUSES_OVERWRITE:{final_receipt}")
    base_geometry = args.base_runtime_geometry_manifest.resolve()
    required_paths = [
        reference,
        reference_v2,
        object_mesh,
        strict_mask,
        native_contact,
        base_geometry,
    ]
    if object_usd is not None:
        required_paths.append(object_usd)
    for path in required_paths:
        _artifact(path)

    steps: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        steps.append(
            _run_step(
                "prepare_support",
                [
                    sys.executable,
                    "scripts/physics/prepare_physical_support.py",
                    "--sequence",
                    args.clip_id,
                    "--support",
                    "auto",
                    "--reference-root",
                    str(reference_root),
                    "--object-root",
                    str(object_root),
                    "--source-sequence-dir",
                    str(Path(str(rows[0]["raw_path"])).resolve()),
                    "--support-asset-root",
                    str(support_asset_root),
                    "--output-root",
                    str(support_report),
                    "--runtime-geometry-manifest",
                    str(base_geometry),
                    "--source-contact-native",
                    str(native_contact),
                    "--static",
                    "--replay",
                ],
                log_root=support_report / "logs",
                expected_artifacts=(support_resolution, support_geometry, support_plane_fit),
            )
        )
        resolution = _json(support_resolution)
        geometry = _json(support_geometry)
        plane_fit = _json(support_plane_fit)
        preflight_reasons: list[str] = []
        if resolution.get("support_type") != "INFERRED_PLANAR_SUPPORT":
            preflight_reasons.append(
                f"SUPPORT_TYPE_NOT_RUNTIME_PLANAR:{resolution.get('support_type')}"
            )
        if plane_fit.get("support_patch_type") != "AREA_SUPPORT":
            preflight_reasons.append(
                f"SUPPORT_PATCH_NOT_AREA:{plane_fit.get('support_patch_type')}"
            )
        if geometry.get("status") != "PASS":
            preflight_reasons.append(f"SUPPORT_GEOMETRY:{geometry.get('status')}")
        if not support_asset.is_file() or not support_proxy.is_file():
            preflight_reasons.append("SUPPORT_RUNTIME_ARTIFACT_MISSING")
        candidate_audit = resolution.get("diagnostics", {}).get("candidate_interval_audit", [])
        later_area = [
            row
            for row in candidate_audit
            if isinstance(row, dict)
            and row.get("status") == "ELIGIBLE"
            and row.get("support_patch_type") == "AREA_SUPPORT"
            and row.get("support_inference_authorized") is not True
        ]
        repair_requirements: list[str] = []
        if plane_fit.get("support_patch_type") in {"POINT_SUPPORT", "EDGE_SUPPORT"}:
            repair_requirements.append("SOURCE_OBJECT_INITIAL_POSE_AUTHORITY_REQUIRED")
            if later_area:
                repair_requirements.append("LATER_AREA_SUPPORT_IS_POST_CONTACT_DIAGNOSTIC_ONLY")
        if resolution.get("support_type") == "UNRESOLVED":
            repair_requirements.append("SOURCE_SUPPORT_OR_UNMODELED_FORCE_AUTHORITY_REQUIRED")
        unmodeled_start = [
            side
            for side, hand in source_contact.get("support_contact_authority", {})
            .get("per_hand", {})
            .items()
            if isinstance(hand, dict)
            and hand.get("modeled_by_target_robot") is False
            and hand.get("contact_or_proximity_at_window_start") is True
        ]
        if unmodeled_start:
            repair_requirements.append(
                "UNMODELED_"
                + "_AND_".join(side.upper() for side in unmodeled_start)
                + "_HAND_CONTACT_SUPPORT"
            )
        hand_geometry = geometry.get("hand_table", {})
        if isinstance(hand_geometry, dict) and hand_geometry.get("status") == "FAIL":
            repair_requirements.append("SUPPORT_AWARE_HAND_RETARGET_REQUIRED")
        atomic_write_json(
            support_preflight,
            {
                "schema_version": "IndependentPhysicalSupportPreflightV1",
                "status": "PASS" if not preflight_reasons else "BLOCKED",
                "clip_id": args.clip_id,
                "contact_mask_used": (resolution.get("stable_interval") or {}).get(
                    "contact_mask_used"
                ),
                "support_type": resolution.get("support_type"),
                "support_patch_type": plane_fit.get("support_patch_type"),
                "support_patch_projected_area_m2": plane_fit.get("support_patch_projected_area_m2"),
                "geometry_status": geometry.get("status"),
                "native_contact_authority": _artifact(native_contact),
                "candidate_interval_audit": candidate_audit,
                "repair_requirements": repair_requirements,
                "reasons": preflight_reasons,
                "gpu_physx_authorized": not preflight_reasons,
            },
        )
        if args.preflight_only:
            preflight_receipt = {
                "schema_version": "IndependentPhysicalSupportPreflightReceiptV1",
                "status": "PASS" if not preflight_reasons else "BLOCKED",
                "clip_id": args.clip_id,
                "selection_manifest_sha256": manifest["manifest_sha256"],
                "source_authority_receipt": _artifact(source_receipt_path),
                "source_authority_schema": source_schema,
                "terminal_scope": "CPU_SUPPORT_PREFLIGHT_ONLY",
                "gpu_physx_authorized": not preflight_reasons,
                "l0_training_authorized": not preflight_reasons,
                "ppo_optimizer_steps": 0,
                "artifacts": {
                    "support_resolution": _artifact(support_resolution),
                    "geometry_validation": _artifact(support_geometry),
                    "plane_fit": _artifact(support_plane_fit),
                    "preflight": _artifact(support_preflight),
                    "native_contact_authority": _artifact(native_contact),
                    "support_proxy": (
                        _artifact(support_proxy) if support_proxy.is_file() else None
                    ),
                    "support_asset": (
                        _artifact(support_asset) if support_asset.is_file() else None
                    ),
                },
                "reasons": preflight_reasons,
                "repair_requirements": repair_requirements,
                "later_area_support_diagnostic": later_area,
                "stages": steps,
                "productive_run_seconds": time.perf_counter() - started,
                "technical_retry_seconds": 0.0,
                "retry_count": 0,
                "cache_hit": False,
            }
            atomic_write_json(final_receipt, preflight_receipt)
            print(json.dumps(preflight_receipt, indent=2, sort_keys=True))
            return 0
        if preflight_reasons:
            raise RuntimeError(
                "INDEPENDENT_SUPPORT_PREFLIGHT_BLOCKED:" + ",".join(preflight_reasons)
            )
        if args.gpu_preflight_receipt is None:
            raise ValueError("GPU_PREFLIGHT_RECEIPT_REQUIRED_BEFORE_SUPPORT_PHYSX")
        gpu_preflight_path = args.gpu_preflight_receipt.resolve()
        validate_gpu_preflight_receipt(gpu_preflight_path)
        assert object_usd is not None and object_usd_parent is not None
        for case in ("with_support", "without_support"):
            command = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "toporetarget-isaaclab",
                "python",
                "scripts/physics/evaluate_physical_support.py",
                "--clip",
                args.clip_id,
                "--case",
                case,
                "--steps",
                "360",
                "--reference",
                str(reference_root),
                "--object-usd",
                str(object_usd_parent),
                "--output",
                str(physics_root / f"{case}.json"),
                "--accept-eula",
            ]
            if case == "with_support":
                command.extend(
                    ("--support-asset", str(support_asset), "--proxy-json", str(support_proxy))
                )
            steps.append(
                _run_step(
                    f"evaluate_{case}",
                    command,
                    log_root=support_report / "logs",
                    expected_artifacts=(physics_root / f"{case}.json",),
                )
            )
        steps.append(
            _run_step(
                "summarize_support",
                [
                    sys.executable,
                    "scripts/physics/summarize_physical_support.py",
                    "--clip",
                    args.clip_id,
                    "--output-root",
                    str(support_report),
                ],
                log_root=support_report / "logs",
                expected_artifacts=(support_report / "final_summary.json",),
            )
        )
        summary = _json(support_report / "final_summary.json")
        if summary.get("overall_status") != "PASS_WITH_TRANSFER_DEFERRED":
            raise RuntimeError("INDEPENDENT_SUPPORT_PHYSICAL_GATE_BLOCKED")
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
                    str(reference),
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
                log_root=support_report / "logs",
                expected_artifacts=(contracts_root / "physical_contract_receipt.json",),
            )
        )
    except BaseException as error:
        atomic_write_json(
            support_report / "support_failure.json",
            {
                "schema_version": "IndependentPhysicalSupportFailureV1",
                "status": "FAIL",
                "clip_id": args.clip_id,
                "reason": f"{type(error).__name__}:{error}",
                "completed_stages": steps,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise

    receipt = {
        "schema_version": "IndependentPhysicalSupportReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "source_policy_receipt": _artifact(source_receipt_path),
        "artifacts": {
            "gpu_preflight": _artifact(gpu_preflight_path),
            "support_proxy": _artifact(support_proxy),
            "support_asset": _artifact(support_asset),
            "with_support": _artifact(physics_root / "with_support.json"),
            "without_support": _artifact(physics_root / "without_support.json"),
            "summary": _artifact(support_report / "final_summary.json"),
            "physical_contracts": _artifact(contracts_root / "physical_contract_receipt.json"),
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


if __name__ == "__main__":
    raise SystemExit(main())
