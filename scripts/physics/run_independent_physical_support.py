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
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--source-policy-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--strict-v4-contract", type=Path, required=True)
    parser.add_argument("--base-runtime-geometry-manifest", type=Path, required=True)
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


def _run_step(name: str, command: list[str], *, log_root: Path) -> dict[str, Any]:
    receipt_path = log_root / f"{name}.receipt.json"
    if receipt_path.is_file():
        previous = _json(receipt_path)
        if previous.get("status") == "PASS" and previous.get("command") == command:
            return {**previous, "resumed_from_pass_receipt": True}
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
    receipt = {
        "stage": name,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "command": command,
        "started_utc": started,
        "ended_utc": _utc(),
        "wall_seconds": time.perf_counter() - tick,
        "returncode": result.returncode,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }
    atomic_write_json(receipt_path, receipt)
    if result.returncode != 0:
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
    if (
        source.get("schema_version") != "IndependentSourcePolicyReceiptV1"
        or source.get("status") != "PASS"
        or source.get("clip_id") != args.clip_id
        or source.get("selection_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("INDEPENDENT_SUPPORT_SOURCE_POLICY_RECEIPT_INVALID")

    source_run = args.run_root.resolve() / args.clip_id / "source_policy"
    reference_root = source_run / "references"
    object_root = source_run / "objects"
    object_usd_parent = source_run / "object_usd"
    contact_root = source_run / "source_contact"
    reference = reference_root / f"{args.clip_id}.world_wrist.stage16.npz"
    reference_v2 = reference_root / f"{args.clip_id}.reference_kinematics_v2.npz"
    object_mesh = object_root / f"{args.clip_id}.obj"
    object_usd = object_usd_parent / args.clip_id / f"{args.clip_id}.usda"
    strict_mask = contact_root / f"strict_source_contact_mask_{args.clip_id}.npz"
    support_report = args.report_root.resolve() / "clips" / args.clip_id / "support"
    support_asset_root = args.run_root.resolve() / args.clip_id / "support_assets"
    support_asset = support_asset_root / args.clip_id / "support_proxy.usda"
    support_proxy = support_report / "inference" / args.clip_id / "table_proxy.json"
    physics_root = support_report / "physics" / args.clip_id
    contracts_root = args.report_root.resolve() / "clips" / args.clip_id / "physical_contracts"
    final_receipt = support_report / "support_receipt.json"
    if final_receipt.exists():
        raise FileExistsError(f"INDEPENDENT_SUPPORT_REFUSES_OVERWRITE:{final_receipt}")
    base_geometry = args.base_runtime_geometry_manifest.resolve()
    strict_contract = args.strict_v4_contract.resolve()
    for path in (
        reference,
        reference_v2,
        object_mesh,
        object_usd,
        strict_mask,
        base_geometry,
        strict_contract,
    ):
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
                    "--static",
                    "--replay",
                ],
                log_root=support_report / "logs",
            )
        )
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
            steps.append(_run_step(f"evaluate_{case}", command, log_root=support_report / "logs"))
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
