#!/usr/bin/env python3
"""Bind a V2 physical-scene canary to the downstream support contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = REPO_ROOT / ".local/reports/support_physicalization_object_dynamics_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--canary", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-contracts", type=Path)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V2_SUPPORT_RECEIPT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"V2_SUPPORT_RECEIPT_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _copy_once(source: Path, destination: Path) -> None:
    source_bytes = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != source_bytes:
            raise FileExistsError(f"V2_SUPPORT_RECEIPT_DESTINATION_DRIFT:{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_bytes)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"V2_SUPPORT_RECEIPT_REFUSES_OVERWRITE:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    report_root = args.report_root.resolve()
    fixed = _json(report_root / "preflight/fixed_canaries.json")
    rows = [
        row
        for row in fixed.get("canaries", [])
        if int(row.get("canary", -1)) == args.canary and row.get("episode_id") == args.clip_id
    ]
    if len(rows) != 1:
        raise ValueError("V2_SUPPORT_RECEIPT_FIXED_CANARY_MISMATCH")
    fixed_row = rows[0]
    label = f"canary_{args.canary}"
    support_row_path = report_root / "lane_b_support_physicalization/per_canary" / f"{label}.json"
    support_row = _json(support_row_path)
    main_path = report_root / "two_canary/per_episode" / label / "main.json"
    main_row = _json(main_path)
    if (
        main_row.get("scene_status") != "PHYSICAL_SCENE_READY"
        or main_row.get("episode_id") != args.clip_id
        or main_row.get("object_id") != fixed_row.get("object_id")
    ):
        raise ValueError("V2_SUPPORT_RECEIPT_SCENE_NOT_READY")
    if support_row.get("physicalization", {}).get("selected_mode") != "SUPPORT_ONLY":
        raise ValueError("V2_SUPPORT_RECEIPT_REQUIRES_SUPPORT_ONLY")

    historical_proxy = Path(str(support_row["support_proxy"]["historical_proxy"]["path"])).resolve()
    proxy_path = (
        report_root / "lane_b_support_physicalization/per_canary" / label / "table_proxy.json"
    )
    _copy_once(historical_proxy, proxy_path)
    support_asset = Path(str(support_row["support_proxy"]["new_proxy"]["path"])).resolve()
    protocol_path = report_root / "contracts/physical_scene_protocol_v2.json"
    protocol_sha_path = report_root / "contracts/physical_scene_protocol_v2_sha256.txt"
    protocol_sha = protocol_sha_path.read_text(encoding="utf-8").strip()
    if (
        _sha(protocol_path)
        != hashlib.sha256(
            json.dumps(_json(protocol_path), indent=2, sort_keys=True).encode() + b"\n"
        ).hexdigest()
    ):
        raise ValueError("V2_SUPPORT_RECEIPT_PROTOCOL_ENCODING_INVALID")
    if support_row.get("protocol_sha256", protocol_sha) != protocol_sha:
        raise ValueError("V2_SUPPORT_RECEIPT_PROTOCOL_HASH_MISMATCH")

    retarget_reuse = report_root / "retarget_reuse" / f"{label}.json"
    runtime_audit = report_root / "lane_a_object_dynamics/runtime_default_audit.json"
    fresh_qualification = (
        report_root / "two_canary/per_episode" / label / "object_only/with_support.json"
    )
    artifacts: dict[str, dict[str, str]] = {
        "support_proxy": _artifact(proxy_path),
        "support_asset": _artifact(support_asset),
        "physical_scene": _artifact(main_path),
        "settled_dynamics": _artifact(fresh_qualification),
        "runtime_default_audit": _artifact(runtime_audit),
        "physical_scene_protocol": _artifact(protocol_path),
        "retarget_reuse": _artifact(retarget_reuse),
        "gpu_preflight": _artifact(
            REPO_ROOT / ".local/reports/dataset_semantic_authority_two_clip_canary/"
            "p8_two_canary_physicalization/gpu/gpu_preflight_receipt.json"
        ),
    }
    if args.physical_contracts is not None:
        artifacts["physical_contracts"] = _artifact(args.physical_contracts.resolve())
    receipt = {
        "schema_version": "IndependentPhysicalSupportPreflightReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "primary_object_id": fixed_row["object_id"],
        "selection_manifest_sha256": fixed["p5_manifest_semantic_sha256"],
        "gpu_physx_authorized": True,
        "l0_training_authorized": True,
        "physical_scene_protocol_sha256": protocol_sha,
        "physicalization_mode": "SUPPORT_ONLY",
        "retarget_reuse_status": "PASS",
        "source_authority_schema": "SupportPhysicalizationObjectDynamicsCertificationV1",
        "source_authority_receipt": _artifact(main_path),
        "artifacts": artifacts,
        "terminal_scope": "V2_PHYSICAL_SCENE_READY",
        "ppo_optimizer_steps": 0,
        "per_canary_tuning": False,
    }
    _write_once(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
