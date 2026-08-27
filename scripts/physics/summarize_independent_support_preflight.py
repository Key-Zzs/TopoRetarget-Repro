#!/usr/bin/env python3
"""Aggregate five fail-closed independent support preflight receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_episode_manifest,
    atomic_write_json,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_SUPPORT_PREFLIGHT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _validated_artifact(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file() or row.get("sha256") != sha256_file(path):
        raise ValueError("INDEPENDENT_SUPPORT_PREFLIGHT_ARTIFACT_DRIFT")
    return path


def _row(*, clip_id: str, receipt_path: Path, selection_manifest_sha256: str) -> dict[str, Any]:
    receipt = _json(receipt_path)
    if (
        receipt.get("schema_version") != "IndependentPhysicalSupportPreflightReceiptV1"
        or receipt.get("clip_id") != clip_id
        or receipt.get("selection_manifest_sha256") != selection_manifest_sha256
        or receipt.get("status") not in {"PASS", "BLOCKED"}
        or int(receipt.get("ppo_optimizer_steps", -1)) != 0
        or receipt.get("terminal_scope") != "CPU_SUPPORT_PREFLIGHT_ONLY"
    ):
        raise ValueError(f"INDEPENDENT_SUPPORT_PREFLIGHT_RECEIPT_INVALID:{clip_id}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"INDEPENDENT_SUPPORT_PREFLIGHT_ARTIFACTS_INVALID:{clip_id}")
    for name in (
        "support_resolution",
        "geometry_validation",
        "plane_fit",
        "preflight",
        "native_contact_authority",
    ):
        value = artifacts.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"INDEPENDENT_SUPPORT_PREFLIGHT_ARTIFACT_MISSING:{clip_id}:{name}")
        _validated_artifact(value)
    preflight = _json(_validated_artifact(artifacts["preflight"]))
    status = str(receipt["status"])
    authorized = status == "PASS"
    if (
        bool(receipt.get("gpu_physx_authorized")) is not authorized
        or bool(receipt.get("l0_training_authorized")) is not authorized
        or preflight.get("status") != status
    ):
        raise ValueError(f"INDEPENDENT_SUPPORT_PREFLIGHT_DECISION_MISMATCH:{clip_id}")
    return {
        "clip_id": clip_id,
        "status": status,
        "support_type": preflight.get("support_type"),
        "support_patch_type": preflight.get("support_patch_type"),
        "support_patch_projected_area_m2": preflight.get("support_patch_projected_area_m2"),
        "geometry_status": preflight.get("geometry_status"),
        "contact_mask_used": preflight.get("contact_mask_used"),
        "l0_training_authorized": authorized,
        "gpu_physx_authorized": authorized,
        "reasons": list(receipt.get("reasons") or []),
        "repair_requirements": list(receipt.get("repair_requirements") or []),
        "later_area_support_diagnostic": list(receipt.get("later_area_support_diagnostic") or []),
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
        },
    }


def build_aggregate(*, manifest: dict[str, Any], report_root: Path) -> dict[str, Any]:
    assert_frozen_episode_manifest(manifest)
    rows = [
        _row(
            clip_id=str(item["clip_id"]),
            receipt_path=(
                report_root
                / "clips"
                / str(item["clip_id"])
                / "support"
                / "support_preflight_receipt.json"
            ),
            selection_manifest_sha256=str(manifest["manifest_sha256"]),
        )
        for item in manifest["clips"]
    ]
    authorized = [row["clip_id"] for row in rows if row["status"] == "PASS"]
    blocked = [row["clip_id"] for row in rows if row["status"] == "BLOCKED"]
    overall = (
        "PASS"
        if len(authorized) == len(rows)
        else "BLOCKED"
        if not authorized
        else "PARTIAL_PASS_FAIL_CLOSED"
    )
    return {
        "schema_version": "IndependentPhysicalSupportPreflightAggregateV1",
        "status": overall,
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "clip_count": len(rows),
        "pass_count": len(authorized),
        "blocked_count": len(blocked),
        "l0_training_authorized_clips": authorized,
        "gpu_physx_authorized_clips": authorized,
        "blocked_clips": blocked,
        "rows": rows,
        "downstream": {
            "source_policy_l0": "AUTHORIZED_FOR_PASS_ROWS_ONLY",
            "support_physx": "AUTHORIZED_FOR_PASS_ROWS_AFTER_SOURCE_POLICY_V3_ONLY",
            "frozen_physical_evaluation": "NOT_RUN",
            "grouped_multiplicative_rse_ppo": "NOT_RUN",
            "standalone_strict_v4_ppo": "FORBIDDEN",
        },
        "prohibited_changes": {
            "per_clip_tuning": False,
            "support_extent_enlarged_to_pass": False,
            "friction_tuned_to_pass": False,
            "geometry_threshold_relaxed": False,
            "object_pose_edited_to_pass": False,
        },
    }


def _markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Independent held-out HOCap support preflight",
        "",
        f"Status: `{aggregate['status']}`.",
        "",
        "| Clip | Decision | Patch | Geometry | L0/GPU | Reasons |",
        "|---|---|---|---|---|---|",
    ]
    for row in aggregate["rows"]:
        reasons = ", ".join(row["reasons"]) or "none"
        lines.append(
            f"| {row['clip_id']} | {row['status']} | {row['support_patch_type']} | "
            f"{row['geometry_status']} | "
            f"{'AUTHORIZED' if row['gpu_physx_authorized'] else 'NOT_RUN'} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "Only PASS rows may resume source-policy L0. Blocked rows are terminal "
            "for the current frozen support authority and must not enter physical PPO.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"INDEPENDENT_SUPPORT_PREFLIGHT_REFUSES_OVERWRITE:{output}")
    aggregate = build_aggregate(manifest=manifest, report_root=args.report_root.resolve())
    atomic_write_json(output, aggregate)
    output.with_suffix(".md").write_text(_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
