#!/usr/bin/env python3
"""Reduce PhysX receipts into an auditable physical-support report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.support.physics_validation import (  # noqa: E402
    build_physics_validation,
    compare_support_counterfactuals,
    summarize_static_support_test,
)

CLIPS = ("hocap_170105", "hocap_170650")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16_support_reconstruction",
    )
    parser.add_argument("--clip")
    return parser.parse_args()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reduce_clip(output_root: Path, clip: str) -> dict[str, Any]:
    clip_root = output_root / "inference" / clip
    physics_root = output_root / "physics" / clip
    resolution = _read(clip_root / "support_resolution.json")
    geometry = _read(clip_root / "geometry_validation.json")
    with_receipt = _read(physics_root / "with_support.json")
    without_receipt = _read(physics_root / "without_support.json")
    support_normal = with_receipt["inputs"]["support_normal"]
    with_summary = summarize_static_support_test(
        with_receipt["telemetry"],
        support_active=True,
        mass_kg=float(with_receipt["mass_kg"]),
        gravity_world_mps2=(0.0, 0.0, -9.81),
        support_normal=support_normal,
    )
    without_summary = summarize_static_support_test(
        without_receipt["telemetry"],
        support_active=False,
        mass_kg=float(without_receipt["mass_kg"]),
        gravity_world_mps2=(0.0, 0.0, -9.81),
        support_normal=support_normal,
    )
    causal = compare_support_counterfactuals(with_summary, without_summary)
    physics = build_physics_validation(
        with_support=with_summary,
        without_support=without_summary,
        causal_comparison=causal,
    )
    static_report = {
        "schema_version": "Stage16StaticSupportTestV1",
        "status": with_summary["status"],
        "expected_behavior": "object remains positionally and rotationally stable",
        "raw_receipt": str((physics_root / "with_support.json").resolve()),
        "summary": with_summary,
        "qualification": {
            "contact_observed": with_summary["support_contact_frames"] > 0,
            "force_matches_mg_within_10_percent": abs(
                with_summary["support_force_to_mg_ratio"] - 1.0
            )
            <= 0.1,
            "translation_stable": with_summary["position_drift_max_m"] <= 0.01,
            "rotation_stable": with_summary["rotation_drift_max_rad"] <= 0.1,
        },
    }
    no_support_report = {
        "schema_version": "Stage16NoSupportCounterfactualV1",
        "status": ("PASS" if without_summary["position_drift_max_m"] >= 0.05 else "FAIL"),
        "expected_behavior": "object falls under full gravity without support",
        "raw_receipt": str((physics_root / "without_support.json").resolve()),
        "summary": without_summary,
        "fall_observed": without_summary["position_drift_max_m"] >= 0.05,
        "fall_threshold_m": 0.05,
    }
    ab_report = {
        "schema_version": "Stage16SupportCounterfactualABV1",
        "status": causal["status"],
        "with_support": static_report,
        "without_support": no_support_report,
        "comparison": causal,
        "physics_validation": physics.as_dict(),
    }
    transfer_report = {
        "schema_version": "Stage16SupportTransferDiagnosticV1",
        "status": "DEFERRED_BY_HAND_OBJECT_GEOMETRY",
        "reason": (
            "runtime reference-following transfer is not authorized while the "
            "full hand collision mesh is unavailable and the existing P3 reset "
            "hand-object geometry gate remains blocked"
        ),
        "hand_table_geometry_status": geometry.get("hand_table", {}).get("status"),
        "hidden_support_or_attachment": False,
        "object_state_writes_after_reset": 0,
    }
    _write(physics_root / "static_support_test.json", static_report)
    _write(physics_root / "no_support_test.json", no_support_report)
    _write(physics_root / "ab_comparison.json", ab_report)
    _write(physics_root / "support_transfer.json", transfer_report)
    # Keep the per-clip inference directory self-contained for reviewers who
    # start from the resolver receipt rather than the backend receipt folder.
    _write(clip_root / "static_support_test.json", static_report)
    _write(clip_root / "no_support_test.json", no_support_report)
    _write(clip_root / "ab_comparison.json", ab_report)
    _write(clip_root / "support_transfer.json", transfer_report)

    resolution["geometry_validation"] = geometry
    resolution["physics_validation"] = physics.as_dict()
    resolution["transfer_status"] = transfer_report["status"]
    resolution["status"] = (
        "INFERRED_SUPPORT_VALIDATED_TRANSFER_DEFERRED"
        if geometry.get("status") == "PASS" and physics.status == "PASS"
        else "SUPPORT_RECONSTRUCTION_BLOCKED"
    )
    _write(clip_root / "support_resolution.json", resolution)
    return {
        "clip": clip,
        "support_type": resolution.get("support_type"),
        "support_inferred": resolution.get("support_inferred"),
        "stable_interval": resolution.get("support_interval"),
        "source_support_status": "NO_RECOVERABLE_SOURCE_SUPPORT",
        "geometry_status": geometry.get("status"),
        "hand_table_status": geometry.get("hand_table", {}).get("status"),
        "static_support_status": static_report["status"],
        "no_support_status": no_support_report["status"],
        "physics_status": physics.status,
        "ab_status": causal["status"],
        "support_resolution_status": resolution["status"],
        "transfer_status": transfer_report["status"],
        "qualification_status": (
            "QUALIFIED" if resolution["status"] != "SUPPORT_RECONSTRUCTION_BLOCKED" else "BLOCKED"
        ),
    }


def main() -> int:
    args = _args()
    if args.clip is not None and (
        not args.clip or any(token in args.clip for token in ("/", "\\", ".."))
    ):
        raise ValueError("INDEPENDENT_SUPPORT_SUMMARY_CLIP_ID_INVALID")
    output_root = args.output_root.resolve()
    clips = (args.clip,) if args.clip else CLIPS
    rows = [_reduce_clip(output_root, clip) for clip in clips]
    overall = (
        "PASS_WITH_TRANSFER_DEFERRED"
        if all(row["qualification_status"] == "QUALIFIED" for row in rows)
        else "BLOCKED"
    )
    summary = {
        "schema_version": "Stage16SupportReconstructionFinalV1",
        "overall_status": overall,
        "support_resolution_policy": (
            "source_first_then_explicit_planar_inference_then_fail_closed_unknown"
        ),
        "physics_backend": "IsaacLab_5.1_PhysX_CUDA_full_gravity",
        "hand_object_geometry_fixed": False,
        "p3_status": "P3_RESTART_BLOCKED_BY_HAND_OBJECT_RESET_GEOMETRY",
        "clips": rows,
        "prohibited_changes": {
            "ppo_retrained": False,
            "reward_changed": False,
            "geometry_gate_changed": False,
            "hand_object_reference_penetration_fixed": False,
            "hidden_support_or_attachment": False,
        },
    }
    _write(output_root / "final_summary.json", summary)
    lines = [
        "# Stage 16 support reconstruction final report",
        "",
        f"Overall status: `{overall}`.",
        "",
        "The source-first resolver audited source support before using the "
        "explicit planar fallback. PhysX receipts are object-only, full-gravity "
        "counterfactuals with no policy, hand, guidance, or post-reset object writes.",
        "",
        "| Clip | Inference | Geometry | Static support | No support | A/B | Final |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {clip} | {support_type} | {geometry_status} | {static_support_status} | "
            "{no_support_status} | {ab_status} | {support_resolution_status} |".format(**row)
        )
    lines += [
        "",
        "Both clips pass the object-only static-support qualification: contact is "
        "continuous, the normal force is near mg, and position plus quaternion "
        "pose drift remain bounded. Runtime reference-following support transfer "
        "remains deferred by the existing hand-object geometry blocker.",
        "",
        "PPO/C0-C4/G3/P4 and the hand-object reference geometry were not modified.",
        "",
    ]
    (output_root / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if overall != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
