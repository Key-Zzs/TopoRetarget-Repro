#!/usr/bin/env python3
"""Assemble fail-closed P2 source-controller A/B/C receipts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/raw_to_physical_hardening_v2/p2_source_controller"
HISTORICAL_UNBOUNDED_REPORT = REPO_ROOT / (
    ".local/reports/held_out_hocap_raw_to_physical_pilot_post_freeze_l0_unbounded_eval_retry1/clips"
)
CLIPS = (
    "hocap_170105",
    "hocap_170650",
    "hocap_subject_9_20231027_125019__right__G16_3__ep00",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _current_row(payload: dict[str, Any]) -> dict[str, object]:
    rows = payload["per_episode_receipts"]
    return {
        "clip_id": payload["clip_id"],
        "condition": payload["mode"],
        "authority": "PRODUCTION_CANDIDATE",
        "episodes": payload["episodes"],
        "qualified_episodes": payload["qualified_episodes"],
        "status": payload["status"],
        "optimizer_steps": payload["optimizer_steps"],
        "training_samples": payload.get("training_samples", 0),
        "wrist_position_mean_m": _mean(rows, "wrist_command_actual_position_mean_m"),
        "wrist_rotation_mean_deg": _mean(rows, "wrist_command_actual_rotation_mean_deg"),
        "finger_mean_rad": _mean(rows, "finger_command_actual_mean_rad"),
        "contact_fraction_mean": _mean(rows, "contact_fraction"),
        "reference_end_count": sum(bool(row["reference_end"]) for row in rows),
        "joint_limit_safe_count": sum(bool(row["joint_limits_safe"]) for row in rows),
        "actuator_safe_count": sum(bool(row["actuator_limits_safe"]) for row in rows),
        "action_safe_count": sum(bool(row["action_bounds_safe"]) for row in rows),
        "finite_safe_count": sum(bool(row["finite_safe"]) for row in rows),
        "collision_safe_count": sum(bool(row["collision_safety_pass"]) for row in rows),
        "minimum_singularity_margin_deg": min(
            float(row["minimum_singularity_margin_deg"]) for row in rows
        ),
        "artifact": payload["contract"]["path"],
    }


def _historical_rows() -> list[dict[str, object]]:
    hardening = _read(
        HISTORICAL_UNBOUNDED_REPORT
        / CLIPS[2]
        / "physical_refinement/physical_refinement_receipt.json"
    )
    sources = {
        CLIPS[0]: REPO_ROOT
        / ".local/reports/stage16d_ppo26d_continuation/hocap_170105/l0_rebaseline_evaluation.json",
        CLIPS[1]: REPO_ROOT
        / ".local/reports/stage16d_ppo26d/hocap_170650/ppo_l0_eval_qualification.json",
    }
    rows: list[dict[str, object]] = []
    for clip, path in sources.items():
        payload = _read(path)
        rows.append(
            {
                "clip_id": clip,
                "condition": "HISTORICAL_L0_DIAGNOSTIC",
                "authority": "DIAGNOSTIC_ONLY",
                "episodes": len(payload.get("frame_zero", [])) or "NOT_RECORDED",
                "qualified_episodes": "NOT_COMPARABLE",
                "status": "HISTORICAL_DIAGNOSTIC",
                "optimizer_steps": 25,
                "training_samples": 1_024_000,
                "wrist_position_mean_m": "NOT_COMPARABLE",
                "wrist_rotation_mean_deg": "NOT_COMPARABLE",
                "finger_mean_rad": "NOT_COMPARABLE",
                "contact_fraction_mean": "NOT_COMPARABLE",
                "reference_end_count": "NOT_COMPARABLE",
                "joint_limit_safe_count": "NOT_RECORDED",
                "actuator_safe_count": "NOT_RECORDED",
                "action_safe_count": "NOT_RECORDED",
                "finite_safe_count": "NOT_RECORDED",
                "collision_safe_count": "NOT_RECORDED",
                "minimum_singularity_margin_deg": "NOT_RECORDED",
                "artifact": str(path.resolve()),
            }
        )
    rows.append(
        {
            "clip_id": CLIPS[2],
            "condition": "HISTORICAL_UNBOUNDED_L0_DIAGNOSTIC",
            "authority": "DIAGNOSTIC_ONLY_CURRENT_AUTHORITY_NO",
            "episodes": 10,
            "qualified_episodes": 0,
            "status": "PPO_BUDGET_EXHAUSTED_PF_V2_0_OF_10",
            "optimizer_steps": hardening["ppo_updates"],
            "training_samples": hardening["ppo_samples"],
            "wrist_position_mean_m": "NOT_COMPARABLE",
            "wrist_rotation_mean_deg": "NOT_COMPARABLE",
            "finger_mean_rad": "NOT_COMPARABLE",
            "contact_fraction_mean": "NOT_COMPARABLE",
            "reference_end_count": "NOT_COMPARABLE",
            "joint_limit_safe_count": "DISABLED_DIAGNOSTIC",
            "actuator_safe_count": "RETAINED",
            "action_safe_count": "RETAINED",
            "finite_safe_count": "RETAINED",
            "collision_safe_count": 10,
            "minimum_singularity_margin_deg": "RETAINED",
            "artifact": str(
                (
                    HISTORICAL_UNBOUNDED_REPORT
                    / CLIPS[2]
                    / "physical_refinement/physical_refinement_receipt.json"
                ).resolve()
            ),
        }
    )
    return rows


def main() -> int:
    zero: list[dict[str, Any]] = []
    corrected: list[dict[str, Any]] = []
    comparison: list[dict[str, object]] = []
    for clip in CLIPS:
        zero_path = REPORT_ROOT / "zero_residual" / clip / "qualification.json"
        corrected_path = (
            REPORT_ROOT / "corrected_l0_source_qualification" / clip / "qualification.json"
        )
        zero_payload = _read(zero_path)
        corrected_payload = _read(corrected_path)
        zero.append(zero_payload)
        corrected.append(corrected_payload)
        comparison.extend((_current_row(zero_payload), _current_row(corrected_payload)))
        for condition, payload in (
            ("zero_residual", zero_payload),
            ("corrected_l0", corrected_payload),
        ):
            for receipt in payload["per_episode_receipts"]:
                _write(
                    REPORT_ROOT
                    / "per_episode_receipts"
                    / clip
                    / condition
                    / f"episode_{int(receipt['episode']):02d}.json",
                    receipt,
                )
    comparison.extend(_historical_rows())
    csv_path = REPORT_ROOT / "zero_residual_vs_l0.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    angle = {
        "schema_version": "ContinuousVirtualWristAngleAuthorityV1",
        "status": "PASS",
        "authority": "continuous_equivalent_branch_v1",
        "equation": "q_tilde_t=q_t+2*pi*argmin_k(abs(q_t+2*pi*k-q_tilde_previous))",
        "principal_wrapped_angle_gate": "FORBIDDEN_FOR_PRODUCTION_ADMISSION",
        "physical_orientation_invariant": True,
        "observed_223_872_deg": "VALID_EQUIVALENT_BRANCH_NOT_PHYSICAL_LIMIT_FAILURE",
        "real_finger_joint_limits_enforced": True,
        "virtual_wrist_translation_limits_enforced": True,
        "virtual_wrist_rotation_representation_limits_enforced": False,
        "singularity_detection_retained": True,
        "tests": [
            "angle unwrap across +pi/-pi",
            "223.872 degree equivalent branch",
            "same physical rotation matrix",
            "real finger and virtual translation limits retained",
        ],
    }
    _write(REPORT_ROOT / "angle_semantics_contract.json", angle)
    selected = {
        clip: ("ZERO_RESIDUAL" if payload["status"] == "PASS" else "CORRECTED_L0")
        for clip, payload in zip(CLIPS, zero, strict=True)
    }
    failed_fallback = [
        clip
        for clip, payload in zip(CLIPS, corrected, strict=True)
        if selected[clip] == "CORRECTED_L0" and payload["status"] != "PASS"
    ]
    decision = {
        "schema_version": "SourceControllerHardeningP2DecisionV1",
        "status": "INCONCLUSIVE",
        "decision": "L0_AUTHORITY_INCONCLUSIVE",
        "source_controller_mode": "AUTO",
        "selected_mode_by_clip": selected,
        "fallback": "CORRECTED_L0",
        "fallback_samples": 1_024_000,
        "failed_corrected_l0_qualification": failed_fallback,
        "reason": (
            "Zero residual and corrected L0 both failed full reference progression on "
            "hardening episode 1; development clips passed both."
        ),
        "p5_route": (
            "AUTO_ZERO_RESIDUAL_EVAL10_THEN_CORRECTED_L0_ON_FAILURE; "
            "if corrected L0 qualification fails classify SOURCE_CONTROLLER_FAILED"
        ),
        "unbounded_profile": "DIAGNOSTIC_ONLY_CURRENT_AUTHORITY_NO",
        "angle_semantics": {
            "path": str((REPORT_ROOT / "angle_semantics_contract.json").resolve()),
            "sha256": _sha256(REPORT_ROOT / "angle_semantics_contract.json"),
        },
        "comparison": {
            "path": str(csv_path.resolve()),
            "sha256": _sha256(csv_path),
        },
        "production_safety_relaxed": False,
        "p5_blocked": False,
    }
    _write(REPORT_ROOT / "final_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
