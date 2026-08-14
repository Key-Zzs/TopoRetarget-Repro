#!/usr/bin/env python3
"""Finalize the P3-B.6 physical scene and RSI evidence receipt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_p3b6_scene_rsi_requalification"
CLIPS = ("hocap_170105", "hocap_170650")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    offline = read_json(REPORT_ROOT / "offline_summary.json")
    dynamic = {
        clip: read_json(REPORT_ROOT / clip / "dynamic_reset_qualification.json") for clip in CLIPS
    }
    joint = {
        clip: read_json(REPORT_ROOT / clip / "joint_zero_replay_qualification.json")
        for clip in CLIPS
    }
    formal_failures = [
        clip for clip in CLIPS if not bool(offline[clip]["formal_trajectory_geometry_valid"])
    ]
    dynamic_failures = {
        clip: int(dynamic[clip]["dynamic_state_count"])
        - int(dynamic[clip]["dynamic_safe_state_count"])
        for clip in CLIPS
    }
    decision = "P3_RESTART_BLOCKED_REFERENCE_GEOMETRY"
    blockers = [
        "REFERENCE_TRAJECTORY_FORMAL_GEOMETRY_GATE_FAILED",
        "JOINT_ZERO_REPLAY_NOT_AUTHORIZED",
    ]
    if any(dynamic_failures.values()):
        blockers.append("DYNAMIC_RESET_SUPPORT_OR_OBJECT_STABILITY_REJECTIONS_PRESENT")
    if any(
        any(
            row.get("first_termination_reason") == "FAILURE_JOINT_LIMIT"
            for chunk in dynamic[clip]["chunks"]
            for row in read_json(Path(chunk["path"])).get("rows", [])
        )
        for clip in CLIPS
    ):
        blockers.append("DYNAMIC_RESET_JOINT_LIMIT_FAILURES_PRESENT")
    final = {
        "schema_version": "PhysicalSceneRSIFinalReceiptV1",
        "decision": decision,
        "ppo_started": False,
        "ppo_gravity_training_decision": "NO_PPO_GRAVITY_TRAINING",
        "offline": offline,
        "dynamic": dynamic,
        "joint": joint,
        "blockers": blockers,
        "formal_geometry_failed_clips": formal_failures,
        "dynamic_reset_failed_state_count": dynamic_failures,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "final_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision_payload = {
        "schema_version": "P3RestartDecisionV1",
        "decision": decision,
        "next_step": "REFERENCE_GEOMETRY_REPAIR",
        "ppo_started": False,
        "blockers": blockers,
        "evidence": {
            "offline_summary": str((REPORT_ROOT / "offline_summary.json").resolve()),
            "dynamic_reports": {
                clip: str((REPORT_ROOT / clip / "dynamic_reset_qualification.json").resolve())
                for clip in CLIPS
            },
            "joint_reports": {
                clip: str((REPORT_ROOT / clip / "joint_zero_replay_qualification.json").resolve())
                for clip in CLIPS
            },
        },
    }
    (REPORT_ROOT / "p3_restart_decision.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    handoff = """# P3-B.6 Physical Scene and RSI Requalification

Decision: `P3_RESTART_BLOCKED_REFERENCE_GEOMETRY`

Both clips were scanned over all 321 reference frames using the complete 21-body
Wuji collision reconstruction, the exact python-fcl runtime manifest, finite
inferred support, and the frozen H-O/H-T/O-T/inter-finger gates. The formal
reference trajectory geometry gate fails on both clips. The support-aware
offline banks and dynamic PhysX reset receipts are retained as evidence; the
dynamic bank is not a blacklist.

PhysX dynamic reset used 1g, nominal friction, active finite table actors, zero
residual actions, and zero rollout object/wrist-root state writes. Joint zero
replay started at the earliest physically valid PRE_CONTACT frame with four
replicas, but both traces stopped at a runtime `FAILURE_JOINT_LIMIT` and
therefore remain `FULL_FRAME_ZERO_REPLAY_NOT_AUTHORIZED`.

PPO was not started. Next step: `REFERENCE_GEOMETRY_REPAIR`.
"""
    (REPORT_ROOT / "handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps(decision_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
