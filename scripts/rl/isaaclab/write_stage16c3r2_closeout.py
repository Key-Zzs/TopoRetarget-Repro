#!/usr/bin/env python3
"""Write the fail-closed C.3R2-to-C.5 closeout status from real reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r2_c5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "final_summary.json")
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"STAGE16C3R2_CLOSEOUT_MISSING_EVIDENCE: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _c2_contract(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "num_envs": report["num_envs"],
        "steps": report["steps"],
        "environment_steps_per_s": report["environment_steps_per_s"],
        "no_wrist_root_write_during_step": report["contract"]["wrist_root_state_writes_during_step"]
        == 0,
        "no_object_rollout_write": report["contract"]["object_rollout_state_writes"] == 0,
    }


def main() -> int:
    args = parse_args()
    frozen = _read(REPORT_ROOT / "frozen_baseline.json")
    platform = _read(REPORT_ROOT / "preflight/platform_short_smoke/final_summary.json")
    c2_1 = _read(REPORT_ROOT / "contact/c2_1_after_object_centric_profile.json")
    c2_128 = _read(REPORT_ROOT / "contact/c2_128_after_object_centric_profile.json")
    contact = _read(REPORT_ROOT / "contact/c3_contact_readout_summary.json")
    c3_0 = _read(REPORT_ROOT / "c3/c3_0_fully_kinematic.json")
    input_hash_match = _read(REPORT_ROOT / "preflight/input_hash_verify.json")
    reference_gate_passed = c3_0["status"] == "C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED"
    contact_gate_passed = contact["status"] == "C3_CONTACT_READOUT_VALIDATED"
    result = {
        "status": (
            "STAGE16C3R2_CLOSEOUT_READY_FOR_WRIST"
            if reference_gate_passed and contact_gate_passed
            else "STAGE16C3R2_CLOSEOUT_BLOCKED_REFERENCE_FRAME"
        ),
        "input_freeze": {
            "baseline_status": frozen["status"],
            "post_c1_hash_status": input_hash_match["status"],
        },
        "platform_short_smoke": {
            "status": platform["status"],
            "scope": platform["qualification_scope"],
            "c0_qualification_status_unchanged": platform["c0_qualification_status_unchanged"],
        },
        "c2_contact_enabled_profile": {
            "one_env": _c2_contract(c2_1),
            "vector_128": _c2_contract(c2_128),
        },
        "contact_readout": {
            "status": contact["status"],
            "precision": contact["precision"],
            "passes": contact["passes"],
        },
        "c3_0_fully_kinematic": {
            "status": c3_0["status"],
            "mode": c3_0["mode"],
            "tolerances": c3_0["tolerances"],
            "clips": c3_0["clips"],
        },
        "wrist_architecture": {
            "path_a": "NOT_RUN_GATE_BLOCKED_BY_C3_REFERENCE_FRAME",
            "path_b": "NOT_RUN_GATE_BLOCKED_BY_C3_REFERENCE_FRAME",
            "decision": (
                "No actuator architecture is selected. C3-0 must be repaired before the "
                "fixed Path A/Path B decision tree can begin."
            ),
        },
        "downstream": {
            "c3_semantic": "NOT_RUN_GATE_BLOCKED_BY_C3_REFERENCE_FRAME",
            "contact_momentum_causality": "NOT_RUN_GATE_BLOCKED_BY_C3_REFERENCE_FRAME",
            "c4_gpu_benchmark": "NOT_RUN_GATE_BLOCKED_BY_C3",
            "c5_physx_oracle": "NOT_RUN_GATE_BLOCKED_BY_C3",
            "c6_gpu_ppo": "NOT_AUTHORIZED",
            "ppo_started": False,
        },
        "next_required_recovery": {
            "gate": "C3_REFERENCE_OR_FRAME_CONTRACT_FAILURE",
            "frozen_failure": (
                "hocap_170105 frame=9 r_pinky_distal tracked-link=0.00018068931240122765 m"
            ),
            "rule": (
                "repair frame/reference semantics; do not tune or qualify a wrist controller first"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
