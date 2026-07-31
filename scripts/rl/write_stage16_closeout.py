#!/usr/bin/env python3
"""Write fail-closed Stage-16 closeout artifacts from executed evidence.

This tool intentionally writes explicit BLOCKED/UNAVAILABLE results instead of
inventing physical-training or HOCap-evaluation metrics when their input gate
is not satisfied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".local/reports/stage16_reference_tracking_ppo"),
    )
    parser.add_argument("--test-status", default="NOT_RUN")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    inventory = read_json(root / "reference_inventory.json")
    environment = read_json(root / "environment_validation.json")
    ppo = read_json(root / "ppo_smoke.json")
    pd = read_json(root / "pd_qualification.json")
    selection = read_json(root / "reference_selection.json")
    penspin = read_json(root / "penspin_availability.json")
    penspin_status = penspin.get("status", penspin.get("penspin_status", "UNKNOWN"))
    recovery = read_json(root / "recovery_summary.json")
    environment_qualification = (
        read_json(root / "environment_qualification.json")
        if (root / "environment_qualification.json").is_file()
        else {"status": "NOT_RUN"}
    )
    resource_use = (
        read_json(root / "resource_use.json")
        if (root / "resource_use.json").is_file()
        else {"status": "NOT_RUN"}
    )
    eligible = bool(inventory["accepted_dynamic_hocap_robot_reference_available"])
    now = datetime.now(UTC).isoformat()
    blocked_reason = inventory["blocker"]

    reference_validation = {
        "status": "REFERENCE_VALIDATION_BLOCKED" if not eligible else "REFERENCE_VALIDATION_READY",
        "reference_is_dynamic": eligible,
        "sampling_hz": 20 if eligible else None,
        "raw_contactpose_accepted": False,
        "source": inventory,
        "reason": None if eligible else blocked_reason,
    }
    kinematic = {
        "status": "KINEMATIC_REPLAY_BLOCKED" if not eligible else "PENDING_EXECUTION",
        "kinematic_replay_is_rl_result": False,
        "reason": None if eligible else blocked_reason,
    }
    reward = {
        "status": "REWARD_IMPLEMENTATION_UNIT_TESTED",
        "paper_reward": (
            "Table 4 literal components are unit tested; no HOCap reference replay was run."
        ),
        "real_reference_validation": "BLOCKED" if not eligible else "PENDING_EXECUTION",
    }
    randomization = {
        "status": "TABLE5_RANDOMIZATION_IMPLEMENTATION_UNIT_TESTED",
        "physical_robustness_result": "BLOCKED" if not eligible else "PENDING_EXECUTION",
        "reason": None if eligible else blocked_reason,
    }
    training = {
        "status": "SINGLE_CLIP_TRACKING_BLOCKED" if not eligible else "PENDING_EXECUTION",
        "physical_training": False,
        "numerical_smoke": ppo["status"],
        "reason": None if eligible else blocked_reason,
    }
    robust = {
        "status": "SINGLE_CLIP_ROBUSTNESS_BLOCKED" if not eligible else "PENDING_EXECUTION",
        "physical_training": False,
        "reason": None if eligible else blocked_reason,
    }
    multi = {
        "status": "MULTI_CLIP_TRAINING_BLOCKED" if not eligible else "PENDING_EXECUTION",
        "physical_training": False,
        "reason": None if eligible else blocked_reason,
    }
    hocap = {
        "status": "HOCAP_32_EVALUATION_BLOCKED" if not eligible else "PENDING_EXECUTION",
        "required_episode_count": 32,
        "actual_episode_count": 0,
        "selection": selection,
        "success_rate": None,
        "tracking_error": None,
        "drop_rate": None,
        "reason": None if eligible else blocked_reason,
    }
    for filename, payload in {
        "reference_validation.json": reference_validation,
        "kinematic_replay.json": kinematic,
        "kinematic_reference_replay.json": kinematic,
        "reward_validation.json": reward,
        "randomization_validation.json": randomization,
        "single_clip_training.json": training,
        "single_clip_robust.json": robust,
        "single_clip_robustness.json": robust,
        "multi_clip_training.json": multi,
        "hocap_32_evaluation.json": hocap,
        "hocap_evaluation.json": hocap,
    }.items():
        write_json(root / filename, payload)

    with (root / "evaluation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol",
                "required_episode_count",
                "actual_episode_count",
                "success_rate",
                "tracking_error",
                "drop_rate",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "protocol": "HOCap_32",
                "required_episode_count": 32,
                "actual_episode_count": 0,
                "success_rate": "",
                "tracking_error": "",
                "drop_rate": "",
                "status": hocap["status"],
                "reason": hocap["reason"],
            }
        )

    fidelity = {
        "status": "PAPER_PROTOCOL_REPRODUCTION_WITH_ASSUMPTIONS",
        "paper_exact_components": [
            "base-frame reference",
            "residual action",
            "observation layout",
            "Table 4 reward and termination",
            "Table 5 ranges",
            "Table 6 architecture and listed PPO values",
        ],
        "engineering_assumptions": [
            "MuJoCo 3.3.6 correctness backend",
            "tracked-link profile and 5cm object-axis offsets",
            "PD gain qualification candidates",
            "undisclosed PPO clip/value/gradient settings",
        ],
        "unresolved": [
            "author simulator/control",
            "author-exact HOCap clip IDs",
            penspin_status,
        ],
    }
    write_json(root / "paper_fidelity_ledger.json", fidelity)
    test_report = {"status": args.test_status, "scope": "repository test/quality commands"}
    write_json(root / "tests.json", test_report)
    write_json(root / "test_summary.json", test_report)
    summary = {
        "generated_at_utc": now,
        "status": "STAGE16_IMPLEMENTATION_COMPLETE_EVALUATION_PARTIAL",
        "branch_scope": "feature/reference-tracking-ppo",
        "git": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "base": git_value("merge-base", "HEAD", "origin/main"),
        },
        "environment": environment,
        "environment_qualification": environment_qualification,
        "ppo_numerical_smoke": ppo,
        "pd_qualification": pd,
        "reference_validation": reference_validation,
        "hocap_32_evaluation": hocap,
        "penspin": penspin,
        "recovery": recovery,
        "resource_use": resource_use,
        "comparison_to_paper": (
            "NOT_COMPARABLE: no physical HOCap evaluation episodes or private Pen-Spin data."
        ),
        "next_gate": (
            "operator approval for new final jobs plus an accepted post-source-contract "
            "dynamic HOCap RobotReference and object collision asset"
        ),
    }
    write_json(root / "final_summary.json", summary)
    summary_md = "\n".join(
        [
            "# Stage 16 closeout",
            "",
            f"- Status: `{summary['status']}`",
            (
                f"- E2 generic free-object smoke: `{environment['status']}`; "
                "this is not a HOCap evaluation."
            ),
            (
                f"- T0 numerical PPO: `{ppo['status']}` with "
                f"`{ppo['checkpoint_validation']['status']}`."
            ),
            f"- HOCap 32 evaluation: `{hocap['status']}` (0/32 episodes).",
            f"- Pen-Spin: `{penspin_status}`; no substitute was used.",
            f"- Blocker: {blocked_reason}",
            (
                "- Comparison to paper: not comparable; physical-reference protocol results "
                "were not fabricated."
            ),
            "",
        ]
    )
    (root / "final_summary.md").write_text(summary_md, encoding="utf-8")
    handoff = "\n".join(
        [
            "# Stage 16 handoff",
            "",
            "## Completed",
            "",
            "- Branch closeout and dedicated feature branch creation.",
            (
                "- Appendix A.5 MDP/reward/termination/DR/PPO implementation, MuJoCo "
                "free-object smoke, and numerical PPO checkpoint/reload."
            ),
            "",
            "## Gate remaining",
            "",
            f"- {blocked_reason}",
            (
                "- Obtain approved dynamic 20 Hz RobotReference and matching object collision "
                "asset before PD qualification or training."
            ),
            "",
            "## Non-claims",
            "",
            "- Generic free-object smoke is not HOCap evaluation.",
            "- Numerical PPO smoke is not a trained physical policy.",
            "- Pen-Spin data are unavailable and were not substituted.",
            "",
        ]
    )
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    integrity_paths = sorted(
        path for path in root.iterdir() if path.is_file() and path.name != "artifact_integrity.json"
    )
    integrity = {
        "status": "ARTIFACT_INTEGRITY_PASS",
        "generated_at_utc": now,
        "files": [{"path": str(path), "sha256": sha256(path)} for path in integrity_paths],
    }
    write_json(root / "artifact_integrity.json", integrity)
    print(json.dumps({"status": summary["status"], "root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
