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


def read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {"status": "NOT_RUN"}


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
    environment_qualification = read_optional_json(root / "environment_qualification.json")
    resource_use = read_optional_json(root / "resource_use.json")
    t1 = read_optional_json(root / "hocap_t1_training.json")
    t2 = read_optional_json(root / "hocap_t2_training.json")
    t3 = read_optional_json(root / "hocap_t3_training.json")
    nominal_evaluation = read_optional_json(root / "hocap_t3_nominal_evaluation.json")
    robust_evaluation = read_optional_json(root / "hocap_t3_robust_evaluation.json")
    kinematic_reports = [
        read_optional_json(root / "hocap_170105_kinematic_replay.json"),
        read_optional_json(root / "hocap_170650_kinematic_replay.json"),
    ]
    eligible = bool(inventory["accepted_dynamic_hocap_robot_reference_available"])
    functional_complete = (
        eligible
        and t3["status"] == "HOCAP_REFERENCE_PPO_BOUNDED_FUNCTIONAL_PASS"
        and nominal_evaluation["status"] == "HOCAP_REFERENCE_POLICY_EVALUATION_COMPLETE"
        and robust_evaluation["status"] == "HOCAP_REFERENCE_POLICY_EVALUATION_COMPLETE"
    )
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
    if functional_complete:
        reference_validation = {
            "status": "REFERENCE_VALIDATION_ACCEPTED_TWO_CLIP_PASS",
            "reference_is_dynamic": True,
            "sampling_hz": 20,
            "raw_contactpose_accepted": False,
            "source": inventory,
            "reason": None,
        }
        kinematic = {
            "status": "E0_REFERENCE_CONTRACT_REPLAY_TWO_CLIP_PASS",
            "kinematic_replay_is_rl_result": False,
            "reports": kinematic_reports,
            "reason": None,
        }
        reward["real_reference_validation"] = "EXECUTED_BOUNDED_FUNCTIONAL"
        randomization["physical_robustness_result"] = "EXECUTED_BOUNDED_FUNCTIONAL"
        training = {
            "status": "SINGLE_CLIP_TRACKING_COMPLETE",
            "physical_training": True,
            "report": t1,
            "reason": None,
        }
        robust = {
            "status": "SINGLE_CLIP_ROBUSTNESS_COMPLETE",
            "physical_training": True,
            "report": t2,
            "reason": None,
        }
        multi = {
            "status": "MULTI_CLIP_TRAINING_COMPLETE",
            "physical_training": True,
            "report": t3,
            "reason": None,
        }
        nominal = nominal_evaluation["summary"]
        hocap = {
            "status": "HOCAP_BOUNDED_TWO_CLIP_EVALUATION_COMPLETE",
            "required_episode_count": 32,
            "actual_episode_count": nominal["episode_count"],
            "functional_clip_count": len(inventory.get("accepted_references", [])),
            "selection": selection,
            "success_rate": nominal["success_rate"],
            "tracking_error_cm": nominal["object_position_error_cm_all"],
            "drop_rate": 1.0 - nominal["success_rate"],
            "nominal_evaluation": nominal_evaluation,
            "robust_evaluation": robust_evaluation,
            "reason": "Functional two-clip protocol only; paper HOCap-32 protocol was not run.",
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
        "status": (
            "STAGE16_FUNCTIONAL_HOCAP_COMPLETE_PAPER_EVALUATION_PARTIAL"
            if functional_complete
            else "STAGE16_IMPLEMENTATION_COMPLETE_EVALUATION_PARTIAL"
        ),
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
            "NOT_COMPARABLE: bounded two-clip CPU protocol, not HOCap-32 or Pen-Spin."
            if functional_complete
            else "NOT_COMPARABLE: no physical HOCap evaluation episodes or private Pen-Spin data."
        ),
        "next_gate": (
            "Optional paper-scale expansion requires an approved 32-clip HOCap selection."
            if functional_complete
            else (
                "operator approval for new final jobs plus an accepted post-source-contract "
                "dynamic HOCap RobotReference and object collision asset"
            )
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
            (
                f"- HOCap evaluation: `{hocap['status']}` "
                f"({hocap['actual_episode_count']}/{hocap['required_episode_count']} episodes)."
            ),
            f"- Pen-Spin: `{penspin_status}`; no substitute was used.",
            f"- Remaining scope boundary: {hocap['reason'] or blocked_reason}",
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
                "- Optional paper-scale expansion requires a 32-clip HOCap selection; this "
                "functional run uses exactly the two user-approved clips."
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
